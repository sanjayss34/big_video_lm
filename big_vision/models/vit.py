# Copyright 2025 big_video_lm authors - Changes for flash attention and LLaVA support
# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from typing import Optional, Sequence, Union
import functools
import inspect
import math

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
from flax.linen.attention import combine_masks
from flax.linen.module import compact
import flax.training.checkpoints
import jax
from jax import lax
import jax.numpy as jnp
from jax._src.core import ShapedArray
from big_vision.flash_attention import flash_attention, BlockSizes
from jax.experimental.shard_map import shard_map
import numpy as np
import scipy.ndimage
from einops import rearrange
from easydel.etils.errors import EasyDeLBlockWiseFFNError

from jax_array_info import sharding_info
from scalax.sharding import with_sharding_annotation

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]

def posemb_sincos_1d(seqlen, width, temperature=10_000., dtype=jnp.float32):
  y = jnp.mgrid[:seqlen]
  assert width % 2 == 0
  omega = jnp.arange(width // 2) / (width // 2 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]

def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  elif typ == "sincos1d":
    return posemb_sincos_1d(seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")

def flash_attention_output(query, key, value, attention_bias):
  return flash_attention(query, key, value, attention_bias)[0]

def block_wise_ffn(remat_ffn, inputs, chunk_size: int, deterministic: bool, axis: int = 1):
  generating = inputs.shape[1] == 1
  try:
    if generating:
      return remat_ffn(inputs, deterministic)
    else:
      scan_axis = None
      if axis == 1:
        inputs = rearrange(inputs, "b (c n) d -> b c n d", c=chunk_size)
        scan_axis = 2
      else:
        assert axis == 0
        inputs = rearrange(inputs, "(b c) n d -> b c n d", c=chunk_size)
        scan_axis = 0

      def scan_ffn(remat_ffn_, carry, hidden_states):
        outputs = remat_ffn_(hidden_states, deterministic)
        return carry, outputs

      _, output = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
      )(remat_ffn, None, inputs)
      if axis == 1:
        output = rearrange(output, "b c n d -> b (c n) d")
      else:
        output = rearrange(output, "b c n d -> (b c) n d")
      return output
  except Exception as e:
      raise EasyDeLBlockWiseFFNError(
          "You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
          "input arguments in wrong way in case that you don't want to use this just pass `use_scan_mlp=False` in "
          "model config or in config_kwargs in AutoEasyDeLModelForCausalLM or change `scan_mlp_chunk_size` "
          f"in configs for more information read Docs.\nOriginal Error\n{e}"
      )

class LlavaMlp(nn.Module):
  mlp_dim: int
  dtype_mm: str = "float32"
  # precision: Optional[jax.lax.Precision] = None
  inits: dict = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    x1 = x = nn.Dense(self.mlp_dim, dtype=self.dtype_mm, name="head1", **self.inits)(x)
    x = nn.gelu(x, approximate=False)
    x = nn.Dense(self.mlp_dim, dtype=self.dtype_mm, name="head2", **self.inits)(x)
    return x

class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype_mm: str = "float32"
  precision: Optional[jax.lax.Precision] = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x1 = x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits, precision=self.precision)(x)
    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype_mm, **inits, precision=self.precision)(x)
    return x

class DenseWithDeterministic(nn.Dense):
  @nn.compact
  def __call__(self, x, deterministic=True):
    return super().__call__(x)

class MultiHeadDotProductAttentionCombinedProj(nn.MultiHeadDotProductAttention):
  mesh: Optional[jax.sharding.Mesh] = None
  
  @compact
  def __call__(
    self,
    inputs_q,
    inputs_k = None,
    inputs_v = None,
    *,
    inputs_kv = None,
    mask = None,
    deterministic = None,
    dropout_rng = None,
    sow_weights = False,
  ):
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. '
            'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
            'value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads


    dense = functools.partial(
      nn.Dense,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      features=qkv_features, # (self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query_layer = dense(name='query')
    query, key, value = (
      query_layer(inputs_q).reshape(*inputs_q.shape[:-1], self.num_heads, head_dim),
      dense(name='key')(inputs_k).reshape(*inputs_k.shape[:-1], self.num_heads, head_dim),
      dense(name='value')(inputs_v).reshape(*inputs_v.shape[:-1], self.num_heads, head_dim),
    )
    query = nn.with_logical_constraint(query, ("flattened_images", "act_patches", "act_emb", "act_emb2"))
    key = nn.with_logical_constraint(key, ("flattened_images", "act_patches", "act_emb", "act_emb2"))
    value = nn.with_logical_constraint(value, ("flattened_images", "act_patches", "act_emb", "act_emb2"))

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = nn.LayerNorm(
        name='query_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(query)  # type: ignore[call-arg]
      key = nn.LayerNorm(
        name='key_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[int | jax.Array, ...] = (zero,) * len(
          batch_dims
        ) + (
          cur_index,
          zero,
          zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index,
            tuple(batch_dims) + (1, 1, max_length),
          ),
        )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # `qk_attn_weights_einsum` and `attn_weights_value_einsum` are optional
    # arguments that can be used to override the default `jnp.einsum`. They
    # exist for quantized einsum support in AQT.
    qk_attn_weights_einsum = (
        self.qk_attn_weights_einsum_cls()
        if self.qk_attn_weights_einsum_cls
        else None
    )
    attn_weights_value_einsum = (
        self.attn_weights_value_einsum_cls()
        if self.attn_weights_value_einsum_cls
        else None
    )
    # apply attention
    attn_args = (query, key, value)
    # This kwargs list match the default nn.dot_product_attention.
    # For custom `attention_fn`s, invalid kwargs will be filtered.
    attn_kwargs = dict(
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=m_deterministic,
      dtype=self.dtype,
      precision=self.precision,
      force_fp32_for_softmax=self.force_fp32_for_softmax,
      qk_attn_weights_einsum=qk_attn_weights_einsum,
      attn_weights_value_einsum=attn_weights_value_einsum,
    )
    attn_kwargs = {
        k: v
        for k, v in attn_kwargs.items()
        if k in inspect.signature(self.attention_fn).parameters
    }
    block_size = 128
    attention_bias = jnp.concatenate((
      jnp.zeros((query.shape[0], 1, query.shape[1], query.shape[1]), dtype=query.dtype),
      -10000.*jnp.ones((query.shape[0], 1, query.shape[1], block_size-(query.shape[1] % block_size)), dtype=query.dtype)
    ), axis=-1)
    attention_bias = jnp.concatenate((
      attention_bias,
      -10000.*jnp.ones((query.shape[0], 1, block_size-(query.shape[1] % block_size), attention_bias.shape[-1]), dtype=query.dtype)
    ), axis=-2)
    attention_bias = nn.with_logical_constraint(attention_bias, ('flattened_images', 'act_emb1', 'act_patches', 'act_patches2'))
    query = jnp.transpose(query, (0, 2, 1, 3))
    key = jnp.transpose(key, (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))
    query_padded = jnp.concatenate((
      query,
      jnp.zeros((query.shape[0], query.shape[1], block_size-(query.shape[2] % block_size), query.shape[3]), dtype=query.dtype),
    ), axis=2)
    key_padded = jnp.concatenate((
      key,
      jnp.zeros((key.shape[0], key.shape[1], block_size-(key.shape[2] % block_size), key.shape[3]), dtype=key.dtype),
    ), axis=2)
    value_padded = jnp.concatenate((
      value,
      jnp.zeros((value.shape[0], value.shape[1], block_size-(value.shape[2] % block_size), value.shape[3]), dtype=value.dtype),
    ), axis=2)
    flash_attn_1head = shard_map(
        functools.partial(flash_attention, sm_scale=1.0/math.sqrt(query.shape[-1])),
        mesh=self.mesh,
        in_specs=(
          jax.sharding.PartitionSpec(('fsdp', 'sequence'), 'tensor', None, None),
          jax.sharding.PartitionSpec(('fsdp', 'sequence'), 'tensor', None, None),
          jax.sharding.PartitionSpec(('fsdp', 'sequence'), 'tensor', None, None),
          jax.sharding.PartitionSpec(('fsdp', 'sequence'), None, None, None),
        ),
        out_specs=jax.sharding.PartitionSpec(('fsdp', 'sequence'), 'tensor', None, None),
        check_rep=False,
    )
    x = flash_attn_1head(
      query_padded,
      key_padded,
      value_padded,
      attention_bias
    )
    x = jnp.transpose(x, (0, 2, 1, 3))[:,:query.shape[2],:,:]
    # back to the original inputs dimensions
    x = x.reshape(*x.shape[:-2], -1)
    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))
    out = nn.Dense(
      features=features,
      kernel_init=self.out_kernel_init or self.kernel_init,
      bias_init=self.out_bias_init or self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      name='out',  # type: ignore[call-arg]
    )(x)
    return out

class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"
  built_in_attn_class: bool = True
  precision: Optional[jax.lax.Precision] = None
  mesh: Optional[jax.sharding.Mesh] = None
  mlp_chunk_size: int = None
  remat_policy: str = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))
    y = nn.LayerNorm()(x)
    out["+ln"] = y
    attn_class = nn.MultiHeadDotProductAttention if self.built_in_attn_class else MultiHeadDotProductAttentionCombinedProj
    y = out["sa"] = attn_class(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
        name="MultiHeadDotProductAttention_0",
        precision=self.precision,
        **({'mesh': self.mesh} if not self.built_in_attn_class else {}),
    )(y, y)
    y = nn.with_logical_constraint(y, ("flattened_images", "act_patches", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    mlp_block = None
    if self.remat_policy is None:
      mlp_block = MlpBlock(
          mlp_dim=self.mlp_dim, dropout=self.dropout,
          dtype_mm=self.dtype_mm,
          precision=self.precision,
      )
    else:
      mlp_block = nn.remat(
          MlpBlock,
          prevent_cse=True,
          static_argnums=(2,),
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )(mlp_dim=self.mlp_dim, dropout=self.dropout, dtype_mm="bfloat16", precision=self.precision, name="MlpBlock_0")
    if self.mlp_chunk_size is None:
      y = out["mlp"] = mlp_block(y, deterministic)
    else:
      y = out["mlp"] = block_wise_ffn(mlp_block, y, self.mlp_chunk_size, deterministic, axis=1)
    y = nn.with_logical_constraint(y, ("flattened_images", "act_patches", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  built_in_attn_class: bool = True
  precision: Optional[jax.lax.Precision] = None
  mesh: Optional[jax.sharding.Mesh] = None
  mlp_chunk_size: int = None
  post_layer_norm: bool = True

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    sharding_info(x, "vit encoder input")

    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(2,),  # 0=self, 2=deterministic
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      x, scan_out = nn.scan(
          block,
          variable_axes={"params": 0},
          split_rngs={"params": True, "dropout": True},
          in_axes=nn.broadcast,
          length=self.depth)(
              name="encoderblock",
              dtype_mm=self.dtype_mm,
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout=self.dropout,
              built_in_attn_class=self.built_in_attn_class,
              precision=self.precision,
              mesh=self.mesh,
              mlp_chunk_size=self.mlp_chunk_size,
              remat_policy=self.remat_policy)(x, deterministic)
      for lyr in range(self.depth):
        out[f"block{lyr:02d}"] = jax.tree.map(lambda o, l=lyr: o[l], scan_out)
    else:
      block = nn.remat(
        Encoder1DBlock,
        prevent_cse=True,
        static_argnums=(2,),
        policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
      )
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = block( # Encoder1DBlock(
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim, num_heads=self.num_heads,
            dropout=self.dropout,
            built_in_attn_class=self.built_in_attn_class,
            precision=self.precision,
            mesh=self.mesh,
            mlp_chunk_size=self.mlp_chunk_size,
            remat_policy=None) # self.remat_policy)
        x, block_out = block_cur(x, deterministic)
        if lyr == self.depth-1:
          out[f"block{lyr:02d}"] = block_out

    final_encoder_out = nn.LayerNorm(name="encoder_norm")(x) if self.post_layer_norm else x
    return final_encoder_out, out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  pool_mlp2xgelu: bool = False
  head_zeroinit: bool = False # True
  add_image_newline: bool = False
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  built_in_attn_class: bool = True
  precision: str = "default"
  img_token_pooling: dict = None
  mesh: Optional[jax.sharding.Mesh] = None
  mlp_chunk_size: int = None

  @nn.compact
  def __call__(self, image, *, train=False, **kwargs):
    out = {}

    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction
    x = out["stem"] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding", dtype=self.dtype_mm, precision=utils.PRECISION_MAP[self.precision])(image)
    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        built_in_attn_class=self.built_in_attn_class,
        precision=utils.PRECISION_MAP[self.precision] if self.precision is not None else None,
        mesh=self.mesh,
        mlp_chunk_size=self.mlp_chunk_size,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    elif self.pool_type == "none":
      pass
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    
    x = nn.with_logical_constraint(x, ("flattened_images", "act_patches", "act_emb"))
    x_2d = nn.with_logical_constraint(x_2d, ("flattened_images", "act_patches_height", "act_patches_width", "act_emb"))
    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x
    sharding_info(x, "pre_logits")

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      kw['precision'] = utils.PRECISION_MAP[self.precision]
      if self.pool_mlp2xgelu:
        dense_block = nn.remat(
            DenseWithDeterministic,
            # LlavaMlp,
            # prevent_cse=False,
            static_argnums=(1,),  # 0=self
            policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
        )
        print('self.remat_policy', self.remat_policy)

        head_layer1 = dense_block(self.num_classes, name="head1", **kw)
        head_layer2 = dense_block(self.num_classes, name="head2", **kw)
        max_layer_num = max([int(key.split('block')[1]) for key in out['encoder'] if 'block' in key])
        x = encoded = out['encoder']['block'+str(max_layer_num)]['+mlp']
        """x_2d = nn.with_logical_constraint(
          head_layer1(x_2d) if self.mlp_chunk_size is None else block_wise_ffn(head_layer1, x_2d, self.mlp_chunk_size, not train),
          ("flattened_images", "act_patches_height", "act_patches_width", "act_emb")
        )"""
        if True: # self.mlp_chunk_size is None:
          x = nn.with_logical_constraint(head_layer1(x), ("flattened_images", "act_patches", "act_emb"))
          x = head_layer2(nn.gelu(x, approximate=False))
        else:
          x = nn.with_logical_constraint(
            block_wise_ffn(head_layer1, x, self.mlp_chunk_size, not train),
            ("flattened_images", "act_patches", "act_emb"),
          )
          x = block_wise_ffn(head_layer2, nn.gelu(x, approximate=False), self.mlp_chunk_size, not train)
        x_2d = out["logits_2d"] = nn.with_logical_constraint(
          jnp.reshape(x, [n, h, w, -1]),
          ("flattened_images", "act_patches_height", "act_patches_width", "act_emb"),
        )
      else:
        head = nn.Dense(self.num_classes, name="head", **kw)
        x_2d = out["logits_2d"] = head(x_2d)
    if self.add_image_newline:
      image_newline = nn.Embed(
          1,
          self.num_classes,
          dtype=self.dtype_mm,
          param_dtype=self.dtype_mm,
          name="image_newline"
      )
      newline_emb = image_newline(jnp.zeros((x.shape[0],), dtype=jnp.int32))
      out["image_newline"] = newline_emb[:,None,:]

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  cfg = {**decode_variant(variant)}
  cfg.update({**kw})
  return _Model(num_classes, **cfg)


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {"mu": 32, "Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "So400m": 1152, "H": 1280, "g": 1408, "g-opt": 1536, "G": 1664, "G-opt": 1536, "e": 1792}[v],
      "depth": {"mu": 1, "Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "So400m": 27, "H": 32, "g": 40, "g-opt": 40, "G": 48, "G-opt": 48, "e": 56}[v],
      "mlp_dim": {"mu": 128, "Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "So400m": 4304, "H": 5120, "g": 6144, "g-opt": 6144, "G": 8192, "G-opt": 8192, "e": 15360}[v],
      "num_heads": {"mu": 2, "Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "So400m": 16, "H": 16, "g": 16, "g-opt": 16, "G": 16, "G-opt": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)
  return grid


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Original ViT paper variant had posemb in a module:
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # Widely used version before 2022 had posemb in Encoder:
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # Old vit.py used to first concat [cls] token, then add posemb.
  # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
  # so we changed to add posemb then concat [cls]. We can recover the old
  # checkpoint by manually summing [cls] token and its posemb entry.
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP-head variants during ViT-G development had it inlined:
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def pyloop_to_scan(params_pyloop):
  """Converts a python for-loop ViT checkpoint to a lax.scan based one."""
  # On a high level, they are the same except that the for loop has separate
  # array pytrees for each encoderblock, while the scan one has just one
  # encoderblock pytree, with all block's params concatenated.

  params_scan = jax.tree.map(lambda x: x, params_pyloop)  # Structural copy
  t = params_scan["Transformer"]

  # Find highest index of encoderblocks in the checkpoint (they start at 0):
  encoderblocks = {k for k in t if k.startswith("encoderblock_")}
  depth = 1 + max({int(k.split("_")[-1]) for k in encoderblocks})

  def stack(*values):
    return np.stack(values)

  # Stack all encoderblocks into a single one:
  t["encoderblock"] = jax.tree.map(
      stack, *[t[f"encoderblock_{lyr}"] for lyr in range(depth)])

  for lyr in range(depth):
    del t[f"encoderblock_{lyr}"]

  return params_scan


def scan_to_pyloop(params_scan):
  """Converts a lax.scan ViT checkpoint to a python for-loop based one."""
  # See comment in pyloop_to_scan.

  params_scan = jax.tree.map(lambda x: x, params_scan)  # Structural copy
  t = params_scan["Transformer"]

  # Find out how many encoderblocks there are
  depth = len(t["encoderblock"]["LayerNorm_0"]["bias"])

  # Create that many encoderblocks, each with their slice of their sub-pytree.
  for lyr in range(depth):
    block = jax.tree.map(lambda x, lyr=lyr: x[lyr], t["encoderblock"])
    t[f"encoderblock_{lyr}"] = block

  del t["encoderblock"]
  return params_scan


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(init_file)

  restored_params = fix_old_checkpoints(restored_params)

  # Detect attempts to load non-scan checkpoint into scan model.
  if (model_cfg.get("scan") and
      "encoderblock" not in restored_params["Transformer"]):
    restored_params = pyloop_to_scan(restored_params)
  if (not model_cfg.get("scan")
      and "encoderblock" in restored_params["Transformer"]):
    restored_params = scan_to_pyloop(restored_params)

  print("image model config", model_cfg)
  if (not model_cfg.add_image_newline) and "image_newline" in restored_params:
    del restored_params["image_newline"]
  
  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load)


  # resample posemb if needed.
  # TODO: Take this from model_cfg to avoid need for init_params.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",

    # SigLIP image encoder checkpoints from https://arxiv.org/abs/2303.15343
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz:img",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz:img",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz:img",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz:img",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz:img",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz:img",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz:img",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz:img",
    # pylint: enable=line-too-long
}
