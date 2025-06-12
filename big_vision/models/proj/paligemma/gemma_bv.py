# Copyright 2025 big_video_lm authors - Changes for supporting non-PaliGemma models
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

"""Gemma wrapper to make it work for us."""

import functools
from typing import Optional

import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
import time
from jax_array_info import sharding_info

import easydel as ed
from easydel.etils.partition_module import PartitionAxis
from easydel.modules.qwen2.qwen_configuration import Qwen2Config

import big_vision.utils as u
from big_vision.models import common
from big_vision.models.ppp import gemma
from big_vision.models.ppp.qwen2 import NewFlaxQwen2ForCausalLMModule as FlaxQwen2ForCausalLMModule # Qwen2Config


def _get_config(model):
  if "gemma" in model.variant:
    config = gemma.get_config(model.variant)
    config.remat_policy = model.remat_policy
    if model.vocab_size is not None:
      config.vocab_size = model.vocab_size
    config.dropout = model.dropout
    config.dropout_bdims = model.dropout_bdims
    config.cache_dtype = model.cache_dtype
    config.canonical_attn_types = model.canonical_attn_types
    config.rope_max_wavelength = model.rope_max_wavelength
  elif "qwen2" in model.variant:
    size = model.variant.split('_')[1].split('b')[0]
    assert size in {'0.5', '7', '72'}
    config = Qwen2Config.from_pretrained(f'Qwen/Qwen2-{size}B-Instruct')
    config.vocab_size = model.vocab_size # 152064
    config.hidden_size = 3584 if size == '7' else (896 if size == '0.5' else 8192)
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.intermediate_size = 18944 if size == '7' else (4864 if size == '0.5' else 29568)
    config.num_attention_heads = 28 if size == '7' else (14 if size == '0.5' else 64)
    config.num_hidden_layers = 28 if size == '7' else (24 if size == '0.5' else 80)
    config.num_key_value_heads = 4 if size == '7' else (2 if size == '0.5' else 8)
    config.rope_theta = 1000000.0
    config.attn_mechanism = model.attention_impl
    config.scan_ring_attention = False
    if "ring" in config.attn_mechanism:
      config.use_scan_mlp = True
      config.scan_mlp_chunk_size = model.scan_mlp_chunk_size
    config.gradient_checkpointing = model.remat_policy
    mesh = model.mesh
    config.axis_names = tuple(mesh.axis_names)
    config.axis_dims = tuple([s for s in mesh.device_ids.shape])
    config.partition_axis = PartitionAxis(batch_axis=('replica', 'fsdp'), sequence_axis='sequence', query_sequence_axis='sequence', head_axis=None, key_sequence_axis='sequence', hidden_state_axis='tensor', attention_dim_axis='tensor', bias_head_sequence_axis=None, bias_key_sequence_axis=None, generation_query_sequence_axis=None, generation_head_axis=None, generation_key_sequence_axis='sequence', generation_attention_dim_axis='tensor')
    config.attention_axis_name = 'sequence'
    # config.convert_to_head_sharding = False
    # config.sliding_window_size = None
    # config.sliding_window_section_start = None
    # config.sliding_window_section_end = None
  else:
    raise NotImplementedError(f"{model.variant} not implemented yet.")
  return config

def _get_width(config):
  if hasattr(config, 'width'):
    return config.width
  return config.hidden_size

@jax.vmap
def _left_to_right_align(x, input_mask, attn_mask):
  """Converts input from left-align to right-aligned."""
  # Due to vmap, this is operating in a single example (not batch level).
  assert x.ndim == 2 and input_mask.ndim == 1 and attn_mask.ndim == 2
  assert x.shape[0] == input_mask.shape[0]
  assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
  seqlen = jnp.sum(input_mask)
  x = jnp.roll(x, -seqlen, axis=0)
  input_mask = jnp.roll(input_mask, -seqlen, axis=0)
  attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
  return x, input_mask, attn_mask

def compute_position_ids_from_segment_ids(segment_ids, prefix_mask, max_segment=100):
   batch_size, seq_length = segment_ids.shape
   # max_segment = jnp.max(segment_ids)
   position_indices = jnp.arange(seq_length, dtype=jnp.int32)

   # Create mask for positions >= prefix_size
   mask = (position_indices >= prefix_mask.shape[1]).astype(int)
   mask = jnp.broadcast_to(mask, (batch_size, seq_length))

   # One-hot encode segment IDs and apply mask
   one_hot_segments = jax.nn.one_hot(segment_ids, num_classes=max_segment, dtype=jnp.int32) # max_segment + 1)
   masked_one_hot = one_hot_segments * jnp.expand_dims(mask, axis=-1)

   # Cumulative counts for each segment in post-prefix regions
   cum_counts = jnp.cumsum(masked_one_hot, axis=1)

   # Gather counts for each position's segment
   indices = jnp.expand_dims(segment_ids, axis=-1)
   selected_counts = jnp.take_along_axis(cum_counts, indices, axis=-1).squeeze(axis=-1)

   # Compute post-prefix positions
   post_positions = jnp.expand_dims(jnp.sum(prefix_mask, axis=-1), axis=-1) + (selected_counts - 1)

   # Compute prefix positions (0 to prefix_size-1)
   prefix_positions = jnp.cumsum(jnp.concatenate((prefix_mask, jnp.ones((batch_size, seq_length-prefix_mask.shape[1]), dtype=prefix_mask.dtype)), axis=1), axis=-1) - 1

   # Combine prefix and post positions
   position_ids = jnp.where(
       position_indices < prefix_mask.shape[1],
       prefix_positions,
       post_positions
   )

   return position_ids

class Model(nn.Module):
  """Wrapping gemma big_vision model."""
  variant: str = "gemma_2b"
  scan: bool = True
  remat_policy: str = "nothing_saveable"
  vocab_size: int | None = None
  true_vocab_size: int | None = None
  rope_max_wavelength: int = 10_000

  dropout: float = 0.0
  dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
  cache_dtype: str | None = "bfloat16"  # bfloat16 to save memory and transfers.
  cache_size: int = None

  canonical_attn_types: tuple[str, ...] | None = None
  mesh: Optional[jax.sharding.Mesh] = None

  image_start_index: int = None
  image_end_index: int = None
  attention_impl: str = "sharded_vanilla"
  scan_mlp_chunk_size: int = 1024


  precision: str = 'default'

  def setup(self):
    # The parent+name avoids an unnecessary nesting in params pytree.
    if 'gemma' in self.variant:
      self.model = gemma.Model(**_get_config(self), parent=self.scope, name="", mesh=self.mesh)
    else:
      assert 'qwen2' in self.variant
      config = _get_config(self)
      self.model = FlaxQwen2ForCausalLMModule(config, parent=self.scope, name="", precision=self.precision)

  def embed_tokens(self, tokens, train=False, image_emb=None):
    # Turns int32[B,T] tokens into float32[B,T,d_model] embeddings.
    # Really just the vocab embedding.
    # if not train:
    #   jax.debug.print('tokens {s}', s=tokens)
    if 'gemma' in self.variant:
      return self.model(tokens, embed_only=True, deterministic=not train)
    assert 'qwen2' in self.variant
    embedding = self.model.model.embed_tokens(tokens.astype("i4"))
    return embedding

  def compute_logits(self, pre_logits, train=False):
    if "gemma" in self.variant:
      return self.model(None, pre_logits=pre_logits, deterministic=not train)[0]
    if pre_logits.shape[1] == 1:
      pre_logits = nn.with_logical_constraint(pre_logits, ("act_batch", "single", "act_emb"))
    logits = self.model.lm_head(pre_logits).astype(jnp.float32)
    return logits

  def __call__(self, embs, mask=None, train=False, extra_input=None, image_tokens=None, segment_ids=None, position_ids=None, input_mask=None):
    # Turns float32[B,T,d_model] embedding sequence to logits.
    # call(emb_tokens(tokens)) should be a forward pass.
    # Allow for specifying int32[B,T,T] attention masks. For convenience
    # default to triangular autorgressive mask when None, but not P0.
    # Return float32[B,T,vocab_size] logits and out-dict.

    batch_size, seq_length, d_model = embs.shape
    assert d_model == self.embdim
    tokens = jnp.zeros([batch_size, 0], dtype=jnp.int32)
    if "gemma" in self.variant:
      logits, out = self.model(
          tokens=tokens,
          embedded_prefix=embs,
          mask=mask,
          deterministic=not train,
          extra_input=extra_input,
          image_tokens=image_tokens,
      )
    else:
      if position_ids is None:
        if segment_ids is not None:
          prefix_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
          prefix_mask = input_mask[:,:self.image_end_index]
          position_ids = compute_position_ids_from_segment_ids(segment_ids, prefix_mask)
        else:
          position_ids = jnp.cumsum(input_mask, axis=-1) - 1

      mask = jnp.expand_dims(mask, axis=1)
      print('mask shape', mask.shape)
      print('tokens embs shapes', tokens.shape, embs.shape)
      mask = nn.with_logical_constraint(mask, ("act_batch", "act_heads", "act_len", "act_len2"))
      model_outputs = self.model.model(
        input_ids=tokens,
        inputs_embeds=embs,
        position_ids=position_ids,
        attention_mask=mask,
        deterministic=not train,
      )
      pre_logits = model_outputs.last_hidden_state
      text_pre_logits = pre_logits[:,self.image_end_index-self.image_start_index:,:]
      text_pre_logits = nn.with_logical_constraint(text_pre_logits, ("act_batch", "act_len", "act_emb"))
      logits = self.compute_logits(text_pre_logits)
      out = {"pre_logits": pre_logits}
      if input_mask is not None:
        last_token_idx = jnp.argmax(jnp.where(input_mask, position_ids, -1*jnp.ones_like(position_ids)), axis=-1)
    return logits, out

  def prefill_cache(self, x, input_mask, attn_mask, position_ids, *, cache_size):
    """Initializes decoding cache with `x` [B, N, E] as prompt.

    IMPORTANT: Inputs MUST be left-aligned and attn_mask should not allow
    input tokens to attend to padding tokens.

    TODO: Relax left-align requirement by converting any input into
    a right aligned input with no attention to padding tokens.

    Args:
      x: float[B, N, E] with prompt tokens.
      input_mask: bool[B, N]. True indicates tokens are part of the prompt.
        False indicates padding tokens. This class doesn't combine this with
        attn_mask, so mask out the attention to padding tokens beforehand.
      attn_mask: bool[B, N, N]. Indicates which tokens can attend to which while
        processing the prompt tokens. During extend_cache tokens, it is assumed
        that tokens can attend all previous valid tokens.
      cache_size: int. Indicates the size of the cache. The prompt will consume
        the first N entries of the cache. Each subsequent extend_cache will
        consume one entry. Behaviour is undefined when prefill_len plus number
        of extend_cache exceeds the cache_size.

    Returns:
      logits of the last valid token (i.e. last logits where input_mask=True).
    """
    # To call the model with decode=True we need to be able to provide:
    #   (a) positions of tokens [B, N], ([B, 1] for extend)
    #   (b) attention mask [B, N, cache_size] ([B, 1, cache_size] for extend)
    #
    # To do so we track how many tokens each example has seen so far, and we
    # align the prompt to the right so that cache usage for each example is in
    # a continuous subsequent of (cache_begin, cache_end] such that cache_end
    # is the same for all sequences (this allows to do faster row updates of
    # the cache during decoding).
    num_text_tokens = 0
    num_image_tokens = self.image_end_index
    if num_image_tokens is None: #  or self.pack_sequence:
      x, input_mask, attn_mask = _left_to_right_align(x, input_mask, attn_mask)
    else:
      num_text_tokens = jnp.sum(input_mask[:,num_image_tokens:], axis=-1)
      sharding_info(x, "x")
      sharding_info(input_mask, "input_mask")
      sharding_info(attn_mask, "attn_mask")
      print('num image tokens', input_mask.shape[1], num_image_tokens)
      x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
      input_mask = nn.with_logical_constraint(input_mask, ("act_batch", "act_len"))
      attn_mask = nn.with_logical_constraint(attn_mask, ("act_batch", "act_len", "act_len2"))

    if self.cache_size is not None:
      cache_size = self.cache_size
    # Track sequence len
    seq_len = jnp.sum(input_mask, axis=-1)
    img_padded_seq_len = seq_len
    if position_ids is not None:
      img_padded_seq_len = jnp.max(
        jnp.where(input_mask, position_ids, jnp.zeros_like(position_ids)),
        axis=-1
      )+1
    self.put_variable("cache", "seq_len", img_padded_seq_len)
    positions = position_ids
    if positions is None:
      positions = jnp.cumsum(input_mask, axis=-1) - 1

    # Initialize cache_begin and cache_end. Note: cache_end is the same for all
    # sequences but we keep it per example to allow easy sharding rules with
    # batch as the first axis.
    batch_size, prefill_len, _ = x.shape
    self.put_variable("cache", "cache_begin", jnp.full((batch_size,), prefill_len, jnp.int32))
    self.put_variable(
        "cache", "cache_end", jnp.full((batch_size,), prefill_len, jnp.int32)
    )

    # Pad attention to set the cache size.
    mask_len = attn_mask.shape[-1]
    print('cache size', cache_size, 'mask_len', mask_len)
    mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, cache_size - mask_len)))
    last_token = jnp.argmax(jnp.where(input_mask, positions, jnp.zeros_like(positions)), axis=-1)[:,None]
    self.put_variable(
        "cache", "mask", mask[jnp.expand_dims(jnp.arange(batch_size), axis=1), last_token]
    )
    full_position_ids = jnp.concatenate((
      positions,
      img_padded_seq_len[:,None]+jnp.arange(cache_size-mask_len)[None,:]
    ), axis=1)
    self.put_variable(
        "cache", "full_position_ids", full_position_ids
    )

    if "gemma" in self.variant:
      _, aux = self.model(
          tokens=None,
          embedded_prefix=x,
          positions=positions,
          mask=mask,
          decode=True,
          image_tokens=x[:,:num_image_tokens]
      )
      pre_logits = aux["pre_logits"]
    elif "qwen2" in self.variant:
      tokens = jnp.zeros([batch_size, 0], dtype=jnp.int32),
      mask = jnp.expand_dims(mask, axis=1)
      pre_logits = self.model.model(
        input_ids=tokens,
        inputs_embeds=x,
        position_ids=positions,
        attention_mask=mask,
        init_cache=True,
      )[0]
    else:
      raise NotImplementedError(f"{self.variant} not implemented yet.")
    shifted_pre_logits = pre_logits
    shifted_input_mask = input_mask
    self.put_variable(
        "cache", "pre_logits", shifted_pre_logits[:,self.image_start_index:self.image_end_index,:],
    )
    self.put_variable(
        "cache", "pre_logits_mask", shifted_input_mask[:,self.image_start_index:self.image_end_index],
    )
    last_pre_logits = pre_logits[jnp.expand_dims(jnp.arange(batch_size), axis=1), last_token]
    logits = self.compute_logits(last_pre_logits[:,-1:])
    return logits

  def extend_cache(self, x, extend_mask=None):
    """Extends decoding cache with `x` [B, 1, E] and returns logits."""
    # assert x.shape[1] == 1, f"Only supports extend the cache by one token but shape is {str(x.shape)}."
    cache_size = self.cache_size
    if cache_size is None:
      if "gemma" in self.variant:
        if self.model.scan:
          cache_size = self.variables["cache"]["layers"]["attn"]["k_cache"].shape[2]
        else:
          cache_size = self.variables["cache"]["layers"]["0"]["attn"]["k_cache"].shape[1]
      elif "qwen2" in self.variant:
        cache_size = self.variables["cache"]["model"]["layers"]["0"]["self_attn"]["cached_key"].shape[1]
      else:
        raise NotImplementedError("Not implemented yet.")
    if extend_mask is None:
      assert x.shape[1] == 1
      extend_mask = jnp.ones((x.shape[0], 1), dtype=jnp.int32)

    # Lookup current token position and increment by one for next call.
    positions = self.get_variable("cache", "seq_len")
    self.put_variable("cache", "seq_len", positions + jnp.sum(extend_mask, axis=1))

    # Update which cache positions are in use and construct attention mask.
    # Tokens can attend to all cache positions which are in use including self.
    cache_begin = self.get_variable("cache", "cache_begin")
    cache_end = self.get_variable("cache", "cache_end") + jnp.sum(extend_mask, axis=1)
    self.put_variable("cache", "cache_end", cache_end)
    mask = jnp.logical_and(
      jnp.arange(cache_size)[None,None,:] < cache_end[:,None,None],
      jnp.arange(cache_size)[None,None,:] >= cache_begin[:, None, None],
    )
    mask = jnp.logical_or(mask, self.get_variable("cache", "mask"))
    causal_extend_mask = jnp.arange(extend_mask.shape[1])[None,:,None]+positions[:,None,None] >= self.get_variable("cache", "full_position_ids")[:,None,:]
    mask = jnp.logical_and(
      mask,
      causal_extend_mask,
    )
    last_token = jnp.sum(extend_mask, axis=1)[:,None] - 1
    self.put_variable("cache", "mask", mask[jnp.arange(last_token.shape[0])[:,None],last_token])

    if "gemma" in self.variant:
      logits, _ = self.model(
          tokens=None, embedded_prefix=x,
          positions=positions[:, None], mask=mask, decode=True)
    elif "qwen2" in self.variant:
      tokens = jnp.zeros([x.shape[0], 0], dtype=jnp.int32)
      mask = jnp.expand_dims(mask, axis=1)
      pre_logits = self.model.model(
        input_ids=tokens,
        inputs_embeds=x,
        position_ids=positions[:,None],
        attention_mask=mask
      )[0]
      last_pre_logits = pre_logits[jnp.arange(pre_logits.shape[0])[:,None], last_token]
      logits = self.compute_logits(last_pre_logits)
    return logits

  @property
  def embdim(self):
    return _get_width(_get_config(self))


# load = gemma.load
def load(init_params, init_file, model_cfg=None, dont_load=()):
  """Loads existing weights."""
  model_cfg = model_cfg or {}
  variant = model_cfg.get("variant", "gemma_2b")
  if "gemma" in variant:
    return gemma.load(init_params, init_file, model_cfg=model_cfg, dont_load=dont_load)
  init_variant = f"{init_file} {variant}"
  params = u.load_params(init_file)

  def extend_rows(emb1, target_rows):
    if (missing_rows := target_rows - emb1.shape[0]) == 0:
      return emb1
    assert missing_rows > 0, "You're asking to shrink vocab?!"
    new_rows = np.random.randn(missing_rows, emb1.shape[1])
    new_rows = (new_rows * 0.02).astype(emb1.dtype)
    new_rows += np.expand_dims(np.asarray(emb1).mean(axis=0), axis=0)
    return np.r_[np.asarray(emb1), new_rows]

  if "vocab_size" in model_cfg:
    params["model"]["embed_tokens"]["embedding"] = extend_rows(
        params["model"]["embed_tokens"]["embedding"][:model_cfg["true_vocab_size"],:],
        model_cfg["vocab_size"],
    )
    params["lm_head"]["kernel"] = np.transpose(
        extend_rows(
            np.transpose(params["lm_head"]["kernel"][:,:model_cfg["true_vocab_size"]]),
            model_cfg["vocab_size"]
        )
    )
  print('vocab size', params["model"]["embed_tokens"]["embedding"].shape)
  if "copy_interval" in model_cfg:
    if model_cfg["copy_interval"] > 0:
      dont_load = dont_load+("copy_layer/bias", "copy_layer/kernel", "copy_layer_text/bias", "copy_layer_text/kernel")

  return common.merge_params(params, init_params, dont_load)
