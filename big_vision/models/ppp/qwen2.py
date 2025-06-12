# Copyright 2025 big_video_lm authors - modified to work with 4D attention mask and the custom attention module

# Copyright 2025 Easydel Authors
# Licensed under the Apache 2.0 License

import math
from typing import Optional, Tuple, Union

import numpy as np
import chex
import fjformer
import flax.linen
from flax.linen.dtypes import promote_dtype
import jax
import jax.numpy as jnp
import einops
from fjformer import linen as nn
from fjformer.linen import Dense
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxCausalLMOutput,
    FlaxSequenceClassifierOutput,
)

from easydel.modules.common import RMSNorm as RMSNorm
from easydel.modules.easydel_modelling_utils import EasyDeLFlaxPretrainedModel

from easydel.modules.flax_modelling_utils import (
    BaseJAXAttentionModule,
    apply_rotary_pos_emb,
    control_mlp_sharding,
    get_dot_general_by_bits,
    get_gradient_checkpoint_policy,
    precompute_freq_cis,
    with_sharding_constraint,
    quantize_kv_cache,
    dequantize_kv_cache,
)
from easydel.modules.qwen2.modelling_qwen_flax import *
from easydel.etils.errors import EasyDeLBlockWiseFFNError
from easydel.modules.qwen2.qwen_configuration import Qwen2Config as Qwen2OriginalConfig
from jax_array_info import sharding_info

from big_vision.models.ppp.easydel_attention import AttentionModule, AttentionOutput


class NewFlaxQwen2Attention(FlaxQwen2Attention):

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads
        self.q_proj = Dense(
            config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.k_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.v_proj = Dense(
            config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
        self.o_proj = Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )

        self.rotary = FlaxQwen2Embedding(self.dtype)
        self.attention_performer = AttentionModule(
            use_sharding_constraint=self.config.use_sharding_constraint,
            block_k_major=self.config.block_k_major,
            block_b=self.config.block_b,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
            block_q_major_dkv=self.config.block_q_major_dkv,
            block_k_major_dkv=self.config.block_k_major_dkv,
            block_k_major_dq=self.config.block_k_major_dq,
            block_k_dkv=self.config.block_k_dkv,
            block_q_dkv=self.config.block_q_dkv,
            block_q_dq=self.config.block_q_dq,
            block_k_dq=self.config.block_k_dq,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            head_dims=self.head_dim,
            shard_attention_computation=self.config.shard_attention_computation,
            precision=self.precision,
            force_float32_tpu=True,
            attn_mechanism=self.config.attn_mechanism,
            dtype=self.config.attn_dtype,
            partition_axis=self.config.partition_axis,
            scan_ring_attention=self.config.scan_ring_attention,
            mesh=self.config.get_mesh(),
            sm_scale=1 / math.sqrt(self.head_dim),
            axis_name=self.config.attention_axis_name,
            backward_pass_impl=self.config.flash_attention_backward_pass_impl,
        )
        self.resid_dropout = flax.linen.Dropout(rate=config.resid_pdrop)

    def __call__(
        self,
        hidden_states: chex.Array,
        freq_cis: Tuple[chex.Array, chex.Array],
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array,
        segment_ids: Optional[chex.Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        """The __call__ function is the main function of a JAX module. It defines how the module behaves when called
        with inputs. The __call__ function can be thought of as a &quot;forward pass&quot; through the model,
        and it should return all outputs that are needed for training or inference.

        Args:
            self: Access variables that belong to the class
            hidden_states: chex.Array: Pass the hidden states of the
                previous layer
            freq_cis: Tuple[chex.Array, chex.Array],: Pass in the
                frequency coefficients for each position
            attention_mask: chex.Array: Mask out certain tokens in the
                input sequence
            position_ids: chex.Array: Determine the position of each
                token in a sequence
            causal_mask: chex.Array: Mask out the future tokens in the
                decoder
            deterministic: bool: Determine whether to use dropout or not
            init_cache: bool: Initialize the cache
            output_attentions: bool: Determine whether to return the
                attention weights or not
            fcm_mask: Mask out the attention weights between the input
                and output tokens
        :param : Determine if the attention is causal or not

        Returns:
            A tuple of two arrays
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size, sequence_length, self.config.num_attention_heads, self.head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim
        )

        query_states, key_states, value_states = self.apply_rotary(
            query=query_states,
            key=key_states,
            value=value_states,
            position_ids=position_ids,
            freq_cis=freq_cis,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        query_length, key_length = query_states.shape[1], key_states.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"][0]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal = False
        if attention_mask.ndim == 2:
            causal_mask = jnp.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)
            if attention_mask.ndim == 2:
                attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            causal = True

        dropout_rng = None

        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        if self.has_variable("cache", "cached_key") or init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )
        attention_mask = attention_mask[:,:,:query_length,:key_states.shape[1]]

        key_states, value_states = self.repeat_key_value(key_states, value_states, self.num_key_value_groups)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                self.dtype
            ),
        )
        query_length, key_length = query_states.shape[1], key_states.shape[1]

        num_query_padding_heads = 0
        num_kv_padding_heads = 0
        attentions = self.attention_performer.__call__(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            bias=attention_bias,
            attention_mask=attention_mask,
            causal=causal,
            dropout_rng=dropout_rng,
            deterministic=deterministic,
            query_sequence_length=query_length,
            key_value_sequence_length=key_length,
            uses_cache=self.has_variable("cache", "cached_key") or init_cache,
            segment_ids=segment_ids,
            causal_mask=causal_mask if causal else None,
        )

        attentions = AttentionOutput(
            attention_weights=attentions.attention_weights, attention_outputs=with_sharding_constraint(attentions.attention_outputs[:,:,:-num_query_padding_heads,:], PartitionSpec(("dp", "fsdp"), "sp" if attentions.attention_outputs.shape[1] != 1 else None, "tp")) if num_query_padding_heads > 0 else attentions.attention_outputs
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        if self.config.shard_attention_computation:
            attn_output = with_sharding_constraint(
                attn_output,
                PartitionSpec(
                    ("dp", "fsdp"), "sp" if attn_output.shape[1] != 1 else None, "tp"
                ),
            )
        attn_output = self.o_proj(attn_output)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (
            (attn_output, attentions.attention_weights)
            if output_attentions
            else (attn_output,)
        )
        return outputs

    @nn.compact
    def _concatenate_to_cache(self, key, value, query_states, attention_mask):
        """The _concatenate_to_cache function is used to concatenate the key and value vectors
        of a query_states with those of previous queries. This allows for the attention mechanism to
        look at all previous queries when computing its output. The function takes in three
        arguments: key, value, and query_states. It also uses two variables that are stored in the cache:
        cached_key and cached_value.

        Args:
            self: Access the variables stored in the cache
            key: Store the keys of the encoder-decoder attention
            value: Initialize the cached_value variable
            query_states: Determine the number of cache vectors to
                update
            attention_mask: Mask out the padded vectors in the cache

        Returns:
            The key, value and attention_mask
        """
        do_quantize_kv_cache = self.config.quantize_kv_cache
        is_initialized = self.has_variable("cache", "cached_key")
        # jax.debug.print('cache initialized {s}', s=is_initialized)
        if is_initialized:
            cache_size = self.variables["cache"]["cached_key"].shape[1]
        else:
            cache_size = attention_mask.shape[-1]
        # print(cache_size)
        # print(attention_mask.shape)
        # jax.debug.print('cache attention mask shape {s}', s=attention_mask.shape)
        k_shape = key.shape[:1]+(cache_size,)+key.shape[2:]
        v_shape = value.shape[:1]+(cache_size,)+value.shape[2:]
        if do_quantize_kv_cache:
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, k_shape, jnp.uint8
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, v_shape, jnp.uint8
            )
            cached_key_scale = self.variable(
                "cache",
                "cached_key_scale",
                jnp.zeros,
                k_shape[0:-1] + (1,),
                key.dtype,
            )
            cached_value_scale = self.variable(
                "cache",
                "cached_value_scale",
                jnp.zeros,
                v_shape[0:-1] + (1,),
                value.dtype,
            )
            cached_key_minval = self.variable(
                "cache",
                "cached_key_minval",
                jnp.zeros,
                k_shape[0:-1] + (1,),
                key.dtype,
            )
            cached_value_minval = self.variable(
                "cache",
                "cached_value_minval",
                jnp.zeros,
                v_shape[0:-1] + (1,),
                value.dtype,
            )
            cache_index = self.variable(
                "cache", "cache_index", jnp.zeros, k_shape[:1], jnp.int32 # lambda: jnp.array(0, dtype=jnp.int32)
            )
        else:
            cached_key_scale = None
            cached_value_scale = None
            cached_value_minval = None
            cached_key_minval = None
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, k_shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, v_shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", jnp.zeros, k_shape[:1], jnp.int32 # lambda: jnp.array(0, dtype=jnp.int32)
            )
        paxs = self.config.partition_axis
        is_initialized = True
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value[0]
            if query_states.shape[1] == 1 and self.config.use_sharded_kv_caching:
                mesh = self.config.get_mesh()

                def fn(_cached_key, _cached_value, _key, _value, _cur_index):
                    assert _key.shape[1] == 1 and _value.shape[1] == 1, (
                        _key.shape,
                        _value.shape,
                    )
                    sp_size = max_length // mesh.shape[paxs.key_sequence_axis]
                    axis_index = jax.lax.axis_index(paxs.key_sequence_axis)
                    _cur_index = _cur_index - axis_index * sp_size
                    _key, _value = jax.lax.cond(
                        jnp.logical_and(_cur_index >= 0, _cur_index < sp_size),
                        lambda: (
                            _cached_key.at[:, _cur_index].set(_key[:, -1]),
                            _cached_value.at[:, _cur_index].set(_value[:, -1]),
                        ),
                        lambda: (_cached_key, _cached_value),
                    )
                    return _key, _value

                fn = shard_map(
                    fn,
                    mesh=mesh,
                    in_specs=(
                        PartitionSpec(
                            paxs.batch_axis,
                            paxs.key_sequence_axis,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                        PartitionSpec(
                            paxs.batch_axis,
                            paxs.key_sequence_axis,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                        PartitionSpec(
                            paxs.batch_axis,
                            None,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                        PartitionSpec(
                            paxs.batch_axis,
                            None,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                        PartitionSpec(),
                    ),
                    out_specs=(
                        PartitionSpec(
                            paxs.batch_axis,
                            paxs.key_sequence_axis,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                        PartitionSpec(
                            paxs.batch_axis,
                            paxs.key_sequence_axis,
                            paxs.head_axis,
                            paxs.attention_dim_axis,
                        ),
                    ),
                    check_rep=False,
                )
                key, value = fn(
                    cached_key.value, cached_value.value, key, value, cur_index
                )
            else:
                *batch_dims, max_length, num_heads, depth_per_head = (
                    cached_key.value.shape
                )
                cur_index = cache_index.value[0]
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)  # type:ignore
                if do_quantize_kv_cache:
                    key_val = dequantize_kv_cache(
                        cached_key.value,
                        cached_key_scale.value,
                        cached_key_minval.value,
                        key.dtype,
                    )
                    value_val = dequantize_kv_cache(
                        cached_value.value,
                        cached_value_scale.value,
                        cached_key_minval.value,
                        value.dtype,
                    )
                else:
                    key_val = cached_key.value
                    value_val = cached_value.value

                key = jax.lax.dynamic_update_slice(key_val, key, indices)
                value = jax.lax.dynamic_update_slice(value_val, value, indices)
                num_updated_cache_vectors = query_states.shape[1]
            if do_quantize_kv_cache:
                kq, ks, km = quantize_kv_cache(key)
                vq, vs, vm = quantize_kv_cache(value)

                cached_key.value = kq
                cached_key_scale.value = ks.astype(self.dtype)
                cached_key_minval.value = km.astype(self.dtype)

                cached_value.value = vq
                cached_value_scale.value = vs.astype(self.dtype)
                cached_value_minval.value = vm.astype(self.dtype)
            else:
                cached_key.value = key
                cached_value.value = value

            num_updated_cache_vectors = query_states.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
        return key, value, attention_mask


class NewFlaxQwen2Block(FlaxQwen2Block):

    def setup(self) -> None:
        attn_block = NewFlaxQwen2Attention
        if self.config.gradient_checkpointing != "":
            attn_block = nn_partitioning.remat(
                NewFlaxQwen2Attention,
                static_argnums=(1, 3, 4, 6, 7, 8, 9),
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
            )

        self.self_attn = attn_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        mlp_block = FlaxQwen2MLP

        if self.config.gradient_checkpointing != "":
            mlp_block = nn_partitioning.remat(
                FlaxQwen2MLP,
                static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(
                    self.config.gradient_checkpointing
                ),
            )

        self.mlp = mlp_block(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.input_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )


class NewFlaxQwen2BlockCollection(FlaxQwen2BlockCollection):

    def setup(self):
        self.blocks = [
            NewFlaxQwen2Block(
                config=self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            )
            for i in range(self.config.num_hidden_layers)
        ]


class NewFlaxQwen2Module(FlaxQwen2Module):

    def setup(self):

        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = flax.linen.Dropout(rate=self.config.embd_pdrop)
        self.layers = NewFlaxQwen2BlockCollection(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        config = self.config
        self.causal_mask = make_causal_mask(
            jnp.ones(
                (
                    1,
                    getattr(
                        config,
                        "c_max_position_embeddings",
                        config.max_position_embeddings,
                    ),
                ),
                dtype="bool",
            ),
            dtype="bool",
        )

        initial_rope_kwargs = dict(rope_type="none")
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            initial_rope_kwargs = dict(
                scaling_factor=scaling_factor, rope_type=scaling_type
            )
        self.freq_cis = precompute_freq_cis(
            max_position_embeddings=(
                getattr(
                    self.config,
                    "freq_max_position_embeddings",
                    self.config.max_position_embeddings,
                )
            ),
            dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            **initial_rope_kwargs,
        )


class NewFlaxQwen2ForCausalLMModule(FlaxQwen2ForCausalLMModule):

    def setup(self):
        self.model = NewFlaxQwen2Module(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            parent=self.scope,
        )

        self.lm_head = Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            precision=self.precision,
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method),
        )
