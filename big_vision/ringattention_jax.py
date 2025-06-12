# Copyright 2025 big_video_lm authors - added ring_window_attention
# Copyright 2023 ringattention authors
import numpy as np
import flax.linen as nn
from flax.linen import partitioning
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from einops import rearrange
from functools import partial
import dataclasses
import functools
from typing import Any, NamedTuple, Optional
import os

# Ring window attention is an implementation meant for cases in which the sliding window size is less than or equal to the sequence length per device.
# In this case, we only need to compute query-key products for adjacent devices.
def _ring_window_attention_standard_fwd(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    # scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, mask_slice, kv_rotation):
        prev_max_score, numerator, denominator, k, v = carry
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + kv_rotation) % axis_size) for i in range(axis_size)]), (k, v))
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) + mask_slice) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights + mask
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        numerator = numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v)
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, k, v), _ = scan_kv_block((prev_max_score, numerator, denominator, k, v), 0, 0)
    (max_score, numerator, denominator, k, v), _ = scan_kv_block((max_score, numerator, denominator, k, v), -1, 1)
    (max_score, numerator, denominator, k, v), _ = scan_kv_block((max_score, numerator, denominator, k, v), 1, -2)
    k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
        (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    # return output.astype(v.dtype)
    return output.astype(v.dtype), (output, q, k, v, attn_mask, numerator, denominator, max_score)

def _ring_window_attention_standard_bwd(axis_name, float32_logits, scale, res, g):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    # scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, mask_slice, kv_rotation):
        dq, dk, dv, k, v = carry
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + kv_rotation) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) + mask_slice) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights + mask
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + jnp.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + jnp.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = scan_kv_block((dq, dk, dv, k, v), 0, 0)
    (dq, dk, dv, k, v), _ = scan_kv_block((dq, dk, dv, k, v), -1, 1)
    (dq, dk, dv, k, v), _ = scan_kv_block((dq, dk, dv, k, v), 1, -2)
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None, None

@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7])
def ring_window_attention_standard(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale):
    y, _ = _ring_window_attention_standard_fwd(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale)
    return y

ring_window_attention_standard.defvjp(_ring_window_attention_standard_fwd, _ring_window_attention_standard_bwd)

def _ring_attention_standard_fwd(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    batch, q_len, num_heads, _ = q.shape
    batch, kv_len, num_heads, dim_per_head = k.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(q.dtype)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    axis_size = lax.psum(1, axis_name)
    # scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights + mask
        max_score = jnp.maximum(prev_max_score, jnp.max(attn_weights, axis=-1))
        exp_weights = jnp.exp(attn_weights - max_score[..., None])
        correction = rearrange(jnp.exp(prev_max_score - max_score), 'b h q -> b q h')[..., None]
        numerator = numerator * correction + jnp.einsum("bhqk,bkhd->bqhd", exp_weights, v)
        denominator = denominator * jnp.exp(prev_max_score - max_score) + jnp.sum(exp_weights, axis=-1)
        k, v = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v))
        return (max_score, numerator, denominator, k, v), None
    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    (max_score, numerator, denominator, _, _), _ = lax.scan(scan_kv_block,
    init=(prev_max_score, numerator, denominator, k, v), xs=jnp.arange(0, axis_size))
    output = numerator / rearrange(denominator, 'b h q -> b q h')[..., None]
    return output.astype(v.dtype), (output, q, k, v, attn_mask, numerator, denominator, max_score)

def _ring_attention_standard_bwd(axis_name, float32_logits, scale, res, g):
    del float32_logits
    axis_size = lax.psum(1, axis_name)
    output, q, k, v, attn_mask, numerator, denominator, max_score = res
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    batch, kv_len, num_heads, dim_per_head = k.shape
    # scale = jnp.sqrt(q.shape[-1])
    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        mask = lax.dynamic_slice_in_dim(attn_mask,
            (lax.axis_index(axis_name) - idx) % axis_size * kv_len, kv_len, axis=-1)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / scale
        # attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights + mask
        exp_weights = jnp.exp(attn_weights - max_score[..., None]) / denominator[..., None]
        ds = jnp.einsum("bqhd,bkhd->bhqk", g, v)
        dl = (ds - jnp.einsum("bqhd,bqhd->bhq", g, output)[..., None]) * exp_weights
        dq = dq + jnp.einsum("bhqk,bkhd->bqhd", dl, k) / scale
        dk = dk + jnp.einsum("bqhd,bhqk->bkhd", q, dl) / scale
        dv = dv + jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g)
        k, v, dk, dv = map(lambda x: lax.ppermute(x, axis_name, perm=[(i,
            (i + 1) % axis_size) for i in range(axis_size)]), (k, v, dk, dv))
        return (dq, dk, dv, k, v), None
    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    return dq, dk, dv, None, None

@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7])
def ring_attention_standard(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale):
    y, _ = _ring_attention_standard_fwd(q, k, v, attn_mask, segment_ids, axis_name, float32_logits, scale)
    return y

ring_attention_standard.defvjp(_ring_attention_standard_fwd, _ring_attention_standard_bwd)
