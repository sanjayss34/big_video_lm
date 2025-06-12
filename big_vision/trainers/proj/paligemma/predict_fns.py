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

"""Prediction functions for PaliGemma."""

import collections
import functools

from big_vision.pp import registry
import big_vision.utils as u
import einops
import jax
import jax.numpy as jnp
import numpy as np
import time

from jax_array_info import sharding_info

P = jax.sharding.PartitionSpec
# pylint: disable=missing-function-docstring


def get_all(model):
  """Returns `predict_fns` for evaluators."""
  fns = {
      "decode_with_logp": _decode_with_logp,
  }
  return {name: functools.partial(fn, model=model) for name, fn in fns.items()}


def _decode_with_logp(
    train_state, batch, *, model, devices, max_decode_len, eos_token,
    best_of_n=1, sampler="greedy", replicate_out=False, eos_look_behind=0,
    extra_model_args=tuple(), sharding_config=None):
  """Sample token continuations to the input sequences."""
  if sharding_config is None:
    mesh = jax.sharding.Mesh(devices, ("devices",))
  else:
    mesh_helper = sharding_config.get_mesh()
    mesh = mesh_helper.mesh
  replicate_sharding = jax.sharding.NamedSharding(mesh, P())
  prefill_batch = {
      "image": batch["image"],
      "text": batch["text"],
      "mask_input": batch["mask_input"],
      "mask_ar": batch["mask_ar"],
      **{extra_model_arg: batch.get(extra_model_arg, None) for extra_model_arg in extra_model_args},
  }
  if sharding_config is None:
    out_sharding = jax.sharding.NamedSharding(
        mesh, P() if replicate_out else P("devices")
    )
    out_sharding = (out_sharding, out_sharding)
    input_batch = prefill_batch
  else:
    logits_sharding = jax.sharding.NamedSharding(
        mesh, P() if replicate_out else P(("replica", "fsdp",))
    )
    cache_template = sharding_config.get_cache_template(batch["image"].shape[0], max_decode_len, train_state)
    cache_sharding = jax.tree.map(
      lambda spec: jax.sharding.NamedSharding(mesh, spec),
      sharding_config.get_cache_sharding().apply(cache_template)
    )
    out_sharding = (
      logits_sharding,
      cache_sharding,
    )
    sharded_batch = prefill_batch
    input_batch = sharded_batch

  # Prefill the model cache and generate logits for first token.
  prefill = jax.jit(
      _prefill_cache,
      out_shardings=out_sharding,
      static_argnames=("model", "max_decode_len", "extra_embed_args"),
  )
  logits, cache = prefill(
      train_state["params"],
      input_batch,
      model=model,
      max_decode_len=max_decode_len,
      extra_embed_args=tuple(extra_model_args),
  )

  # Mask indicating real examples. False if example is used to pad the batch.
  mask = batch["_mask"]
  sharded_mask = mask
  if sharding_config is not None and isinstance(batch["_mask"], np.ndarray):
    sharded_mask = u.reshard(mask, jax.sharding.NamedSharding(mesh, P(("replica", "fsdp"))))

  print('mask shape before bon repeat', mask.shape, 'logits shape', logits.shape)
  if sharding_config is None:
    logits, cache, mask = jax.jit(
        _bon_repeat,
        static_argnames=("n",)
    )((logits, cache, mask), n=best_of_n)
  else:
    logits, cache, mask = jax.jit(
        _bon_repeat,
        out_shardings=(*out_sharding, out_sharding[0]),
        static_argnames=("n",)
    )((logits, cache, mask), n=best_of_n)

  decode_sample_output = jax.jit(
      _decode_sample_output,
      out_shardings=replicate_sharding,
      static_argnames=("max_decode_len", "sampler"),
  )
  decode_early_stop = jax.jit(
      _decode_early_stop,
      out_shardings=replicate_sharding,
      static_argnames=("eos_token",),
  )
  extend_cache = jax.jit(
      _extend_cache,
      donate_argnums=1,
      static_argnames=("model",),
  )
  if sharding_config is not None:
    extend_cache = jax.jit(
      _extend_cache,
      out_shardings=(
        jax.sharding.NamedSharding(mesh, P(("replica", "fsdp"))),
        cache_sharding
      ),
      donate_argnums=1,
      static_argnames=("model",),
    )
    decode_sample_output = jax.jit(
      _decode_sample_output,
      out_shardings=jax.sharding.NamedSharding(mesh, P(("replica", "fsdp"))),
      static_argnames=("max_decode_len", "sampler"),
    )

  # Keep sampling tokens from last logits until EOS or max_decode_len.
  state = None
  # Setting `eos_look_behind>0` removes blocking transfer with small batches.
  stops = collections.deque(maxlen=1 + eos_look_behind)
  for idx in range(max_decode_len):
    tokens, state = decode_sample_output(
        state, logits, max_decode_len=max_decode_len, sampler=sampler
    )

    if idx + 1 >= max_decode_len:
      break
   

    stops.append(decode_early_stop(state, mask, eos_token=eos_token))
    if len(stops) == stops.maxlen and jax.device_get(stops[0]):
      break
    # Compute logits for next token
    logits, cache = extend_cache(
        train_state["params"], cache, tokens, model=model,
    )

  _, tokens, logp = state
  # Select the best of n sample for each example.
  tokens, logp = jax.jit(
    _reshape_outputs,
    out_shardings=jax.sharding.NamedSharding(mesh, P(None, ("replica", "fsdp"))),
    static_argnames=("n",),
  )(state, n=best_of_n)

  return tokens, logp

def _decode(train_state, batch, **kwargs):
  tokens, _ = _decode_with_logp(train_state, batch, **kwargs)
  return tokens

def _bon_repeat(tree, *, n):
  return jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), tree)

def _compute_score(tokens, logp, eos_token):
  """Compute log-probability of each sequence up to first eos (including it)."""
  seqlen = jnp.sum(jnp.cumsum(tokens == eos_token, axis=-1) == 0, axis=-1) + 1
  token_mask = jnp.arange(tokens.shape[-1]) < seqlen[..., None]
  scores = jnp.sum(logp * token_mask, axis=-1)
  return scores

def _bon_select(state, *, n, eos_token):
  """Pick the sampled sequence with the highest likelihood for each example."""
  (_, tokens, logp) = state

  # Filter state to only keep the best of each example.
  scores = _compute_score(tokens, logp, eos_token)
  scores = einops.rearrange(scores, "(b n) -> b n", n=n)
  state = jax.tree.map(
      lambda x: einops.rearrange(x, "(b n) l -> b n l", n=n), state)
  best_indices = jnp.argmax(scores, -1)  # [b]
  state = jax.tree.map(
      lambda x: jnp.take_along_axis(x, best_indices[:, None, None], axis=1),
      state)
  state = jax.tree.map(lambda x: x[:, 0], state)

  return state

def _reshape_outputs(state, *, n):
  (_, tokens, logp) = state
  tokens = jnp.transpose(jnp.reshape(tokens, (-1, n, tokens.shape[-1])), (1, 0, 2))
  logp = jnp.transpose(jnp.reshape(logp, (-1, n, tokens.shape[-1])), (1, 0, 2))
  return (tokens, logp)

def _decode_sample_output(state, logits, *, max_decode_len, sampler):
  if state is None:
    # Decode state keeps track of sampled tokens and their logp.
    bs = logits.shape[0]
    seqlen = jnp.zeros((bs, 1), dtype=jnp.int32)
    tokens = jnp.zeros((bs, max_decode_len), dtype=jnp.int32)
    logp = jnp.zeros((bs, max_decode_len), dtype=logits.dtype)
  else:
    (seqlen, tokens, logp) = state

  # Sample tokens.
  sampled_tokens, sampled_logp = _sample_logits(logits, sampler=sampler)

  # Update state with sampled outputs.
  new_len = seqlen + 1
  new_tokens = _put_along_last_axis(tokens, seqlen, sampled_tokens)
  new_logp = _put_along_last_axis(logp, seqlen, sampled_logp)
  new_state = (new_len, new_tokens, new_logp)

  return sampled_tokens, new_state

def _decode_early_stop(state, mask, *, eos_token):
  (seqlen, tokens, unused_logp) = state
  token_mask = jnp.arange(tokens.shape[-1])[None, :] < seqlen
  has_eos = jnp.any(jnp.logical_and(tokens == eos_token, token_mask), axis=-1)
  done = jnp.logical_or(has_eos, jnp.logical_not(mask))
  return jnp.all(done)

def _put_along_last_axis(arr, indices, values):
  """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
  assert arr.ndim == indices.ndim == values.ndim, (
      arr.ndim, indices.ndim, values.ndim)
  onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
  put_mask = jnp.einsum("...i,...in->...n",
                        jnp.ones(values.shape, jnp.int32), onehot)
  put_values = jnp.einsum("...i,...in->...n", values, onehot)
  return jnp.where(put_mask, put_values, arr)

def _prefill_cache(params, batch, *, model, max_decode_len, extra_embed_args):
  """Initialize the model cache for decoding with the prompts."""
  variables = {"params": params}
  pre_prefill_time = time.time()
  (x, input_mask, mask_ar), aux = model.apply(
      variables, batch["image"], batch["text"],
      input_mask=batch["mask_input"],
      mask_ar=batch["mask_ar"],
      method=model.embed_image_and_text,
      **{arg: batch.get(arg, None) for arg in extra_embed_args},
  )
  num_image_tokens = aux["img/zimg"].shape[1]
  cache_size = x.shape[1] + max_decode_len
  last_logits, variables = model.apply(
      variables, x, input_mask, mask_ar,
      cache_size=cache_size,
      method=model.prefill_cache,
      mutable=("cache",))
  return last_logits, variables["cache"]

def _extend_cache(params, cache, tokens, *, model):
  """Extend the model cache for decoding with one token per sequence."""
  variables = {"params": params, "cache": cache}
  x, _ = model.apply(variables, tokens, method=model.embed_text)
  # jax.debug.print('in extend cache {s}', s=x.shape)
  last_logits, variables = model.apply(
      variables, x, method=model.extend_cache, mutable=("cache",))
  # jax.debug.print("_extend_cache last_logits {s}", s=last_logits.shape)
  return last_logits, variables["cache"]

def _sample_logits(logits, sampler):
  """Returns a sampled token and its logp from logits."""
  # Note: Consider making it possible for evaluators to pass rng seed to
  # decode functions. For now generate it from jax.lax and avoid evaluators
  # having to deal with it.
  rng = jax.random.PRNGKey(
      jax.lax.rng_uniform(0, np.iinfo(np.int32).max, tuple()))

  # Use Registry to support specifying things like:
  #  "greedy", "nucleus(0.2)", "temperature(t=1.0)"
  sampled_tokens = registry.Registry.lookup("paligemma_sampler." + sampler)(
      logits=logits, rng=rng)

  # Find the log probability (normalized logits) of selected tokens.
  sampled_logp = jnp.take_along_axis(
      jax.nn.log_softmax(logits, axis=-1),
      sampled_tokens[..., None], -1)[..., 0]

  return sampled_tokens, sampled_logp


@registry.Registry.register("paligemma_sampler.greedy")
def _greedy_sampling(*, logits, rng):
  del rng
  # jax.debug.print("_greedy_sampling logits {s}", s=logits.shape)
  return jnp.argmax(logits, axis=-1)


@registry.Registry.register("paligemma_sampler.temperature")
# def _temperature_sampling(t, *, logits, rng):
def _temperature_sampling(logits, rng, t=1):
  return jax.random.categorical(rng, logits / t)


@registry.Registry.register("paligemma_sampler.nucleus")
def _nucleus_sampling(p: float, t: float = 1.0, *, logits, rng):
  logits = logits / t
  neg_inf = np.array(-1.0e7)  # Effective negative infinity.
  logits_sorted = jnp.sort(logits, axis=-1, descending=True)
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < p, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, neg_inf), logits)
  return jax.random.categorical(rng, logits)
