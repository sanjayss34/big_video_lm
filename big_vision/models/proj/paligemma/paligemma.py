# Copyright 2025 big_video_lm authors - changes for supporting non-PaliGemma models
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

"""Image encoder + AR-decoder LLM."""

import os
import importlib
from typing import Any, Optional
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.core import ShapedArray
from scalax.sharding import with_sharding_annotation

from jax_array_info import sharding_info

ConfigDict = Any

def make_attn_mask(input_mask, mask_ar, text_causal, vision_causal, vision_attend_to_text, num_image_tokens=None, segment_ids=None):
  """Returns attention mask bool[B, N, N] to use in transformer.

  Tokens can attend to valid inputs tokens which have a cumulative mask_ar
  smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
  setup several types of attention, for example:

    [[1 1 1 1 1 1]]: pure causal attention.

    [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
        themselves and the last 3 tokens have a causal attention. The first
        entry could also be a 1 without changing behaviour.

    [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
        block can attend all previous blocks and all tokens on the same block.

  Args:
    input_mask: bool[B, N] true if its part of the input, false if padding.
    mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
      it and 0 where it shares the same attention mask as the previous token.
  """
  cumsum = jnp.cumsum(mask_ar, axis=1)
  attn_mask = (cumsum[:, None, :] <= cumsum[:, :, None])
  valid_mask = (input_mask[:, None, :] * input_mask[:, :, None])
  attn_mask = jnp.logical_and(attn_mask, valid_mask)
  if text_causal and num_image_tokens is not None:
    attn_mask = jnp.concatenate((
      attn_mask[:,:num_image_tokens,:],
      jnp.tril(attn_mask)[:,num_image_tokens:,:],
    ), axis=1)
  if vision_causal and num_image_tokens is not None:
    attn_mask = jnp.concatenate((
      jnp.tril(attn_mask)[:,:num_image_tokens,:],
      attn_mask[:,num_image_tokens:,:],
    ), axis=1)
  if not (text_causal and vision_causal):
    if (not vision_attend_to_text) and num_image_tokens is not None:
      attn_mask = jnp.where(
        jnp.repeat((jnp.expand_dims(jnp.expand_dims(jnp.arange(attn_mask.shape[-1]), 0), 0) >= num_image_tokens) & (jnp.expand_dims(jnp.expand_dims(jnp.arange(attn_mask.shape[-1]), 1), 0) < num_image_tokens), attn_mask.shape[0], axis=0),
        jnp.zeros_like(attn_mask),
        attn_mask
      )
  attn_mask = nn.with_logical_constraint(attn_mask, ("act_batch", "act_len", "act_len2"))
  if segment_ids is not None:
    segment_equal_mask = (jnp.expand_dims(segment_ids, axis=1) == jnp.expand_dims(segment_ids, axis=2))
    segment_equal_mask = nn.with_logical_constraint(segment_equal_mask, ("act_batch", "act_len", "act_len2"))
    segment_equal_mask = jnp.concatenate((
      jnp.ones_like(segment_equal_mask[:,:,:num_image_tokens]),
      segment_equal_mask[:,:,num_image_tokens:]
    ), axis=-1)
    segment_equal_mask = nn.with_logical_constraint(segment_equal_mask, ("act_batch", "act_len", "act_len2"))
    attn_mask = jnp.logical_and(attn_mask, segment_equal_mask)
  return attn_mask

class PatchDownsampler(nn.Module):
  img_token_pooling: dict
  
  @nn.compact
  def __call__(self, zimg_2d):
    pool_type = "average"
    antialias = False
    if "type" in self.img_token_pooling:
      pool_type = self.img_token_pooling["type"]
    if "antialias" in self.img_token_pooling:
      antialias = self.img_token_pooling["antialias"]
    if pool_type == "average":
      zimg_2d = nn.pooling.avg_pool(zimg_2d, **{key: self.img_token_pooling[key] for key in self.img_token_pooling if key not in ["type", "antialias"]})
    else:
      downsample_factor = self.img_token_pooling["strides"][1]
      assert self.img_token_pooling["strides"][2] == downsample_factor
      zimg_2d = jax.image.resize(
        zimg_2d,
        (*zimg_2d.shape[:-3], int(math.ceil(zimg_2d.shape[-3] / downsample_factor)), int(math.ceil(zimg_2d.shape[-2] / downsample_factor)), zimg_2d.shape[-1]),
        pool_type,
        antialias
      )
    zimg_2d = nn.with_logical_constraint(zimg_2d, ("flattened_images", "act_patches_height", "act_patches_width", "act_emb"))
    return zimg_2d

class Model(nn.Module):
  """Two towers transformer."""
  img_model: str = "vit"
  img: Optional[ConfigDict] = None
  llm_model: str = "proj.paligemma.gemma_bv"
  llm: Optional[ConfigDict] = None
  mesh: Optional[jax.sharding.Mesh] = None
  img_token_pooling: Optional[dict] = None
  img_token_length: int = None
  rope_max_wavelength: int = 10_000
  text_causal: bool = False
  vision_causal: bool = False
  vision_attend_to_text: bool = True
  insert_images_at_index: int = 0

  def setup(self):
    self._llm = importlib.import_module(
        f"big_vision.models.{self.llm_model}"
    ).Model(**(self.llm or {}), name="llm", mesh=self.mesh, image_start_index=self.insert_images_at_index)

    img_config = {"num_classes": self._llm.embdim, **(self.img or {})}
    self._img_model = importlib.import_module(
        f"big_vision.models.{self.img_model}"
    ).Model(**img_config, name="img", img_token_pooling=self.img_token_pooling, mesh=self.mesh)
    if self.img_token_pooling is not None:
      self.pooler = nn.remat(
        PatchDownsampler,
        static_argnums=(1,),  # 0=self
        policy=getattr(jax.checkpoint_policies, "nothing_saveable", None),
      )(img_token_pooling=self.img_token_pooling)

  def embed_image(self, image, mask_image=None, train=False):
    out = {}

    # if we have video, fold frame dimension into the batch dimension
    image_shape = image.shape
    if len(image_shape) == 5:  # video frames
      image = nn.with_logical_constraint(
        jnp.reshape(image, (-1, *image.shape[-3:])),
        ("flattened_images", "act_height", "act_width", "act_channels")
      )

    # Do we want to normalize? are they huge?
    _, out_img = self._img_model(image, train=train)
    
    zimg_2d = out_img["logits_2d"]

    zimg_2d_high_res = None
    add_image_newline_after_video = None
    if self.img_token_pooling:
      zimg_2d = self.pooler(zimg_2d)
      if mask_image is not None:
        mask_image = jnp.reshape(
          jnp.repeat(mask_image[:,:,None], axis=-1, repeats=zimg_2d.shape[1]*zimg_2d.shape[2]),
          (mask_image.shape[0], -1)
        )
    if self.img["add_image_newline"]:
      add_image_newline_after_video = out_img["image_newline"][:1,:,:]
    zimg = jnp.reshape(zimg_2d, (zimg_2d.shape[0], zimg_2d.shape[-3]*zimg_2d.shape[-2], zimg_2d.shape[-1]))
    zimg = nn.with_logical_constraint(zimg, ("flattened_images", "act_patches", "act_emb"))

    if len(image_shape) == 5:  # concatenate tokens from all video frames
      zimg = jnp.reshape(zimg, (image_shape[0], -1, zimg.shape[-1]))
      if zimg_2d_high_res is not None:
        zimg_high_res = jnp.reshape(zimg_2d_high_res, (image_shape[0], -1, zimg_2d_high_res.shape[-1]))
        zimg = jnp.concatenate((
          zimg,
          zimg_high_res
        ), axis=1)
    if add_image_newline_after_video is not None:
      out["img/image_newline"] = out_img["image_newline"] = jnp.repeat(add_image_newline_after_video, repeats=zimg.shape[0], axis=0)
    out["img/num_tokens"] = zimg.shape[1]
    zimg = nn.with_logical_constraint(zimg, ("act_batch", "act_len", "act_emb"))

    out["img/zimg"] = zimg
    for k, v in out_img.items():
      out[f"img/{k}"] = v

    if mask_image is not None:
      out["img/mask_image"] = mask_image
    return zimg, out

  def embed_text(self, tokens, train=False, image_emb=None):
    out = {}
    ztxt = out["llm/ztxt"] = self._llm.embed_tokens(tokens, train=train, image_emb=image_emb)
    return ztxt, out

  def embed_image_and_text(self, image, text, *,
                           input_mask=None, mask_ar=None, mask_image=None,
                           train=False):
    """Concats image/text into a sequence of embeded tokens to pass to `llm`.

    Args:
      image: float[B, H, W, 3] image to be embedded by the `img` model and used
        as prefix to the sequence passed to the `llm` model.
      text: int32[B, T] token sequence to embedded by the `llm`.
      input_mask: bool[B, T] true if the text token is a valid token and false
        if its a token to pad the sequence. Defaults to all being input tokens.
      mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
        causally, and 0 where it can be attended to with full self-attention.
        Defaults to all text tokens being auto-regressive.
      train: bool whether we're in train or test mode (dropout etc).

    Returns:
      Tuple (x: float[B, N, E], input_mask: bool[B, N], mask_ar: int[B, N]) and
      auxiliary outputs.
    """
    zimg, out_img = self.embed_image(image, train=train, mask_image=mask_image)
    ztxt, out_txt = self.embed_text(text, train=train, image_emb=zimg)

    if input_mask is None:
      input_mask = jnp.full(text.shape, True)
    if mask_ar is None:
      mask_ar = jnp.full(text.shape, 1)

    # Concatenate embeded image and text into a single token sequence.
    x = jnp.concatenate([zimg, ztxt], axis=1)
    img_len = out_img["img/num_tokens"]
    pad_width = ((0, 0), (img_len, 0))
    text_input_mask = input_mask.copy()
    text_mask_ar = mask_ar.copy()
    remainder = 0
    if mask_image is None:
      if self.img_token_length is not None:
        pad_width = ((0, 0), (self.img_token_length, 0))
      input_mask = jnp.pad(input_mask, pad_width, constant_values=True)
    else:
      mask_image = jnp.expand_dims(mask_image, axis=-1)
      # remainder is for the image newline if there is only one for the full sequence, rather than
      # one per image
      if "img/image_newline" in out_img:
        remainder = out_img["img/image_newline"].shape[1]
      tokens_per_image = img_len // mask_image.shape[1]
      mask_image = out_img["img/mask_image"]
      if self.img_token_length is not None:
        pad_width = ((0, 0), (self.img_token_length, 0))
        input_mask = jnp.concatenate((
          mask_image,
          jnp.repeat(jnp.zeros_like(mask_image[:,:1]), axis=1, repeats=self.img_token_length-mask_image.shape[1]-remainder),
          jnp.ones((mask_image.shape[0], remainder), dtype=mask_image.dtype),
          input_mask
        ), axis=1)
      else:
        input_mask = jnp.concatenate((mask_image, input_mask), axis=1)
    if self.img_token_length is not None:
      x = jnp.concatenate((
        zimg,
        jnp.repeat(jnp.zeros_like(zimg[:,:1,:]), axis=1, repeats=self.img_token_length-zimg.shape[1]-remainder),
        out_img["img/image_newline"] if "img/image_newline" in out_img else jnp.zeros_like(zimg[:,:0,:]),
        ztxt
      ), axis=1)
      img_len = self.img_token_length
    mask_ar = jnp.pad(mask_ar, pad_width, constant_values=0)
    out_img["img/zimg"] = x[:,:img_len]
    if self.insert_images_at_index > 0:
      x = jnp.concatenate((x[:,img_len:img_len+self.insert_images_at_index], x[:,:img_len], x[:,img_len+self.insert_images_at_index:]), axis=1)
      input_mask = jnp.concatenate((input_mask[:,img_len:img_len+self.insert_images_at_index], input_mask[:,:img_len], input_mask[:,img_len+self.insert_images_at_index:]), axis=1)
      mask_ar = jnp.concatenate((mask_ar[:,img_len:img_len+self.insert_images_at_index], mask_ar[:,:img_len], mask_ar[:,img_len+self.insert_images_at_index:]), axis=1)
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    mask_ar = nn.with_logical_constraint(mask_ar, ("act_batch", "act_len"))
    input_mask = nn.with_logical_constraint(input_mask, ("act_batch", "act_len"))

    return (x, input_mask, mask_ar), {**out_img, **out_txt}

  def __call__(self, image, text, mask_ar, mask_image=None, segment_ids=None, train=False):
    """Concats image/text and returns text logits.

    Args:
      image: float32[B, H, W, 3] image that can be passed to the `img` model.
      text: int32[B, T] token sequence that can be embedded by the `txt` model.
      mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
        causally, and 0 where it can be attended to with full self-attention.
      train: bool whether we're in train or test mode (dropout etc).

    Returns:
      float32[B, T, V] logits for the `text` input, and an out-dict of named
      intermediates.
    """
    # Embed the image and text.
    (x, input_mask, mask_ar), out = self.embed_image_and_text(
        image, text, mask_ar=mask_ar, train=train, mask_image=mask_image)
    zimg = out["img/zimg"]

    if segment_ids is not None:
      segment_ids = segment_ids[:,:-1]
      segment_ids = jnp.concatenate((
        segment_ids[:,:self.insert_images_at_index],
        jnp.repeat(jnp.zeros_like(segment_ids[:,:1]), repeats=zimg.shape[1], axis=1),
        segment_ids[:,self.insert_images_at_index:]+1
      ), axis=1)
    # Call transformer on the embedded token sequence.
    attn_mask = out["attn_mask"] = make_attn_mask(input_mask, mask_ar, segment_ids=segment_ids, text_causal=self.text_causal, vision_causal=self.vision_causal, vision_attend_to_text=self.vision_attend_to_text, num_image_tokens=zimg.shape[1]+self.insert_images_at_index)
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    print('zimg shape', zimg.shape)
    logits, out_llm = self._llm(x, mask=attn_mask, train=train, image_tokens=zimg, segment_ids=segment_ids, input_mask=input_mask)
    # Extract the logits for the text tokens.
    text_pre_logits = out_llm["pre_logits"]
    text_logits = logits
    for k, v in out_llm.items():
      out[f"llm/{k}"] = v

    text_logits = nn.with_logical_constraint(text_logits, ("act_batch", "act_len", "act_emb"))
    out["text_logits"] = text_logits
    out["text_tokens"] = jnp.argmax(text_logits, axis=-1)
    return text_logits, out

  def prefill_cache(self, x, input_mask, mask_ar, *, cache_size):
    """Initializes decoding cache with `x` [B, N, E] as prompt."""
    position_ids = None
    if hasattr(self._llm, "prefill_cache"):
      attn_mask = make_attn_mask(input_mask, mask_ar, text_causal=self.text_causal, vision_causal=self.vision_causal, vision_attend_to_text=self.vision_attend_to_text, num_image_tokens=self.img_token_length+self.insert_images_at_index if self.img_token_length is not None else None)
      return self._llm.prefill_cache(
          x, input_mask, attn_mask, position_ids, cache_size=cache_size)
    else:
      return self._fallback_prefill_cache(x, input_mask, mask_ar, cache_size)

  def extend_cache(self, x, *args):
    """Advances decoding cache with `x` [B, 1, E]."""
    if hasattr(self._llm, "prefill_cache"):
      return self._llm.extend_cache(x, *args)
    else:
      return self._fallback_extend_cache(x)

  def _fallback_prefill_cache(self, x, input_mask, mask_ar, cache_size):
    # FALLBACK: only cache inputs and call the model with the full sequence
    # for each and every decode step. Very slowwww...
    #
    # This very slow codepath does not requires the model to implement caching.
    # It is intended to allow to plug any model under development quite early
    # into some decoding tasks and not as a long term decoding solution.
    attn_mask = make_attn_mask(input_mask, mask_ar)
    logits, _ = self._llm(x, mask=attn_mask)

    # Save the prefill inputs for subsequent extend_calls in the cache.
    # Unused entries are zero-initialized.
    pad_size = cache_size - x.shape[1]
    x = jnp.pad(jnp.where(input_mask[..., None], x, 0),
                [(0, 0), (0, pad_size), (0, 0)])
    mask_ar = jnp.pad(jnp.where(input_mask, mask_ar, 0),
                      [(0, 0), (0, pad_size)])
    input_mask = jnp.pad(input_mask, [(0, 0), (0, pad_size)])
    self.put_variable("cache", "x_cache", x)
    self.put_variable("cache", "input_mask_cache", input_mask)
    self.put_variable("cache", "mask_ar_cache", mask_ar)

    # Extract logits of the last token (using einsum).
    last_pos = jnp.sum(input_mask, axis=1)[:, None] - 1
    last_onehot = jax.nn.one_hot(last_pos, logits.shape[1], dtype=jnp.int32)
    last_logits = jnp.einsum("bnh,ben->beh", logits, last_onehot)

    return last_logits

  def _fallback_extend_cache(self, x):
    # FALLBACK: append inputs to cache and call the model with the full sequence
    # for each and every decode step. Very slowwww...
    assert x.shape[1] == 1
    mask_ar = jnp.full(x.shape[:-1], 1)
    input_mask = jnp.full(x.shape[:-1], True)

    # Append inputs to cache by add/or on the next available cache position,
    # which is zero-initialized.
    c_x = self.get_variable("cache", "x_cache")
    c_input_mask = self.get_variable("cache", "input_mask_cache")
    c_mask_ar = self.get_variable("cache", "mask_ar_cache")
    next_pos = jnp.sum(c_input_mask, axis=1)[:, None]
    move_onehot = jax.nn.one_hot(next_pos, c_x.shape[1], dtype=jnp.int32)
    x = jnp.add(c_x, jnp.einsum("beh,ben->bnh", x, move_onehot))
    mask_ar = jnp.add(c_mask_ar, jnp.einsum("be,ben->bn", mask_ar, move_onehot))
    input_mask = jnp.logical_or(
        c_input_mask, jnp.einsum("be,ben->bn", input_mask, move_onehot))
    self.put_variable("cache", "x_cache", x)
    self.put_variable("cache", "input_mask_cache", input_mask)
    self.put_variable("cache", "mask_ar_cache", mask_ar)

    # Call model on the full cached sequence.
    attn_mask = make_attn_mask(input_mask, mask_ar)
    logits, _ = self._llm(x, mask=attn_mask)

    # Extract logits of the last token.
    last_pos = jnp.sum(input_mask, axis=1)[:, None] - 1
    last_onehot = jax.nn.one_hot(last_pos, logits.shape[1], dtype=jnp.int32)
    last_logits = jnp.einsum("bnh,ben->beh", logits, last_onehot)

    return last_logits

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "PLEASE_SET_CHECKPOINTS_DIR")


def load(init_params, init_files, model_cfg, img_load_kw={}, llm_load_kw={}):  # pylint: disable=dangerous-default-value
  """Loads both pieces, `init_files` is now a dict with `img` and `llm` keys."""

  # A slight shortcut when loading an already combined model:
  if isinstance(init_files, str):
    init_files = os.path.join(CHECKPOINTS_DIR, init_files+'.npz')
    init_files = {"img": f"{init_files}:img", "llm": f"{init_files}:llm"}

  if not init_params:  # Convenience to skip checks in colab.
    init_params = {"img": None, "llm": None}
  restored_params = {**init_params}

  init_files = {**init_files}  # Needed because ConfigDict but we'll pop stuff.

  if img_init := init_files.pop("img", None):
    restored_params["img"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('img_model', 'vit')}"
    ).load(init_params["img"], img_init, model_cfg.img, **img_load_kw)

  if llm_init := init_files.pop("llm", None):
    restored_params["llm"] = importlib.import_module(
        f"big_vision.models.{model_cfg.get('llm_model', 'proj.paligemma.gemma_bv')}"
    ).load(init_params["llm"], llm_init, model_cfg.llm, **llm_load_kw)

  assert not init_files, (
      f"There's something unused left in `config.model_init`. You probably got "
      f"a typo. Here it is: {init_files}")

  return restored_params
