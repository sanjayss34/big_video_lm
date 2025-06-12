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

"""Preprocessing for videos."""

from big_vision.pp import utils
from big_vision.pp.registry import Registry

import tensorflow as tf


@Registry.register('preprocess_ops.video_decode')
def video_decode(res):
  """Preprocessing."""

  def _pp_per_image(img):
    # decode
    return tf.image.resize(tf.io.decode_jpeg(img), (res, res))

  def _pp(data):
    images = data['episodic_images']
    # resize
    images = tf.map_fn(_pp_per_image, images, fn_output_signature=tf.float32)
    # rescale
    images = 2 * (images / 255.) - 1.0
    data['image'] = images
    return data

  return _pp


@Registry.register('preprocess_ops.video_ensure_shape')
def video_ensure_shape(key, shape):
  """Preprocessing."""
  def _video_ensure_shape(data):
    data[key] = tf.ensure_shape(data[key], shape)
    return data

  return _video_ensure_shape


@Registry.register('preprocess_ops.video_replicate_img')
def video_replicate_img(replicas, num_frames):
  """Ensure that for short videos, we have the correct number of frames.

  We replicate and select.

  Args:
    replicas: num_replicas before selection. Should be less than num_frames.
    num_frames: number of frames

  Returns:
    _replicate_img: preprocessing function
  """

  def _replicate_img(data):
    # visual analogies + query
    image = data['image']
    image = tf.tile(image, [replicas, 1, 1, 1])
    data['image'] = image[:num_frames]
    return data

  return _replicate_img


@Registry.register('preprocess_ops.video_choice')
@utils.InKeyOutKey()
def video_choice(empty_fallback=None):
  """Randomly takes one entry out of a tensor after flattening."""

  def _choice(x):
    x = tf.reshape(x, (-1,))  # Ensure it's a 1D array

    # Append the fallback value so we gracefully handle empty cases.
    x0 = tf.zeros(1, x.dtype) if empty_fallback is None else [empty_fallback]
    x = tf.concat([x, x0], axis=0)

    num_choices = tf.maximum(tf.shape(x)[0] - 1, 1)  # Don't sample x0.
    return x[tf.random.uniform([], 0, num_choices, dtype=tf.int32)]

  return _choice


@Registry.register('preprocess_ops.stack_images')
def stack_images(inkeys=(), outkey='image'):

  def _pp(data):
    images = tf.stack([data[inkey] for inkey in inkeys])
    data[outkey] = images
    return data

  return _pp

@Registry.register('preprocess_ops.concat_images')
def concat_images(inkeys=(), outkey='image'):
  def _pp_concat(data):
    images = tf.concat([data[inkey] for inkey in inkeys], axis=0)
    data[outkey] = images
    return data
  return _pp_concat

@Registry.register('preprocess_ops.qwen_patchify')
def qwen_patchify(inkey='image', outkey_patches='image', outkey_thw='thw', outkey_mask='mask_image', temporal_patch_size=2, patch_size=14, merge_size=2, max_num_patches=None):
  def _qwen_patchify(data):
    data['image'] = tf.transpose(data['image'], perm=(0, 3, 1, 2))
    padding_images = tf.repeat(data['image'][-1:,:,:,:], repeats=temporal_patch_size - (data.shape[0] % temporal_patch_size), axis=0)
    data['image'] = tf.concat([data['image'], padding_images], axis=0)
    grid_t = data['image'].shape[0] // temporal_patch_size
    grid_h = data['image'].shape[-2] // patch_size
    grid_w = data['image'].shape[-1] // patch_size
    channels = data['image'].shape[1]
    patches = tf.reshape(data['image'], (grid_t, temporal_patch_size, channels, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size))
    patches = tf.transpose(patches, perm=(0, 3, 6, 4, 7, 2, 1, 5, 8))
    flattened_patches = tf.reshape(patches, (grid_t * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size))
    data[outkey_patches] = flattened_patches
    data[outkey_thw] = tf.repeat([grid_t, grid_h, grid_w], repeats=1, axis=0)
    if max_num_patches is not None:
      data[outkey_mask] = tf.concat([tf.ones(data[outkey_patches].shape[0], dtype=tf.int32), tf.zeros(max_num_patches-data[outkey_patches].shape[0], dtype=tf.int32)], axis=0)
      data[outkey_patches] = tf.concat(
        [
          data[outkey_patches],
          tf.repeat(data[outkey_patches][-1:,:], repeats=max_num_patches-data[outkey_patches].shape[0], axis=0)
        ],
        axis=0
      )
    return data
  return _qwen_patchify

@Registry.register('preprocess_ops.mask_video')
# @utils.InKeyOutKey()
def mask_video(key, num_images, second_key=None, second_num_images=None, dtype=tf.int32):
    def _mask_video(data):
        num_unmasked = data[key]
        num_unmasked_t = tf.convert_to_tensor(num_unmasked)
        video_mask = tf.concat([tf.ones(num_unmasked_t, dtype=dtype), tf.zeros(num_images-num_unmasked_t, dtype=dtype)], axis=0)
        if second_key is not None:
          second_num_unmasked_t = tf.convert_to_tensor(data[second_key+'_mask'])
          # tf.print("video_mask", video_mask)
          # tf.print("second num unmasked ones", tf.ones(second_num_unmasked_t, dtype=dtype))
          # tf.print("second num images zeros", tf.zeros(second_num_images-second_num_unmasked_t, dtype=dtype))
          video_mask = tf.concat([video_mask, tf.ones(second_num_unmasked_t, dtype=dtype), tf.zeros(second_num_images-second_num_unmasked_t, dtype=dtype)], axis=0)
        # tf.debugging.assert_type(video_mask, dtype=dtype)
        # return video_mask
        data[key] = video_mask
        return data
    return _mask_video
