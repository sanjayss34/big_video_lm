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

"""Image-centric preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.

The key named "image" is commonly used for the image, and is a 3D tensor of
shape (height x width x channels).
"""
from typing import List

from big_vision.pp import utils
from big_vision.pp.registry import Registry

from PIL import Image
import tensorflow as tf


@Registry.register("preprocess_ops.decode")
@utils.InKeyOutKey()
def get_decode(channels=3, precise=False, raw=False, max_bytes=None):
  """Decode an encoded image string, see tf.io.decode_image.

  Args:
    channels: see tf.io.decode_image.
    precise: if False, use default TF image decoding algorithm.
        If True, change DCT method for JPEG decoding to match PIL/cv2/PyTorch.
        See also (internal link) for a concrete example.

  Returns:
    The decoded image.
  """

  def _decode(image):
    if raw:
      im = tf.io.decode_raw(image, out_type=tf.uint8)
      # im = image
      # tf.print(str(im))
      # tf.print(str(im.shape))
      if max_bytes is not None:
        im = tf.cond(
            tf.shape(im)[0] >= max_bytes,
            lambda: im[:max_bytes],
            lambda: tf.pad(
                im, [(0, max_bytes - tf.shape(im)[0])],
                constant_values=256),
        )
      return im
    elif precise:
      return tf.image.decode_jpeg(  # Also supports png btw.
          image, channels=channels, dct_method="INTEGER_ACCURATE")
    else:
      return tf.io.decode_image(
          image, channels=channels, expand_animations=False)

  return _decode


@Registry.register("preprocess_ops.resize")
@utils.InKeyOutKey()
def get_resize(size, method="bilinear", antialias=False):
  """Resizes image to a given size.

  Args:
    size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
    method: resize method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function for resizing an image.

  """
  size = utils.maybe_repeat(size, 2)

  def _resize(image):
    """Resizes image to a given size."""
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    # In particular it was not equivariant with rotation and lead to the network
    # to learn a shortcut in self-supervised rotation task, if rotation was
    # applied after resize.
    dtype = image.dtype
    tf_dtype = tf.type_spec_from_value(image).dtype
    # tf.print(image)
    image = tf.image.resize(image, size, method=method, antialias=antialias)
    return tf.cast(tf.clip_by_value(image, tf_dtype.min, tf_dtype.max), dtype)

  return _resize

@Registry.register("preprocess_ops.resize_llava_square")
@utils.InKeyOutKey()
def get_resize_llava_square(size, method="bicubic", antialias=False):
  size = utils.maybe_repeat(size, 2)
  # def _resize_numpy(image):
  #   pil_image = Image.fromarray(image)
  #   pil_resized = pil_image.resize(size, resample=getattr(Image.Resampling, method.upper()))
  #   np_resized = np.array(pil_resized)
  #   return np_resized
  def _resize_square(image):
    image_shape = tf.shape(image)
    original_width = image_shape[1]
    original_height = image_shape[0]
    dtype = image.dtype
    tf_dtype = tf.type_spec_from_value(image).dtype
    num_channels = image_shape[2]
    new_width = tf.cond(original_width > original_height, lambda: original_width, lambda: original_height)
    new_height = new_width
    color = 127
    top_padding = color*tf.ones(((new_height - original_height) // 2, original_width, num_channels), dtype=dtype)
    bottom_padding = color*tf.ones(((new_height - original_height) // 2 + (new_height - original_height) % 2, original_width, num_channels), dtype=dtype)
    left_padding = color*tf.ones((new_height, (new_width - original_width) // 2, num_channels), dtype=dtype)
    right_padding = color*tf.ones((new_height, (new_width - original_width) // 2 + (new_width - original_width) % 2, num_channels), dtype=dtype)
    height_padded_image = tf.concat([
      top_padding,
      image,
      bottom_padding
    ], axis=0)
    width_padded_image = tf.concat([
      left_padding,
      height_padded_image,
      right_padding
    ], axis=1)
    # tf.print('pre resize', width_padded_image[419:430,:,:])
    resized_image = tf.image.resize(width_padded_image, size, method=method, antialias=antialias)
    # resized_image = tf.numpy_function(_resize_numpy, [width_padded_image], tf_dtype)
    # tf.print('post resize', resized_image[100:110,:,:])
    return tf.cast(tf.math.round(tf.clip_by_value(resized_image, tf_dtype.min, tf_dtype.max)), dtype)
  return _resize_square

@Registry.register("preprocess_ops.resize_anyres")
@utils.InKeyOutKey()
def get_resize_anyres(size, method="bilinear", antialias=False):
  possible_resolutions = tf.convert_to_tensor([
    [
      384,
      384
    ],
    [
      384,
      768
    ],
    [
      384,
      1152
    ],
    [
      384,
      1536
    ],
    [
      384,
      1920
    ],
    [
      384,
      2304
    ],
    [
      768,
      384
    ],
    [
      768,
      768
    ],
    [
      768,
      1152
    ],
    [
      768,
      1536
    ],
    [
      768,
      1920
    ],
    [
      768,
      2304
    ],
    [
      1152,
      384
    ],
    [
      1152,
      768
    ],
    [
      1152,
      1152
    ],
    [
      1152,
      1536
    ],
    [
      1152,
      1920
    ],
    [
      1152,
      2304
    ],
    [
      1536,
      384
    ],
    [
      1536,
      768
    ],
    [
      1536,
      1152
    ],
    [
      1536,
      1536
    ],
    [
      1536,
      1920
    ],
    [
      1536,
      2304
    ],
    [
      1920,
      384
    ],
    [
      1920,
      768
    ],
    [
      1920,
      1152
    ],
    [
      1920,
      1536
    ],
    [
      1920,
      1920
    ],
    [
      1920,
      2304
    ],
    [
      2304,
      384
    ],
    [
      2304,
      768
    ],
    [
      2304,
      1152
    ],
    [
      2304,
      1536
    ],
    [
      2304,
      1920
    ],
    [
      2304,
      2304
    ]
  ], dtype=tf.int32)
  def _resize_anyres(image):
    original_width = image.shape[1]
    original_height = image.shape[0]
    num_channels = image.shape[2]
    dtype = image.dtype
    tf_dtype = tf.type_spec_from_value(image).dtype
    scale = tf.math.minimum(possible_resolutions[:,0] / original_height, possible_resolutions[:,1] / original_width)
    downscaled_height = tf.cast(tf.math.floor(original_height * scale), dtype=tf.int32)
    downscaled_width = tf.cast(tf.math.floor(original_width * scale), dtype=tf.int32)
    effective_resolution = tf.math.minimum(downscaled_width * downscaled_height, original_width * original_height * tf.ones_like(downscaled_width))
    wasted_resolution = (possible_resolutions[:,0] * possible_resolutions[:,1]) - effective_resolution
    max_effective_resolution = tf.math.reduce_max(effective_resolution, axis=0)
    min_wasted_resolution = tf.reduce_min(
      tf.where(
        effective_resolution == max_effective_resolution,
        wasted_resolution,
        tf.reduce_max(wasted_resolution, axis=0)*tf.ones_like(wasted_resolution)
      ),
      axis=0
    )
    best_fit_index = tf.math.argmin(min_wasted_resolution, axis=0)
    target_height = possible_resolutions[best_fit_index, 0]
    target_width = possible_resolutions[best_fit_index, 1] 
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    new_width = tf.cond(scale_w < scale_h, target_width, tf.math.minimum(tf.math.ceil(original_width * scale_h), target_width))
    new_height = tf.cond(scale_w < scale_h, tf.math.minimum(tf.math.ceil(original_height * scale_w), target_height), target_height)
    resized_image = tf.image.resize(image, (new_height, new_width), method=method, antialias=antialias)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    height_padded_image = tf.concat([
      tf.zeros((paste_y, new_width, num_channels), dtype=dtype),
      resized_image,
      tf.zeros((paste_y, new_width, num_channels), dtype=dtype),
    ], axis=0)
    width_padded_image = tf.concat([
      tf.zeros((target_height, paste_x, num_channels), dtype=dtype),
      resized_image,
      tf.zeros((target_height, paste_x, num_channels), dtype=dtype),
    ], axis=1)
    new_image = width_padded_image
    new_image = tf.cast(tf.clip_by_value(new_image, tf_dtype.min, tf_dtype.max), dtype) # [H, W, C]
    new_image = tf.reshape(new_image, [new_image.shape[0]*new_image.shape[1] / size.shape[1], size.shape[1], new_image.shape[2]])
    new_image = tf.reshape(new_image, [new_image.shape[0]*new_image.shape[1] / (size.shape[0] * size.shape[1]), size.shape[0], size.shape[1], new_image.shape[2]])
    return new_image
  return _resize_anyres

@Registry.register("preprocess_ops.qwen_smart_resize")
@utils.InKeyOutKey()
def get_qwen_smart_resize(factor: int = 28, min_pixels: int = 4*28*28, max_pixels: int = 16384*28*28, scale_factor: float = 1. / 255, image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073], image_std: List[float] = [0.26862954, 0.26130258, 0.27577711]):
  def _qwen_smart_resize(image):
    original_width = image.shape[1]
    original_height = image.shape[0]
    num_channels = image.shape[2]
    h_bar = tf.math.maximum(factor, tf.math.round(original_height / factor) * factor)
    w_bar = tf.math.maximum(factor, tf.math.round(original_width / factor) * factor)
    beta = tf.cond(
      h_bar * w_bar > max_pixels,
      tf.math.sqrt((original_height * original_width) / max_pixels),
      tf.cond(
        h_bar * w_bar < min_pixels,
        tf.math.sqrt(min_pixels / (height * width)),
        1
      )
    )
    h_bar = tf.cond(
      h_bar * w_bar > max_pixels,
      tf.math.floor((original_height / beta) / factor) * factor,
      tf.cond(
        h_bar * w_bar < min_pixels,
        tf.math.ceil((original_height * beta) / factor) * factor,
        h_bar
      )
    )
    w_bar = tf.cond(
      h_bar * w_bar > max_pixels,
      tf.math.floor((original_width / beta) / factor) * factor,
      tf.cond(
        h_bar * w_bar < min_pixels,
        tf.math.ceil((original_width * beta) / factor) * factor,
        w_bar
      )
    )
    new_image = tf.image.resize(image, (h_bar, w_bar), method='bicubic', antialias=True)
    new_image = new_image * scale_factor
    for i in range(3):
      new_image[...,i] = (new_image[...,i] - image_mean[i]) / image_std[i]

    return new_image
  return _qwen_smart_resize

"""@Registry.register("preprocess_ops.resize_video")
@utils.InKeyOutKey()
def get_resize_video(size, method="bilinear", antialias=False):
    size = utils.maybe_repeat(size, 2)
    def _resize(video):
      dtype = video.dtype
      tf_dtype = tf.type_spec_from_value(video).dtype
      resized_frames = []
      for frame in tf.unstack(video, axis=0):
        resized_frame = tf.image.resize(frame, size, method=method, antialias=antialias)
        resized_frames.append(resized_frame)"""

# This functionality is used by resize_small and resize_long. But we're not
# registering it as a pp op yet, as there is no need for it. However, it can
# probably be slightly generalized into "scale augmentation" eventually.
def _resize_factor(image, factor, method="area", antialias=True):
  """Resizes the image by a (float) `factor`, keeping the aspect ratio fixed."""
  h, w = tf.shape(image)[0], tf.shape(image)[1]

  h = tf.cast(tf.round(tf.cast(h, tf.float32) * factor), tf.int32)
  w = tf.cast(tf.round(tf.cast(w, tf.float32) * factor), tf.int32)

  dtype = image.dtype
  tf_dtype = tf.type_spec_from_value(image).dtype
  image = tf.image.resize(image, (h, w), method=method, antialias=antialias)
  return tf.cast(tf.clip_by_value(image, tf_dtype.min, tf_dtype.max), dtype)


@Registry.register("preprocess_ops.resize_small")
@utils.InKeyOutKey()
def get_resize_small(smaller_size, method="area", antialias=False):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.

  Note:
    backwards-compat for "area"+antialias tested here:
    (internal link)
  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    factor = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    return _resize_factor(image, factor, method=method, antialias=antialias)
  return _resize_small


@Registry.register("preprocess_ops.resize_long")
@utils.InKeyOutKey()
def get_resize_long(longer_size, method="area", antialias=True):
  """Resizes the longer side to `longer_size` keeping aspect ratio.

  Args:
    longer_size: an integer, that represents a new size of the longer side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.
  """

  def _resize_long(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    factor = (
        tf.cast(longer_size, tf.float32) /
        tf.cast(tf.maximum(h, w), tf.float32))
    return _resize_factor(image, factor, method=method, antialias=antialias)
  return _resize_long


@Registry.register("preprocess_ops.inception_crop")
@utils.InKeyOutKey()
def get_inception_crop(size=None, area_min=5, area_max=100,
                       method="bilinear", antialias=False):
  """Makes inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    size: Resize image to [size, size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    method: rezied method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image):  # pylint: disable=missing-docstring
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, crop_size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if size:
      crop = get_resize(size, method, antialias)({"image": crop})["image"]
    return crop

  return _inception_crop


@Registry.register("preprocess_ops.decode_jpeg_and_inception_crop")
@utils.InKeyOutKey()
def get_decode_jpeg_and_inception_crop(size=None, area_min=5, area_max=100,
                                       ratio_min=0.75, ratio_max=1.33,
                                       method="bilinear", antialias=False):
  """Decode jpeg string and make inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    size: Resize image to [size, size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    ratio_min: minimal aspect ratio.
    ratio_max: maximal aspect ratio.
    method: rezied method, see tf.image.resize docs for options.
    antialias: see tf.image.resize. Ideally set to True for all new configs.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        aspect_ratio_range=(ratio_min, ratio_max),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(crop_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if size:
      image = get_resize(size, method, antialias)({"image": image})["image"]

    return image

  return _inception_crop


@Registry.register("preprocess_ops.random_crop")
@utils.InKeyOutKey()
def get_random_crop(crop_size):
  """Makes a random crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.

  Returns:
    A function, that applies random crop.
  """
  crop_size = utils.maybe_repeat(crop_size, 2)

  def _crop(image):
    return tf.image.random_crop(image, (*crop_size, image.shape[-1]))

  return _crop


@Registry.register("preprocess_ops.central_crop")
@utils.InKeyOutKey()
def get_central_crop(crop_size=None):
  """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively. If `crop_size` is not
      specified, then the largest possible center crop will be taken.

  Returns:
    A function, that applies central crop.
  """
  if crop_size:
    crop_size = utils.maybe_repeat(crop_size, 2)

  def _crop(image):
    if crop_size:
      h, w = crop_size[0], crop_size[1]
    else:
      h = w = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

  return _crop


@Registry.register("preprocess_ops.flip_lr")
@utils.InKeyOutKey()
def get_random_flip_lr():
  """Flips an image horizontally with probability 50%."""

  def _random_flip_lr_pp(image):
    return tf.image.random_flip_left_right(image)

  return _random_flip_lr_pp


@Registry.register("preprocess_ops.vgg_value_range")
@utils.InKeyOutKey()
def get_vgg_value_range(
    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
    std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
):
  """VGG-style preprocessing, subtracts mean and divides by stddev.

  This preprocessing is very common for ImageNet pre-trained models since VGG,
  and to this day the standard for models coming from most PyTorch codes.

  Args:
    mean: Tuple of values to be subtracted. Default to widespread VGG values.
    std: Tuple of values to be divided by. Default to widespread VGG values.

  Returns:
    A function to rescale the values.
  """
  mean = tf.constant(mean, tf.float32)
  std = tf.constant(std, tf.float32)

  def _vgg_value_range(image):
    return (tf.cast(image, tf.float32) - mean) / std
  return _vgg_value_range


@Registry.register("preprocess_ops.clip_value_range")
@utils.InKeyOutKey()
def get_clip_value_range():
  mean = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
  std = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)

  def _clip_value_range(image):
    return (tf.cast(image, tf.float32) - mean) / std
  return _clip_value_range

@Registry.register("preprocess_ops.llava_value_range")
@utils.InKeyOutKey()
def get_llava_value_range(
  mean=0.5,
  std=0.5
):
  def _llava_value_range(image):
    # tf.print('in image', image)
    # tf.print('scaled centered image', tf.cast(image, tf.float32) / 255.0 - mean)
    return (tf.cast(image, tf.float32) / 255.0 - mean) / std
  return _llava_value_range

@Registry.register("preprocess_ops.convert_to_video")
@utils.InKeyOutKey()
def get_convert_to_video(num_frames):
  """Converts an image to a video with zero padded frames.

  Args:
    num_frames: total number of frames that the video should have.

  Returns:
    A function for converting an image to a video.
  """

  def _convert_to_video(image):
    return tf.pad(
        tf.expand_dims(image, axis=0),
        [[0, num_frames - 1], [0, 0], [0, 0], [0, 0]],
    )

  return _convert_to_video
