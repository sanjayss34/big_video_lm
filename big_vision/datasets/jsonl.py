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

"""Simple data input from .jsonl files."""

import hashlib
import json
from multiprocessing.pool import ThreadPool
import os
import math
import tempfile
import urllib.request

from absl import logging
import big_vision.datasets.core as ds_core
from big_vision.pp import utils
import imageio.v3 as iio
import jax
import numpy as np
import overrides
import tensorflow as tf
import tensorflow_io as tfio
from tqdm import tqdm
# import random
# random.seed(42)

DEFAULT_NUM_PARALLEL_CALLS = 100
def cached_download(url, dest=None, verbose=True):
  """Download `url` to local file and return path to that, but with caching."""
  # NOTE: there is a small chance of saving corrupted data if the process is
  # interrupted in the middle of writing the file. Then, reading in the input
  # pipeline will fail, and the fix is to nuke the temp folder.

  # Compute a temp name based on the URL, so we can check if we already
  # downloaded it before.
  dest = dest or os.path.join(tempfile.gettempdir(), "bv")
  os.makedirs(dest, exist_ok=True)
  dest = os.path.join(dest, hashlib.md5(url.encode()).hexdigest())

  # NOTE: we should use last-modified header to know whether to re-download.
  if os.path.isfile(dest):
    return dest

  if verbose:
    print(f"\rRetrieving {url} into {dest}", end="", flush=True)

  with urllib.request.urlopen(url) as f:
    data = f.read()
  with open(dest, "wb+") as f:
    f.write(data)
  return dest


class DataSource(ds_core.DataSource):
  """.jsonl DataSource."""

  def __init__(self, fname, *, fopen_keys=(), download_keys=(),
               start=0, stop=None,
               split=None, video_num_frames=None, video_frame_size=None, max_segments=None):
    """Create data-source that's jsonl + data files (eg images).

    This correctly supports multi-host in that each host only reads a subset of
    the dataset automatically. However, currently, all hosts download all items
    if `download_keys` is specified. TODO: b/lbeyer - This can be improved.

    Args:
      fname: str, the path to the jsonl file that holds the dataset.
      fopen_keys: collection of str or dict, the keys in the dataset whose
        string value actually is a file-path that should be opened and read,
        and its content is what goes into the batch (eg image filenames
        commonly ["image"]).
        If a dict, the values are folders prefixed to the filenames.
        Supports gs:// for reading from buckets.
      download_keys: collection of str, the keys in the dataset whose string
        value actually is a URL from which the file should be downloaded first.
        files are downloaded to a persistent tmp folder using the URL hash as
        filename. If the file already exists, the download is skipped.
        Must be a subset of `fopen_keys`.
      start: int, index of the first row to use; use for slicing the data.
      stop: int or inf, index of the row after the last one to use.
      video_num_frames: int, number of video frames to extract, if this is a video dataset

    Note:
      This simple data input does not allow for nested/hierarchical values,
      or in any way more complicated values like vectors. Use TFDS for that.

      The way start/stop arguments are used is as in list slicing[start:stop].
    """
    self.examples = []
    self.video_num_frames = video_num_frames
    self.video_fps = 30
    self.video_frame_size = video_frame_size

    with tf.io.gfile.GFile(fname) as f:
      for i, line in enumerate(f):
        if (start or 0) <= i < (stop or float("inf")):
          try:
            self.examples.append(json.loads(line))
          except json.decoder.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in line {i}:\n{line}") from e

    for datum in self.examples:
        # datum['question'] = 'How many people are there? Write only the number and no other text.'
        if 'images' in datum:
            assert 'image_root' in datum
            for i, im in enumerate(datum['images']):
                if isinstance(im, int):
                    datum[f'image_{i}'] = os.path.join(datum['image_root'], 'frame_{:04d}.jpg'.format(im))
                else:
                    datum[f'image_{i}'] = im
            del datum['image_root']
            del datum['images']
        if max_segments is not None:
            # datum['num_segments'] = len([k for k in datum if 'answer' in k])
            datum['segment_flags'] = [1 for _ in range(len([k for k in datum if 'answer' in k]))]+[0 for _ in range(max_segments-len([k for k in datum if 'answer' in k]))]
            if max_segments > 1:
              for i in range(max_segments):
                  if f'answer{i}' not in datum:
                      assert f'question{i}' not in datum
                      datum[f'question{i}'] = ""
                      datum[f'answer{i}'] = ""
        if "question" in datum:
          datum["question_str"] = datum["question"]
    print('len examples', len(self.examples))
    if self.video_num_frames is not None:
        for i in tqdm(range(len(self.examples))):
            if 'image' in self.examples[i]:
                assert all(['image_' not in key for key in self.examples[i]])
                self.examples[i]['image_0'] = self.examples[i]['image']
                del self.examples[i]['image']
            max_frame_idx = max([int(key.split('image_')[1]) for key in self.examples[i] if 'image_' in key])
            if max_frame_idx >= self.video_num_frames:
                new_images = {}
                for j in range(self.video_num_frames):
                    new_images[f'image_{j}'] = self.examples[i][f'image_{int(j*max_frame_idx/self.video_num_frames)}']
                for j in range(self.video_num_frames, max_frame_idx+1):
                    del self.examples[i][f'image_{j}']
                for key in new_images:
                    self.examples[i][key] = new_images[key]
                max_frame_idx = self.video_num_frames-1
            for j in range(max_frame_idx+1, self.video_num_frames):
                self.examples[i][f'image_{j}'] = self.examples[i]['image_0']
            self.examples[i]['mask_image'] = max_frame_idx+1
    # random.shuffle(self.examples)

    if download_keys:
      for k in download_keys:
        assert k in fopen_keys, (
            f"{k} in download_keys but missing from fopen_keys {fopen_keys}")

      # TODO: b/lbeyer - use info from trainer instead, move that to utils.
      logging.info(  # pylint: disable=logging-fstring-interpolation
          f"\u001b[33mNOTE\u001b[0m: Downloading {download_keys} "
          f"for dataset {fname} ({len(self.examples)} examples) ...")

      def _dl_one(ex):
        for k in download_keys:
          ex[k] = cached_download(ex[k])

      ThreadPool(100).map(_dl_one, self.examples)
      print("Done")
      logging.info("\u001b[33mNOTE\u001b[0m: Done downloading.")

    # Normalize.
    if isinstance(fopen_keys, (list, tuple)):
      self.fopen_keys = {k: "" for k in fopen_keys}
    else:
      self.fopen_keys = fopen_keys or {}

    # We need to apply fopen path prefix here already, because doing so while
    # actually reading the files in TF, things are symbolic :(
    if any([len(dirname) > 0 for dirname in self.fopen_keys.values()]):
      for ex in tqdm(self.examples):
        for k, dirname in self.fopen_keys.items():
          if len(dirname) > 0:
            ex[k] = os.path.join(dirname, ex[k])

  def _indices(self, *, process_split=True, process_index=None, split_by_modulus=False):
    indices = np.arange(len(self.examples))

    if not process_split:
      return list(indices)

    pid = jax.process_index() if process_index is None else process_index
    if split_by_modulus:
      return indices[pid::jax.process_count()]
    return list(np.array_split(indices, jax.process_count())[pid])

  @overrides.overrides
  def get_tfdata(self, ordered=False, *, process_split=True, allow_cache=True, split_by_modulus=False):
    del allow_cache  # We don't cache anything anyways.
    assert not process_split or len(self.examples) >= jax.process_count(), (
        f"Process splitting the data with fewer examples than processes ({str(len(self.examples))}, {str(jax.process_count())})!?")

    my_idxs = self._indices(process_split=process_split, split_by_modulus=split_by_modulus)
    if not ordered:
      np.random.shuffle(my_idxs)

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: ({"id": str(i), **self.examples[i]} for i in my_idxs),
        output_signature={
            "id": _guess_signature("0"),
            **{k: _guess_signature(v) for k, v in self.examples[0].items()},
            })

    def _read_files(example):
      for k in self.fopen_keys:
        example[k] = tf.io.read_file(example[k])
      return example
    dataset = dataset.map(_read_files)
    # dataset = dataset.map(_read_files, num_parallel_calls=DEFAULT_NUM_PARALLEL_CALLS)

    return dataset

  @property
  @overrides.overrides
  def total_examples(self):
    return len(self.examples)

  @overrides.overrides
  def num_examples_per_process(self):
    return [len(self._indices(process_index=pid))
            for pid in range(jax.process_count())]


def _guess_signature(value):
  return tf.TensorSpec.from_tensor(tf.constant(value))
