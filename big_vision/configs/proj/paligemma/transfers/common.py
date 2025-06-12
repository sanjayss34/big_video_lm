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

"""Common things across all transfer configs."""


TOKENIZER = 'gemma(tokensets=("loc", "seg"))'


def tok(tokenizer_str, **kw):
  """Creates the tokenization preprocessing string."""
  # Single entry point so that it's consistent everywhere and easier to switch.
  kw.setdefault('model', tokenizer_str)
  kw = ', '.join(f'{k}={repr(v)}' for k, v in kw.items())
  return f'tok({kw})'


def combine_and_keep_train(text_len, before=(), sep='\n', keep=tuple(), keep_len=tuple(), tokenizer_str=TOKENIZER, bos=True, eos=True, num_segments=1, prefix_truncation_length=0, img_segment_key=None, num_images=None, logp_key=None, preamble_key=None):
  masked_concat_keys = "["+", ".join([f"\"prefix{s}\", \"septok{s}\", \"suffix{s}\"" for s in range(num_segments)])+"]"
  mask_ar_list = mask_loss_list = "["+", ".join(["0, 0, 1" for _ in range(num_segments)])+"]"
  extra_masked_concat_args = ""
  if img_segment_key is not None:
    extra_masked_concat_args += f", img_segment_key=\"{img_segment_key}\""
  if preamble_key is not None:
    extra_masked_concat_args += f", preamble_key=\"{preamble_key}\""
    masked_concat_keys = masked_concat_keys.replace("[", f"[\"{preamble_key}\", ")
    mask_ar_list = mask_ar_list.replace("[", f"[0, ")
    mask_loss_list = mask_ar_list
    before = (*before, tok(key=preamble_key, bos='yes' if bos else 'no', tokenizer_str=tokenizer_str))
  return '|'.join([
      *before,
    ]+[
      tok(key='septok'+str(s), text=sep, tokenizer_str=tokenizer_str)
      for s in range(num_segments)
    ]+[
      tok(key='prefix'+str(s), bos='yes' if bos else 'no', tokenizer_str=tokenizer_str)
      for s in range(num_segments)
    ]+[
      tok(key='suffix'+str(s), eos='yes' if eos else 'no', tokenizer_str=tokenizer_str)
      for s in range(num_segments)
    ]+[
      f'masked_concat({masked_concat_keys}, segment_key=\"segment_ids\", mask_ar={mask_ar_list}, mask_loss={mask_loss_list}, prefix_truncation_length={prefix_truncation_length}{extra_masked_concat_args})',
      f'tolen({text_len+1}, pad_value=0, key="text")',  # Value doesn't matter.
      f'tolen({text_len+1}, pad_value=1, key="mask_ar")',
      f'tolen({text_len+1}, pad_value=0, key="mask_loss")',
      f'tolen({text_len+1}, pad_value={num_segments+2}, key="segment_ids")',
    ]+(
      [f'tolen({text_len+1}, pad_value=0, key="{logp_key}")']
      if logp_key is not None else []
    )+[
        f'tolen({l}, pad_value=0, key="{key}")'
        for key, l in zip(keep, keep_len)
        if l is not None
    ]+[
        'keep(' + ', '.join(f'"{x}"' for x in (
            'image', 'text', 'mask_ar', 'mask_loss', 'segment_ids') + tuple(keep) + ((img_segment_key,) if img_segment_key is not None else tuple())) + ')',
    ]
  )
  """return '|'.join([
      *before,
      tok(key='prefix', bos='yes' if bos else 'no', tokenizer_str=tokenizer_str),
      tok(key='suffix', eos='yes' if eos else 'no', tokenizer_str=tokenizer_str),
      tok(key='septok', text=sep, tokenizer_str=tokenizer_str),
      # If masks confuse you, see (internal link)
      'masked_concat(["prefix", "septok", "suffix"], mask_ar=[0, 0, 1], mask_loss=[0, 0, 1])',  # pylint: disable=line-too-long
      # For training, we +1 since the trainer removes EOS.
      f'tolen({text_len+1}, pad_value=0, key="text")',  # Value doesn't matter.
      f'tolen({text_len+1}, pad_value=1, key="mask_ar")',
      f'tolen({text_len+1}, pad_value=0, key="mask_loss")',
      # 'keep("image", "text", "mask_ar", "mask_loss")',
  ]+[
      f'tolen({l}, pad_value=0, key="{key}")'
      for key, l in zip(keep, keep_len)
      if l is not None
  ]+[
      'keep(' + ', '.join(f'"{x}"' for x in (
          'image', 'text', 'mask_ar', 'mask_loss') + tuple(keep)) + ')',
  ])"""


def combine_and_keep_eval(text_len, keep=tuple(), before=(), sep='\n', keep_len=tuple(), tokenizer_str=TOKENIZER, bos=True, eos=False, img_segment_key=None, num_images=None, preamble_key=None, prefix_truncation_length=0):
  mask_ar_list = "[0, 0, 1]"
  masked_concat_keys = "[\"prefix\", \"septok\", \"suffix\"]"
  extra_masked_concat_args = ""
  if img_segment_key is not None:
    extra_masked_concat_args += f", img_segment_key=\"{img_segment_key}\""
  if preamble_key is not None:
    extra_masked_concat_args += f", segment_key=\"segment_ids\", preamble_key=\"{preamble_key}\""
    masked_concat_keys = masked_concat_keys.replace("[", f"[\"{preamble_key}\", ")
    mask_ar_list = mask_ar_list.replace("[", f"[0, ")
    before = (*before, tok(key=preamble_key, bos='yes' if bos else 'no', tokenizer_str=tokenizer_str))
  mask_input_list = "["+",".join(["1" for _ in range(mask_ar_list.count(",")+1)])+"]"
  return '|'.join([
      *before,
      # Same as training, except that suffix is now the empty string.
      # Meaning, we create text as [prefix separator pad],
      # and the mask accordingly as [0 0 1] (with repeats of respective lengths)
      tok(key='prefix', bos='yes' if bos else 'no', tokenizer_str=tokenizer_str),
      tok(key='septok', text=sep, tokenizer_str=tokenizer_str),
      # At eval time, there can be also a suffix key in the data. If so it is
      # tokenized without EOS and decoding will continue from it.
      'setdefault("suffix", "")',
      tok(key='suffix', eos='yes' if eos else 'no', tokenizer_str=tokenizer_str),
      # If masks confuse you, see (internal link)
      f'masked_concat({masked_concat_keys}, mask_ar={mask_ar_list}, mask_input={mask_input_list}, prefix_truncation_length={prefix_truncation_length}{extra_masked_concat_args})',
      f'tolen({text_len}, pad_value=0, key="text")',  # value doesn't matter.
      f'tolen({text_len}, pad_value=1, key="mask_ar")',
      f'tolen({text_len}, pad_value=0, key="mask_input")',
  ]+[
      f'tolen({l}, pad_value=0, key="{key}")'
      for key, l in zip(keep, keep_len)
      if l is not None
  ]+[
      # And we need to keep everything that makes our evaluator happy.
      'keep(' + ', '.join(f'"{x}"' for x in (
          'image', 'text', 'mask_ar', 'mask_input') + tuple(keep) + ((img_segment_key,) if img_segment_key is not None else tuple())) + ')',
  ])
