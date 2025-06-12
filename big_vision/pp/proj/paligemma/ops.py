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

"""pp ops."""

import functools
import string

from big_vision.pp import ops_text
from big_vision.pp import utils
from big_vision.pp.registry import Registry
import big_vision.pp.tokenizer as bv_tok
import numpy as np
import tensorflow as tf


@Registry.register('tokenizers.gemma')
def get_tokenizer_gemma(
    tokensets=(),
    model='gs://big_vision/gemma_tokenizer.model',
):
  # See (internal link) for colab playground.
  return ops_text.SentencepieceTokenizer(model=model, tokensets=tokensets)

@Registry.register('tokenizers.qwen2')
def get_tokenizer_qwen2(
  tokensets=(),
  model='Qwen/Qwen2-7B-Instruct',
  num_unused_tokens=0,
  num_extra_tokens=0,
):
  return ops_text.HuggingfaceTokenizer(model=model, tokensets=tokensets, num_unused_tokens=num_unused_tokens, num_extra_tokens=num_extra_tokens)

@functools.cache
def tokenize_constant(model, text, bos='no', eos='no', length=None):
  """Tokenize a constant string, with memoization."""
  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  tokenizer = bv_tok.get_tokenizer(model)
  tokens = tokenizer.to_int(
      text, bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

  if length is None:
    return tokens

  if len(tokens) > length:
    if eos == 'sticky':
      return np.r_[tokens[:length-1], tokens[-1]]
    else:
      return tokens[:length]
  else:
    return np.pad(tokens, [(0, length - len(tokens))],
                  constant_values=tokenizer.pad_token)


@Registry.register('preprocess_ops.tolen')
@utils.InKeyOutKey(indefault=None, outdefault=None, with_data=True)
def get_tolen(length, *, sticky_end=False, pad_value=None, pad_key=None):
  """Gets token to a fixed length."""
  def _tolen(x, data):
    if not length:
      return x

    xlen = tf.shape(x)[0]

    if sticky_end:
      trunc_fn = lambda: tf.concat([x[:length - 1], x[-1:]], axis=0)
    else:
      trunc_fn = lambda: x[:length]

    # Potentially get the pad value from a data key (to be tokenizer agnostic).
    pad_value_ = pad_value
    if pad_key:
      pad_value_ = data[pad_key]
      # If coming from a previous tokenization op, it's probably 1D; take first.
      if getattr(pad_value_, 'ndim', 0) == 1:
        pad_value_ = pad_value_[0]
    assert pad_value_ is not None, 'Need either pad_value or pad_key.'

    pad_fn = lambda: tf.pad(x, [(0, length - xlen)], constant_values=pad_value_)
    out = tf.cond(xlen >= length, trunc_fn, pad_fn)
    out.set_shape([length])
    return out
  return _tolen


@Registry.register('preprocess_ops.tok')
def get_tokenize(model, length=None, *, bos='no', eos='no',
                 text=None, key=None, inkey=None, outkey=None):
  """Tokenizes and optionally truncates/pads a string."""

  assert eos in ('no', 'yes', 'sticky')
  assert bos in ('no', 'yes')
  outkey_ = outkey or key
  inkey_ = inkey or key

  if text is not None:
    assert inkey is None, 'Either inkey or text, not both.'
    tokens = tokenize_constant(model, text, bos=bos, eos=eos, length=length)
    def _pp_tokenize_text(data):
      data[outkey_] = tokens
      return data
    return _pp_tokenize_text

  tokenizer = bv_tok.get_tokenizer(model)

  def _pp_tokenize(data):
    assert getattr(data[inkey_], 'ndim', 0) == 0, (
        f'Can only tokenize single string ({inkey_}, {data[inkey_].ndim}-D)')

    # tf.print("data inkey_", data[inkey_])
    toks = tokenizer.to_int_tf_op(
        data[inkey_], bos=bos == 'yes', eos=eos in ('yes', 'sticky'))

    tolen = get_tolen(
        length, sticky_end=eos == 'sticky',
        pad_value=bv_tok.get_tokenizer(model).pad_token,
        key='tmp',
    )
    toks = tolen({'tmp': toks})['tmp']

    data[outkey_] = toks
    return data
  return _pp_tokenize

@Registry.register('preprocess_ops.masked_concat')
def get_masked_concat(keys, outkey='text', id_based_masking=False,
                      user_token=872, assistant_token=77091, segment_key=None,
                      img_segment_key=None, logp_key=None, preamble_key=None, prefix_truncation_length=0, **masks):
  assert all(len(keys) == len(m) for m in masks.values()), (keys, masks)
  
  # Precompute segment numbers by extracting numeric characters from each key.
  segments = []
  for k in keys:
    # tf.print('key', k)
    if k == preamble_key:
      segments.append(-1)
      continue
    segment_num_str = ''.join(c for c in k if c.isnumeric())
    segment_num = int(segment_num_str) if segment_num_str else 0
    segments.append(segment_num)

  def _masked_concat(data):
    truncate = False
    text_list = []
    for k in keys:
      if not truncate:
        text_list.append(data[k])
        truncate = True
      elif 'prefix' in k:
        text_list.append(data[k][prefix_truncation_length:])
      else:
        text_list.append(data[k])
    """text_list = [
      data[k]
      if prefix_truncation_length == 0 or 'prefix' not in k or len([c for c in k if c.isnumeric()]) == 0 or int(''.join([c for c in k if c.isnumeric()])) == 0
      else data[k][prefix_truncation_length:]
      for k in keys
    ]"""
    data[outkey] = tf.concat(text_list, axis=0)
    if logp_key is not None:
      data[logp_key] = tf.concat((
        tf.zeros((data[outkey].shape[0]-data[logp_key].shape[0]), dtype=data[logp_key].dtype),
        data[logp_key]
      ), axis=0)
    # data[outkey] = tf.concat([data[k] for k in keys], axis=0)
    if id_based_masking:
      cumulative_num_user_tokens = (data[outkey] == user_token).astype(tf.int32).cumsum(axis=-1)
      cumulative_num_assistant_tokens = (data[outkey] == assistant_token).astype(tf.int32).cumsum(axis=-1)
    else:
      for mask_name, mask_vals in masks.items():
        m = []
        start_index = 0
        if preamble_key is not None:
          m.append(
            tf.fill(tf.shape(text_list[0]), mask_vals[0])
          )
          start_index = 1
          # tf.print(keys[0], mask_vals[0], segments[0], len(text_list[0]))
        # for j in range(start_index, len(keys)):
        #   tf.print('mask info', mask_name, keys[j], mask_vals[j], segments[j], len(text_list[j]))
        if mask_name == "mask_loss":
          m += [tf.fill(tf.shape(t), v)*data["segment_flags"][s] for k, v, s, t in zip(keys[start_index:], mask_vals[start_index:], segments[start_index:], text_list[start_index:])]
        else:
          m += [tf.fill(tf.shape(t), v) for k, v, t in zip(keys[start_index:], mask_vals[start_index:], text_list[start_index:])]
        data[mask_name] = tf.concat(m, axis=0)
        # if mask_name == 'mask_loss':
        #   tf.print('tf mask_loss:', data[mask_name])
    # tf.print('tf text', data[outkey], summarize=-1)
    if segment_key is not None:
      data[segment_key] = tf.concat([tf.fill(tf.shape(text_list[i]), segments[i]) for i in range(len(keys))], axis=0)
    return data

  return _masked_concat


@Registry.register('preprocess_ops.strfmt')
def get_strfmt(template, outkey='text', chars2bar=set()):
  """Formats a string template with content form the data dict."""
  for c in chars2bar:
    template = template.replace(c, '|')

  def _template(data):
    outputs = []
    parts = string.Formatter().parse(template)
    for (literal_text, field_name, format_spec, conversion) in parts:
      # For now, we keep it simple and don't support fancy format specs.
      # But we can add support to that via py_func as soon as we need it.
      assert not format_spec and not conversion
      outputs.append(tf.constant(literal_text))
      if field_name:
        value = data[field_name]
        # Convert any non-strings (numbers, vectors) to a string.
        if tf.convert_to_tensor(value).dtype != tf.string:
          value = tf.strings.format('{}', value, summarize=-1)
        outputs.append(value)
    data[outkey] = tf.strings.join(outputs)
    # tf.print("stfmt output", data[outkey])
    return data

  return _template


@Registry.register('preprocess_ops.strjoin')
@utils.InKeyOutKey()
def get_strjoin(glue):
  def _strjoin(x):
    return tf.strings.reduce_join(x, separator=glue)
  return _strjoin


@Registry.register('preprocess_ops.majority')
@utils.InKeyOutKey()
def get_majority():
  def _majority(x):
    val, _, count = tf.unique_with_counts(x)  # Sadly, stablesorted.
    return val[tf.argmax(count)]
  return _majority


@Registry.register('preprocess_ops.getidx')
def getidx(inkey, index_key, outkey=None):
  """Indexes a tensor and stores result in outkey."""
  def _getidx(data):
    idx = data[index_key]
    array = data[inkey]
    data[outkey or inkey] = array[idx]
    return data
  return _getidx
