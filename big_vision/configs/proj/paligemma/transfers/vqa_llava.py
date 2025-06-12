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

# pylint: disable=line-too-long
r"""Example config for finetuning PaliGemma to a task stored in the JSON-L file, designed to fit on four L4 GPU.

Can be used as a starting point to finetune PaliGemma model. If you prefer to 
use tfds-based data input, check out other transfer configs as examples.

Command to run this config:

```
env BV_GEMMA_DIR=ckpts/ python -m big_vision.trainers.proj.paligemma.train \
    --config big_vision/configs/proj/paligemma/transfers/forkme.py \
    --workdir workdirs/`date '+%m-%d_%H%M'`
```
"""

import os
import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers.common import combine_and_keep_train, combine_and_keep_eval, TOKENIZER


NUM_IMAGES = int(os.environ['NUM_IMAGES'])
QWEN_TOKENIZER = 'qwen2()'
BASE_VOCAB_SIZE = 151647
num_unused_tokens = int(os.environ['VOCAB_SIZE'])-BASE_VOCAB_SIZE
QWEN_TOKENIZER_EXTRA = f'qwen2(num_unused_tokens={num_unused_tokens}, num_extra_tokens=1024)'
NUM_CONVERSATION_SEGMENTS = int(os.environ['NUM_CONVERSATION_SEGMENTS'] if 'NUM_CONVERSATION_SEGMENTS' in os.environ else 1)

input_template = "system\\nYou are a helpful assistant.<?im_end?>\\n<?im_start?>user\\n{question}<?im_end?>\\n<?im_start?>assistant"

def training_data(res, text_len, prefix_truncation_length=0, num_images=NUM_IMAGES, data_limit=None):
  """Creates training data config."""
  c = bvcc.parse_arg('')  # Just make a configdict without extra import.
  c.data = dict(
      name='bv:jsonl',
      fname=os.path.join(os.environ['REMOTE_PROJECT_HOME'], f'combined_{NUM_IMAGES}frames', 'train.jsonl'),
      fopen_keys={f'image_{i}': '' for i in range(num_images)},
      video_num_frames=num_images,
      max_segments=NUM_CONVERSATION_SEGMENTS,
      stop=data_limit,
  )
  fmt_question_answer = [
    'strfmt("'+input_template+'", outkey="prefix0")',
    'copy(inkey="answer", outkey="suffix0")',
  ]
  if NUM_CONVERSATION_SEGMENTS > 1:
    fmt_question_answer = [
      'strfmt("'+input_template.replace('{question}', '{question'+str(s)+'}')+'", outkey="prefix'+str(s)+'", chars2bar={"?"})'
      for s in range(NUM_CONVERSATION_SEGMENTS)
    ]+[
      'copy(inkey="answer'+str(s)+'", outkey="suffix'+str(s)+'")'
      for s in range(NUM_CONVERSATION_SEGMENTS)
    ]
  preamble_key = None
  c.pp = '|'.join([
      # Read and prepare the image by just resizing it:
      f'decode(key="image_{str(i)}", precise=True)|resize_llava_square({res}, key="image_{str(i)}", antialias=True)'
      for i in range(num_images)
    ]+[
      f'stack_images(inkeys=['+', '.join([f'"image_{str(i)}"' for i in range(num_images)])+'], outkey="image")',
      'llava_value_range(0.5, 0.5)',
      f'video_ensure_shape("image", {(num_images, res, res, 3)})',
      f'mask_video(key="mask_image", num_images={num_images})',
      f'video_ensure_shape("mask_image", {(num_images,)})',
    ]+fmt_question_answer+[
      combine_and_keep_train(text_len, keep=('mask_image',), keep_len=(num_images,), tokenizer_str=QWEN_TOKENIZER_EXTRA, bos=True, eos=True, num_segments=NUM_CONVERSATION_SEGMENTS, prefix_truncation_length=prefix_truncation_length),
  ])
  # Keep the whole dataset in RAM after first pass. Useful optimization for
  # small/mid-size datasets, but risks a host OOM for large datasets.
  c.cache_raw = False
  c.ordered = True
  return c

def add_eval_acc(c, res, text_len, dataset_name, answer_key="answer", is_image=False, to_lower=False, eval_frequency=1/10, start=None, stop=None, task="vqa", decode_fn="decode_with_logp", prefix_truncation_length=0):
  """Perplexity evaluator to test runs before implementing the real deal."""
  num_images = NUM_IMAGES
  c_data = training_data(res, text_len, num_images=num_images)  # Use mostly same settings as training.
  # is_image = True
  if is_image:
    if 'NUM_IMAGES_FOR_SINGLE_IMAGE' in os.environ and len(os.environ['NUM_IMAGES_FOR_SINGLE_IMAGE']) > 0:
      num_images = int(os.environ['NUM_IMAGES_FOR_SINGLE_IMAGE'])
  fmt_question = ['strfmt("'+input_template+'", outkey="prefix", chars2bar={"?"})']
  c_data.pp = '|'.join([
      f'decode(key="image_{str(i)}", precise=True)|resize_llava_square({res}, key="image_{str(i)}", antialias=True)'
      for i in range(num_images)
    ]+[
      f'stack_images(inkeys=['+', '.join([f'"image_{str(i)}"' for i in range(num_images)])+'], outkey="image")',
      'llava_value_range(0.5, 0.5)',
      f'video_ensure_shape("image", {(num_images, res, res, 3)})',
      f'mask_video(key="mask_image", num_images={num_images})',
      f'video_ensure_shape("mask_image", {(num_images,)})',
    ]+fmt_question+[
      combine_and_keep_eval(text_len, keep=(answer_key, 'question_str', 'question_id', 'mask_image'), keep_len=(None, None, None, num_images), tokenizer_str=QWEN_TOKENIZER_EXTRA, bos=True, eos=False, prefix_truncation_length=prefix_truncation_length),
  ])
  c_data.data['video_num_frames'] = num_images
  c_data.data['fopen_keys'] = {f'image_{i}': '' for i in range(num_images)}
  if start is not None:
    c_data.data["start"] = start
  if stop is not None:
    c_data.data["stop"] = stop

  c.evals[f'val/{dataset_name}/acc'] = dict(
      type=f'proj.paligemma.transfers.{task}', pred=decode_fn,
      pred_kw={'max_decode_len': text_len if 'MAX_DECODE_LEN' not in os.environ or len(os.environ['MAX_DECODE_LEN']) == 0 else int(os.environ['MAX_DECODE_LEN'])}, # , 'beam_size': 1},
      log_percent=eval_frequency,
      tokenizer=QWEN_TOKENIZER_EXTRA,
      outfile=f'{{workdir}}/val_{dataset_name}.json',
      data={
        **c_data.data,
        'fname': os.path.join(os.environ['REMOTE_PROJECT_HOME'], f'combined_{NUM_IMAGES}frames', f'val_{dataset_name}.jsonl'),
        'split': 'val',
      },
      to_lower=to_lower,
      pp_fn=c_data.pp,
      num_samples=1 if 'EVAL_SAMPLES' not in os.environ or len(os.environ['EVAL_SAMPLES']) == 0 else int(os.environ['EVAL_SAMPLES']),
  )

def get_config(arg=None):
  """Config for training."""
  # You probably do NOT want to add settings here. The `arg` way of settings is
  # really only for things you'd want to sweep and which affect MULTIPLE config
  # settings at once or go into the pp string.
  c = bvcc.parse_arg(arg, res=384, text_len=int(os.environ['TEXT_LEN']), batch_size=int(os.environ['BATCH_SIZE']), eval_text_len=int(os.environ['EVAL_TEXT_LEN']), max_decode_len=int(os.environ['MAX_DECODE_LEN']),
                     freeze_vit=False, freeze_llm=False,
                     run_local=False)

  insert_images_at_index = 14
  if 'RESOLUTION' in os.environ and len(os.environ['RESOLUTION']) > 0:
    c.res = int(os.environ['RESOLUTION'])
  data_limit = None
  if 'DATA_LIMIT' in os.environ and len(os.environ['DATA_LIMIT']) > 0:
    data_limit = int(os.environ['DATA_LIMIT'])
  c.input = training_data(c.res, c.text_len, prefix_truncation_length=insert_images_at_index, data_limit=data_limit)

  # Instead of epochs, you can also use `total_examples` or `total_steps`.
  if 'STEPS' in os.environ and len(os.environ['STEPS']) > 0 and os.environ['STEPS'].isnumeric():
    c.total_steps = int(os.environ['STEPS'])
  else:
    c.total_epochs = int(os.environ['EPOCHS'])
  c.input.batch_size = c.batch_size
  c.optax_name = 'scale_by_adam'
  c.lr = 1e-5
  c.wd = 3e-7
  c.grad_clip_norm = 1.0
  c.gradient_accumulation_steps = 1
  if 'GRADIENT_ACCUMULATION_STEPS' in os.environ and len(os.environ['GRADIENT_ACCUMULATION_STEPS']) > 0:
    c.gradient_accumulation_steps = int(os.environ['GRADIENT_ACCUMULATION_STEPS'])
  if 'LEARNING_RATE' in os.environ and len(os.environ['LEARNING_RATE']) > 0:
    c.lr = float(os.environ['LEARNING_RATE'])
  c.label_smoothing = 0.0
  eval_frequency = 1/20

  if 'FREEZE_VIT' in os.environ and os.environ['FREEZE_VIT'] == '1':
    c.freeze_vit = True

  # Learning-rate schedule. Probably is fine like this.
  sched = dict(decay_type='cosine', warmup_percent=0.05)
  c.schedule = [
      # ('img/.*', None if c.freeze_vit else sched),
      ('img/Transformer/.*', None if c.freeze_vit else sched),
      ('img/embedding/.*', None if c.freeze_vit else sched),
      ('img/pos_embedding', None if c.freeze_vit else sched),
      ('img/head1/.*', sched),
      ('img/head2/.*', sched),
      ('img/image_newline/.*', sched),
      ('llm/.*', None if c.freeze_llm else sched),
  ]
  c.prefetch_to_device = 1

  decode_fn = "decode_with_logp"
  c.evals = {}
  for dataset_name, answer_key, is_image, to_lower, task in zip(['perception_test'], ['answers']*1, [False]*1, [False]*1, ['vqa']*1):
      add_eval_acc(c, c.res, c.eval_text_len, dataset_name, answer_key=answer_key, is_image=is_image, to_lower=to_lower, eval_frequency=eval_frequency, task=task, stop=data_limit, decode_fn=decode_fn, prefix_truncation_length=insert_images_at_index)
  
  # Model section.
  c.model_name = 'proj.paligemma.paligemma'
  c.model = {}
  c.model_init = os.environ['PRETRAINED_CKPT'] # 'raw_gemma_actual2_2b_siglip_so400m_224'
  if 'raw' in c.model_init:
    c.model_load = {'img_load_kw': {'dont_load': ['head/bias', 'head/kernel']}}
  downsample = int(os.environ['DOWNSAMPLE'])
  if downsample > 1:
    c.model.img_token_pooling = {"window_shape": (1, downsample, downsample), "strides": (1, downsample, downsample), "type": "bilinear", "antialias": False}
  PRECISION = 'default'
  add_image_newline = True
  if 'IMAGE_NEWLINE' in os.environ and int(os.environ['IMAGE_NEWLINE']) == 0:
    add_image_newline = False
  c.model.img = dict(variant='So400m/14', pool_type='none', scan=False, pool_mlp2xgelu=True, depth=26, add_image_newline=add_image_newline, built_in_attn_class=False, precision=PRECISION, mlp_chunk_size=None)
  c.model.llm = dict(
    variant=os.environ['VARIANT'],
    vocab_size=int(os.environ['VOCAB_SIZE']), # 152064, # 256_000 + 1024 + 128,
    true_vocab_size=BASE_VOCAB_SIZE,
    dropout=0.0,
    canonical_attn_types=(os.environ['ATTN_TYPE'], os.environ['ATTN_TYPE']),
    precision=PRECISION,
  ) # , sliding_window_prefix_size=256, sliding_window_size=64)
  c.model.llm['rope_max_wavelength'] = 1000000
  c.model.insert_images_at_index = insert_images_at_index
  c.model.text_causal = True
  c.model.vision_causal = True
  c.model.vision_attend_to_text = False
  if 'ROPE_MAX_WAVELENGTH' in os.environ and len(os.environ['ROPE_MAX_WAVELENGTH']) > 0:
    c.model.rope_max_wavelength = int(os.environ['ROPE_MAX_WAVELENGTH'])
  if 'IMG_TOKEN_LENGTH' in os.environ and len(os.environ['IMG_TOKEN_LENGTH']) > 0:
    c.model.img_token_length = int(os.environ['IMG_TOKEN_LENGTH'])
    c.model.llm["cache_size"] = c.model.img_token_length+c.eval_text_len+c.max_decode_len
    c.model.llm["image_end_index"] = c.model.img_token_length+c.model.insert_images_at_index
  else:
    c.model.llm["cache_size"] = None
  if 'REMAT_POLICY' in os.environ and len(os.environ['REMAT_POLICY']) > 0:
    c.model.llm["remat_policy"] = os.environ['REMAT_POLICY']
    # c.model.img["remat_policy"] = os.environ['REMAT_POLICY']
  if 'ATTENTION_IMPL' in os.environ and len(os.environ['ATTENTION_IMPL']) > 0:
    c.model.llm['attention_impl'] = os.environ['ATTENTION_IMPL']
  if 'MLP_CHUNK_SIZE' in os.environ and len(os.environ['MLP_CHUNK_SIZE']) > 0:
    c.model.llm['mlp_chunk_size'] = int(os.environ['MLP_CHUNK_SIZE'])

  # FSDP strategy.
  # fsdp_axis = 'fsdp'
  fsdp_axis = os.environ['FSDP_AXIS']
  sequence_axis = 'sequence'
  sequence_sharding = 1
  tensor_sharding = 1
  if 'SEQUENCE_SHARDING' in os.environ and len(os.environ['SEQUENCE_SHARDING']) > 0:
    sequence_sharding = int(os.environ['SEQUENCE_SHARDING'])
  if 'TENSOR_SHARDING' in os.environ and len(os.environ['TENSOR_SHARDING']) > 0:
    tensor_sharding = int(os.environ['TENSOR_SHARDING'])
  c.mesh_dims = f'1,-1,{sequence_sharding},{tensor_sharding}'
  c.mesh = [(fsdp_axis, -1)]
  c.sharding_strategy = [('.*', f'fsdp(axis="{fsdp_axis}")')]
  c.sharding_rules = [('act_batch', fsdp_axis), ('act_len', sequence_axis), ('flattened_images', (fsdp_axis, sequence_axis)), ('dp', 'replica'), ('tp', 'tensor'), ('sp', sequence_axis), ('act_emb', 'tensor')]

  c.input.shuffle_buffer_size = 1000
  c.log_training_steps = 50
  c.ckpt_steps = int(os.environ['CKPT_STEPS']) # 2_000
  c.keep_ckpt_steps = c.ckpt_steps
  c.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'proj.paligemma.ops', 'proj.paligemma.video']

  c.extra_model_args = ('mask_image', 'segment_ids')
  c.eval_extra_model_args = ('mask_image',)

  c.eval_only = False
  if 'EVAL_ONLY' in os.environ and os.environ['EVAL_ONLY'] == '1':
    c.eval_only = True

  if "RESUME" in os.environ and len(os.environ["RESUME"]) > 0:
    c.resume = os.environ["RESUME"]

  c.seed = 0

  return c
