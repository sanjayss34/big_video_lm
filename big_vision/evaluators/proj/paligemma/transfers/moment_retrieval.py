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

import functools
import string

import numpy as np
import jax
from jax.experimental import multihost_utils
import flax.linen as nn

import big_vision.evaluators.common as c
import big_vision.pp.tokenizer
import big_vision.utils as u
import editdistance


P = jax.sharding.PartitionSpec
# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = "jit"


class Evaluator:
  """Evaluator for moment retrieval

  This evaluator expects the batch to contain a field `question_id` and a field
  `answer` for the ground-truth. Both the ground-truth and the prediction are
  expected to be formatted such that they contain two integers, of which the
  first is the start of the interval and the second is the end of the interval.

  The field names used when writting the json result can be controlled with
  `out_question_key` and `out_answer_key`.
  """

  def __init__(
      self, predict_fn, tokenizer, to_lower=False,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id", out_answer_key="answer", full_logp=True, num_samples=1,
      *, data, devices, extra_model_args=[], sharding_config=None, sharding_rules=None, **kw):
    self.non_numeric_keys = {"answers", "answer", "question_id", "question_str"}
    self.data_options = data
    self.data_kw = kw
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu=self.non_numeric_keys,
        keep_all_on_cpu=False,
        sharding_config=sharding_config,
        data=self.data_options, devices=devices, **self.data_kw)

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))
    self.out_question_key = out_question_key
    self.out_answer_key = out_answer_key
    self.full_logp = full_logp
    self.num_samples = num_samples

    self.extra_model_args = extra_model_args

    # We'll need the tokenizer to detokenize the model outputs later.
    print('tokenizer', tokenizer)
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.postproc = (lambda s: ''.join([c for c in s.lower() if c not in string.punctuation])) if to_lower else lambda s: ''.join([c for c in s if c not in string.punctuation])
    self.sharding_config = sharding_config
    self.sharding_rules = sharding_rules
    self.mesh = None
    if self.sharding_config is not None:
      mesh_helper = self.sharding_config.get_mesh()
      self.mesh = mesh_helper.mesh
    self.devices=devices
    self.decode = functools.partial(
        predict_fn, devices=devices, eos_token=self.tok.eos_token, best_of_n=self.num_samples, extra_model_args=self.extra_model_args, sharding_config=sharding_config, sampler=("greedy" if self.num_samples == 1 else "temperature"))

  def run(self, train_state, outfile_suffix="", start_index=None, end_index=None):
    """Does one evaluation run, yields metrics."""

    recall1_by_iou = {3: 0, 5: 0, 7: 0}
    total = 0
    json_out = []
    get_data_iter = self.get_data_iter
    steps = self.steps
    if start_index is not None or end_index is not None:
      data_options = self.data_options
      if start_index is not None:
        data_options["start"] = start_index
      if end_index is not None:
        data_options["end"] = end_index
      get_data_iter, steps = c.eval_input_pipeline(
          keep_on_cpu=self.non_numeric_keys,
          keep_all_on_cpu=False,
          sharding_config=self.sharding_config,
          data=data_options, devices=self.devices, **self.data_kw
      )
    for s, batch in zip(range(steps), get_data_iter()):
      print('Batch', s, 'out of', steps)
      if self.sharding_config is not None:
        sharding_map = jax.tree.map(
          lambda spec: jax.sharding.NamedSharding(self.mesh, spec),
          self.sharding_config.get_batch_sharding().apply({key: batch[key] for key in batch if key not in self.non_numeric_keys})
        )
        sharded_batch = batch
      else:
        sharded_batch = batch
      if self.sharding_config is not None:
        with self.mesh, nn.logical_axis_rules(self.sharding_rules):
          tokens_n_samples, logp_n_samples = self.decode(train_state, sharded_batch)
      else:
        tokens_n_samples, logp_n_samples = self.decode(train_state, sharded_batch)
      for sample_i in range(self.num_samples):
        with jax.transfer_guard("allow"):
          tokens = tokens_n_samples[sample_i]
          logp = logp_n_samples[sample_i]
        # print('after decode')
        sharded_mask = sharded_batch['_mask']

        # (local_batch,) that indicates padding examples (0) vs real examples (1).
        text = None
        if self.sharding_config is None:
          tokens = u.get_local_slice_from_fsarray(tokens)
          ex_masks = u.get_local_slice_from_fsarray(sharded_mask)
          logp = u.get_local_slice_from_fsarray(logp)
          print('tokens', tokens.shape, 'ex_masks', ex_masks.shape, 'answer', len(batch['answer']))
        else:
          with jax.transfer_guard("allow"):
            tokens = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(tokens, self.mesh, tokens.sharding.spec))
            ex_masks = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(sharded_mask, self.mesh, sharded_mask.sharding.spec))
            logp = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(logp, self.mesh, logp.sharding.spec))
          print('tokens', tokens.shape, 'ex_masks', ex_masks.shape, 'answer', len(batch['answer']))

        text = batch["text"]
        # print('going to postprocessing')
        # Turn predictions into texts and then scores, one by one.
        for i in range(len(tokens)):
          if ex_masks[i] == 0:  # Skip last-batch padding examples
            continue

          answer = self.postproc(self.tok.to_str(tokens[i], stop_at_eos=True))
          print('prediction', answer)
          print('pred tokens', tokens[i])
          # print(i, 'question:', self.tok.to_str(text[i], stop_at_eos=True))
          # print('output:', answer)
          print('ground truth', self.tok.to_int(batch['answer'][i]))

          # Now we have two commonly used VQA evaluation modes:
          gts = []
          iou = 0
          if "answer" in batch:
            gt = self.postproc(batch["answer"][i])
            numeric_gt_tokens = []
            for token in gt.split():
              if token.isnumeric():
                numeric_gt_tokens.append(token)
              elif token[0] == "<" and token[-1] == ">" and token[1:-1].isnumeric():
                numeric_gt_tokens.append(token[1:-1])
              elif token[:6] == "unused" and token[6:].isnumeric():
                  numeric_gt_tokens.append(token[6:])
            numeric_tokens = []
            for token in answer.split():
              if token.isnumeric():
                numeric_tokens.append(token)
              elif token[0] == "<" and token[-1] == ">" and token[1:-1].isnumeric():
                numeric_tokens.append(token[1:-1])
              elif token[:6] == "unused" and token[6:].isnumeric():
                numeric_tokens.append(token[6:])
            try:
              if len(numeric_gt_tokens) >= 2 and len(numeric_tokens) >= 2:
                start = int(numeric_tokens[-2])
                end = int(numeric_tokens[-1])
                gt_start = int(numeric_gt_tokens[-2])
                gt_end = int(numeric_gt_tokens[-1])
                overlap_start = max(start, gt_start)
                overlap_end = min(end, gt_end)
                if overlap_start < overlap_end:
                  iou = (overlap_end-overlap_start) / (end-start + gt_end-gt_start - (overlap_end-overlap_start))
            except ValueError:
              pass
            for key in recall1_by_iou:
                if iou >= key / 10:
                    recall1_by_iou[key] += 1
            if len(numeric_gt_tokens) == 1 and len(numeric_tokens) >= 1:
                answer_frame = int(numeric_tokens[-1])
                gt_frame = int(numeric_gt_tokens[0])
                for key in recall1_by_iou:
                    if abs(answer_frame-gt_frame) < int(key/2):
                        recall1_by_iou[key] += 1
            gts = [gt]

          eos_index = tokens.shape[1]-1
          for j in range(tokens.shape[1]):
            if tokens[i,j] == self.tok.eos_token:
              eos_index = j
          json_out.append({
              self.out_question_key: batch["question_id"][i].item()+(f"_sample{sample_i}" if self.num_samples > 1 else ""),
              self.out_answer_key: answer, "iou": [iou], "logp": (float(np.sum(logp[i,:eos_index+1])) if not self.full_logp else logp[i,:eos_index+1].tolist()), "question": batch["question_str"][i].item()} | ({"gts": gts} if gts else {}))
          print('logp', logp[i,:], json_out[-1]['logp'])

    # print('local total', len(json_out), len(accuracies))
    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    # print(sum(accuracies), sum(accuracies_any), sum(anls_values), len(accuracies), len(json_out))
    recall_sums = {key: c.process_sum(recall1_by_iou[key]) for key in recall1_by_iou}
    total = c.process_sum(len(json_out))
    # sum_accs = c.process_sum({k: sum(v) for k, v in accuracies_by_type.items()})
    # num_accs = c.process_sum({k: len(v) for k, v in accuracies_by_type.items()})
    # num = c.process_sum(len(json_out))

    # Yielding metric_name, value means logging the metric.
    if total:
      for key in recall1_by_iou:
        yield "r@1 (0."+str(key)+")", recall_sums[key] / total

    yield "num", total  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile.split(".json")[0]+outfile_suffix+".json", json_out)


def anls_metric(target: str, prediction: str, theta: float = 0.5):
  """Calculates ANLS for DocVQA.

  There does not seem to be an official evaluation script.
  Public implementation on which this implementation is based:
  https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

  Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

  Args:
    target: Target string.
    prediction: Predicted string.
    theta: Filter threshold set to 0.5 for DocVQA.

  Returns:
    ANLS score.
  """
  if target:
    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0
  else:
    return float(prediction == "")
