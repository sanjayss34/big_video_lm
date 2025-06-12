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

"""Evaluator for simple VQA variants (OCR-VQA, OKVQA, A-OKVQA).

According to the (A-)OKVAQ papers, the eval for these datasets should follow
VQAv2. But here we don't track different answer-types, and don't do any
leave-one-out averaging, as this isn't done in the official implementation at
https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py
either.

Please read the description of how evaluators work at (internal link).
This evaluator follows the pattern of also parallelizing the CPU computations
(ie postprocessing, score computation) across hosts for more scalability.

For now, simple decoding is implemented as part of the evaluator. We'll soon
unify and move to a library of decoding functions, including fancier and more
efficient ones.
"""
import functools

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
  """Evaluator for simple VQA tasks.

  This evaluator expects the batch to contain a field `question_id` and a field
  `answer` for single ground truth or `answers` for multiple ground truths.

  The field names used when writting the json result can be controlled with
  `out_question_key` and `out_answer_key`.
  """

  def __init__(
      self, predict_fn, tokenizer, to_lower=False,
      outfile="{workdir}/{split}.json",
      out_question_key="question_id", out_answer_key="answer", yes_no=False,
      *, data, devices, extra_model_args=[], sharding_config=None, sharding_rules=None, num_samples=1, answer_prefix=None, **kw):
    self.non_numeric_keys = {"answers", "answer", "question_id", "question_str"}
    self.data_options = data
    self.data_kw = kw
    self.get_data_iter, self.steps = c.eval_input_pipeline(
        keep_on_cpu=self.non_numeric_keys,
        keep_all_on_cpu=False, # sharding_config is not None,
        sharding_config=sharding_config,
        data=self.data_options, devices=devices, **self.data_kw)

    self.outfile = c.resolve_outfile(outfile, split=data.get("split"))
    self.out_question_key = out_question_key
    self.out_answer_key = out_answer_key

    self.extra_model_args = extra_model_args

    # We'll need the tokenizer to detokenize the model outputs later.
    print('tokenizer', tokenizer)
    self.tok = big_vision.pp.tokenizer.get_tokenizer(tokenizer)
    self.postproc = (lambda s: s.lower()) if to_lower else lambda s: s
    self.yes_no = yes_no
    self.num_samples = num_samples
    self.answer_prefix = answer_prefix
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

    accuracies = []
    accuracies_any = []
    json_out = []
    predicted_counts = []
    true_counts = []
    correct_counts = []
    # print('eval going to loop')
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
    for step_num, batch in zip(range(steps), get_data_iter()):
      print('Evaluation step', step_num)
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
          tokens, logp = self.decode(train_state, sharded_batch)
      else:
        tokens, logp = self.decode(train_state, sharded_batch)
      sharded_mask = sharded_batch['_mask']

      # (local_batch,) that indicates padding examples (0) vs real examples (1).
      text = None
      if self.sharding_config is None:
        sharded_mask = u.reshard(batch['_mask'], jax.sharding.NamedSharding(jax.sharding.Mesh(self.devices, ("devices",)), P("devices")))
        tokens = u.get_local_slice_from_fsarray(tokens)
        ex_masks = u.get_local_slice_from_fsarray(sharded_mask)
        logp = u.get_local_slice_from_fsarray(logp)
        # print('tokens', tokens.shape, 'ex_masks', ex_masks.shape)
      else:
        with jax.transfer_guard("allow"):
          text = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(sharded_batch["text"], self.mesh, sharded_batch["text"].sharding.spec))
          tokens = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(tokens, self.mesh, tokens.sharding.spec))[:,0,:]
          ex_masks = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(sharded_mask, self.mesh, sharded_mask.sharding.spec))
          logp = np.array(jax.experimental.multihost_utils.global_array_to_host_local_array(logp, self.mesh, logp.sharding.spec))[:,0,:]
        print('tokens', tokens.shape, 'ex_masks', ex_masks.shape) # , 'answer', len(batch['answer']))

      # text = batch["text"]
      # print('going to postprocessing')
      # Turn predictions into texts and then scores, one by one.
      for i in range(len(tokens)):
        if ex_masks[i] == 0:  # Skip last-batch padding examples
          continue

        answer = self.tok.to_str(tokens[i], stop_at_eos=True)
        print(answer)
        if self.answer_prefix is not None:
          if self.answer_prefix in answer:
            answer = answer.split(self.answer_prefix)[1].strip()
        answer = self.postproc(answer)
        print(tokens[i])
        # print(i, 'question:', self.tok.to_str(text[i], stop_at_eos=True))
        # print('output:', answer)

        # Now we have two commonly used VQA evaluation modes:
        if "answer" in batch:
          # single GT (eg ocrvqa): just compare to that answer, done.
          gt = batch["answer"][i]
          if self.answer_prefix is not None:
            gt = gt.split(self.answer_prefix)[1].strip()
          gts = [gt]
          accuracies.append(float(answer == gt))
          accuracies_any.append(float(answer == gt))
          if self.yes_no:
            if gt.lower() == 'yes':
              true_counts.append(1)
            else:
              true_counts.append(0)
            if answer.lower() == 'yes':
              predicted_counts.append(1)
              if gt.lower() == 'yes':
                correct_counts.append(1)
              else:
                correct_counts.append(0)
            else:
              predicted_counts.append(0)
              correct_counts.append(0)
        elif "answers" in batch and (gt_answers := batch["answers"][i]).size:
          # multiple GTs (eg okvqa): introduced by VQA, compare to each of them
          # with a threshold, see also: https://visualqa.org/evaluation.html
          if self.answer_prefix is not None:
            gts = [a.split(self.answer_prefix)[1].strip() for a in gt_answers]
          else:
            gts = gt_answers
          gts = [self.postproc(a) for a in gts]
          print('ground truth:', gts)
          num_match = sum([answer == gt for gt in gts])
          accuracies.append(min(1.0, num_match / 3.0))
          accuracies_any.append(min(1.0, float(num_match)))
        else:
          gts = []

        json_out.append({
            self.out_question_key: batch["question_id"][i].item(), 'question': batch["question_str"][i].item(),
            self.out_answer_key: answer, "logp": float(np.sum(logp[i,:1]))} | ({"gts": gts} if gts else {}))
        print('logp', logp[i,:], json_out[-1]['logp'])

    # At this point `accuracies` is a list of per-example scores. However,
    # remember that each host holds a different subset of the examples! So if
    # we were to just return the mean accuracy here, we would effectively only
    # have evaluated on the main host's (who writes metrics) subset!
    # So now, we need to compute global means.
    # There is one more caveat: `process_sum` needs the summands on each host
    # to have the same size. So we either need to include dummy values for
    # the padding examples (last batch, annoying), or we only sum scalars as in
    # sufficient statistics, which we do here.
    sum_accs, sum_accs_any, num_accs, num = c.process_sum(
        [sum(accuracies), sum(accuracies_any),
         len(accuracies), len(json_out)])
    if self.yes_no:
      sum_true_counts, sum_predicted_counts, sum_correct_counts = c.process_sum(
        [sum(true_counts), sum(predicted_counts), sum(correct_counts)]
      )
      if sum_predicted_counts > 0:
        precision = sum_correct_counts / sum_predicted_counts
      else:
        precision = 1
      if sum_true_counts > 0:
        recall = sum_correct_counts / sum_true_counts
      else:
        recall = 0
      if precision+recall > 0:
        f1 = 2 * precision * recall / (precision+recall)
      else:
        f1 = 0

    # print('global total', num, num_accs)
    # Yielding metric_name, value means logging the metric.
    if num_accs:
      yield "acc", sum_accs / num_accs
      yield "acc_any", sum_accs_any / num_accs
      if self.yes_no:
        yield "micro_precision", precision
        yield "micro_recall", recall
        yield "micro_f1", f1

    yield "num", num  # Just for sanity checks.
    c.multiprocess_write_json(self.outfile.split(".json")[0]+outfile_suffix+".json", json_out)
