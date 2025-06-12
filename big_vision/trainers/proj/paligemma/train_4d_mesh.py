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

"""Training loop for PaliGemma-style VLM."""
# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

import functools
import importlib
import multiprocessing.pool
import os
import sys

from absl import app
from absl import flags
from absl import logging
import big_vision.datasets.core as ds_core
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
import big_vision.trainers.proj.paligemma.predict_fns as predict_fns
from big_vision.models.proj.paligemma.llava_ov_sharding import LlavaOneVisionShardingConfig
import big_vision.utils as u
from clu import parameter_overview
import flax
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
import jax.numpy as jnp
import ml_collections as mlc
from ml_collections import config_flags
import numpy as np
import optax
try:
  import lorax
except:
  pass
import tensorflow as tf

from tensorflow.io import gfile
from jax_smi import initialise_tracking
from jax_array_info import sharding_info

jnp.set_printoptions(threshold=sys.maxsize)

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
jax.config.update("jax_transfer_guard", "disallow")


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec

def flatten_dict(d):
    if not isinstance(d, dict):
        return {'': d}
    flattened = {}
    for k in d:
        flattened_k = flatten_dict(d[k])
        for k2 in flattened_k:
            flattened[k+('/' if len(k2) > 0 else '')+k2] = flattened_k[k2]
    return flattened

def unflatten_dict(d):
    result = {}
    for k, v in d.items():
        keys = k.split('/')
        current_dict = result
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[keys[-1]] = v
    return result

@jax.vmap
def roll_by_amount(x, amount):
  x = jnp.roll(x, amount, axis=0)
  return x

def main(argv):
  del argv

  # This is needed on multihost systems, but crashes on non-TPU single-host.
  if os.environ.get("BV_JAX_INIT"):
    jax.distributed.initialize()
  initialise_tracking()

  # Make sure TF does not touch GPUs.
  tf.config.set_visible_devices([], "GPU")

################################################################################
#                                                                              #
#                                Set up logging                                #
#                                                                              #
################################################################################

  # Set up work directory and print welcome message.
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  save_ckpt_path = None
  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)
    save_ckpt_path = os.path.join(workdir, "checkpoint.bv")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # Setup up logging and experiment manager.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  # Allow for things like timings as early as possible!
  u.chrono.inform(measure=mw.measure, write_note=write_note)

################################################################################
#                                                                              #
#                                Set up Mesh                                   #
#                                                                              #
################################################################################

  sharding_config = LlavaOneVisionShardingConfig(None, mesh_dims=config.get("mesh_dims"), img_scan=config.get("model").get("img").get("scan", True))
  write_note("Creating device mesh...")
  mesh_helper = sharding_config.get_mesh()
  mesh = mesh_helper.mesh
  device_mesh = mesh.devices
  devices_flat = device_mesh.flatten()
  # We rely on jax mesh_utils to organize devices, such that communication
  # speed is the fastest for the last dimension, second fastest for the
  # penultimate dimension, etc.

  # Sharding rules with the default of doing full data sharding.
  sharding_rules = config.get("sharding_rules", [("act_batch", "data")])

################################################################################
#                                                                              #
#                                Input Pipeline                                #
#                                                                              #
################################################################################

  write_note("Initializing train dataset...")
  batch_size = config.input.batch_size
  # if batch_size % jax.device_count() != 0:
  #   raise ValueError(f"Batch size ({batch_size}) must "
  #                    f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  train_ds, ntrain_img = input_pipeline.training(config.input, divide_by_process_count=True) # False)
  print(train_ds.element_spec)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible, this will kick-start filling
  # shuffle buffers and get the first batch in a background thread.
  n_prefetch = config.get("prefetch_to_device", 1)
  # train_iter = input_pipeline.start_global(
  #     train_ds, devices_flat, n_prefetch, warmup=n_prefetch > 0, keep_all_on_cpu=True)

  # For mixed data, add per-dataset epoch and examples seen measurements.
  if isinstance(config.input.data.get("name"), str):
    measure_per_dataset_times = lambda step: None  # No-op
  else:
    nexamples = {
        name: ds_core.get(**config.input[name].data).total_examples
        for name in config.input.data
    }
    def measure_per_dataset_times(step):
      total = sum(config.input.data.values())
      for name, w in config.input.data.items():
        w = w / total
        mw.measure(f"examples_seen_{name}", u.chrono.accum_examples_seen * w)
        mw.measure(f"epoch_{name}", step * batch_size * w / nexamples[name])

################################################################################
#                                                                              #
#                           Create Model & Optimizer                           #
#                                                                              #
################################################################################
  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"big_vision.models.{config.model_name}")
  print('MESH', mesh)
  model = model_mod.Model(**mlc.FrozenConfigDict(config.get("model", {})), mesh=mesh)

  def init(rng, partial_params=None, batch=None):
    if batch is None:
      # batch = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype.as_numpy_dtype),
      batch = jax.tree.map(lambda x: jnp.repeat(jnp.zeros(x.shape, x.dtype.as_numpy_dtype), axis=0, repeats=jax.process_count()),
                           train_ds.element_spec)
    jax.debug.print('unsharded init before model apply')
    # jax.debug.print('unsharded init params max scale {s}', s=jnp.max(partial_params['llm']['layers']['5']['post_ffw_norm']['scale']))
    jax.debug.print('unsharded init batch image max {s}', s=jnp.max(batch['image']))
    _, variables = model.apply(  # flax init is just apply with mutable.
        {"params": partial_params or {}},
        batch["image"], batch["text"][:, :-1], batch["mask_ar"][:, :-1],
        *[batch.get(extra_arg_key, None) for extra_arg_key in (config.extra_model_args if hasattr(config, 'extra_model_args') else [])],
        rngs={"params": rng, "dropout": rng},
        mutable=["params"])
    return flax.core.unfreeze(variables["params"])

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  # See (internal link) for a fun read on what it takes.
  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))

  write_note("Inferring parameter shapes...")
  rng, rng_init = jax.random.split(rng)
  params_shape = jax.eval_shape(init, rng_init)
  params_shape = nn.unbox(params_shape)

  write_note("Inferring optimizer state shapes...")
  tx, sched_fns = bv_optax.make(config, params_shape, sched_kw=dict(
      total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
  if getattr(config, "gradient_accumulation_steps", 1) > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps)
  opt_shape = jax.eval_shape(tx.init, params_shape)
  # We jit this, such that the arrays are created on the CPU, not device[0].
  sched_fns_cpu = [u.jit_cpu()(sched_fn) for sched_fn in sched_fns]

  if jax.process_index() == 0:
    num_params = sum(np.prod(p.shape) for p in jax.tree.leaves(params_shape))
    mw.measure("num_params", num_params)

  if config.get("lora_k", None) is not None:
    # Simple_spec is a helper to do this, but you can also create the label pytree yourself
    lora_spec = lorax.simple_spec(params_shape, decision_fn=decision_fn, tune_vectors=True)

    # Split the parameters up into tunable and frozen ones, and initialize a pair of LoRA matrices for each parameter
    # which had a spec value other than LORA_FULL or LORA_FREEZE
    lora_params = lorax.init_lora(params_shape, lora_spec, jax.random.PRNGKey(0))

    # `wrap_optimizer` uses the spec to freeze the appropriate subset
    # of parameters.
    # The frozen parameters won't have optimizer states etc
    # created for them
    optimizer = lorax.wrap_optimizer(tx, lora_spec)
    model = lorax.lora(model)

################################################################################
#                                                                              #
#                    Init and/or load model onto devices                       #
#                                                                              #
################################################################################
  repl_sharding = jax.sharding.NamedSharding(mesh, P())

  write_note("Inferring shardings...")
  train_state_shape = {"params": params_shape, "opt": opt_shape}

  strategy = config.get("sharding_strategy", [(".*", "replicate")])
  train_state_sharding = bv_sharding.infer_sharding(
      train_state_shape, strategy=strategy, mesh=mesh)
  dummy_batch = jax.tree.map(lambda x: np.zeros(x.shape, x.dtype.as_numpy_dtype),
                       train_ds.element_spec)
  train_state_sharding = {}
  train_state_sharding["params"] = jax.tree.map(
    lambda spec: jax.sharding.NamedSharding(mesh, spec),
    sharding_config.get_model_sharding_rule().apply(train_state_shape["params"])
  )
  train_state_sharding["opt"] = jax.tree.map(
    lambda spec: jax.sharding.NamedSharding(mesh, spec),
    sharding_config.get_model_sharding_rule().apply(train_state_shape["opt"])
  )
  print(train_state_sharding["params"])
  batch_sharding = jax.tree.map(
    lambda spec: jax.sharding.NamedSharding(mesh, spec),
    sharding_config.get_batch_sharding().apply(dummy_batch)
  )

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from scratch or from something, e.g. fine-tuning job.
  resume_ckpt_path = None
  if save_ckpt_path and gfile.exists(f"{save_ckpt_path}-LAST"):
    resume_ckpt_path = save_ckpt_path
  if config.get("resume"):
    resume_ckpt_path = fillin(config.resume)

  if resume_ckpt_path:
    write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
    shardings = {
        **train_state_sharding,
        "chrono": jax.tree.map(lambda _: repl_sharding, u.chrono.save()),
    }
    shardings = {
      'params': shardings['params'],
      'chrono': shardings['chrono']
    }
    load_chrono = True
    if 'checkpoint-' not in resume_ckpt_path.split('/')[-1]:
      load_chrono = False
      shardings = {'params': shardings['params']}
    print('chrono' in shardings, load_chrono)
    loaded = u.load_checkpoint_ts(
        resume_ckpt_path, tree=shardings, shardings=shardings)
    train_state = {key: loaded[key] for key in train_state_sharding.keys() if key in loaded}
    if load_chrono:
      u.chrono.load(jax.device_get(loaded["chrono"]))
    params = train_state["params"]
    params = jax.tree.map(lambda x: u.float_tensor_to_dtype(x, dtype=jnp.float32), params)
    del loaded
  else:
    write_note(
        f"Initialize model from {config.get('model_init') or 'scratch'}...")

    # To avoid holding two copies of parameters we first call `model.load`
    # and then initialize the missing variables.
    if config.get("model_init"):
      # We call `model.load` with params shape, so it can know all model params
      # including their shapes and dtypes (also shardings once wired).
      params = model_mod.load(
          params_shape, config.model_init, config.get("model"),
          **config.get("model_load", {}))

      # Keep only params loaded by `model.load` and shard them into devices.
      mask = jax.tree.map(
          lambda x: not isinstance(x, jax.ShapeDtypeStruct), params)
      print('params', {k: (type(t), t.shape) for k, t in flatten_dict(params).items()})
      if config.get("scalax_load", False):
        shard_fns, _ = mesh_helper.make_shard_and_gather_fns(u.tree_filter(params, mask), sharding_config.get_model_sharding_rule())
        params = u.scalax_load_checkpoint(u.tree_filter(params, mask), shard_fns=shard_fns)
      else:
        params = u.reshard(u.tree_filter(params, mask),
                           u.tree_filter(train_state_sharding["params"], mask))
    else:
      params = {}
    train_state = {"params": params}

  # Init will initialize any missing params.
  rng_init = u.reshard(rng_init, repl_sharding)

  with mesh, nn.logical_axis_rules(sharding_rules):
    params = jax.jit(
      init, donate_argnums=2, out_shardings=train_state_sharding["params"])(
          rng_init, params, None)
  params = nn.unbox(params)

  if "opt" not in train_state:
    # Initialize optimizer and construct train_state.
    opt = jax.jit(tx.init, out_shardings=train_state_sharding["opt"])(params)
    train_state = {"params": params, "opt": opt}
  del params, opt  # Delete to avoid memory leak or accidental reuse.

  parameter_overview.log_parameter_overview(
      train_state["params"], msg="Parameter overview",
      include_stats="global", jax_logging_process=0)

  rng, rng_loop = jax.random.split(rng, 2)
  rng_loop = u.reshard(rng_loop, repl_sharding)
  del rng, rng_init  # not used anymore, so delete it.

  train_iter = input_pipeline.start_global(
      train_ds,
      devices_flat,
      n_prefetch,
      warmup=n_prefetch > 0,
      keep_all_on_cpu=False, # True,
      sharding_map=batch_sharding,
      full_batch_size=batch_size)

################################################################################
#                                                                              #
#                                 Update Step                                  #
#                                                                              #
################################################################################

  @functools.partial(
      jax.jit,
      donate_argnums=(0,),
      out_shardings=(
        train_state_sharding,
        repl_sharding,
      )
  )
  def update_fn(train_state, rng, batch):
    """Update step."""

    step_count = bv_optax.get_count(train_state["opt"], jittable=True)
    rng = jax.random.fold_in(rng, step_count)
    assert "mixup" not in config, "Mixup is not supported for SigLIP."

    # Get device-specific loss rng.
    _, rng_model = jax.random.split(rng, 2)

    imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]
    extra_model_args = {extra_arg_key: batch.get(extra_arg_key, None) for extra_arg_key in config.extra_model_args}

    def loss_fn1(params):
      text_logits, outputs = model.apply(
          {"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], **extra_model_args,
          train=True, rngs={"dropout": rng_model})

      logp = jax.nn.log_softmax(text_logits, axis=-1)
      targets = jax.nn.one_hot(txts[:,1:], text_logits.shape[-1])
      off_value = config.get("label_smoothing", 0.0)
      if off_value > 0:
        denom = text_logits.shape[-1] - 1
        targets = jnp.where(
            targets == 1.0, 1.0 - off_value, off_value / denom)

      # Sum across vocab.
      token_pplx = jnp.sum(logp * targets, axis=-1)

      # Shift by one since the loss is on the _next_ token.
      mask_loss = batch["mask_loss"][:,1:] # [:, model.insert_images_at_index+1:]
      token_pplx = token_pplx * mask_loss
      mask_loss_cumulative = jnp.cumsum(mask_loss, axis=-1)
      mask_loss_prefix = (mask_loss_cumulative == 0).astype(jnp.int32).sum(axis=-1)
      pplx = -jnp.sum(token_pplx, axis=-1)
      pplx /= jnp.clip(jnp.sum(mask_loss, axis=-1), 1)
      training_loss = jnp.mean(pplx)

      # In this dict the (outer) reduction is along batch.
      measurements = dict(
          training_loss=training_loss,
          avg_sup_seqlen=jnp.mean(jnp.sum(mask_loss, axis=-1)),
          max_sup_seqlen=jnp.max(jnp.sum(mask_loss, axis=-1)),
      )

      return measurements["training_loss"], measurements

    def loss_fn2(params):
      l = jnp.mean(imgs) * jnp.mean(imgs)
      measurements = dict(training_loss=l, avg_sup_seqlen=1, max_sup_seqlen=1)
      return l, measurements

    loss_fn = loss_fn1
    params, opt = train_state["params"], train_state["opt"]
    (_, measurements), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    gs = jax.tree.leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
    measurements["l2_grads"] = jnp.sqrt(sum([jnp.sum(g * g) for g in gs]))
    ps = jax.tree.leaves(params)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.sum(p * p) for p in ps]))
    us = jax.tree.leaves(updates)
    measurements["l2_updates"] = jnp.sqrt(sum([jnp.sum(u * u) for u in us]))
    measurements["nonzero_updates"] = sum([jnp.count_nonzero(u) for u in us])

    return {"params": params, "opt": opt}, measurements

################################################################################
#                                                                              #
#                                 Setup Evals                                  #
#                                                                              #
################################################################################

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config,
        predict_fns.get_all(model),
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices_flat,
        sharding_config=sharding_config,
        sharding_rules=sharding_rules
    )

  # At this point we need to know the current step to see whether to run evals.
  write_note("Inferring the first step number...")
  first_step_device = bv_optax.get_count(train_state["opt"], jittable=True)
  first_step = int(jax.device_get(first_step_device))
  u.chrono.inform(first_step=first_step)

  # Note that training can be pre-empted during the final evaluation (i.e.
  # just after the final checkpoint has been written to disc), in which case we
  # want to run the evals.
  if first_step in ((total_steps,) if not config.get("eval_only", False) else (total_steps, 0)):
    write_note("Running initial or final evals...")
    mw.step_start(first_step)
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      write_note(f"{name} evaluation...\n{u.chrono.note}")
      with u.chrono.log_timing(f"z/secs/eval/{name}"):
        for key, value in evaluator.run(train_state):
          mw.measure(f"{prefix}{key}", value)

################################################################################
#                                                                              #
#                                  Train Loop                                  #
#                                                                              #
################################################################################

  if config.get("eval_only", False):
    exit()

  prof = None  # Keeps track of start/stop of profiler state.
  ckpt_mngr = None

  write_note("Starting training loop, compiling the first step...")
  batch = next(train_iter)
  batch_sharding = jax.tree.map(
    lambda spec: jax.sharding.NamedSharding(mesh, spec),
    sharding_config.get_batch_sharding().apply(batch)
  )
  sharded_batch = batch
  for step in range(first_step+1, total_steps+1):
    mw.step_start(step)
    with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
      with u.chrono.log_timing("z/secs/update", noop=(step % 50 != 0)):
        with mesh, nn.logical_axis_rules(sharding_rules):
          train_state, measurements = update_fn(
              train_state, rng_loop, sharded_batch
          )
    with jax.profiler.StepTraceAnnotation("fetch_batch", step_num=step):
      with u.chrono.log_timing("z/secs/fetch_batch", noop=(step % 50 != 0)):
        batch = next(train_iter)
    with jax.profiler.StepTraceAnnotation("shard_batch", step_num=step):
      with u.chrono.log_timing("z/secs/shard_batch", noop=(step % 50 != 0)):
        sharded_batch = batch

    # On the first host, let's always profile a handful of early steps.
    if jax.process_index() == 0:
      prof = u.startstop_prof(prof, step, first_step, get_steps("log_training"))

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        # or u.chrono.warmup and jax.process_index() == 0):
        and jax.process_index() == 0):
      for i, sched_fn_cpu in enumerate(sched_fns_cpu):
        mw.measure(f"global_schedule{i if i else ''}",
                   sched_fn_cpu(u.put_cpu(step - 1)))
      measurements = jax.device_get(measurements)
      for name, value in measurements.items():
        mw.measure(name, value)
      u.chrono.tick(step)
      measure_per_dataset_times(step)

      for k in ("training_loss", "l2_params", "l2_grads"):
        if not np.isfinite(measurements.get(k, 0.0)):
          raise RuntimeError(f"{k} became nan or inf somewhere within steps "
                             f"[{step - get_steps('log_training')}, {step}]")

    # Checkpoint saving
    keep_last = total_steps if get_steps("ckpt", None) else None
    keep_ckpt_steps = get_steps("keep_ckpt", None) or keep_last
    if save_ckpt_path and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=False)
    ):
      info("%s", "Checkpoint line 1")
      u.chrono.pause(wait_for=train_state)

      # Copy because we add extra stuff to the checkpoint.
      info("%s", "Checkpoint line 2")
      ckpt = {**train_state}

      # To save chrono state correctly and safely in a multihost setup, we
      # broadcast the state to all hosts and convert it to a global array.
      info("%s", "Checkpoint line 3")
      with jax.transfer_guard("allow"):
        chrono_ckpt = multihost_utils.broadcast_one_to_all(u.chrono.save())
      info("%s", "Checkpoint line 4")
      chrono_shardings = jax.tree.map(lambda _: repl_sharding, chrono_ckpt)
      info("%s", "Checkpoint line 5")
      ckpt = ckpt | {"chrono": u.reshard(chrono_ckpt, chrono_shardings)}
      info("%s", "Checkpoint line 6")

      ckpt_mngr = ckpt_mngr or array_serial.GlobalAsyncCheckpointManager()
      info("%s", "Checkpoint line 7")
      u.save_checkpoint_ts(ckpt_mngr, ckpt, save_ckpt_path, step, keep)
      info("%s", "Checkpoint line 8")
      u.chrono.resume()
      info("%s", "Checkpoint line 9")

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.pause(wait_for=train_state)
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          with mesh, nn.logical_axis_rules(sharding_rules):
            for key, value in evaluator.run(train_state, outfile_suffix=str(step)):
              mw.measure(f"{prefix}{key}", jax.device_get(value))
        u.chrono.resume()
    mw.step_end()

  print('Exiting')
  write_note('Exiting')
  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  print('Stopping trace')

  # Last note needs to happen before the pool's closed =)
  write_note(f"Done!\n{u.chrono.note}")

  pool.close()
  pool.join()
  mw.close()
  if ckpt_mngr:
    ckpt_mngr.wait_until_finished()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
