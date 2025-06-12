export TPU_LIBRARY_PATH=$REMOTE_PROJECT_HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
export BV_JAX_INIT=true
export JAX_TRACEBACK_FILTERING=off
export CHECKPOINTS_DIR=$REMOTE_ALL_CHECKPOINTS_DIR
export LIBTPU_INIT_ARGS="--xla_tpu_impure_oom_fast_exit_threshold=-1"
python -m big_vision.trainers.proj.paligemma.$TRAIN_SCRIPT --config big_vision/configs/proj/paligemma/transfers/$CONFIG_NAME.py --workdir $REMOTE_CHECKPOINT_DIR
