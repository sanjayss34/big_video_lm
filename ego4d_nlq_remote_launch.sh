#!/bin/bash

# ===== TPU Configuration =====
export TPU_NAME="" # TODO: name of the TPU pod
export REMOTE_USER="" # TODO: username on the TPU hosts
export GCLOUD_KEY_PATH="" # TODO: path to Google Cloud key (e.g. id_rsa)
export GCLOUD_PROJECT_NAME="" # name of your Google Cloud project

# ===== Storage Configuration =====
export CHECKPOINTS_BUCKET="" # TODO: name of Google Cloud bucket where checkpoints should be saved
export LOCAL_PROJECT_PATH="" # TODO: path to the directory containing this file and the rest of the repo
export REMOTE_PROJECT_HOME="/home/$REMOTE_USER"
export REMOTE_ALL_CHECKPOINTS_DIR="$REMOTE_PROJECT_HOME/buckets/checkpoints/"
export REMOTE_CHECKPOINT_SUBDIR="" # TODO: name of this experiment
export REMOTE_CHECKPOINT_DIR="$REMOTE_ALL_CHECKPOINTS_DIR/$REMOTE_CHECKPOINT_SUBDIR"
export HF_TOKEN="" # TODO: your Huggingface token

# Whether to run the initial setup steps
export FIRST_TIME_SETUP=1  # Set to 0 for subsequent runs

# ===== Model Configuration =====
# Base model to start from
export PRETRAINED_CKPT="llava_video_7b"
# Model variant to use (qwen2_7b or gemma_2b)
export VARIANT="qwen2_7b"
# export RESUME="<CHECKPOINT_PATH>"

# ===== Training Parameters =====
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
export TRAIN_SCRIPT=train_4d_mesh
export CONFIG_NAME=retrieval_llava
export BATCH_SIZE=16
export LEARNING_RATE=2e-5
export GRADIENT_ACCUMULATION_STEPS=1
# export STEPS=1000
# Setting EPOCHS instead of STEPS
export EPOCHS=1
# Set to 1 for evaluation only, 0 for training
export EVAL_ONLY=0
# Set to "inf" for no limit, or a number to limit dataset size
export DATA_LIMIT="inf"
# How often to save checkpoints
export CKPT_STEPS=4000

# ===== Model Architecture =====
# Number of frames to process per video
export NUM_IMAGES=496
# Number of conversation segments to process
export NUM_CONVERSATION_SEGMENTS=32
# Maximum length of text input/output
export TEXT_LEN=1024
export EVAL_TEXT_LEN=1024
# Maximum length of generated text
export MAX_DECODE_LEN=128
# Number of tokens used to represent images
export IMG_TOKEN_LENGTH=5120
# Downsampling factor for images (higher = smaller images)
export DOWNSAMPLE=9
# Vocabulary size of the model, padded to a multiple of 256 for sharding purposes
# You can increase it further if you want more randomly initialized token embeddings
export VOCAB_SIZE=152064
# Input image resolution
export RESOLUTION=384
# Whether to add newline tokens between images
export IMAGE_NEWLINE=1
# Freeze ViT parameters
export FREEZE_VIT=0

# ===== Attention Configuration =====
# Type of attention to use (only GLOBAL is supported)
export ATTN_TYPE="GLOBAL"
# Implementation of attention (ring or sharded_vanilla)
export ATTENTION_IMPL="ring"

# ===== Sharding Configuration =====
# Number of sequence shards for parallel processing
export SEQUENCE_SHARDING=8
# Number of tensor shards for model parallelism
export TENSOR_SHARDING=1
# Mesh dimensions for distributed training
export MESH_DIMS="1,-1,$SEQUENCE_SHARDING,$TENSOR_SHARDING"
# FSDP (Fully Sharded Data Parallel) axis
export FSDP_AXIS="fsdp"
# Rematerialization policy for memory efficiency
export REMAT_POLICY="nothing_saveable"

# ===== Data Configuration =====
# Local path to data directory containing jsonl files
export LOCAL_DATA_DIR=/path/to/formatted/data/dir/ # TODO: change this to your formatted data dir

# ===== Helper Functions =====
cleanup_tpu() {
    python -m tpu_scripts.local_launch ssh "tmux kill-session -t launch ; pkill -9 python ; pkill -f -9 VLM" --project=$TPU_NAME
    python -m tpu_scripts.local_launch ssh "bash scripts/clean_tpu.sh" --project=$TPU_NAME
}

cleanup_tpu_before_train() {
    python -m tpu_scripts.local_launch ssh "tmux kill-session -t launch ; pkill -9 python ; pkill -f -9 VLM" --project=$TPU_NAME
    python -m tpu_scripts.local_launch ssh "bash clean_tpu.sh" --project=$TPU_NAME
    python -m tpu_scripts.local_launch ssh "tmux kill-session -t launch ; pkill -9 python ; pkill -f -9 VLM" --project=$TPU_NAME
}

setup_remote_env() {
    # Copy setup files
    if [ "$FIRST_TIME_SETUP" = "1" ]; then
        echo "First time setup - installing all dependencies..."
        python -m tpu_scripts.local_launch scp $LOCAL_PROJECT_PATH/requirements.txt $REMOTE_PROJECT_HOME/ --project=$TPU_NAME
        python -m tpu_scripts.local_launch scp $LOCAL_PROJECT_PATH/tpu_scripts/mount_disk.py $REMOTE_PROJECT_HOME/mount_disk.py --project=$TPU_NAME
        # Inject environment variables
        inject_env_vars $REMOTE_PROJECT_HOME/tpu_vm_setup_initial.sh
        python -m tpu_scripts.local_launch scp $LOCAL_PROJECT_PATH/tpu_scripts/tpu_vm_setup_initial.sh $REMOTE_PROJECT_HOME/tpu_vm_setup.sh --project=$TPU_NAME
    else
        echo "Subsequent run - skipping disk mounting and package installations..."
    fi
}

upload_code() {
    python -m tpu_scripts.local_launch ssh "rm -rf $REMOTE_PROJECT_HOME/big_vision_repo" --project=$TPU_NAME
    gsutil -m rm -rf gs://$CHECKPOINTS_BUCKET/big_vision_repo
    gsutil -m mkdir gs://$CHECKPOINTS_BUCKET/big_vision_repo
    gsutil -m cp -r $LOCAL_PROJECT_PATH/big_vision gs://$CHECKPOINTS_BUCKET/big_vision_repo/big_vision
    gsutil cp -r $LOCAL_PROJECT_PATH/scripts gs://$CHECKPOINTS_BUCKET/big_vision_repo/scripts
    gsutil cp -r $LOCAL_PROJECT_PATH/tpu_scripts gs://$CHECKPOINTS_BUCKET/big_vision_repo/tpu_scripts
    python -m tpu_scripts.local_launch ssh "cp -r $REMOTE_ALL_CHECKPOINTS_DIR/big_vision_repo $REMOTE_PROJECT_HOME/" --project=$TPU_NAME
}

setup_model() {
    # Initialize Qwen model
    python -m tpu_scripts.local_launch ssh "python -c 'from transformers import AutoConfig; tokenizer = AutoConfig.from_pretrained(\"Qwen/Qwen2-7B-Instruct\")'" --project=$TPU_NAME
    python -m tpu_scripts.local_launch ssh "python -c 'from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained(\"Qwen/Qwen2-7B-Instruct\")'" --project=$TPU_NAME
}

inject_env_vars() {
    local file="$1"
    if [[ -z "$file" ]]; then
        echo "inject_env_vars: missing file path" >&2
        return 1
    fi

    local vars=(
        "CHECKPOINTS_BUCKET" "VIDEO_DATASETS_BUCKET" "REMOTE_PROJECT_HOME" "REMOTE_USER"
        "HF_TOKEN" "WANDB_USER" "WANDB_API_KEY" "KAGGLE_USERNAME" "KAGGLE_KEY"
        "REMOTE_CHECKPOINT_DIR" "BATCH_SIZE" "GRADIENT_ACCUMULATION_STEPS" "EPOCHS"
        "TRAIN_SCRIPT" "NUM_IMAGES" "TEXT_LEN" "IMG_TOKEN_LENGTH" "FSDP_AXIS"
        "ATTN_TYPE" "DOWNSAMPLE" "CKPT_STEPS" "PRETRAINED_CKPT" "EVAL_ONLY"
        "VARIANT" "IMAGE_NEWLINE" "VOCAB_SIZE" "SEQUENCE_SHARDING" "TENSOR_SHARDING"
        "NUM_CONVERSATION_SEGMENTS" "EVAL_TEXT_LEN" "MAX_DECODE_LEN" "REMAT_POLICY"
        "LEARNING_RATE" "STEPS" "RESUME" "ATTENTION_IMPL" "MESH_DIMS" "DATA_LIMIT"
        "CONFIG_NAME" "REMOTE_ALL_CHECKPOINTS_DIR"
    )

    for var in "${vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            python -m tpu_scripts.local_launch ssh \
              "sed -i '1s@^@export $var=${!var}\n@' \"$file\"" \
              --project="$TPU_NAME"
        fi
    done
}

# ===== Main Execution =====
echo "Starting TPU training setup..."

# Initial cleanup
cleanup_tpu

# Setup environment
setup_remote_env

# Upload code
upload_code

# Setup model
setup_model

# Create checkpoint directory
python -m tpu_scripts.local_launch ssh "mkdir -p $REMOTE_CHECKPOINT_DIR" --project=$TPU_NAME --host_index 0

# Inject environment variables
inject_env_vars $REMOTE_PROJECT_HOME/big_vision_repo/scripts/train.sh
python -m tpu_scripts.local_launch ssh "cp $REMOTE_PROJECT_HOME/big_vision_repo/scripts/train.sh train.sh" --project=$TPU_NAME

# Copy data
python -m tpu_scripts.local_launch ssh "rm -rf $REMOTE_PROJECT_HOME/combined_$((NUM_IMAGES))frames" --project=$TPU_NAME
gsutil -m rm -rf gs://$CHECKPOINTS_BUCKET/combined_$((NUM_IMAGES))frames
gsutil -m cp -r $LOCAL_DATA_DIR gs://$CHECKPOINTS_BUCKET/combined_$((NUM_IMAGES))frames
python -m tpu_scripts.local_launch ssh "cp -r $REMOTE_ALL_CHECKPOINTS_DIR/combined_$((NUM_IMAGES))frames $REMOTE_PROJECT_HOME/" --project=$TPU_NAME

# Final cleanup and launch
cleanup_tpu_before_train
python -m tpu_scripts.local_launch launch "bash train.sh" --project=$TPU_NAME

echo "Training launched!"
