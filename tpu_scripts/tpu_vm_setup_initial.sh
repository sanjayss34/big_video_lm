# Source environment variables from env.sh
. ./env.sh
echo "!!!! Sourced env.sh !!!!"
# Add local bin directory to PATH
export PATH=$REMOTE_PROJECT_HOME/.local/bin/:$PATH

#! /bin/bash

# Install essential system packages
sudo apt-get update && sudo apt-get install -y \
    build-essential \    # Basic build tools
    python-is-python3 \  # Ensure python3 is default
    tmux \              # Terminal multiplexer
    htop \              # System monitor
    git \               # Version control
    bmon \              # Network bandwidth monitor
    p7zip-full \        # Archive manager
    ffmpeg \            # Video processing
    libsm6 \            # X11 libraries
    libxext6            # X11 extension libraries

# Install gcsfuse - allows mounting Google Cloud Storage buckets as local filesystem
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install -y gcsfuse
sudo apt-get install -y openjdk-21-jre-headless  # Java runtime for some dependencies

python -m tpu_scripts.local_launch ssh "python $REMOTE_PROJECT_HOME/mount_disk.py" --project=$TPU_NAME

# Setup GCS bucket mounts
mkdir $REMOTE_PROJECT_HOME/buckets
mkdir $REMOTE_PROJECT_HOME/buckets/checkpoints
fusermount -u $REMOTE_PROJECT_HOME/buckets/checkpoints
# Mount checkpoints bucket and create symlink
gcsfuse --implicit-dirs $CHECKPOINTS_BUCKET $REMOTE_PROJECT_HOME/buckets/checkpoints/
ls $REMOTE_PROJECT_HOME/buckets/checkpoints

# Install Python dependencies
python3 -m pip install --upgrade pip
pip install imageio[pyav]~=2.34.0
pip uninstall -y jax
pip uninstall -y jaxlib
pip uninstall -y optax
pip3 install  jax[tpu]==0.4.33 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax==0.10.0 jax==0.4.33 jax-lorax
pip install clu
pip install -q kagglehub
pip uninstall -y tensorflow_io
pip install -q tensorflow_io
pip install -q ipython
pip install -r $REMOTE_PROJECT_HOME/requirements.txt
pip install editdistance
pip install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece"
pip install -q git+https://github.com/google-deepmind/gemma.git@af38d6eb413cb98446b78a906c77cf5ba28be149
pip install haliax
pip install jax-smi
pip install git+https://github.com/forhaoliu/ringattention.git
pip install git+https://github.com/forhaoliu/tux.git
pip install scalax>=0.2.1
pip install mlxu>=0.2.0
pip install git+https://github.com/Findus23/jax-array-info.git
# pip install easydel
pip install fjformer==0.0.69 easydel==0.0.69

# Install project-specific requirements
cd $REMOTE_PROJECT_HOME/big_vision_repo
