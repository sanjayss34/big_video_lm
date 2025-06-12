# Copyright 2025 big_video_lm authors - slight modifications for our launch script
# Copyright 2024 Charlie Snell

import textwrap
import os
from tpu_scripts.tpu_pod_launcher import TPUPodClient, TPUPodProject, create_cli

SETUP_SCRIPT = """\
cd ~/
# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda create -n llama_train3 python=3.10 -y
conda activate llama_train3
cd ~/llama_train
python -m pip install -e .
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install git+https://github.com/davisyoshida/lorax.git

# clean up
cd ~/
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
""".strip()

CHECK_DEVICES = r"""
source ~/miniconda3/bin/activate llama_train3
python -c "import jax; print(jax.devices())"
""".strip()

def setup(project: TPUPodProject, verbose: bool=False):
    project.copy(verbose=verbose)
    project.ssh(SETUP_SCRIPT, verbose=verbose)
    project.ssh(f'mkdir /home/{os.environ["REMOTE_USER"]}/.config/', verbose=verbose)
    project.ssh(f'mkdir /home/{os.environ["REMOTE_USER"]}/.config/gcloud/', verbose=verbose)
    project.scp(os.environ["GCLOUD_KEY_PATH"], f'/home/{os.environ["REMOTE_USER"]}/.config/gcloud/', verbose=verbose)

def check_devices(project: TPUPodProject, verbose: bool=False):
    project.ssh(CHECK_DEVICES, verbose=verbose)

def debug(project: TPUPodProject, verbose: bool=False):
    import IPython; IPython.embed()

def create_project(tpu_name: str, zone: str) -> TPUPodProject:
    return TPUPodProject(
        client=TPUPodClient(
            tpu_project=os.environ['GCLOUD_PROJECT_NAME'],
            tpu_zone=zone,
            user=os.environ["REMOTE_USER"],
            # key_path=f'/home/{os.environ["LOCAL_USER"]}/.ssh/google_compute_engine',
            key_path=os.environ['GCLOUD_KEY_PATH'],
        ),
        tpu_name=tpu_name,
        copy_dirs=[(os.environ["LOCAL_PROJECT_PATH"], f'{os.environ["REMOTE_PROJECT_HOME"]}/big_vision_repo')],
        working_dir=f'{os.environ["REMOTE_PROJECT_HOME"]}/big_vision_repo',
        copy_excludes=['.git', '__pycache__'],
        kill_commands=['pkill -9 python'],
    )

if __name__ == "__main__":
    available_tpus = [
        # Example:
        # ('sanjayss-europe-128-2', 'europe-west4-a'), # v3-64
    ]

    tpu_projects = {name: create_project(name, zone) for name, zone in available_tpus}

    create_cli(
        projects=tpu_projects,
        setup=setup,
        custom_commands={'check_devices': check_devices, 'debug': debug},
    )
