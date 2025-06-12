# Video LM finetuning/inference in JAX

The goal of this repo is to enable you to easily run inference and finetuning on video LMs in JAX.

## Converting model weights from Huggingface PyTorch to Flax/JAX
Here's an example command for converting model weights:
```
python convert_llavaov_to_jax.py --model-name lmms-lab/LLaVA-Video-7B-Qwen2 --output-path path/to/llava_video_7b.npz
```
Note that this script supports models with the [LLaVA OneVision architecture](https://arxiv.org/abs/2408.03326), which includes the original LLaVA OneVision models as well as others like the [LLava Video models](https://arxiv.org/abs/2410.02713).

## Prerequisites
Currently, our code supports running models on TPUs. If you do not already have access to TPUs and are an academic researcher, you can try to obtain access via Google's [TPU research cloud program](https://sites.research.google/trc/about/).

For each task, you will need to download the data from the original source, and extract the video frames to images (we extract frames at 2 FPS). These frames will need to be available on the remote TPU hosts. To this end, we recommend using a [persistent disk](https://cloud.google.com/persistent-disk?hl=en). See [this link](https://cloud.google.com/tpu/docs/attach-durable-block-storage) to attach the persistent disks to the TPU hosts.

## Example 1: Video QA Inference
In this example, we evaluate LLaVA Video 7B on the Perception Test benchmark.
First, in `vqa_inference_remote_launch.sh`, set the variables marked as TODO.

Next, format the data to match our format:
```
python format_perception_test.py --data-dir /path/to/perception/test/data --output-dir /path/to/output/dir/with/formatted/data --remote-path-to-frames /path/to/frames/on/hosts/where/inference/occurs/ --num-frames 64 --extracted-fps 2 --shuffle --choice-delimiter \\n
```
Finally, launch the job by running:
```
bash vqa_inference_remote_launch.sh
```

## Example 2: Ego4D moment retrieval
In this example, we finetune LLaVA Video 7B on the Ego4D NLQ (natural language query) task for retrieving moments from an egocentric video.

First, in `ego4d_nlq_remote_launch.sh`, set the variables marked as TODO.

Next, format the data to match our format. We recommend using the [NAQ data](https://github.com/srama2512/NaQ/blob/main/PREPARE_NAQ_DATASETS.md) for training data, since it's larger than the NLQ training set alone.
```
python format_ego4d_nlq.py --data-path /path/to/naq/train.json --output-dir /home/shang/ego4d_nlq --num-frames 496 --extracted-fps 2 --max-segments 32 --augment-delay --num-epochs 10
python format_ego4d_nlq.py --data-path /path/to/nlq_val.json --output-dir /home/shang/ego4d_nlq --num-frames 496 --extracted-fps 2 --max-segments 32 --augment-delay --num-epochs 10
```
Finally, launch the job by running:
```
bash ego4d_nlq_remote_launch.sh
```

## Support
Please contact Sanjay at sanjayss@berkeley.edu if you're having issues with this repo. We want to make it useful for you!

## Contributors
Sanjay Subramanian, Chuyi Shang, Anish Kachinthaya

## Acknowledgements
Thanks to Charlie Snell, Jessy Lin, Jiayi Pan, and Zineng Tang for help in working with JAX and TPUs!

