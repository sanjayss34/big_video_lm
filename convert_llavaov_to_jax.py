import os
import argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file   # pip install safetensors
import torch


def load_all_safetensors(dir_path: str, device: str = "cpu") -> OrderedDict:
    """
    Args
    ----
    dir_path : str or Path
        Directory that contains one or more *.safetensors shards.
    device : "cpu", "cuda", "meta", etc.
        Where to put the tensors.  Use "meta" if you only need the keys / shapes.

    Returns
    -------
    params : OrderedDict[str, torch.Tensor]
        All tensors in a single mapping (no duplicates).
    """
    dir_path = Path(dir_path)
    shard_files = sorted(dir_path.glob("*.safetensors"))
    if not shard_files:
        raise ValueError("No *.safetensors files found in", dir_path)

    params = OrderedDict()
    for shard in shard_files:
        # Each shard is itself an OrderedDict
        shard_dict = load_file(shard, device=device)        # <-- memory-mapped, fast
        overlap = set(params).intersection(shard_dict)
        if overlap:
            raise ValueError(f"Duplicate keys across shards: {overlap}")
        params.update(shard_dict)

    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~")+'/.cache/')
    parser.add_argument("--output-path")
    args = parser.parse_args()

    local_path = snapshot_download(
        repo_id=args.model_name,
        revision="main",                                 # branch / tag / commit hash
        cache_dir=args.cache_dir,
        allow_patterns=["*.safetensors", "config.json"], # optio grab only what you need
        local_dir_use_symlinks=False                     # set True to save disk via symlinks
    )
    params = load_all_safetensors(local_path, device="cpu")
    lm_layers = max([int(key.split('.')[2]) for key in params if key.startswith("model.layers")])
    weights_fp32 = {}
    weights_fp32['params/llm/model/embed_tokens/embedding'] = params[f"model.embed_tokens.weight"].detach().float().cpu().numpy()
    for l in range(lm_layers):
        weights_fp32[f'params/llm/model/layers/{l}/self_attn/k_proj/kernel'] = params[f"model.layers.{l}.self_attn.k_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        # weights_fp32[f'params/llm/model/layers/{l}/self_attn/k_proj/bias'] = params[f"model.model.layers.{l}.self_attn.k_proj.bias"].detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/self_attn/q_proj/kernel'] = params[f"model.layers.{l}.self_attn.q_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        # weights_fp32[f'params/llm/model/layers/{l}/self_attn/q_proj/bias'] = model.model.layers.{l}.self_attn.q_proj.bias"].detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/self_attn/v_proj/kernel'] = params[f"model.layers.{l}.self_attn.v_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        # weights_fp32[f'params/llm/model/layers/{l}/self_attn/v_proj/bias'] = model.model.layers.{l}.self_attn.v_proj.bias"].detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/self_attn/o_proj/kernel'] = params[f"model.layers.{l}.self_attn.o_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/mlp/gate_proj/kernel'] = params[f"model.layers.{l}.mlp.gate_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/mlp/up_proj/kernel'] = params[f"model.layers.{l}.mlp.up_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/mlp/down_proj/kernel'] = params[f"model.layers.{l}.mlp.down_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/input_layernorm/kernel'] = params[f"model.layers.{l}.input_layernorm.weight"].detach().float().cpu().numpy()
        weights_fp32[f'params/llm/model/layers/{l}/post_attention_layernorm/kernel'] = params[f"model.layers.{l}.post_attention_layernorm.weight"].detach().float().cpu().numpy()
    weights_fp32[f'params/llm/lm_head/kernel'] = params[f"lm_head.weight"].permute(1, 0).detach().float().cpu().numpy()
    weights_fp32[f'params/img/image_newline/embedding'] = params[f"model.image_newline"].unsqueeze(0).detach().float().cpu().numpy()
    weights_fp32[f'params/llm/model/norm/kernel'] = params[f"model.norm.weight"].detach().float().cpu().numpy()
    weights_fp32['params/img/embedding/bias'] = params[f"model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.bias"].detach().float().cpu().numpy()
    weights_fp32['params/img/embedding/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"].permute(2, 3, 1, 0).detach().float().cpu().numpy()
    weights_fp32['params/img/pos_embedding'] = params[f"model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight"].unsqueeze(0).detach().float().cpu().numpy()
    vision_layers = max([int(key.split('.')[6]) for key in params if key.startswith("model.vision_tower.vision_tower.vision_model.encoder.layers")])
    for i in range(vision_layers):
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"].permute(1, 0).detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"].permute(1, 0).detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"].permute(1, 0).detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"].detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"].permute(1, 0).detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"].detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"].permute(1, 0).detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"].detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/kernel'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"].permute(1, 0).detach().float().cpu().numpy() 
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/LayerNorm_0/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/LayerNorm_0/scale'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/LayerNorm_1/bias'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"].detach().float().cpu().numpy()
      weights_fp32[f'params/img/Transformer/encoderblock_{i}/LayerNorm_1/scale'] = params[f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"].detach().float().cpu().numpy() 
    weights_fp32[f'params/img/Transformer/encoder_norm/bias'] = params[f"model.vision_tower.vision_tower.vision_model.post_layernorm.bias"].detach().float().cpu().numpy()
    weights_fp32[f'params/img/Transformer/encoder_norm/scale'] = params[f"model.vision_tower.vision_tower.vision_model.post_layernorm.weight"].detach().float().cpu().numpy()
    weights_fp32[f'params/img/head1/kernel'] = params[f"model.mm_projector.0.weight"].permute(1, 0).detach().float().cpu().numpy()
    weights_fp32[f'params/img/head1/bias'] = params[f"model.mm_projector.0.bias"].detach().float().cpu().numpy()
    weights_fp32[f'params/img/head2/kernel'] = params[f"model.mm_projector.2.weight"].permute(1, 0).detach().float().cpu().numpy()
    weights_fp32[f'params/img/head2/bias'] = params[f"model.mm_projector.2.bias"].detach().float().cpu().numpy()
    np.savez(args.output_path, **weights_fp32)
