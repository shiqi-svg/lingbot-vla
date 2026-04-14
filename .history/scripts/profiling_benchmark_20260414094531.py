#!/usr/bin/env python3
"""
LingBot-VLA Inference Profiling Benchmark Script
=================================================
Measures the following metrics (matching the profiling output format):
  - Total inference time (ms)
  - Video  loop (prefix encoding: vision + language) time, steps, percentage
  - Action  loop (iterative denoising) time, steps, percentage
  - Latent embed (MLP): action_time_mlp total/avg time
  - Action embed (linear): action_in_proj + action_out_proj total/avg time
  - TFLOPs total
  - TFLOPs/s overall, video, action
  - Params latent/action (parameter counts)
  - GPU memory before, after, peak
  - GPU utilization
  - Action generation bandwidth (actions/sec, bytes/sec)

Usage:
    cd /home/user/lerobot/lingbot-vla
    python -m scripts.profiling_benchmark \
        --model_path ./checkpoints/lingbot-vla-4b-posttrain-robotwin \
        --num_warmup 2 \
        --num_runs 5 \
        --num_steps 10
"""

import argparse
import json
import os
import sys
import time
import random
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig as HFPretrainedConfig

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lerobot.configs.policies import PreTrainedConfig
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.data.vla_data.transform import Normalizer, prepare_images, prepare_language, prepare_state
from lingbotvla.models import build_processor
from lingbotvla.utils.count_flops import LingBotFlopsCounter, get_device_flops


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


BASE_MODEL_PATH = {
    'lingbotvla': os.environ.get('QWEN25_PATH', './checkpoints/Qwen2.5-VL-3B-Instruct/'),
}


def load_model_weights(policy, path_to_pi_model, strict=True):
    all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
    merged_weights = {}
    for file_path in tqdm(all_safetensors, desc="Loading weights"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)
    policy.load_state_dict(merged_weights, strict=strict)


def merge_qwen_config(policy_config, qwen_config):
    if hasattr(qwen_config, 'to_dict'):
        config_dict = qwen_config.to_dict()
    else:
        config_dict = qwen_config
    text_keys = {
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
        "rope_theta", "vocab_size", "max_position_embeddings",
        "hidden_act", "tie_word_embeddings", "tokenizer_path",
    }
    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    return policy_config


def resize_with_pad_item(img, width, height, pad_value=0):
    if img.ndim != 3:
        raise ValueError(f"(c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[1:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0), size=(resized_height, resized_width),
        mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def count_parameters(module):
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class InferencePolicyForProfiling(LingbotVlaPolicy):
    """Extended policy class with profiling hooks for detailed timing."""
    pass


def load_model(model_path, use_bf16=True):
    """Load model following the same logic as QwenPiServer."""
    print(f"Loading model from: {model_path}")
    config = PreTrainedConfig.from_pretrained(model_path)

    # print(f"-------查看config:--------\n {config}\n----------结束查看config------------------")

    training_config_path = Path(model_path) / 'lingbotvla_cli.yaml'
    if not training_config_path.exists():
        # Try different levels up
        for i in range(1, 5):
            candidate = Path(model_path).parents[i] / 'lingbotvla_cli.yaml'
            if candidate.exists():
                training_config_path = candidate
                break
        if not training_config_path.exists():
            training_config_path = Path(model_path) / 'lingbotvla_cli.yaml'

    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    training_model_config = training_config['model']
    training_model_config.update(training_config['train'])
    for k, v in training_model_config.items():
        v = getattr(config, k, training_model_config[k])
        setattr(config, k, v)

    # config.attention_implementation = 'flex'

    base_model_path = BASE_MODEL_PATH['lingbotvla']
    config.tokenizer_path = base_model_path

    qwen_config = AutoConfig.from_pretrained(base_model_path)
    config = merge_qwen_config(config, qwen_config)
    # print(f"-------查看merged  qwen_config 后的config:--------\n {config}\n----------结束查看merged config------------------")
    if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
        config.vocab_size = training_config['model']['vocab_size']

    processor = build_processor(base_model_path)
    language_tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    data_config = SimpleNamespace(**training_config['data'])

    print('Initializing model...')
    policy = InferencePolicyForProfiling(config, tokenizer_path=base_model_path)
    load_model_weights(policy, model_path, strict=True)

    policy.feature_transform = None

    # Setup normalizer
    norm_stats_file = getattr(data_config, 'norm_stats_file', 'assets/norm_stats/robotwin_50.json')
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)
    policy.normalizer = Normalizer(
        norm_stats=norm_stats['norm_stats'],
        from_file=True,
        data_type='robotwin',
        norm_type={
            "observation.images.cam_high": "identity",
            "observation.images.cam_left_wrist": "identity",
            "observation.images.cam_right_wrist": "identity",
            "observation.state": data_config.norm_type,
            "action": data_config.norm_type,
        },
    )

    policy.action_dim = training_config['train']['action_dim']
    policy.chunk_size = training_config['train']['chunk_size']

    if use_bf16:
        policy = policy.cuda().eval().to(torch.bfloat16)
    else:
        policy = policy.cuda().eval()

    return policy, config, language_tokenizer, image_processor, data_config, training_config


def create_dummy_observation(config, language_tokenizer, image_processor):
    """Create a dummy observation for profiling."""
    image_size = getattr(config, 'resize_imgs_with_padding', [224, 224])
    if isinstance(image_size, (list, tuple)):
        h, w = image_size[0], image_size[1]
    else:
        h = w = image_size

    # Create 3 dummy images (cam_high, cam_left_wrist, cam_right_wrist)
    dummy_images = {}
    for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        img = torch.randint(0, 255, (3, h, w), dtype=torch.uint8)
        dummy_images[key] = img

    state_dim = config.action_dim if hasattr(config, 'action_dim') else 16
    dummy_state = torch.randn(state_dim, dtype=torch.float32)

    obs_dict = {
        "image": dummy_images,
        "state": dummy_state,
        "prompt": ["pick up the red block"],
    }

    state = prepare_state(config, obs_dict)
    lang_tokens, lang_masks = prepare_language(config, language_tokenizer, obs_dict)
    images, img_masks, _ = prepare_images(config, image_processor, obs_dict)

    observation = {
        'images': images,
        'img_masks': img_masks,
        'state': state,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
    }
    return observation


def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0.0

def get_hit_time(H, u_0=0.9, alpha=0.7):
    """Compute hit time based on given parameters."""
    i = torch.arange(H, dtype=torch.bfloat16)
    u = u_0 * (1 - (i/(H-1)))**alpha
    return u



def profile_inference(policy, config, observation, num_steps, use_bf16=True):
    """
    Run a single inference with detailed timing instrumentation.
    Returns a dict of all timing measurements.
    """
    device = 'cuda'
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Move observation to device
    obs = {}
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):
            if k in ('img_masks', 'lang_tokens', 'lang_masks'):
                # These should stay as their original dtype (bool/long)
                obs[k] = v.to(device=device)
            else:
                obs[k] = v.to(dtype=dtype, device=device)
        else:
            obs[k] = v

    if len(obs['images'].shape) == 4:
        obs['images'] = obs['images'].unsqueeze(0)
        obs['img_masks'] = obs['img_masks'].unsqueeze(0)

    images = obs['images']
    img_masks = obs['img_masks']
    lang_tokens = obs['lang_tokens'].unsqueeze(0)
    lang_masks = obs['lang_masks'].unsqueeze(0)
    state = obs['state'].unsqueeze(0)

    model = policy.model  # FlowMatching module
    bsize = state.shape[0]

    results = {}

    # =========== Total inference with sub-timings ===========
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    # --- Video  : Prefix Encoding (Vision + Language) ---
    torch.cuda.synchronize()
    video_start = time.perf_counter()

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, vlm_causal=False
    )
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Compute KV cache
    _, past_key_values = model.qwenvl_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=config.use_cache,
        fill_kv_cache=True,
    )

    torch.cuda.synchronize()
    video_end = time.perf_counter()
    video_time_ms = (video_end - video_start) * 1000.0

    # --- Action  : Iterative Denoising Loop ---
    torch.cuda.synchronize()
    action_start = time.perf_counter()

    actions_shape = (bsize, config.n_action_steps, config.max_action_dim)
    noise = torch.randn(actions_shape, device=device, dtype=dtype)
    dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    x_t = noise
    t = torch.tensor(1.0, dtype=dtype, device=device)

    latent_embed_total_ms = 0.0  # MLP time
    action_embed_total_ms = 0.0  # Linear proj time
    latent_embed_calls = 0
    action_embed_calls = 0

    step_count = 0
    while t >= -dt / 2: #t从1-》0
        #make rho 当前循环走到哪了
        rho = (1 - t)
        tau = torch.where()
        step_count += 1
        expanded_time = t.expand(bsize)

        # --- Time embed_suffix (contains Latent Embed MLP and Action Embed Linear) ---
        torch.cuda.synchronize()
        embed_suffix_start = time.perf_counter()

        # Manually decompose embed_suffix for timing
        state_emb = model.state_proj(state)

        from lingbotvla.models.vla.pi0.modeling_lingbot_vla import create_sinusoidal_pos_embedding
        time_emb = create_sinusoidal_pos_embedding(
            expanded_time, config.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=dtype)
        time_emb_ori = time_emb

        # --- Action Embed (Linear): action_in_proj ---
        torch.cuda.synchronize()
        action_linear_start = time.perf_counter()
        action_emb = model.action_in_proj(x_t)
        torch.cuda.synchronize()
        action_linear_end = time.perf_counter()

        # --- Latent Embed (MLP): action_time_mlp ---
        torch.cuda.synchronize()
        mlp_start = time.perf_counter()
        if getattr(config, "separate_time_proj", False):
            time_emb_for_suffix = model.time_mlp_in(time_emb)
            time_emb_for_suffix = F.silu(time_emb_for_suffix)
            time_emb_ori = F.silu(model.time_mlp_out(time_emb_for_suffix))
            action_time_emb = action_emb
        else:
            import einops
            time_emb_expanded = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
            action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=-1)
            action_time_emb = model.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = model.action_time_mlp_out(action_time_emb)
        torch.cuda.synchronize()
        mlp_end = time.perf_counter()

        latent_embed_total_ms += (mlp_end - mlp_start) * 1000.0
        latent_embed_calls += 1

        action_time_dim = action_time_emb.shape[1]
        suffix_embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        suffix_pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        suffix_att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        suffix_att_masks[:, :2] = True

        # predict_velocity (transformer forward pass)
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            bsize, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        ada_cond = time_emb_ori if getattr(config, 'adanorm_time', False) else None

        outputs_embeds, _ = model.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=config.use_cache,
            fill_kv_cache=False,
            ada_cond=ada_cond,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -config.n_action_steps:]

        # --- Action Embed (Linear): action_out_proj ---
        torch.cuda.synchronize()
        action_out_start = time.perf_counter()
        v_t = model.action_out_proj(suffix_out)
        torch.cuda.synchronize()
        action_out_end = time.perf_counter()

        action_embed_total_ms += (action_linear_end - action_linear_start) * 1000.0
        action_embed_total_ms += (action_out_end - action_out_start) * 1000.0
        action_embed_calls += 2  # in_proj + out_proj per step

        # Euler step
        x_t = x_t + dt * v_t
        t = t + dt

    torch.cuda.synchronize()
    action_end = time.perf_counter()
    action_time_ms = (action_end - action_start) * 1000.0

    torch.cuda.synchronize()
    total_end = time.perf_counter()
    total_time_ms = (total_end - total_start) * 1000.0

    results['total_infer_ms'] = total_time_ms
    results['video_ms'] = video_time_ms
    results['video_steps'] = 1  # Prefix computed once (1 forward pass through VLM)
    results['action_ms'] = action_time_ms
    results['action_steps'] = step_count
    results['latent_embed_total_ms'] = latent_embed_total_ms
    results['latent_embed_calls'] = latent_embed_calls
    results['action_embed_total_ms'] = action_embed_total_ms
    results['action_embed_calls'] = action_embed_calls
    results['final_actions'] = x_t

    return results


def compute_flops_inference(config, training_config, num_steps):
    """
    Compute TFLOPs for a single inference pass.
    For inference, we use 2 * N (fwd only) instead of 6 * N (fwd + bwd).
    """
    # VLM (Qwen2.5-VL) parameters
    vlm_hidden = 2048
    vlm_vocab = 151936
    vlm_layers = 36
    vlm_kv_heads = 2
    vlm_attn_heads = 16
    vlm_intermediate = 11008

    # Expert (Action Expert) parameters
    expert_hidden = 768
    expert_vocab = 0
    expert_layers = 36
    expert_kv_heads = 2
    expert_attn_heads = 16
    expert_intermediate = 2752

    def compute_dense_and_attn(hidden_size, vocab_size, num_layers, num_kv_heads, num_attn_heads, intermediate_size):
        head_dim = hidden_size // num_attn_heads
        q_size = num_attn_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim
        mlp_N = hidden_size * intermediate_size * 3  # SwiGLU
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attn_heads * head_dim)
        emb_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_layers + emb_N
        model_attn_flops = head_dim * num_attn_heads * num_layers
        return dense_N, model_attn_flops

    vlm_dense_N, vlm_attn_flops = compute_dense_and_attn(
        vlm_hidden, vlm_vocab, vlm_layers, vlm_kv_heads, vlm_attn_heads, vlm_intermediate
    )
    expert_dense_N, expert_attn_flops = compute_dense_and_attn(
        expert_hidden, expert_vocab, expert_layers, expert_kv_heads, expert_attn_heads, expert_intermediate
    )

    # Vision encoder (Qwen2.5-VL ViT)
    vit_dim = 1280
    vit_depth = 32
    vit_num_heads = 16
    vit_mlp_hidden = 3420
    vit_out_hidden = 2048
    vit_spatial_merge = 2
    vit_head_dim = vit_dim // vit_num_heads

    # Image tokens: 3 images of 224x224 with patch 14, spatial_merge 2
    # token_per_image = (224/14)^2 / (2^2) = 16*16/4 = 64
    tokens_per_image = (224 // 14) ** 2 // (vit_spatial_merge ** 2)
    num_images = 3
    total_image_tokens = tokens_per_image * num_images  # 192

    # Language tokens
    lang_tokens = getattr(config, 'tokenizer_max_length', 24)

    # Prefix tokens
    prefix_len = total_image_tokens + lang_tokens  # ~216

    # Suffix tokens per step: 1 state + n_action_steps
    n_action_steps = getattr(config, 'n_action_steps', 50)
    suffix_len = 1 + n_action_steps  # 51

    # ---- VIDEO FLOPS (prefix encoding) ----
    # ViT FLOPs (fwd only: 2 * N)
    vit_mlp_N = vit_dim * vit_mlp_hidden * 3
    vit_attn_linear_N = vit_dim * (4 * vit_dim)
    vit_patch_embed_N = (vit_out_hidden + (vit_dim * vit_spatial_merge**2)) * (vit_dim * vit_spatial_merge**2)
    vit_dense_N = (vit_mlp_N + vit_attn_linear_N) * vit_depth + vit_patch_embed_N
    # For ViT, total input tokens before spatial merge
    vit_input_tokens = (224 // 14) ** 2 * num_images  # 768
    vit_dense_flops = 2 * vit_dense_N * vit_input_tokens

    # ViT attention flops
    full_attn_layers = 4
    window_attn_layers = vit_depth - full_attn_layers
    vit_attn_flops_total = 0
    # Full attention on 4 layers
    tokens_per_img_before_merge = (224 // 14) ** 2  # 256
    for _ in range(num_images):
        vit_attn_flops_total += 4 * tokens_per_img_before_merge ** 2 * vit_head_dim * vit_num_heads * full_attn_layers
    # Window attention
    vit_attn_flops_total += 4 * vit_input_tokens * (112 ** 2) * vit_head_dim * vit_num_heads * window_attn_layers

    vit_flops = vit_dense_flops + vit_attn_flops_total

    # VLM prefix forward (fwd only: 2 * N)
    vlm_prefix_dense_flops = 2 * vlm_dense_N * prefix_len
    vlm_prefix_attn_flops = 4 * prefix_len ** 2 * (vlm_attn_flops // vlm_layers) * vlm_layers

    video_flops = vit_flops + vlm_prefix_dense_flops + vlm_prefix_attn_flops

    # ---- ACTION FLOPS (denoising loop) ----
    # Per step: process suffix with cross-attention to prefix KV cache
    # Dense flops per step (expert processes suffix tokens)
    expert_suffix_dense_per_step = 2 * expert_dense_N * suffix_len
    # VLM also processes suffix (dual-expert cross-attention)
    # But VLM uses KV cache so only suffix tokens are computed
    vlm_suffix_dense_per_step = 2 * vlm_dense_N * suffix_len

    # Attention: suffix attends to (prefix + suffix) = cross-attention
    total_seq_for_attn = prefix_len + suffix_len
    expert_attn_per_step = 4 * suffix_len * total_seq_for_attn * (expert_attn_flops // expert_layers) * expert_layers
    vlm_attn_per_step = 4 * suffix_len * total_seq_for_attn * (vlm_attn_flops // vlm_layers) * vlm_layers

    action_flops_per_step = (
        expert_suffix_dense_per_step + vlm_suffix_dense_per_step +
        expert_attn_per_step + vlm_attn_per_step
    )
    action_flops = action_flops_per_step * num_steps

    total_flops = video_flops + action_flops

    return {
        'total_flops': total_flops,
        'video_flops': video_flops,
        'action_flops': action_flops,
        'prefix_len': prefix_len,
        'suffix_len': suffix_len,
        'vit_input_tokens': vit_input_tokens,
        'total_image_tokens': total_image_tokens,
    }


def count_model_params(model):
    """Count parameters for latent(VLM) and action(Expert) components."""
    fm = model.model  # FlowMatching

    # Latent params: VLM (Qwen2.5-VL) - processes images + language (prefix path)
    latent_params = 0
    if hasattr(fm.qwenvl_with_expert, 'qwenvl'):
        latent_params += count_parameters(fm.qwenvl_with_expert.qwenvl)

    # Action params: Expert model + projection layers (suffix/denoising path)
    action_params = 0
    if hasattr(fm.qwenvl_with_expert, 'qwen_expert'):
        action_params += count_parameters(fm.qwenvl_with_expert.qwen_expert)
    action_params += count_parameters(fm.state_proj)
    action_params += count_parameters(fm.action_in_proj)
    action_params += count_parameters(fm.action_out_proj)
    if hasattr(fm, 'action_time_mlp_in'):
        action_params += count_parameters(fm.action_time_mlp_in)
        action_params += count_parameters(fm.action_time_mlp_out)
    elif hasattr(fm, 'time_mlp_in'):
        action_params += count_parameters(fm.time_mlp_in)
        action_params += count_parameters(fm.time_mlp_out)

    # Total model parameters
    total_params = count_parameters(fm)

    return latent_params, action_params, total_params


def run_benchmark(args):
    set_seed(42)

    print("=" * 80)
    print("  LingBot-VLA Inference Profiling Benchmark")
    print("=" * 80)

    # Load model
    policy, config, language_tokenizer, image_processor, data_config, training_config = \
        load_model(args.model_path, use_bf16=args.use_bf16)

    num_steps = args.num_steps if args.num_steps > 0 else getattr(config, 'num_steps', 10)
    config.num_steps = num_steps

    # Create dummy observation
    observation = create_dummy_observation(config, language_tokenizer, image_processor)

    # Count parameters
    latent_params, action_params, total_params = count_model_params(policy)

    # Compute theoretical FLOPs
    flops_info = compute_flops_inference(config, training_config, num_steps)

    # GPU memory before
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gpu_mem_before = torch.cuda.memory_allocated() / (1024 ** 3)

    # Warmup runs
    print(f"\nWarmup: {args.num_warmup} runs...")
    for i in range(args.num_warmup):
        with torch.no_grad():
            _ = profile_inference(policy, config, observation, num_steps, use_bf16=args.use_bf16)

    # Benchmark runs
    print(f"Benchmark: {args.num_runs} runs...")
    all_results = []
    gpu_utils = []

    for i in range(args.num_runs):
        torch.cuda.synchronize()
        gpu_util_before = get_gpu_utilization()

        with torch.no_grad():
            result = profile_inference(policy, config, observation, num_steps, use_bf16=args.use_bf16)

        torch.cuda.synchronize()
        gpu_util_after = get_gpu_utilization()

        all_results.append(result)
        gpu_utils.append((gpu_util_before + gpu_util_after) / 2)

    # GPU memory after & peak
    torch.cuda.synchronize()
    gpu_mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Aggregate results
    avg = lambda key: np.mean([r[key] for r in all_results])
    std = lambda key: np.std([r[key] for r in all_results])

    total_infer_ms = avg('total_infer_ms')
    video_ms = avg('video_ms')
    action_ms = avg('action_ms')
    video_steps = all_results[0]['video_steps']
    action_steps = all_results[0]['action_steps']
    latent_embed_total = avg('latent_embed_total_ms')
    latent_embed_calls = all_results[0]['latent_embed_calls']
    action_embed_total = avg('action_embed_total_ms')
    action_embed_calls = all_results[0]['action_embed_calls']

    video_pct = video_ms / total_infer_ms * 100
    action_pct = action_ms / total_infer_ms * 100
    latent_embed_pct = latent_embed_total / total_infer_ms * 100
    action_embed_pct = action_embed_total / total_infer_ms * 100
    latent_embed_avg = latent_embed_total / latent_embed_calls if latent_embed_calls > 0 else 0
    action_embed_avg = action_embed_total / action_embed_calls if action_embed_calls > 0 else 0

    # TFLOPs calculations
    total_tflops = flops_info['total_flops'] / 1e12
    video_tflops = flops_info['video_flops'] / 1e12
    action_tflops = flops_info['action_flops'] / 1e12

    total_infer_s = total_infer_ms / 1000.0
    video_infer_s = video_ms / 1000.0
    action_infer_s = action_ms / 1000.0

    tflops_per_s_overall = total_tflops / total_infer_s if total_infer_s > 0 else 0
    tflops_per_s_video = video_tflops / video_infer_s if video_infer_s > 0 else 0
    tflops_per_s_action = action_tflops / action_infer_s if action_infer_s > 0 else 0

    avg_gpu_util = np.mean(gpu_utils)

    # Action generation bandwidth
    n_action_steps = getattr(config, 'n_action_steps', 50)
    action_dim = training_config['train']['action_dim']
    max_action_dim = training_config['train']['max_action_dim']
    # Bandwidth: how many action steps generated per second
    actions_per_sec = n_action_steps / total_infer_s
    # Bytes: each action is float32 (4 bytes) * action_dim per step
    bytes_per_action = action_dim * 4  # float32
    bandwidth_bytes_per_sec = actions_per_sec * bytes_per_action
    bandwidth_mb_per_sec = bandwidth_bytes_per_sec / (1024 * 1024)

    # Device peak TFLOPs
    device_peak_tflops = get_device_flops("T")
    # Supplement known GPU types not in the original list
    if device_peak_tflops == float('inf'):
        gpu_name = torch.cuda.get_device_name()
        if "B300" in gpu_name or "B200" in gpu_name:
            device_peak_tflops = 2250.0  # BF16 peak TFLOPs
        elif "4090" in gpu_name:
            device_peak_tflops = 330.3
        elif "3090" in gpu_name:
            device_peak_tflops = 142.0

    # Print results
    print("\n")
    print("=" * 80)
    print(f"  [Profiling] {'=' * 66}")
    print("=" * 80)
    print()
    print(f"  Total _infer          : {total_infer_ms:.1f} ms")
    print(f"  Video  loop        : {video_ms:.1f} ms  ({video_steps} steps), percentage:{video_pct:.1f}%")
    print(f"  Action  loop       : {action_ms:.1f} ms  ({action_steps} steps), percentage:{action_pct:.1f}%")
    print(f"  Latent embed (MLP)    : {latent_embed_total:.1f} ms total / {latent_embed_avg:.2f} ms avg ({latent_embed_calls} calls), percentage:{latent_embed_pct:.1f}%")
    print(f"  Action embed (linear) : {action_embed_total:.1f} ms total / {action_embed_avg:.2f} ms avg ({action_embed_calls} calls), percentage:{action_embed_pct:.1f}%")
    print(f"  TFLOPs total          : {total_tflops:.2f} TFLOPs")
    print(f"  TFLOPs/s overall      : {tflops_per_s_overall:.2f} TFLOPs/s")
    print(f"  TFLOPs/s video        : {tflops_per_s_video:.2f} TFLOPs/s")
    print(f"  TFLOPs/s action       : {tflops_per_s_action:.2f} TFLOPs/s")
    print(f"  Params latent/action  : {latent_params} / {action_params}")
    print(f"  GPU mem before        : {gpu_mem_before:.2f} GB")
    print(f"  GPU mem after         : {gpu_mem_after:.2f} GB")
    print(f"  GPU mem peak          : {gpu_mem_peak:.2f} GB")
    print(f"  GPU utilization       : {avg_gpu_util:.0f} %")
    print()
    print(f"  [Bandwidth] {'=' * 64}")
    print(f"  Action chunk size     : {n_action_steps} steps × {action_dim} dims")
    print(f"  Actions/sec           : {actions_per_sec:.2f} steps/s")
    print(f"  Bandwidth (bytes)     : {bandwidth_bytes_per_sec:.0f} B/s ({bandwidth_mb_per_sec:.4f} MB/s)")
    print(f"  Inference latency     : {total_infer_ms:.1f} ms per chunk")
    print(f"  Control freq (chunk)  : {1000.0 / total_infer_ms:.2f} Hz (new chunk per inference)")
    print(f"  Effective ctrl freq   : {n_action_steps * 1000.0 / total_infer_ms:.2f} Hz (if execute full chunk)")
    print()
    print(f"  [Device Info] {'=' * 63}")
    print(f"  GPU                   : {torch.cuda.get_device_name()}")
    print(f"  Device peak TFLOPs    : {device_peak_tflops:.1f} TFLOPs")
    if device_peak_tflops > 0 and device_peak_tflops != float('inf'):
        mfu = tflops_per_s_overall / device_peak_tflops * 100
        print(f"  MFU (overall)         : {mfu:.1f}%")
    print()
    print(f"  [Config] {'=' * 67}")
    print(f"  num_steps (denoising) : {num_steps}")
    print(f"  n_action_steps        : {n_action_steps}")
    print(f"  max_action_dim        : {max_action_dim}")
    print(f"  action_dim            : {action_dim}")
    print(f"  dtype                 : {'bfloat16' if args.use_bf16 else 'float32'}")
    print(f"  num_warmup            : {args.num_warmup}")
    print(f"  num_runs              : {args.num_runs}")
    print()
    print(f"  [Timing Std Dev] {'=' * 59}")
    print(f"  Total infer std       : ±{std('total_infer_ms'):.1f} ms")
    print(f"  Video  std         : ±{std('video_ms'):.1f} ms")
    print(f"  Action  std        : ±{std('action_ms'):.1f} ms")
    print("=" * 80)

    # print(f"查看training_config['model']['vocab_size']: {training_config['model']['vocab_size']}")

    # print(f"-------查看config:--------\n {config}\n----------结束查看config------------------")

    # print(f"\n-------training_config:--------\n {training_config}\n----------结束查看training_config------------------")

    return {
        'total_infer_ms': total_infer_ms,
        'video_ms': video_ms,
        'action_ms': action_ms,
        'latent_embed_total_ms': latent_embed_total,
        'action_embed_total_ms': action_embed_total,
        'tflops_total': total_tflops,
        'tflops_per_s_overall': tflops_per_s_overall,
        'tflops_per_s_video': tflops_per_s_video,
        'tflops_per_s_action': tflops_per_s_action,
        'latent_params': latent_params,
        'action_params': action_params,
        'gpu_mem_before_gb': gpu_mem_before,
        'gpu_mem_after_gb': gpu_mem_after,
        'gpu_mem_peak_gb': gpu_mem_peak,
        'gpu_utilization': avg_gpu_util,
        'actions_per_sec': actions_per_sec,
        'bandwidth_mb_per_sec': bandwidth_mb_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="LingBot-VLA Inference Profiling Benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--num_warmup", type=int, default=2,
                        help="Number of warmup inference runs")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of benchmark inference runs")
    parser.add_argument("--num_steps", type=int, default=0,
                        help="Denoising steps (0=use config default)")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    parser.add_argument("--no_bf16", action="store_true", default=False,
                        help="Use float32 precision")
    args = parser.parse_args()

    # 打印出当前真正在内存里干活的 flash_attn 所在的物理路径和版本
    import sys
    import flash_attn
    print("================= FA 测试 =================")
    print(f"当前加载的模块路径: {flash_attn.__file__}")
    print(f"当前加载的模块版本: {getattr(flash_attn, '__version__', '未知版本')}")
    print("=================================================")

    if args.no_bf16:
        args.use_bf16 = False
    run_benchmark(args)


if __name__ == "__main__":
    main()
