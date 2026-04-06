#!/usr/bin/env python3
"""
Automated benchmark runner for different chunk_size values.
Modifies config.json and lingbotvla_cli.yaml, runs profiling_benchmark,
and collects results.
"""
import json
import re
import subprocess
import sys
import yaml

CONFIG_JSON = "./checkpoints/lingbot-vla-4b-posttrain-robotwin/config.json"
CLI_YAML = "./checkpoints/lingbot-vla-4b-posttrain-robotwin/lingbotvla_cli.yaml"

CHUNK_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

METRICS = {
    "Total_infer": r"Total _infer\s*:\s*([\d.]+)\s*ms",
    "Video_loop": r"Video\s+loop\s*:\s*([\d.]+)\s*ms",
    "Action_loop": r"Action\s+loop\s*:\s*([\d.]+)\s*ms",
    "TFLOPs/s_overall": r"TFLOPs/s overall\s*:\s*([\d.]+)",
    "TFLOPs/s_video": r"TFLOPs/s video\s*:\s*([\d.]+)",
    "TFLOPs/s_action": r"TFLOPs/s action\s*:\s*([\d.]+)",
}

def update_config(chunk_size):
    # Update config.json
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)
    config['chunk_size'] = chunk_size
    config['n_action_steps'] = chunk_size
    with open(CONFIG_JSON, 'w') as f:
        json.dump(config, f, indent=4)

    # Update lingbotvla_cli.yaml
    with open(CLI_YAML, 'r') as f:
        cli_config = yaml.safe_load(f)
    cli_config['train']['chunk_size'] = chunk_size
    with open(CLI_YAML, 'w') as f:
        yaml.dump(cli_config, f, default_flow_style=False, allow_unicode=True)

def update_config_dim(dim):
    # Update lingbotvla_cli.yaml
    with open(CLI_YAML, 'r') as f:
        cli_config = yaml.safe_load(f)
    cli_config['train']['action_dim'] = dim
    with open(CLI_YAML, 'w') as f:
        yaml.dump(cli_config, f, default_flow_style=False, allow_unicode=True)

def run_benchmark():
    cmd = (
        "source /home/user/miniconda3/etc/profile.d/conda.sh && "
        "conda activate lingbot && "
        "cd /home/user/lerobot/lingbot-vla && "
        "python -m scripts.profiling_benchmark "
        "--model_path ./checkpoints/lingbot-vla-4b-posttrain-robotwin "
        "--num_warmup 3 --num_runs 10 --num_steps 10 --use_bf16"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800,
                            executable="/bin/bash")
    return result.stdout + result.stderr

def parse_output(output):
    values = {}
    for name, pattern in METRICS.items():
        m = re.search(pattern, output)
        if m:
            values[name] = float(m.group(1))
        else:
            values[name] = None
    return values

def restore_config(orig_json, orig_yaml):
    with open(CONFIG_JSON, 'w') as f:
        f.write(orig_json)
    with open(CLI_YAML, 'w') as f:
        f.write(orig_yaml)

def run_dims():
    # Save original configs
    with open(CONFIG_JSON, 'r') as f:
        orig_json = f.read()
    with open(CLI_YAML, 'r') as f:
        orig_yaml = f.read()

    results = {name: [] for name in METRICS}
    results['dim'] = []

    try:
        for cs in DIMS:
            print(f"\n{'='*60}")
            print(f"  Running benchmark with dim = {cs}")
            print(f"{'='*60}")
            update_config_dim(cs)
            output = run_benchmark()
            print(output)
            values = parse_output(output)
            results['dim'].append(cs)
            for name in METRICS:
                results[name].append(values[name])
            print(f"  -> Parsed: {values}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Restore original configs
        restore_config(orig_json, orig_yaml)
        print("\nOriginal configs restored.")

    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print(f"chunk_size: {results['chunk_size']}")
    for name in METRICS:
        print(f"{name}: {results[name]}")

def run_chunk_size():
     # Save original configs
    with open(CONFIG_JSON, 'r') as f:
        orig_json = f.read()
    with open(CLI_YAML, 'r') as f:
        orig_yaml = f.read()

    results = {name: [] for name in METRICS}
    results['chunk_size'] = []

    try:
        for cs in CHUNK_SIZES:
            print(f"\n{'='*60}")
            print(f"  Running benchmark with chunk_size = {cs}")
            print(f"{'='*60}")
            update_config(cs)
            output = run_benchmark()
            print(output)
            values = parse_output(output)
            results['chunk_size'].append(cs)
            for name in METRICS:
                results[name].append(values[name])
            print(f"  -> Parsed: {values}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Restore original configs
        restore_config(orig_json, orig_yaml)
        print("\nOriginal configs restored.")

    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    print(f"chunk_size: {results['chunk_size']}")
    for name in METRICS:
        print(f"{name}: {results[name]}")

if __name__ == "__main__":
    run_dims()
