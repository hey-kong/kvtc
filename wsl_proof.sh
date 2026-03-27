#!/bin/bash
set -e
echo "=== KVTC vLLM Proof Benchmark (WSL2) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Setup
export HF_HOME=/mnt/t/huggingface
export PYTHONIOENCODING=utf-8
cd /mnt/c/Users/User/kvtc

# Install deps
echo "[1/3] Installing dependencies..."
pip3 install --user -q vllm torch transformers 2>&1 | tail -5
pip3 install --user -q -e . 2>&1 | tail -3

# Run proof benchmark
echo "[2/3] Running KVTC vLLM proof benchmark..."
python3 proof.py --model Qwen/Qwen2.5-3B-Instruct 2>&1

echo "[3/3] DONE"
