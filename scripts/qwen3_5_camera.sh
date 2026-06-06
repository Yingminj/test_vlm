#!/usr/bin/env bash
# Live video understanding with base Qwen3.5-VL. GPU + a camera required.
#
#   conda activate vlm
#   bash scripts/qwen3_5_camera.sh                                   # default prompt, auto every 4s
#   bash scripts/qwen3_5_camera.sh --prompt "What is the person doing?"
#   bash scripts/qwen3_5_camera.sh --source 0 --interval 0           # webcam 0, manual (SPACE) only
#   bash scripts/qwen3_5_camera.sh --think --max-new-tokens 256      # show reasoning
#
# Optional speedup (linear-attention fast kernels; needs a compiler + matching CUDA):
#   pip install causal-conv1d && pip install flash-linear-attention
# Without them the model still runs via a slower PyTorch fallback.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-qwen3_5_9B}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/qwen3_5_camera.py --model "$MODEL" "$@"
