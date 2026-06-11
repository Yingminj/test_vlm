#!/usr/bin/env bash
# Live video understanding with base Qwen3.6-VL-27B. GPU + a camera required.
#
# This 27B model: ~14 GB in 4bit, ~27 GB in 8bit, ~54 GB in bf16. bf16 does NOT fit on
# a single 48 GB GPU, so the script defaults to --precision 8bit (~30 GB peak, best
# quality that fits). Use --precision 4bit for the lightest footprint (~18 GB).
#
#   conda activate vlm
#   bash scripts/qwen3_6_camera.sh                                   # 8-bit, default prompt, auto every 4s
#   bash scripts/qwen3_6_camera.sh --prompt "What is the person doing?"
#   bash scripts/qwen3_6_camera.sh --source 0 --interval 0           # webcam 0, manual (SPACE) only
#   bash scripts/qwen3_6_camera.sh --think --max-new-tokens 256      # show reasoning
#   bash scripts/qwen3_6_camera.sh --precision 4bit                  # lightest (~18 GB)
#   bash scripts/qwen3_6_camera.sh --precision bf16                  # full bf16 (multi-GPU only)
#
# Optional speedup (linear-attention fast kernels; needs a compiler + matching CUDA):
#   pip install causal-conv1d && pip install flash-linear-attention
# Without them the model still runs via a slower PyTorch fallback.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-model/qwen3_6_27B}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/qwen3_6_camera.py --model "$MODEL" "$@"
