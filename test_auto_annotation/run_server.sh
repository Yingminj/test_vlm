#!/usr/bin/env bash
# Launch the local Qwen3.5-VL OpenAI-compatible server.
#
# Usage:
#   bash run_server.sh                 # serve qwen3_5_9B (bf16) on 0.0.0.0:8000
#   MODEL=27b bash run_server.sh       # serve qwen3_6_27B in 4-bit (fits 48GB)
#   MODEL=27b-bf16 bash run_server.sh  # serve qwen3_6_27B in bf16 (see WARNING)
#   PORT=9000 bash run_server.sh       # override port
#
# Override any of MODEL_PATH / MODEL_NAME / QUANT / HOST / PORT / MAX_FRAMES /
# SAMPLE_FPS / MAX_PIXELS / API_KEY directly (see server.py header).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# Convenience selector: MODEL=9b (default) | 27b | 27b-bf16
MODEL="${MODEL:-9b}"
case "$MODEL" in
  9b)       : "${MODEL_PATH:=$REPO/model/qwen3_5_9B}";  : "${QUANT:=none}" ;;
  27b)      : "${MODEL_PATH:=$REPO/model/qwen3_6_27B}"; : "${QUANT:=4bit}" ;;
  27b-bf16) : "${MODEL_PATH:=$REPO/model/qwen3_6_27B}"; : "${QUANT:=none}"
             echo "[warn] 27B bf16 weights are ~54GB and DO NOT fit one 48GB 4090." >&2
             echo "[warn] device_map=auto will spill layers to CPU RAM (slow), and" >&2
             echo "[warn] needs a large host-RAM budget. Use MODEL=27b (4-bit) to fit on GPU." >&2 ;;
  *)        echo "MODEL must be 9b, 27b, or 27b-bf16 (or set MODEL_PATH directly)"; exit 1 ;;
esac

export MODEL_PATH QUANT
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export MAX_FRAMES="${MAX_FRAMES:-64}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run] MODEL_PATH=$MODEL_PATH QUANT=$QUANT -> $HOST:$PORT"
exec python "$HERE/server.py"
