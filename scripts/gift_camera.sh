#!/usr/bin/env bash
# Real-time gift-packaging state verifier: Qwen3.5-VL-9B + the gift_v1 LoRA adapter.
# GPU + a camera (or a video file) required. Run in the `vlm` conda env.
#
#   conda activate vlm
#   bash scripts/gift_camera.sh                              # auto camera, auto-check every 1.5s
#   bash scripts/gift_camera.sh --source 0 --interval 0      # webcam 0, manual (SPACE) only
#   bash scripts/gift_camera.sh --source clip.mp4 --loop     # replay a recorded clip
#   bash scripts/gift_camera.sh --source data/test_0605/realsense_20260609_150051.mp4 --eval
#                                                            # offline: score vs frame-level GT
#
# Overrides: BASE=<base weights>  ADAPTER=<lora dir>  (defaults track model/).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-model/qwen3_5_9B}
ADAPTER=${ADAPTER:-model/gift_v1}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/gift_camera.py --base "$BASE" --adapter "$ADAPTER" "$@"
