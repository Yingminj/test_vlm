#!/usr/bin/env bash
# Live video understanding with Marlin-2B (dense captioning + temporal grounding).
# GPU + a camera required. Uses ~4.4 GB VRAM (bf16).
#
#   conda activate vlm
#   bash scripts/marlin_camera.sh                                   # live captioning, auto every 5s
#   bash scripts/marlin_camera.sh --interval 0                      # caption only when you press SPACE
#   bash scripts/marlin_camera.sh --mode find --event "a hand picks up an object"
#   bash scripts/marlin_camera.sh --source 0                        # plain USB webcam index 0
#
# Notes:
#  * Loaded as the native Qwen3_5 class (Marlin adds no params) so NO torchcodec /
#    torch>=2.11 is needed -- frames are fed straight to the processor.
#  * Marlin was trained at FPS=2 on <=~2 min clips; event timestamps are relative
#    to the recent ~buffer/sample_fps-second clip, not absolute wall-clock.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-marlin-2b}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/marlin_camera.py --model "$MODEL" "$@"
