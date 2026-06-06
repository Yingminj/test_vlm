#!/usr/bin/env bash
# Live RealSense task-state verification. GPU + RealSense camera required.
# Uses local ./qwen base weights (offline) + a trained LoRA adapter.
#
#   ADAPTER=runs/gift/stage_c SUBTASK="place the cup on the shelf" bash scripts/realtime.sh
#   bash scripts/realtime.sh --subtask "open the drawer"   # extra flags passed through
set -euo pipefail
source "$(dirname "$0")/_env.sh"   # cd repo root + prefer local ./qwen weights

CONFIG=${CONFIG:-configs/train.yaml}
ADAPTER=${ADAPTER:-runs/gift/stage_c}
SUBTASK=${SUBTASK:-}

ARGS=(--config "$CONFIG" --adapter "$ADAPTER")
[ -n "$SUBTASK" ] && ARGS+=(--subtask "$SUBTASK")

python -m verifier.infer.realtime_realsense "${ARGS[@]}" "$@"
