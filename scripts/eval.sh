#!/usr/bin/env bash
# Evaluate a trained adapter on the val shard (Part B.6). GPU required.
set -euo pipefail
source "$(dirname "$0")/_env.sh"   # cd repo root + prefer local ./qwen weights

CONFIG=${CONFIG:-configs/train.yaml}
ADAPTER=${ADAPTER:-runs/stage_c}
VAL=${VAL:-data/processed/val.jsonl}

mkdir -p runs
python -m verifier.eval.evaluate \
  --config "$CONFIG" \
  --adapter "$ADAPTER" \
  --val "$VAL" \
  --out runs/report.json
