#!/usr/bin/env bash
# Evaluate a trained adapter on the val shard (Part B.6). GPU required.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/train.yaml}
ADAPTER=${ADAPTER:-runs/stage_c}
VAL=${VAL:-data/processed/val.jsonl}

mkdir -p runs
python -m verifier.eval.evaluate \
  --config "$CONFIG" \
  --adapter "$ADAPTER" \
  --val "$VAL" \
  --out runs/report.json
