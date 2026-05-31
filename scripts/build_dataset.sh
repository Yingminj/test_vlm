#!/usr/bin/env bash
# End-to-end data build. With no real demos, generates synthetic ones first.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data}

if [ ! -f "$DATA_DIR/demos.jsonl" ]; then
  echo "[*] no demos.jsonl found -> generating synthetic demos"
  python scripts/make_synthetic_demos.py --out "$DATA_DIR" --n 16 --frames 24
fi

python -m verifier.data.build_dataset \
  --demos "$DATA_DIR/demos.jsonl" \
  --out "$DATA_DIR/processed" \
  --failures-per-success 2 \
  --semantic-mismatch-per-demo 1 \
  --window 16 --stride 1 --max-frames-per-demo 64 \
  --ongoing-ratio 2.0 --val-fraction 0.15
