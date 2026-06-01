#!/usr/bin/env bash
# Run the three SFT stages in sequence (Part B.3). GPU required.
set -euo pipefail
source "$(dirname "$0")/_env.sh"   # cd repo root + prefer local ./qwen weights

CONFIG=${CONFIG:-configs/train.yaml}

echo "[*] Stage A: schema cold-start"
python -m verifier.train.train_sft --config "$CONFIG" --stage a

echo "[*] Stage B: per-clip verifier SFT"
python -m verifier.train.train_sft --config "$CONFIG" --stage b

echo "[*] Stage C: streaming/timing SFT (transition-upweighted)"
python -m verifier.train.train_sft --config "$CONFIG" --stage c

echo "[done] adapters in runs/stage_{a,b,c}"
