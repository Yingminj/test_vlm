# Server runbook (validated)

End-to-end verified on the training server (2026-05-31): data build, all 3 SFT
stage entrypoints, streaming inference, and the eval report.

## Environment
- **GPU:** NVIDIA RTX 4090 **48 GB** variant (single GPU), Ubuntu 22.04.
- **conda env `vlm`:** Python 3.10, torch 2.8.0+cu129 (CUDA available).
- **Installed (training):** transformers 5.9, peft 0.19, accelerate 1.13,
  bitsandbytes 0.49, qwen-vl-utils 0.0.14, datasets 4.8.
- **Weights:** Qwen2.5-VL-3B (ModelScope download) at `./qwen` — pointed to via
  `VERIFIER_MODEL_ID` (no HF download needed).

## Install (use a domestic mirror!)
Default PyPI was unusably slow here (a plain `pip install transformers ...` hung
>25 min with nothing downloaded). The Tsinghua mirror finished in seconds:
```bash
conda activate vlm
pip install --progress-bar off -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "transformers>=4.49.0" "accelerate>=0.34" "peft>=0.13" \
  "qwen-vl-utils>=0.0.8" "datasets>=2.20" "bitsandbytes>=0.43"
# light deps for data/eval/tests:
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml pillow pytest
```

## Required env for runs
`scripts/train.sh` and `scripts/eval.sh` source `scripts/_env.sh`, which **auto-uses
local weights**: if `./qwen` exists it sets `VERIFIER_MODEL_ID` + offline flags +
`PYTORCH_CUDA_ALLOC_CONF` for you. So `bash scripts/train.sh` will NOT download.

Only set these manually if you run the python entrypoints directly (not via the .sh),
or your weights live elsewhere:
```bash
export VERIFIER_MODEL_ID=/home/kewei/YING/test_vlm/qwen   # local weights
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1            # weights are local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoid fragmentation OOM
```

## Validated commands
```bash
# tests (CPU)
python -m pytest -q                       # 23 passed

# data (CPU)
bash scripts/build_dataset.sh             # synth demos -> data/processed/{train,val}.jsonl

# train all stages (GPU)
bash scripts/train.sh                     # stage_a -> stage_b -> stage_c

# smoke a single stage quickly: set max_steps in the config, then
python -m verifier.train.train_sft --config configs/train.yaml --stage a

# evaluate (GPU)
python -m verifier.eval.evaluate --config configs/train.yaml \
  --adapter runs/stage_c --val data/processed/val.jsonl --out runs/report.json
```

## Memory notes (the 48 GB 4090)
- **Validated peak:** batch_size 1 + gradient checkpointing + 6 frames/window +
  `max_pixels=128*28*28` → **~16 GB** train peak; streaming inference **~3.9 GB**,
  **0.76–1.25 s** per check.
- `batch_size: 2` with 16-frame windows **OOM'd** (full-logits fp32 loss balloons
  to ~45 GB). Defaults are now `batch_size 1 / grad_accum 16`.
- With 48 GB you have headroom to raise `max_pixels` back to `256*28*28` and/or the
  window; scale up and watch `nvidia-smi`. On a true 24 GB 4090, keep pixels/window low.
- Training disables kv-cache + enables checkpointing automatically (see
  `verifier/train/train_sft.py`).

## Known deprecation
- transformers warns `warmup_ratio` will change; harmless on 5.9. Switch to
  `warmup_steps` if it breaks on a future upgrade.
