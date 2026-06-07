# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A data-processing → training → evaluation codebase for a **real-time task-state
verifier**: a streaming VLM that watches a robot execute a subtask and, at each
check, answers `{"status": "ongoing|done|failed", "anomaly": "...", "reason": "..."}`.
It is the closed-loop monitor between perception and an LLM planner (tells the
planner when to advance or replan). Target deployment: a single RTX 4090, <1 s per
check. See `README.md` for the full design rationale and `SERVER.md` for the
validated server runbook.

The core package is `verifier/` (Qwen2.5-VL-3B + LoRA). Two other model families
live in the repo for *live-demo / exploration only* — they are not part of the
verifier training pipeline (see "Three model families" below).

## Environment & weights

- Runs in the **`vlm` conda env** (Python 3.10, torch 2.8+cu129, transformers 5.9,
  peft, accelerate, bitsandbytes, qwen-vl-utils). `conda activate vlm` first.
- **CPU-only work** (data build, eval metrics, tests) needs only `pyyaml pillow pytest`.
- **Weights are local and gitignored.** `qwen/` (Qwen2.5-VL-3B verifier backbone),
  `qwen3_5_9B/`, and `marlin-2b/` are large weight dirs, not tracked.
- The `.sh` entrypoints source `scripts/_env.sh`, which auto-points
  `VERIFIER_MODEL_ID` at `./qwen` and sets `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`
  so **runs never hit the HF Hub**. Set `VERIFIER_MODEL_ID` yourself only when
  invoking the python `-m` entrypoints directly.
- On this server, PyPI is unusably slow — install via the Tsinghua mirror
  (`-i https://pypi.tuna.tsinghua.edu.cn/simple`). See `SERVER.md`.

## Common commands

```bash
# Tests (CPU, ~23 tests). conftest.py puts repo root on sys.path so `verifier` imports.
python -m pytest -q
python -m pytest tests/test_metrics.py -q          # single file
python -m pytest tests/test_perturb.py::<name> -q  # single test

# Data build (CPU): synth demos if none, then demos.jsonl -> processed/{train,val}.jsonl
bash scripts/build_dataset.sh
# or directly: python -m verifier.data.build_dataset --demos ... --out ...

# Train the 3 SFT stages A->B->C in sequence (GPU)
bash scripts/train.sh
# smoke one stage quickly: set `max_steps: >0` in configs/train.yaml, then
python -m verifier.train.train_sft --config configs/train.yaml --stage a

# Evaluate an adapter on the val shard -> runs/report.json (GPU)
bash scripts/eval.sh

# Live verification on an Intel RealSense stream (GPU + camera)
ADAPTER=runs/gift/stage_c SUBTASK="place the cup on the shelf" bash scripts/realtime.sh
```

## Architecture

The pipeline is a linear flow; everything downstream is derived from a small set of
human-annotated **Demos**.

**`verifier/schema.py`** is the contract for the whole repo. It defines the closed
label sets (`STATUSES`, `ANOMALIES`), the prompt template, a `SCHEMA_REGEX` for
constrained decoding, and `VerifierLabel` (validates on construction). Data
generation, the collator's targets, and inference parsing all go through it — change
a label set or key order here and the regex / collator / parser stay consistent.

**Data (`verifier/data/`)** — `types.py` defines `Demo` (one subtask attempt: frames
+ a single key timestamp — `completion_frame` for success, `deviation_frame` for
failure) and `Sample` (one windowed training example). The build pipeline
(`build_dataset.py`):
1. `perturb.py` synthesizes **failures from successes** (early-stop, slip/drop,
   wrong-object, etc.) and **semantic-mismatch negatives** — so you only ever
   hand-annotate *successful* demos.
2. `streaming_format.py` slices demos into windowed samples with streaming-correct,
   *latched* labels (`ongoing` until the transition frame, then `done`/`failed`).
   The newest frame's status is the target; the transition frame is flagged
   `is_transition`.
3. `balance.py` subsamples the `ongoing` plateau (train only; val keeps the real
   distribution).
4. Split is **by root demo id** so perturbed variants of one demo never straddle the
   train/val boundary (no leakage).

For long, multi-subtask recordings there is a separate ingestion front-end:
`ingest_videos.py` (frames + annotation skeleton) → `scripts/annotate_app.py`
(Streamlit GUI to mark boundaries/transitions) → `segments_to_demos.py` → the build
pipeline above. `video_io.py` extracts frames (decord, opencv fallback).

**Model (`verifier/model/`)** — `loader.py` builds Qwen2.5-VL-3B with 4-bit (nf4) +
LoRA on the LLM projections, **vision tower frozen**, `max_pixels` capped for
latency/VRAM. `collator.py` formats the chat (frames + prompt as the user turn,
target JSON as the assistant turn) and applies **answer-only loss masking** (only the
assistant span contributes loss).

**Training (`verifier/train/train_sft.py`)** — one script, three stages sharing
`train:` defaults and overriding via `stages.{a,b,c}` in `configs/train.yaml`:
A locks the JSON format, B is the discriminative verifier, C is timing (when to flip
`ongoing→done/failed`) and uses `timing_upweight` to oversample transition frames.
Training force-disables kv-cache and enables gradient checkpointing for VRAM.

**Inference & eval** — `infer/streaming_verifier.py` keeps a sliding frame window,
latches terminal states, resets on `set_subtask()`, and has hooks for
constrained decoding (`lm-format-enforcer`) and ReKV bounded memory (`ReKVMemory` is
an intentional stub). `infer/realtime_realsense.py` is the live-camera app (generate
runs in a background thread so the preview never freezes). `eval/metrics.py` reports
done/failed P-R-F1, **false-done/false-failed rates**, anomaly accuracy, and timing
error (early/late asymmetry); `eval/evaluate.py` writes `runs/report.json`.

## Important gotchas

- **Config vs. script path mismatch.** `configs/train.yaml` points at `data/gift/...`
  and `runs/gift/...`, but `scripts/build_dataset.sh` writes to `data/processed/` and
  `scripts/eval.sh` defaults `ADAPTER=runs/stage_c` / `VAL=data/processed/val.jsonl`.
  `scripts/realtime.sh` defaults to `runs/gift/stage_c`. When wiring an end-to-end
  run, make the data-out, train-in, and eval/adapter paths agree (override via the
  `DATA_DIR`/`CONFIG`/`ADAPTER`/`VAL` env vars the scripts expose).
- **Stubs are marked, not bugs.** `ReKVMemory` raises `NotImplementedError`;
  constrained decoding and Stage-D (DPO/RLVR) are hooks. Don't "fix" them blindly.
- **Heavy imports are lazy.** `torch`/`transformers` are imported inside functions so
  the data/eval/test paths run on a CPU box without GPU deps. Keep new code in those
  paths import-light.
- **VRAM is the binding constraint.** Defaults (`batch_size 1`, `grad_accum 16`,
  6-frame windows, `max_pixels=128*28*28`) were tuned so a 48 GB 4090 peaks ~16 GB;
  `batch_size 2` with 16-frame windows OOM'd (full-logits fp32 loss). See `SERVER.md`.

## Three model families (don't conflate)

- **`qwen/` — Qwen2.5-VL-3B**: the verifier backbone. This is what `verifier/` trains
  (LoRA) and evaluates. The whole pipeline above is about this model.
- **`qwen3_5_9B/` — Qwen3.5-VL-9B base** (`Qwen3_5ForConditionalGeneration`, ~19 GB
  bf16): used only for the live demo `scripts/qwen3_5_camera.py`. Not fine-tuned here.
  Pass PIL frames via `processor(videos=[frames])` directly — `process_vision_info`
  rejects the new processor. Use `enable_thinking=False` for fast answers.
- **`marlin-2b/` — Marlin-2B** video VLM (dense caption + temporal grounding), demo
  `scripts/marlin_camera.py`. Load it as the **native `Qwen3_5ForConditionalGeneration`
  class**, not via `trust_remote_code` (the remote `modeling_marlin.py` statically
  requires torchcodec, which the `vlm` env lacks). Import its prompt/parser helpers
  via importlib.

See the auto-memory files (`qwen3_5-deployment`, `marlin-2b-deployment`,
`realtime-verifier-inference`) for the validated load incantations.
