# test_vlm — Streaming VLM Task-State Verifier

A data-processing → training → evaluation codebase for a **real-time task-state
verifier**: a streaming VLM that watches a robot execute a subtask and answers, at
each check,

> Has *\<subtask\>* been completed? Any anomaly?
> → `{"status": "ongoing|done|failed", "anomaly": "...", "reason": "..."}`

It is the closed-loop monitor between a perception stack and an LLM planner: it
tells the planner when to **advance** to the next subtask or **replan** on failure.
Target deployment: a single **RTX 4090 (24 GB)**, < 1 s per check, medium-horizon
(2–15 min) tasks.

This implements **Part B** of the design notes (`streaming-vlm-verifier.md`):
backbone **Qwen2.5-VL-3B** + LoRA, **ReKV**-style bounded memory, the
**VideoLLM-online** per-frame "when to fire" objective, and **FailGen/FailCoT**-style
failure synthesis so you don't hand-annotate per scene.

## Why this design
- **Scalable labels.** You annotate only *successful* demos (subtask + one
  completion frame). Failures are synthesized by procedurally perturbing them
  (early-stop, slip/drop, wrong-object, occlusion, pose-error) and by
  semantic-mismatch relabeling (I-FailSense). No per-scene predicate authoring.
- **Timing-aware.** Status latches `ongoing → done/failed` at a transition frame;
  training up-weights those transitions and eval reports asymmetric early/late
  error (premature `done` is the costly closed-loop mistake).
- **4090-friendly.** 4-bit weights + LoRA, capped tokens/frame, sliding-window KV,
  constrained decoding to a tiny JSON.

## Repo layout
```
verifier/
  schema.py                 # output JSON schema, anomaly taxonomy, parse/validate
  data/
    types.py                # Demo / Sample dataclasses + jsonl IO
    video_io.py             # extract frames from videos (decord/opencv)
    perturb.py              # FailGen-style failure synthesis + semantic-mismatch
    streaming_format.py     # demos -> windowed, streaming-labeled samples
    balance.py              # subsample the 'ongoing' plateau
    build_dataset.py        # CLI: demos.jsonl -> train/val shards
  model/
    loader.py               # Qwen2.5-VL-3B + 4bit + LoRA
    collator.py             # chat formatting + answer-only loss masking
  train/train_sft.py        # SFT stages A/B/C
  infer/streaming_verifier.py  # sliding-window streaming inference (+ ReKV hook)
  eval/metrics.py           # P/R/F1, false-done rate, anomaly acc, timing/ARS
  eval/evaluate.py          # CLI: run model on val -> report.json
configs/                    # data.yaml, train.yaml
scripts/                    # make_synthetic_demos.py, build/train/eval .sh
tests/                      # pytest (schema, perturb, streaming_format, metrics)
```

## Install
```bash
pip install -r requirements.txt          # full (GPU) on the training server
# CPU-only smoke test needs just: pyyaml pillow pytest
```

## Quickstart (no real data — proves the pipeline end-to-end)
```bash
# 1) synth demos + build train/val shards  (CPU)
bash scripts/build_dataset.sh
cat data/processed/stats.json

# 2) train the 3 SFT stages  (GPU / 4090)
bash scripts/train.sh

# 3) evaluate  (GPU)
bash scripts/eval.sh && cat runs/report.json
```

## Ingesting long, multi-subtask videos
If your recordings are **full tasks containing several subtasks** (not pre-cut
single-subtask clips), use the ingestion toolchain to segment them first:

```
manifest.jsonl ──ingest_videos──▶ frames/ + segments.jsonl (skeleton)
segments.jsonl ──(annotate GUI)─▶ segments.jsonl (filled)
segments.jsonl ──segments_to_demos──▶ demos.jsonl ──build_dataset──▶ shards
```

1. **Manifest** (`examples/manifest.example.jsonl`): one JSON/line — `video_id`,
   `video` path, ordered `subtasks` list.
2. **Ingest** — extract frames + emit an annotation skeleton:
   ```bash
   python -m verifier.data.ingest_videos --manifest data/manifest.jsonl \
       --frames-root data/frames --out data/segments.jsonl --fps 2.0
   ```
3. **Annotate** boundaries + transition frames in the GUI (scrub, set
   START/END/TRANSITION, pick outcome + anomaly, save):
   ```bash
   pip install streamlit
   streamlit run scripts/annotate_app.py -- --segments data/segments.jsonl
   ```
   Per subtask you set: `start`/`end` (span) and one `transition` frame —
   **completion** for a success, **deviation** (+ anomaly) for a real failure.
4. **Convert** to demos (validates; video-id prefix keeps each video on one side
   of the train/val split):
   ```bash
   python -m verifier.data.segments_to_demos --segments data/segments.jsonl \
       --out data/demos.jsonl --skip-invalid
   ```
5. Then **`build_dataset`** as below (perturbation + semantic-mismatch augment
   your real successes/failures into the final shards).

## Using your own data (pre-cut single-subtask clips)
Produce `data/demos.jsonl`, one JSON per line (see `verifier/data/types.py`):
```json
{"demo_id": "ep0007", "subtask": "place the cup on the shelf",
 "frames": ["/abs/ep0007/000000.jpg", "..."],
 "outcome": "success", "completion_frame": 41, "fps": 2.0, "source": "teleop"}
```
- Extract `frames` from a recording with `verifier/data/video_io.extract_frames`.
- For **successes**: set `completion_frame` (the frame it becomes done).
- For **real failures**: set `outcome="failure"`, `deviation_frame`, and `anomaly`.
- Synthetic failures + semantic-mismatch negatives are added automatically by
  `build_dataset` — successes are enough to bootstrap.

Then `bash scripts/build_dataset.sh` (it skips synth if `demos.jsonl` exists).

## Training stages (Part B.3)
| Stage | Goal | Key knob |
|---|---|---|
| A | lock the JSON output format | small, `lr=2e-4` |
| B | discriminative skill (done/ongoing/failed + anomaly) | `lr=1e-4` |
| C | **timing**: when to flip ongoing→done/failed | `timing_upweight=4` |
| D (optional) | DPO / streaming-RLVR on timing + anti-hallucination | see notes |

Stage C up-samples transition frames. The principled per-frame "keep-`ongoing`
token" streaming objective (VideoLLM-online LIVE) is documented in
`streaming-vlm-verifier.md` §B.3–B.4; the hook is `--timing-upweight` /
`stages.c` in `configs/train.yaml`.

## Evaluation (Part B.6)
`runs/report.json` contains:
- `done` / `failed` precision-recall-F1, and **false-done / false-failed rates**,
- `anomaly` per-type accuracy among true failures,
- `timing`: mean abs error (s), early/late error, and ARS (asymmetric readiness),
- `latency_s`: p50/p99/mean per check.

## Live demo: gift-packaging verifier (`scripts/gift_camera.sh`)
A separate live-demo path that runs the fine-tuned **gift_v1** adapter
(Qwen3.5-VL-9B base + LoRA from the `vlmtest_training/` gift_ft repo) on a camera or
video file. At each check it answers the closed-set gift-packaging state question and
overlays the parsed `{"state","name"}` (12 classes; class 12 = "rubbish"/reject). It
is *not* part of the Qwen2.5-VL verifier training pipeline above — it shares only the
camera/overlay plumbing from `scripts/qwen3_5_camera.py`.

Requires the `vlm` conda env, a GPU, and local weights at `model/qwen3_5_9B` (base)
and `model/gift_v1` (adapter); override with the `BASE` / `ADAPTER` env vars.

```bash
conda activate vlm
bash scripts/gift_camera.sh                              # auto camera, auto-check every 1.5s
bash scripts/gift_camera.sh --source 0 --interval 0      # webcam 0, manual (SPACE) only
bash scripts/gift_camera.sh --source clip.mp4 --loop     # replay a recorded clip
# offline: score a recording against its frame-level GT (no GUI), writes confusion
# matrix + GT-vs-pred overlay + per-class P/R/F1 + summary.json under results/eval_<tag>/
bash scripts/gift_camera.sh --source data/test_0605/realsense_20260609_150051.mp4 --eval
```
Controls (focus the video window): `q` quit · `SPACE` ask now · `p` pause/resume
auto-asking. The frame window (`--n-frames` / `--window-seconds` / `--sample-fps`)
defaults to the gift_ft training shape (6 frames over 3 s, ~2 fps) — keep these in
sync with the training config or live output gets unstable.

## `vlmtest_training/` — gift_ft LoRA fine-tuning repo (submodule)
A self-contained sibling repo (git submodule, `Yingminj/vlmtest_training`) that LoRA
fine-tunes **Qwen3.5-VL-9B** into the real-time **gift-packaging state verifier** whose
`gift_v1` adapter the live demo above runs. Given the most recent video frames it emits
`{"state": <id>, "name": "<name>"}`, one of **12 classes** (C1–C11 task phases +
C12 `rubbish` = irrelevant/reject frame). It is **independent** of the Qwen2.5-VL
verifier pipeline in this repo — it does not reuse `verifier/` and changes nothing in the
Qwen architecture (LoRA only). Its own `README.md` / `CLAUDE.md` are the source of truth.

```
videos+labels ── build_data ──▶ train/val.jsonl ── train ──▶ LoRA adapter ──▶ infer / eval
```

Design highlights (full rationale in `vlmtest_training/README.md`):
- **LoRA on the LLM only, vision tower frozen** (`q/k/v/o` + MLP, `r=16, α=32`); open the
  vision→LLM merger only if accuracy plateaus.
- **Tiny closed JSON schema + answer-only loss**, greedy decode + regex parse —
  `ontology.py` is the single source of truth for the 12 classes, prompt, and parser.
- **Label engineering**: drop the unlabeled gap (`0`), keep an explicit reject class (`12`),
  median-smooth per-frame labels, oversample short transition phases (3/7/9), and
  downsample static plateaus (2/8/11) while keeping the **real distribution in val**.
- **Causal, leak-free sliding window** (`n_frames=6` over `window_seconds=3.0`, ≈2 fps),
  split **by video**; the identical sampler is used by build, train, infer, and eval.

Pulled in as a submodule — init it before use:
```bash
git submodule update --init vlmtest_training
cd vlmtest_training && bash setup_env.sh   # conda env "gift_ft"; or CONDA_ENV=vlm bash setup_env.sh
PYTHONPATH=src python -m pytest -q          # CPU sanity (no GPU/model needed)
bash scripts/build_data.sh                  # videos+labels -> data/processed/{train,val}.jsonl
bash scripts/train.sh                       # LoRA SFT -> runs/gift_v1
ADAPTER=runs/gift_v1 bash scripts/eval.sh   # per-class P/R/F1 -> runs/report.json
```

## Deployment notes
- Wrap the model in **ReKV** (`verifier/infer/streaming_verifier.py::ReKVMemory`)
  for bounded long-horizon memory; reset memory at subtask boundaries via
  `set_subtask`.
- Enable constrained decoding (`use_constrained_decoding=True`, needs
  `lm-format-enforcer`) so output is always schema-valid and fast.
- Latch terminal states; run checks periodically/event-triggered, not every frame.

## Status / honesty
- **Fully runnable on CPU + unit-tested:** schema, data synthesis, streaming
  labeling, balancing, metrics, synthetic-demo generation.
- **Runs given GPU + model weights:** training (Stages A–C) and streaming eval/
  inference. These follow the HF Transformers Qwen2.5-VL + PEFT APIs; verify the
  installed `transformers` version exposes `Qwen2_5_VLForConditionalGeneration`.
- **Stubs/hooks (clearly marked):** ReKV memory backend, regex-constrained
  decoding integration, and Stage-D preference/RL.

See `streaming-vlm-verifier.md` and `task-state-monitor-design.md` (in the paper
archive) for the full rationale and the streaming-VLM literature map.
