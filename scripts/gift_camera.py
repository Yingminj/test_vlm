"""Real-time gift-packaging state verifier on a live camera / video file.

Drives the fine-tuned **gift_v1** adapter: Qwen3.5-VL-9B base (`qwen3_5_9B/`) + the
LoRA at `model/gift_v1/` trained by the `test_training/` (gift_ft) repo. At each check
it answers the closed-set state question and overlays the parsed `{"state","name"}`.

How it differs from `scripts/qwen3_5_camera.py` (the *base*-model probe):
  * Loads base weights then attaches the PEFT adapter (no merge needed for inference).
  * Uses the gift_ft prompt/ontology + regex parser (12 states, 12="rubbish" reject).
  * Window is sampled the way training did: `n_frames` evenly spaced over the most
    recent `window_seconds` (default 6 frames / 3.0 s, i.e. ~2 fps) -- match these to
    the training config or live output gets unstable.

Camera/video plumbing (RealSense / webcam / video-file replay, threaded inference,
results logging, overlay) is reused from `qwen3_5_camera.py`.

Controls (focus the video window):
  q = quit   SPACE = ask now   p = pause/resume auto-asking

Offline evaluation (`--eval`): instead of the live window, step through a whole video
deterministically, then score the per-check state ids against the same-named
frame-level GT label file (`<video>_lable.txt` / `_label.txt`). Each check's state is
latched forward onto the GT frame grid (time-interval -> frame-level), and we write a
GT-vs-prediction overlay, a confusion matrix, per-class P/R/F1, frame accuracy, and a
summary.json -- the artifact set from inference_show.py, but using the model's decoded
state id directly (no text-similarity matching needed).

Run:
  bash scripts/gift_camera.sh                       # auto camera
  python scripts/gift_camera.py --source path/to.mp4 --loop
  # evaluate a recording against its GT labels:
  python scripts/gift_camera.py --source data/test_0605/realsense_20260609_150051.mp4 --eval
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import textwrap
import threading
import time
from typing import Deque, List

# Reuse camera backends + overlay/result helpers from the base-model live script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import qwen3_5_camera as live  # noqa: E402  (open_camera, ResultWriter, _bar, _text, ...)

# Pull in the gift_ft ontology (prompt + parser + class names) from test_training/.
sys.path.insert(0, os.path.join(_HERE, "..", "test_training", "src"))
from gift_ft.ontology import CLASSES, build_prompt, load_ontology, parse_answer  # noqa: E402


# ===================== offline evaluation (vs frame-level GT) =====================
# The live loop emits a state id at *time intervals* (one per check), while the GT is
# *per-frame*. To score, we latch each check's state forward (streaming semantics) onto
# the GT frame grid, then build the GT-vs-pred overlay + confusion matrix + metrics --
# the artifact set from inference_show.py, but using the model's direct state id (no
# text-similarity matching needed since gift_ft already decodes a class id).

def _find_label_file(source: str):
    """Locate the same-named GT label file next to the video (handles the `_lable`
    misspelling actually used on disk as well as the correct `_label`)."""
    base, _ = os.path.splitext(source)
    for suffix in ("_lable.txt", "_label.txt"):
        cand = base + suffix
        if os.path.isfile(cand):
            return cand
    return None


def _load_gt_labels(path: str) -> dict:
    """`<frame_idx> <class_id>` per line -> {frame_idx: class_id}."""
    gt: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) >= 2:
                gt[int(parts[0])] = int(parts[1])
    if not gt:
        raise ValueError(f"no labels parsed from {path}")
    return gt


def _build_frame_rows(pred_points, gt_frames: dict, classes: dict):
    """Latch checks onto the GT frame grid.

    `pred_points` is a list of `(gt_frame_idx, state_id)` (the newest frame of each
    check). For each GT frame we carry forward the most recent check at or before it;
    frames before the first check are -1 (warmup / unpredicted), counted as misses --
    same convention as inference_show.build_frame_table.
    """
    pts = sorted(pred_points)
    rows = []
    j, cur = 0, -1
    for fi in sorted(gt_frames):
        while j < len(pts) and pts[j][0] <= fi:
            cur = pts[j][1]
            j += 1
        gt_id = gt_frames[fi]
        rows.append({
            "frame_idx": fi,
            "gt_class_id": gt_id,
            "gt_class_text": classes.get(gt_id, "UNKNOWN"),
            "pred_class_id": cur,
            "pred_class_text": classes.get(cur, "UNPREDICTED"),
            "match": int(cur == gt_id),
        })
    return rows


def _confusion_matrix(frame_rows, class_ids):
    idx = {c: i for i, c in enumerate(class_ids)}
    m = [[0] * len(class_ids) for _ in class_ids]
    for r in frame_rows:
        g, p = r["gt_class_id"], r["pred_class_id"]
        if g in idx and p in idx:
            m[idx[g]][idx[p]] += 1
    return m


def _per_class_metrics(frame_rows, class_ids):
    tp = {c: 0 for c in class_ids}
    fp = {c: 0 for c in class_ids}
    fn = {c: 0 for c in class_ids}
    for r in frame_rows:
        g, p = r["gt_class_id"], r["pred_class_id"]
        if g == p:
            if g in tp:
                tp[g] += 1
        else:
            if p in fp:
                fp[p] += 1
            if g in fn:
                fn[g] += 1
    per = {}
    for c in class_ids:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[c] + fn[c]}
    macro_f1 = sum(v["f1"] for v in per.values()) / max(1, len(per))
    return per, macro_f1


def _save_csv(path, rows, fieldnames):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plot_overlay(frame_rows, classes, class_ids, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fidx = [r["frame_idx"] for r in frame_rows]
    gt = [r["gt_class_id"] for r in frame_rows]
    pred = [r["pred_class_id"] for r in frame_rows]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=100)
    ax.step(fidx, gt, where="post", linewidth=2.2, color="limegreen", alpha=0.7,
            label="Ground Truth", zorder=1)
    ax.step(fidx, pred, where="post", linewidth=1.5, color="#E2510D", alpha=0.85,
            label="Prediction", zorder=3)

    ax.set_yticks(class_ids)
    ax.set_yticklabels([f"C{c}: {classes.get(c, '?')}" for c in class_ids], fontsize=9)
    ax.set_xlabel("Frame index", fontsize=11, fontweight="bold")
    ax.set_ylabel("State", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    if fidx:
        ax.set_xlim(0, max(fidx) * 1.02)

    total = len(frame_rows)
    matched = sum(r["match"] for r in frame_rows)
    acc = matched / max(1, total)
    info = f"Accuracy: {acc:.4f}\nFrames: {total}\nMatched: {matched}"
    ax.text(0.995, 0.97, info, transform=ax.transAxes, fontsize=10,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9))
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(matrix, class_ids, accuracy, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalise for colour (so rare classes are still readable); annotate with counts.
    fig, ax = plt.subplots(figsize=(11, 9))
    norm = []
    for row in matrix:
        s = sum(row)
        norm.append([(v / s if s else 0.0) for v in row])
    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    labels = [f"C{c}" for c in class_ids]
    ax.set_xticks(range(len(class_ids)))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion Matrix (row-normalised)", pad=16)
    fig.suptitle(f"Frame Accuracy: {accuracy:.4f}", fontsize=14, y=0.98)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if norm[i][j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_eval(args, infer) -> None:
    """Step through the whole video, run the same windowed inference deterministically
    (no real-time pacing / frame-dropping), then score against the frame-level GT."""
    import collections
    import json
    from datetime import datetime

    import cv2
    from PIL import Image

    src = args.source
    if not os.path.isfile(src):
        raise SystemExit(f"--eval needs --source to be a video file (got {src!r})")

    labels = args.labels
    if labels in ("", "auto", None):
        labels = _find_label_file(src)
    if not labels or not os.path.isfile(labels):
        raise SystemExit(f"GT label file not found for {src!r}; pass --labels <file>")
    gt_frames = _load_gt_labels(labels)

    classes = (load_ontology(args.class_map)
               if args.class_map and os.path.isfile(args.class_map) else dict(CLASSES))

    out_dir = args.eval_out or os.path.join(
        args.results_dir or "results", f"eval_{live._source_tag(src)}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open video {src!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gt_fps = args.gt_fps if args.gt_fps and args.gt_fps > 0 else fps
    print(f"[eval] video={src}")
    print(f"[eval] fps={fps:.2f}  frames={n_total}  gt_fps={gt_fps:.2f}  "
          f"gt_labels={len(gt_frames)}  labels={labels}")

    buf_len = max(args.n_frames, int(round(args.window_seconds * args.sample_fps)) + 1)
    buf: collections.deque = collections.deque(maxlen=buf_len)   # (video_frame_idx, PIL)
    sample_stride = max(1, int(round(fps / max(args.sample_fps, 1e-6))))
    check_stride = (max(1, int(round(args.interval * fps)))
                    if args.interval > 0 else sample_stride)

    def to_gt_idx(vfi: int) -> int:
        return int(round((vfi / fps) * gt_fps))

    pred_points = []        # (gt_frame_idx, state_id)
    vfi, last_check = -1, -(10 ** 9)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        vfi += 1
        if vfi % sample_stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if max(pil.size) > args.max_side:
                s = args.max_side / max(pil.size)
                pil = pil.resize((int(pil.width * s), int(pil.height * s)))
            buf.append((vfi, pil))
        if len(buf) >= 2 and (vfi - last_check) >= check_stride:
            last_check = vfi
            items = list(buf)
            n = min(args.n_frames, len(items))
            if n <= 1:
                picks = items[-1:]
            else:
                idx = [int(round(i * (len(items) - 1) / (n - 1))) for i in range(n)]
                picks = [items[i] for i in idx]
            if len(picks) % 2:                      # temporal_patch_size=2 wants even
                picks = picks[1:]
            if len(picks) < 2:
                continue
            frames = [p[1] for p in picks]
            try:
                sid, ans = infer(frames)
            except Exception as e:                  # keep going, log the gap
                sid, ans = None, f"[infer error] {e}"
            newest = to_gt_idx(picks[-1][0])
            print(f"[eval] frame {picks[-1][0]:5d} ({picks[-1][0] / fps:6.2f}s) -> {ans}")
            if sid is not None:
                pred_points.append((newest, sid))
    cap.release()

    if not pred_points:
        raise SystemExit("[eval] no predictions produced; nothing to score")

    frame_rows = _build_frame_rows(pred_points, gt_frames, classes)
    class_ids = sorted(set(gt_frames.values()) | {p for _, p in pred_points})
    matrix = _confusion_matrix(frame_rows, class_ids)
    per_class, macro_f1 = _per_class_metrics(frame_rows, class_ids)

    total = len(frame_rows)
    correct = sum(r["match"] for r in frame_rows)
    accuracy = correct / max(1, total)
    covered = sum(1 for r in frame_rows if r["pred_class_id"] != -1)
    coverage = covered / max(1, total)

    # --- artifacts ---
    frame_csv = os.path.join(out_dir, "frame_predictions.csv")
    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    overlay_png = os.path.join(out_dir, "gt_vs_pred.png")
    cm_png = os.path.join(out_dir, "confusion_matrix.png")
    summary_json = os.path.join(out_dir, "summary.json")

    _save_csv(frame_csv, frame_rows,
              ["frame_idx", "gt_class_id", "gt_class_text",
               "pred_class_id", "pred_class_text", "match"])
    import csv as _csv
    with open(cm_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["gt\\pred"] + [f"C{c}" for c in class_ids])
        for c, row in zip(class_ids, matrix):
            w.writerow([f"C{c}:{classes.get(c, '?')}"] + row)
    _plot_overlay(frame_rows, classes, class_ids, overlay_png,
                  f"GT vs Prediction by Frame — {os.path.basename(src)}")
    _plot_confusion_matrix(matrix, class_ids, accuracy, cm_png)

    summary = {
        "video": src,
        "labels": labels,
        "fps": fps,
        "gt_fps": gt_fps,
        "checks": len(pred_points),
        "frames_evaluated": total,
        "correct_frames": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "macro_f1": macro_f1,
        "per_class": {f"C{c}": per_class[c] for c in class_ids},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {
            "frame_predictions_csv": frame_csv,
            "confusion_matrix_csv": cm_csv,
            "gt_vs_pred_png": overlay_png,
            "confusion_matrix_png": cm_png,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- console report ---
    print("\n========== EVAL ==========")
    print(f"checks={len(pred_points)}  frames={total}  "
          f"accuracy={accuracy:.4f}  coverage={coverage:.4f}  macro_f1={macro_f1:.4f}")
    print(f"{'class':>5}  {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}   name")
    for c in class_ids:
        m = per_class[c]
        print(f"C{c:>4}  {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
              f"{m['support']:8d}   {classes.get(c, '?')}")
    print(f"\n[eval] artifacts -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="model/qwen3_5_9B", help="base Qwen3.5-VL weights dir")
    ap.add_argument("--adapter", default="model/gift_v1", help="LoRA adapter dir")
    ap.add_argument("--task", default="pack the toy into the gift box",
                    help="task description injected into the prompt")
    ap.add_argument("--source", default="auto",
                    help="'auto' | 'realsense' | webcam index '0' | path to a video file")
    ap.add_argument("--loop", action="store_true",
                    help="when --source is a video file, restart it at EOF")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="video-file playback speed multiplier (1.0 = real time)")
    # Window: match the training config (gift_ft default n_frames=6, window_seconds=3.0).
    ap.add_argument("--n-frames", type=int, default=6, help="frames sent to the model")
    ap.add_argument("--window-seconds", type=float, default=3.0,
                    help="real-time span the window covers (oldest..newest)")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="cadence frames enter the rolling buffer")
    ap.add_argument("--interval", type=float, default=1.5,
                    help="seconds between automatic checks; 0 = manual only (SPACE)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                    help="processor max_pixels per frame (latency/VRAM)")
    ap.add_argument("--max-side", type=int, default=448,
                    help="pre-downscale frames so longest side <= this")
    ap.add_argument("--results-dir", default="results",
                    help="save each check to results/<tag>_<time>.txt; '' disables")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    # --- offline evaluation vs frame-level GT (no GUI) ---
    ap.add_argument("--eval", action="store_true",
                    help="run over the whole --source video and score against the "
                         "same-named frame-level GT label file (no live window)")
    ap.add_argument("--labels", default="auto",
                    help="GT frame-label txt; 'auto' = <source>_lable.txt / _label.txt")
    ap.add_argument("--class-map", default="data/test_0605/gift.txt",
                    help="gift.txt class map for nicer names; '' = built-in ontology")
    ap.add_argument("--gt-fps", type=float, default=0.0,
                    help="fps the GT frame indices use; 0 = use the video fps")
    ap.add_argument("--eval-out", default="",
                    help="dir for eval artifacts; '' = <results-dir>/eval_<tag>")
    args = ap.parse_args()

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    prompt = build_prompt(CLASSES, task=args.task)

    # Processor lives in the adapter dir (chat template + tokenizer were saved there).
    print(f"[load] processor <- {args.adapter}")
    proc = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=False)

    print(f"[load] base {args.base} (bf16) + LoRA {args.adapter} ...")
    t0 = time.time()
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # --- shared state ---
    # Buffer holds frames at sample-fps covering window_seconds; we then pick n_frames
    # evenly spaced over it (newest last) -- the same shape training sampled.
    buf_len = max(args.n_frames, int(round(args.window_seconds * args.sample_fps)) + 1)
    lock = threading.Lock()
    buf: Deque = collections.deque(maxlen=buf_len)      # PIL RGB frames
    ts_buf: Deque = collections.deque(maxlen=buf_len)   # clip time (s) per frame
    state = {"answer": "(warming up)", "state_id": None, "latency": 0.0,
             "paused": args.interval <= 0, "stop": False, "busy": False}
    ask_now = threading.Event()

    writer = (live.ResultWriter(args.results_dir, f"gift_{live._source_tag(args.source)}")
              if args.results_dir and not args.eval else None)
    if writer:
        print(f"[results] saving to {writer.path}")

    def _pick(frames: List) -> List:
        """n_frames evenly spaced over the buffer, newest last (matches gift_ft.step)."""
        n = min(args.n_frames, len(frames))
        if n <= 1:
            return frames[-1:]
        idx = [int(round(i * (len(frames) - 1) / (n - 1))) for i in range(n)]
        return [frames[i] for i in idx]

    def infer(frames: List) -> tuple:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = proc(text=[text], videos=[frames], return_tensors="pt",
                      max_pixels=args.max_pixels).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        raw = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        parsed = parse_answer(raw, CLASSES)
        if parsed:
            return parsed["state"], f'{parsed["state"]}: {parsed["name"]}'
        return None, f"[parse-fail] {raw}"

    # Offline evaluation path: score the whole video against frame-level GT and exit
    # (no camera, no GUI, no threaded worker).
    if args.eval:
        _run_eval(args, infer)
        return

    def worker():
        last = 0.0
        while True:
            triggered = ask_now.wait(timeout=0.1)
            with lock:
                if state["stop"]:
                    return
                paused = state["paused"]
            now = time.time()
            due = (not paused) and (now - last) >= args.interval
            if not (triggered or due):
                continue
            ask_now.clear()
            with lock:
                snap = _pick(list(buf))
                snap_ts = list(ts_buf)
            if len(snap) < 2:
                continue
            if len(snap) % 2:           # temporal_patch_size=2 wants an even count
                snap = snap[1:]
            with lock:
                state["busy"] = True
            t = time.time()
            try:
                sid, ans = infer(snap)
            except Exception as e:
                sid, ans = None, f"[infer error] {e}"
            dt = time.time() - t
            last = time.time()
            with lock:
                state["answer"], state["state_id"] = ans, sid
                state["latency"], state["busy"] = dt, False
            if writer and snap_ts:
                writer.write(snap_ts[0], snap_ts[-1], ans, label="state")

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    cam = live.open_camera(args.source, args.cam_width, args.cam_height, args.cam_fps,
                           loop=args.loop, speed=args.speed)
    win = "Gift verifier  (q quit | SPACE ask | p pause)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sample_period = 1.0 / max(args.sample_fps, 1e-6)
    last_sample = 0.0
    t_start = time.time()
    try:
        while True:
            bgr = cam.read()
            if bgr is None:
                if getattr(cam, "eof", False):
                    print("[video] end of file")
                    break
                continue

            now = time.time()
            if now - last_sample >= sample_period:
                last_sample = now
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                if max(pil.size) > args.max_side:        # downscale for speed/VRAM
                    s = args.max_side / max(pil.size)
                    pil = pil.resize((int(pil.width * s), int(pil.height * s)))
                ts = cam.timestamp() if hasattr(cam, "timestamp") else (now - t_start)
                with lock:
                    buf.append(pil)
                    ts_buf.append(ts)

            with lock:
                answer, sid, lat = state["answer"], state["state_id"], state["latency"]
                paused, busy, nbuf = state["paused"], state["busy"], len(buf)

            disp = bgr.copy()
            W = disp.shape[1]
            stat = "ASKING…" if busy else ("PAUSED" if paused else f"auto {args.interval:.1f}s")
            # green for a parsed state, red for the reject class / parse failures
            ok = sid is not None and sid != 12
            col = (0, 255, 0) if ok else ((0, 165, 255) if sid == 12 else (0, 0, 255))
            live._bar(disp, 0, 0, W, 36)
            live._text(disp, f"{stat}  buf={nbuf}  last={lat*1000:.0f}ms",
                       (10, 25), color=(0, 215, 255) if busy else (0, 255, 0))
            wrapped = textwrap.wrap(f"STATE  {answer}", width=max(20, W // 11))[:3] or [""]
            bh = 30 * len(wrapped) + 16
            live._bar(disp, 0, disp.shape[0] - bh, W, bh)
            for i, ln in enumerate(wrapped):
                live._text(disp, ln, (10, disp.shape[0] - bh + 30 + i * 30),
                           color=col, scale=0.7, thick=2)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                ask_now.set()
            elif k == ord("p"):
                with lock:
                    state["paused"] = not state["paused"]
    finally:
        with lock:
            state["stop"] = True
        ask_now.set()
        cam.release()
        cv2.destroyAllWindows()
        th.join(timeout=2.0)
        if writer:
            writer.close()
            print(f"[results] saved to {writer.path}")


if __name__ == "__main__":
    main()
