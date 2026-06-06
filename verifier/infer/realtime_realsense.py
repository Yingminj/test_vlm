"""Real-time task-state verification from an Intel RealSense color stream.

Loads the fine-tuned verifier (base Qwen2.5-VL-3B + a LoRA adapter, e.g.
`runs/gift/stage_c`) and runs streaming checks on live camera frames, showing the
predicted status / anomaly / reason in an OpenCV overlay.

Design (mirrors infer/streaming_verifier.py and eval/evaluate.py):
  * The camera runs at its native fps for a smooth preview; frames are *sampled*
    into the verifier window at `--sample-fps` (match training, default 2.0).
  * Heavy `model.generate` runs in a background thread so the preview never
    freezes. The capture/display thread only ever holds the lock to copy a tiny
    snapshot of the frame window.
  * Terminal states (done/failed) latch per subtask, exactly like deployment.
    Press 'r' to re-arm the same subtask, 'n' to type a new subtask, 'q' to quit.

Run (GPU + RealSense camera required):
  python -m verifier.infer.realtime_realsense \
      --config configs/train.yaml --adapter runs/gift/stage_c \
      --subtask "place the cup on the shelf"

Prefer launching via scripts/realtime.sh so local ./qwen weights are used offline.
"""
from __future__ import annotations

import argparse
import collections
import threading
import time
from typing import Deque, List, Optional

import numpy as np


def _put_text(img, lines, org=(10, 28), color=(255, 255, 255), scale=0.7, thick=2):
    import cv2
    y = org[1]
    for ln in lines:
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 3, cv2.LINE_AA)  # outline for readability
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)
        y += int(34 * scale / 0.7)
    return img


_STATUS_BGR = {           # OpenCV is BGR
    "ongoing": (0, 215, 255),   # amber
    "done":    (0, 200, 0),     # green
    "failed":  (0, 0, 230),     # red
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--adapter", default="runs/gift/stage_c",
                    help="LoRA adapter dir; '' or 'none' = base model only")
    ap.add_argument("--subtask", default=None,
                    help="standing query; if omitted you'll be prompted")
    ap.add_argument("--window", type=int, default=32, help="frames kept in context")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="cadence frames are fed into the model (match training)")
    ap.add_argument("--check-every", type=float, default=1.0,
                    help="seconds between verifier checks")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--constrained", action="store_true",
                    help="grammar-constrain output to the JSON schema (needs lm-format-enforcer)")
    ap.add_argument("--no-latch", action="store_true",
                    help="keep re-checking after a terminal state instead of latching")
    # camera
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    args = ap.parse_args()

    subtask = args.subtask or input("Subtask to monitor: ").strip()
    if not subtask:
        raise SystemExit("A subtask (the standing query) is required.")

    import cv2
    import yaml
    import pyrealsense2 as rs
    from PIL import Image

    from ..model.loader import ModelConfig, load_for_inference
    from ..infer.streaming_verifier import StreamingVerifier

    # ---- model -------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mcfg = ModelConfig(**raw.get("model", {}))
    adapter = None if args.adapter in ("", "none", "None") else args.adapter
    print(f"[load] base={mcfg.model_id}  adapter={adapter} ...")
    model, processor = load_for_inference(mcfg, adapter_path=adapter)
    verifier = StreamingVerifier(
        model=model, processor=processor, window=args.window,
        max_new_tokens=args.max_new_tokens,
        use_constrained_decoding=args.constrained,
    )
    verifier.set_subtask(subtask)
    print(f"[ready] monitoring: {subtask!r}")

    # ---- shared state ------------------------------------------------------
    lock = threading.Lock()
    frames: Deque = collections.deque(maxlen=args.window)  # sampled PIL frames
    state = {"label": None, "latency": 0.0, "subtask": subtask,
             "reset": False, "stop": False, "latched": False}

    # ---- inference worker --------------------------------------------------
    def worker():
        while True:
            with lock:
                if state["stop"]:
                    return
                if state["reset"]:
                    verifier.set_subtask(state["subtask"])
                    state["reset"] = False
                    state["latched"] = False
                    state["label"] = None
                latched = state["latched"]
                snapshot: List = list(frames)
            if latched or not snapshot:
                time.sleep(0.05)
                continue
            verifier._frames = collections.deque(snapshot, maxlen=args.window)
            t0 = time.time()
            try:
                label = verifier._infer()
            except Exception as e:  # keep the loop alive; surface in overlay
                label = type("E", (), {"status": "ongoing", "anomaly": "none",
                                       "reason": f"infer error: {e}"})()
            dt = time.time() - t0
            with lock:
                state["label"] = label
                state["latency"] = dt
                if not args.no_latch and label.status in ("done", "failed"):
                    state["latched"] = True
            time.sleep(max(0.0, args.check_every - dt))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # ---- camera + display (main thread) ------------------------------------
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.cam_width, args.cam_height,
                         rs.format.bgr8, args.cam_fps)
    pipeline.start(rs_cfg)
    win = "verifier (q quit | r re-arm | n new subtask)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sample_period = 1.0 / max(args.sample_fps, 1e-6)
    last_sample = 0.0
    try:
        while True:
            fs = pipeline.wait_for_frames()
            cframe = fs.get_color_frame()
            if not cframe:
                continue
            bgr = np.asanyarray(cframe.get_data())

            now = time.time()
            if now - last_sample >= sample_period:
                last_sample = now
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                with lock:
                    frames.append(pil)

            with lock:
                label = state["label"]
                lat = state["latency"]
                latched = state["latched"]
                cur_sub = state["subtask"]
                nframes = len(frames)

            disp = bgr.copy()
            status = label.status if label else "…"
            color = _STATUS_BGR.get(status, (255, 255, 255))
            h = disp.shape[0]
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 4), color, -1)
            lines = [f"subtask: {cur_sub}",
                     f"STATUS: {status.upper()}{'  [LATCHED]' if latched else ''}"]
            if label is not None:
                lines.append(f"anomaly: {label.anomaly}")
                lines.append(f"reason: {label.reason}")
                lines.append(f"check: {lat*1000:.0f} ms   window: {nframes} frm")
            else:
                lines.append(f"warming up...  window: {nframes} frm")
            _put_text(disp, lines, color=color)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                with lock:
                    state["reset"] = True
                print("[reset] re-arming subtask")
            elif key == ord("n"):
                cv2.destroyWindow(win)
                new_sub = input("New subtask: ").strip()
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                if new_sub:
                    with lock:
                        state["subtask"] = new_sub
                        state["reset"] = True
                    print(f"[switch] now monitoring: {new_sub!r}")
    finally:
        with lock:
            state["stop"] = True
        pipeline.stop()
        cv2.destroyAllWindows()
        th.join(timeout=2.0)


if __name__ == "__main__":
    main()
