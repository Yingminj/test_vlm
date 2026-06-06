"""Real-time video understanding with Marlin-2B (dense caption + temporal grounding).

Marlin (`MarlinForConditionalGeneration`) is a Qwen3.5-2B fine-tune for video. Its
shipped `.caption()`/`.find()` helpers decode a video *file* via torchcodec
(torch>=2.11). For a live camera we instead feed buffered frames straight to the
processor with Marlin's *canonical* prompts and reuse its output parsers -- this
needs no torchcodec and runs on the installed torch 2.8 stack.

How it works:
  * Weights load as the native `Qwen3_5ForConditionalGeneration` (Marlin adds no
    parameters, only helper methods), so we avoid trust_remote_code's torchcodec
    dependency. Prompts + parsers are imported from the checkpoint's
    `modeling_marlin.py` (its torchcodec import is function-local, so importing the
    module is safe).
  * A rolling buffer of recent frames (sampled at --sample-fps, matching Marlin's
    training FPS=2) is captioned/grounded every --interval seconds in a background
    thread so the preview stays smooth.

Two modes:
  caption : "Scene: ... / Events: <s-e> ..." dense description of the recent clip.
  find    : given --event "<query>", returns the (start,end) span where it occurs.

NOTE on timestamps: feeding raw frames (no file metadata) means event times are
relative to the recent clip, which spans ~ buffer/sample_fps seconds of wall-clock.
Treat them as "seconds into the clip", not absolute.

Controls (focus the window):
  q quit | SPACE ask now | p pause/resume | m toggle caption/find | e set find query

Run:  bash scripts/marlin_camera.sh                       # live captioning
      bash scripts/marlin_camera.sh --mode find --event "a hand picks up an object"
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import textwrap
import threading
import time
from typing import Deque, List

import numpy as np


# ------------------------- camera backends -------------------------
class RealSenseCam:
    def __init__(self, w, h, fps):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self):
        c = self.pipeline.wait_for_frames().get_color_frame()
        return np.asanyarray(c.get_data()) if c else None

    def release(self):
        self.pipeline.stop()


class CV2Cam:
    def __init__(self, index, w, h, fps):
        import cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open webcam index {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


def open_camera(source, w, h, fps):
    if source in ("auto", "realsense"):
        try:
            import pyrealsense2 as rs
            if len(rs.context().query_devices()) > 0:
                print("[cam] using RealSense")
                return RealSenseCam(w, h, fps)
            if source == "realsense":
                raise RuntimeError("no RealSense device found")
        except Exception as e:
            if source == "realsense":
                raise
            print(f"[cam] RealSense unavailable ({e}); falling back to webcam 0")
    idx = 0 if source == "auto" else int(source)
    print(f"[cam] using OpenCV webcam index {idx}")
    return CV2Cam(idx, w, h, fps)


# ------------------------- main -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="marlin-2b", help="path to weights dir")
    ap.add_argument("--mode", choices=["caption", "find"], default="caption")
    ap.add_argument("--event", default="", help="query for find mode")
    ap.add_argument("--source", default="auto", help="'auto'|'realsense'|webcam index")
    ap.add_argument("--buffer", type=int, default=16, help="frames sent to the model")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="cadence frames enter the buffer (Marlin trained at 2.0)")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between automatic runs; 0 = manual only (SPACE)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=448,
                    help="downscale frames so longest side <= this (VIDEO_MAX_PIXELS~448)")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    args = ap.parse_args()

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # canonical prompts + parsers from the checkpoint's custom modeling file
    mpath = os.path.join(args.model, "modeling_marlin.py")
    spec = importlib.util.spec_from_file_location("modeling_marlin", mpath)
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)

    print(f"[load] {args.model} (native Qwen3_5 class, bf16) ...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    print(f"[load] done {time.time()-t0:.1f}s VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    lock = threading.Lock()
    buf: Deque = collections.deque(maxlen=args.buffer)
    state = {"mode": args.mode, "event": args.event, "out": "(warming up)",
             "latency": 0.0, "paused": args.interval <= 0, "stop": False, "busy": False}
    ask_now = threading.Event()

    def run_model(frames: List, prompt: str, max_new: int) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]

    def worker():
        last = 0.0
        while True:
            triggered = ask_now.wait(timeout=0.1)
            with lock:
                if state["stop"]:
                    return
                paused, mode, event = state["paused"], state["mode"], state["event"]
            due = (not paused) and (time.time() - last) >= args.interval
            if not (triggered or due):
                continue
            ask_now.clear()
            with lock:
                snap = list(buf)
            if len(snap) < 2:
                continue
            if len(snap) % 2:
                snap = snap[1:]
            if mode == "find" and not event.strip():
                with lock:
                    state["out"] = "[find] press 'e' to set an event query"
                continue
            with lock:
                state["busy"] = True
            t = time.time()
            try:
                if mode == "caption":
                    raw = run_model(snap, mm.CAPTION_PROMPT, args.max_new_tokens)
                    _, scene, events = mm.parse_caption(raw)
                    lines = [f"Scene: {scene}"] if scene else []
                    if events:
                        lines.append("Events:")
                        lines += [f"  <{e['start']:.1f}-{e['end']:.1f}s> {e['description']}"
                                  for e in events]
                    txt = "\n".join(lines) or mm.strip_thinking(raw) or "(no caption)"
                else:
                    prompt = mm.GROUNDING_PROMPT_TEMPLATE.format(event=event.strip())
                    raw = run_model(snap, prompt, 64)
                    cleaned, span = mm.parse_span(raw)
                    txt = (f'"{event.strip()}"  ->  From {span[0]:.1f}s to {span[1]:.1f}s'
                           if span else f'"{event.strip()}"  ->  {cleaned or "no span"}')
            except Exception as e:
                txt = f"[infer error] {e}"
            dt = time.time() - t
            last = time.time()
            with lock:
                state["out"], state["latency"], state["busy"] = txt, dt, False

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    cam = open_camera(args.source, args.cam_width, args.cam_height, args.cam_fps)
    win = "Marlin-2B live  (q quit | SPACE ask | p pause | m mode | e query)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sample_period = 1.0 / max(args.sample_fps, 1e-6)
    last_sample = 0.0
    try:
        while True:
            bgr = cam.read()
            if bgr is None:
                continue
            now = time.time()
            if now - last_sample >= sample_period:
                last_sample = now
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if max(pil.size) > args.max_side:
                    s = args.max_side / max(pil.size)
                    pil = pil.resize((int(pil.width*s), int(pil.height*s)))
                with lock:
                    buf.append(pil)

            with lock:
                out, lat, mode = state["out"], state["latency"], state["mode"]
                paused, busy, nbuf, event = (state["paused"], state["busy"],
                                             len(buf), state["event"])
            clip_s = nbuf / max(args.sample_fps, 1e-6)

            disp = bgr.copy()
            W = disp.shape[1]
            stat = "RUNNING…" if busy else ("PAUSED" if paused else f"auto {args.interval:.0f}s")
            hdr = f"[{mode}] {stat}  clip~{clip_s:.0f}s ({nbuf}f)  last={lat*1000:.0f}ms"
            if mode == "find":
                hdr += f'  q="{event[:30]}"'
            _bar(disp, 0, 0, W, 30)
            _text(disp, hdr, (10, 21), color=(0, 255, 0) if not busy else (0, 215, 255))

            wrapped: List[str] = []
            for para in out.split("\n"):
                wrapped += textwrap.wrap(para, width=max(24, W // 10)) or [""]
            wrapped = wrapped[:8]
            bh = 24 * len(wrapped) + 14
            _bar(disp, 0, disp.shape[0]-bh, W, bh)
            for i, ln in enumerate(wrapped):
                _text(disp, ln, (10, disp.shape[0]-bh+22+i*24))
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                ask_now.set()
            elif k == ord("p"):
                with lock:
                    state["paused"] = not state["paused"]
            elif k == ord("m"):
                with lock:
                    state["mode"] = "find" if state["mode"] == "caption" else "caption"
            elif k == ord("e"):
                q = input("Find event query: ").strip()
                if q:
                    with lock:
                        state["event"], state["mode"] = q, "find"
                    ask_now.set()
    finally:
        with lock:
            state["stop"] = True
        ask_now.set()
        cam.release()
        cv2.destroyAllWindows()
        th.join(timeout=2.0)


def _bar(img, x, y, w, h, alpha=0.45):
    y = max(0, y)
    sub = img[y:y+h, x:x+w]
    if sub.size:
        sub[:] = (sub * (1 - alpha)).astype(img.dtype)


def _text(img, s, org, color=(255, 255, 255), scale=0.55, thick=1):
    import cv2
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick+2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


if __name__ == "__main__":
    main()
