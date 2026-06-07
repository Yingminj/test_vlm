"""Real-time video understanding with Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

Streams from a camera (Intel RealSense if present, else any OpenCV webcam), keeps a
rolling buffer of recent frames, and periodically (or on demand) asks the model a
question about the live video, showing the answer as an overlay.

This drives the *base* Qwen3.5 model directly (no fine-tuning) so you can probe its
out-of-the-box video understanding.

Key facts baked in (validated on this machine):
  * Loads in bf16 (~19 GB VRAM); no quantization needed on a 48 GB GPU.
  * The clean input path is `processor(text=..., videos=[frames])` with a list of
    PIL frames -- the qwen_vl_utils helper mishandles this processor's fps typing.
  * `enable_thinking=False` (default) skips the <think> reasoning for snappy replies;
    pass --think to see the reasoning trace.
  * Linear-attention layers use a slow PyTorch fallback unless flash-linear-attention
    + causal-conv1d are installed (optional; see scripts/qwen3_5_camera.sh header).

Controls (focus the video window):
  q = quit   SPACE = ask now   n = new question (type in terminal)
  p = pause/resume auto-asking  t = toggle thinking

Run:  bash scripts/qwen3_5_camera.sh --prompt "What is the person doing?"
"""
from __future__ import annotations

import argparse
import collections
import os
import textwrap
import threading
import time
from typing import Deque, List, Optional

import numpy as np


# ------------------------- camera backends -------------------------
class RealSenseCam:
    def __init__(self, w, h, fps):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self):
        fs = self.pipeline.wait_for_frames()
        c = fs.get_color_frame()
        if not c:
            return None
        return np.asanyarray(c.get_data())  # BGR

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
        return frame if ok else None  # BGR

    def release(self):
        self.cap.release()


class VideoFileCam:
    """Replay a video file as if it were a live camera.

    Frames are paced to the file's native fps (x --speed) so the main loop's
    wall-clock sampling (--sample-fps) and the threaded inference behave exactly
    as they do on a real stream -- including dropping frames that elapse while the
    model is busy. At EOF, sets self.eof (the main loop breaks); --loop re-seeks.
    """

    def __init__(self, path, loop=False, speed=1.0):
        import cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video file {path!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop = loop
        self.speed = max(speed, 1e-6)
        self.eof = False
        self._n = 0          # frames emitted since playback start
        self._t0 = None      # wall-clock anchor for pacing

    def read(self):
        if self._t0 is None:
            self._t0 = time.time()
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                import cv2
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._n, self._t0 = 0, time.time()
                ok, frame = self.cap.read()
            if not ok:
                self.eof = True
                return None
        # sleep so emitted-frame time tracks video time -> ~real-time playback
        dt = (self._t0 + self._n / (self.fps * self.speed)) - time.time()
        if dt > 0:
            time.sleep(dt)
        self._n += 1
        return frame  # BGR

    def timestamp(self):
        """Position of the just-read frame within the video, in seconds."""
        return max(0.0, (self._n - 1) / self.fps)

    def release(self):
        self.cap.release()


def open_camera(source, w, h, fps, loop=False, speed=1.0):
    """source: 'auto' | 'realsense' | webcam index string | path to a video file."""
    if source not in ("auto", "realsense") and os.path.isfile(source):
        print(f"[cam] replaying video file: {source} (loop={loop}, speed={speed}x)")
        return VideoFileCam(source, loop=loop, speed=speed)
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
    ap.add_argument("--model", default="qwen3_5_9B", help="path to weights dir")
    ap.add_argument("--prompt", default="Describe what is happening in the scene right now.")
    ap.add_argument("--source", default="auto",
                    help="'auto' | 'realsense' | webcam index '0' | path to a video file")
    ap.add_argument("--loop", action="store_true",
                    help="when --source is a video file, restart it at EOF")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="video-file playback speed multiplier (1.0 = real time)")
    ap.add_argument("--buffer", type=int, default=8, help="frames sent to the model")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="cadence frames enter the buffer")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between automatic questions; 0 = manual only (SPACE)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--results-dir", default="results",
                    help="save each inference to results/<tag>_<time>.txt; '' disables")
    ap.add_argument("--max-side", type=int, default=640,
                    help="downscale frames so longest side <= this (speed/VRAM)")
    ap.add_argument("--think", action="store_true", help="enable <think> reasoning")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    args = ap.parse_args()

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print(f"[load] {args.model} (bf16) ...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # --- shared state ---
    lock = threading.Lock()
    buf: Deque = collections.deque(maxlen=args.buffer)  # PIL RGB frames
    ts_buf: Deque = collections.deque(maxlen=args.buffer)  # clip time (s) per frame
    state = {"answer": "(warming up)", "latency": 0.0, "prompt": args.prompt,
             "think": args.think, "paused": args.interval <= 0, "stop": False,
             "busy": False}
    ask_now = threading.Event()

    writer = (ResultWriter(args.results_dir, f"qwen3_5_{_source_tag(args.source)}")
              if args.results_dir else None)
    if writer:
        print(f"[results] saving to {writer.path}")

    def infer(frames: List, prompt: str, think: bool) -> str:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=think)
        inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        out = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if "</think>" in out:           # show only the post-reasoning answer
            out = out.split("</think>")[-1].strip()
        return out or "(empty)"

    def worker():
        last = 0.0
        while True:
            triggered = ask_now.wait(timeout=0.1)
            with lock:
                if state["stop"]:
                    return
                paused, prompt, think = state["paused"], state["prompt"], state["think"]
            now = time.time()
            due = (not paused) and (now - last) >= args.interval
            if not (triggered or due):
                continue
            ask_now.clear()
            with lock:
                snap = list(buf)
                snap_ts = list(ts_buf)
            if len(snap) < 2:
                continue
            if len(snap) % 2:           # temporal_patch_size=2 wants an even count
                snap, snap_ts = snap[1:], snap_ts[1:]
            with lock:
                state["busy"] = True
            t = time.time()
            try:
                ans = infer(snap, prompt, think)
            except Exception as e:
                ans = f"[infer error] {e}"
            dt = time.time() - t
            last = time.time()
            with lock:
                state["answer"], state["latency"], state["busy"] = ans, dt, False
            if writer and snap_ts:
                writer.write(snap_ts[0], snap_ts[-1], ans)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    cam = open_camera(args.source, args.cam_width, args.cam_height, args.cam_fps,
                      loop=args.loop, speed=args.speed)
    win = "Qwen3.5 live  (q quit | SPACE ask | n new q | p pause | t think)"
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
                    pil = pil.resize((int(pil.width*s), int(pil.height*s)))
                ts = cam.timestamp() if hasattr(cam, "timestamp") else (now - t_start)
                with lock:
                    buf.append(pil)
                    ts_buf.append(ts)

            with lock:
                answer, lat, prompt = state["answer"], state["latency"], state["prompt"]
                think, paused, busy, nbuf = (state["think"], state["paused"],
                                             state["busy"], len(buf))

            disp = bgr.copy()
            W = disp.shape[1]
            # top bar: prompt + status
            mode = "THINK" if think else "fast"
            stat = "ASKING…" if busy else ("PAUSED" if paused else f"auto {args.interval:.0f}s")
            _bar(disp, 0, 0, W, 60)
            _text(disp, f"Q: {prompt}", (10, 24))
            _text(disp, f"[{mode}] {stat}  buf={nbuf}  last={lat*1000:.0f}ms",
                  (10, 50), color=(0, 255, 0) if not busy else (0, 215, 255))
            # bottom: answer (wrapped)
            wrapped = textwrap.wrap(answer, width=max(20, W // 11)) or [""]
            wrapped = wrapped[:5]
            bh = 26 * len(wrapped) + 16
            _bar(disp, 0, disp.shape[0]-bh, W, bh)
            for i, ln in enumerate(wrapped):
                _text(disp, ln, (10, disp.shape[0]-bh+26+i*26), color=(255, 255, 255))
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                ask_now.set()
            elif k == ord("p"):
                with lock:
                    state["paused"] = not state["paused"]
            elif k == ord("t"):
                with lock:
                    state["think"] = not state["think"]
            elif k == ord("n"):
                newq = input("New question: ").strip()
                if newq:
                    with lock:
                        state["prompt"] = newq
                    ask_now.set()
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


def _hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _source_tag(source: str) -> str:
    if os.path.isfile(source):
        return os.path.splitext(os.path.basename(source))[0]
    return f"cam_{source}"


class ResultWriter:
    """Append one line per inference: `HH:MM:SS-HH:MM:SS answer: <text>`.

    One file per run under --results-dir. Times are the span the buffered clip
    covers (video position for a file, elapsed wall-clock for a camera).
    """

    def __init__(self, outdir: str, tag: str):
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(outdir, f"{tag}_{stamp}.txt")
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, t0: float, t1: float, text: str, label: str = "answer") -> None:
        oneline = " ".join((text or "").split())   # collapse newlines -> single line
        self._f.write(f"{_hms(t0)}-{_hms(t1)} {label}: {oneline}\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _bar(img, x, y, w, h, alpha=0.45):
    import cv2
    y = max(0, y)
    sub = img[y:y+h, x:x+w]
    if sub.size:
        sub[:] = (sub * (1 - alpha)).astype(img.dtype)


def _text(img, s, org, color=(255, 255, 255), scale=0.6, thick=1):
    import cv2
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick+2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


if __name__ == "__main__":
    main()
