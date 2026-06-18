#!/usr/bin/env python3
"""OpenAI-compatible inference server for the local Qwen3.5-VL weights.

Exposes a minimal `/v1/chat/completions` (+ `/v1/models`, `/health`) endpoint so any
machine on the intranet can drive the model through the standard `openai` client --
including the base64 `video_url` payload that `video_auto_label.py` already sends.

Why a hand-rolled transformers server (not vLLM):
  The weights are `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`, a hybrid
  linear+full-attention arch). That arch is brand-new; vLLM support is not guaranteed,
  and this repo has only ever validated the plain-transformers load path. So we serve
  with transformers directly -- the same incantation as scripts/qwen3_5_camera.py.

Config is entirely via environment variables (see run_server.sh / README.md):
  MODEL_PATH    weights dir            (default: ../model/qwen3_5_9B)
  MODEL_NAME    name reported to API   (default: basename of MODEL_PATH)
  QUANT         none | 4bit            (default: none; use 4bit for the 27B on 48GB)
  HOST PORT     bind address           (default: 0.0.0.0 : 8000)
  MAX_FRAMES    frames sampled / video (default: 64)
  SAMPLE_FPS    target fps for sampling(default: 0 = keep clip's own fps)
  MAX_PIXELS    per-frame pixel cap    (default: unset -> processor default)
  API_KEY       optional bearer token  (default: unset -> auth disabled)
"""
from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ----------------------------- configuration -----------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_HERE, "..", "model", "qwen3_5_9B"))
MODEL_PATH = os.path.abspath(MODEL_PATH)
MODEL_NAME = os.environ.get("MODEL_NAME", os.path.basename(MODEL_PATH.rstrip("/")))
QUANT = os.environ.get("QUANT", "none").lower()
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "64"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "0"))
MAX_PIXELS = int(os.environ["MAX_PIXELS"]) if os.environ.get("MAX_PIXELS") else None
API_KEY = os.environ.get("API_KEY") or None

# Keep the model off the HF hub -- weights are local.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Heavy imports are deferred to load_model() so `--help`/import stays light.
_MODEL = None
_PROC = None

app = FastAPI(title="qwen3_5-vl local server")


# ------------------------------- model load -------------------------------
def load_model():
    """Load processor + model once, the validated transformers way."""
    global _MODEL, _PROC
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print(f"[load] model={MODEL_PATH} quant={QUANT}", flush=True)
    proc_kwargs = {}
    if MAX_PIXELS:
        proc_kwargs["max_pixels"] = MAX_PIXELS
    _PROC = AutoProcessor.from_pretrained(MODEL_PATH, **proc_kwargs)

    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if QUANT == "4bit":
        # nf4 4-bit -- the only way the 27B fits a single 48GB card.
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs.pop("dtype", None)
    elif QUANT not in ("none", ""):
        raise ValueError(f"unknown QUANT={QUANT!r} (use 'none' or '4bit')")

    t0 = time.time()
    _MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs)
    _MODEL.eval()
    print(f"[load] ready in {time.time() - t0:.1f}s", flush=True)


# --------------------------- request-side helpers ---------------------------
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _materialize_media(url: str, suffix: str) -> str:
    """Return a local file path for a data-URI / http(s) URL / local path."""
    m = _DATA_URI_RE.match(url.strip())
    if m:
        try:
            raw = base64.b64decode(m.group("data"), validate=False)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(400, f"bad base64 in data URI: {e}")
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path
    if url.startswith(("http://", "https://")):
        import urllib.request
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(url, path)  # noqa: S310 (intranet, trusted)
        return path
    if os.path.isfile(url):
        return url
    raise HTTPException(400, f"cannot resolve media url: {url[:80]}...")


def _sample_video_frames(path: str):
    """Decode a video file to a list of RGB PIL frames, bounded by MAX_FRAMES."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, f"could not open video {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Read every frame (or every Nth, if down-sampling fps), then uniformly cap.
    step = 1
    if SAMPLE_FPS > 0 and native_fps > SAMPLE_FPS:
        step = max(1, int(round(native_fps / SAMPLE_FPS)))

    grabbed = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            grabbed.append(frame)
        idx += 1
    cap.release()

    if not grabbed:
        raise HTTPException(400, f"no frames decoded from {path} (total={total})")

    # Uniformly subsample down to MAX_FRAMES.
    if len(grabbed) > MAX_FRAMES:
        sel = [int(round(i * (len(grabbed) - 1) / (MAX_FRAMES - 1))) for i in range(MAX_FRAMES)]
        grabbed = [grabbed[i] for i in sel]

    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in grabbed]


def _parse_messages(messages: List[dict]):
    """Translate OpenAI messages -> Qwen chat messages + collected frame lists.

    Supports content as a plain string, or a list of {type: text|video_url|image_url}.
    Temp files created for base64/url media are returned so the caller can clean up.
    """
    from PIL import Image

    out_msgs = []
    tmp_files: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out_msgs.append({"role": role, "content": content})
            continue
        parts = []
        for item in content or []:
            itype = item.get("type")
            if itype == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif itype == "video_url":
                url = (item.get("video_url") or {}).get("url", "")
                path = _materialize_media(url, suffix=".mp4")
                if path not in tmp_files and _DATA_URI_RE.match(url.strip()):
                    tmp_files.append(path)
                frames = _sample_video_frames(path)
                parts.append({"type": "video", "video": frames})
            elif itype == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                path = _materialize_media(url, suffix=".png")
                if path not in tmp_files and _DATA_URI_RE.match(url.strip()):
                    tmp_files.append(path)
                parts.append({"type": "image", "image": Image.open(path).convert("RGB")})
            else:
                raise HTTPException(400, f"unsupported content type: {itype}")
        out_msgs.append({"role": role, "content": parts})
    return out_msgs, tmp_files


def _collect_media(qwen_msgs):
    """Pull the flat frame / image lists the processor expects."""
    videos, images = [], []
    for m in qwen_msgs:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if part.get("type") == "video":
                videos.append(part["video"])
            elif part.get("type") == "image":
                images.append(part["image"])
    return videos or None, images or None


# ------------------------------- inference --------------------------------
def _thinking_enabled(body: dict) -> bool:
    """Honor `extra_body={'thinking': {'type': 'disabled'}}`; default off (fast)."""
    th = body.get("thinking")
    if isinstance(th, dict):
        return th.get("type") != "disabled"
    return False


def run_inference(body: dict) -> dict:
    import torch

    messages = body.get("messages")
    if not messages:
        raise HTTPException(400, "messages is required")

    qwen_msgs, tmp_files = _parse_messages(messages)
    videos, images = _collect_media(qwen_msgs)
    think = _thinking_enabled(body)
    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature", 0.0) or 0.0)

    try:
        text = _PROC.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
        proc_kwargs = {"text": [text], "return_tensors": "pt"}
        if videos:
            proc_kwargs["videos"] = videos
        if images:
            proc_kwargs["images"] = images
        inputs = _PROC(**proc_kwargs).to(_MODEL.device)

        gen_kwargs = dict(max_new_tokens=max_new)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              top_p=float(body.get("top_p", 1.0) or 1.0))
        else:
            gen_kwargs.update(do_sample=False)

        t0 = time.time()
        with torch.no_grad():
            gen = _MODEL.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen = gen[:, prompt_tokens:]
        completion_tokens = int(gen.shape[1])
        out = _PROC.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if "</think>" in out:
            out = out.split("</think>")[-1].strip()
        finish = "length" if completion_tokens >= max_new else "stop"
        print(f"[infer] {elapsed:.1f}s in={prompt_tokens} out={completion_tokens} "
              f"frames={sum(len(v) for v in videos) if videos else 0} think={think}",
              flush=True)
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# -------------------------------- routes ----------------------------------
def _check_auth(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": _MODEL is not None}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": MODEL_NAME, "object": "model", "owned_by": "local"}
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(400, "streaming is not supported by this server")
    try:
        return JSONResponse(run_inference(body))
    except HTTPException:
        raise
    except Exception as e:  # surface model errors as 500 with a message
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"[boot] {MODEL_NAME} on {HOST}:{PORT} (quant={QUANT}, max_frames={MAX_FRAMES})",
          flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
