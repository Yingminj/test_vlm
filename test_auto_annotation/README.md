# Local Qwen3.5-VL inference server (for auto-labeling)

Deploy the local Qwen weights as an **OpenAI-compatible HTTP API** so any machine on
the intranet can run video inference through the standard `openai` client — including
the base64 `video_url` payload that [`video_auto_label.py`](video_auto_label.py)
already sends to a cloud model. Point that script's `base_url` at this server and it
works unchanged.

Validated end-to-end on this box (RTX 4090, 48 GB): text + base64-video requests,
~19.6 GB VRAM for the 9B in bf16.

## What's here

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server: `/v1/chat/completions`, `/v1/models`, `/health` |
| `run_server.sh` | Launcher (`MODEL=9b`/`27b`, env-var config) |
| `test_client.py` | Smoke test — calls the server exactly like `video_auto_label.py` |
| `environment.yml` / `requirements.txt` | Env setup (clone-or-build) |
| `video_auto_label.py` | The auto-labeler (the client to repoint at this server) |

## Why a transformers server (not vLLM)

Both weight dirs are `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) — a
**brand-new hybrid linear+full-attention** architecture. vLLM support for it is not
guaranteed and isn't installed here, so the server drives the model with plain
`transformers`, the same load path this repo already validated
(`scripts/qwen3_5_camera.py`). Slower than vLLM, but reliable.

## Models & VRAM (single 48 GB 4090)

| Model | Path | Precision | VRAM | Notes |
|-------|------|-----------|------|-------|
| **qwen3_5_9B** (default) | `model/qwen3_5_9B` | bf16 | **~19.6 GB** | fast, validated |
| qwen3_6_27B | `model/qwen3_6_27B` | **4-bit (nf4)** | ~15 GB | `QUANT=4bit`; needs `bitsandbytes` |
| qwen3_6_27B | `model/qwen3_6_27B` | bf16 | ~54 GB | ❌ won't fit one 48 GB card |

The 27B in bf16 (~54 GB of weights) does **not** fit a single 48 GB GPU — run it
4-bit, or use a second GPU / CPU offload.

## 1. Environment

The repo's `vlm` env is already validated for these weights. The fast, reliable way
to get a **fresh isolated env** is to clone it and add `fastapi` (avoids re-resolving
the heavy CUDA stack over the slow PyPI here):

```bash
conda create -y -n autolabel --clone vlm
conda activate autolabel
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi openai
```

> This is exactly how the env in this folder was built and tested.

**From scratch** (slower) — build from `environment.yml`, or into a fresh py3.10 env:

```bash
conda env create -f environment.yml          # creates `autolabel`
# or, into an existing env:
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

`bitsandbytes` is only needed for the 27B 4-bit path; the 9B doesn't use it.

## 2. Start the server (on the GPU machine)

```bash
conda activate autolabel
cd test_auto_annotation

bash run_server.sh                  # qwen3_5_9B, bf16, 0.0.0.0:8000
MODEL=27b bash run_server.sh        # qwen3_6_27B, 4-bit
PORT=9000 bash run_server.sh        # change port
```

First request triggers a ~20–60 s model load; watch the console for `[load] ready`.
Check it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"qwen3_5_9B","loaded":true}
```

Bind address is `0.0.0.0` so other intranet hosts can reach it at
`http://<gpu-machine-ip>:8000/v1`. Open the port in the firewall if needed
(`sudo ufw allow 8000/tcp`).

### Config (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL_PATH` | `../model/qwen3_5_9B` | weights dir |
| `MODEL_NAME` | basename of `MODEL_PATH` | name reported by the API / used in requests |
| `QUANT` | `none` | `none` or `4bit` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | bind address |
| `MAX_FRAMES` | `64` | frames sampled per video (caps VRAM/latency) |
| `SAMPLE_FPS` | `0` | down-sample to this fps before the frame cap (`0` = keep clip's fps) |
| `MAX_PIXELS` | unset | per-frame pixel cap (e.g. `200704` = `256*28*28`) |
| `API_KEY` | unset | if set, requests must send `Authorization: Bearer <key>` |

The server uniformly samples each incoming video down to `MAX_FRAMES`. For minute-long
clips that need fine temporal boundaries, raise `MAX_FRAMES` (watch VRAM) and/or send
the clip pre-trimmed.

## 3. Call it from another machine

Plain OpenAI client — same shape as `video_auto_label.py`:

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://<gpu-machine-ip>:8000/v1")

with open("clip.mp4", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="qwen3_5_9B",
    max_tokens=4096,
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        {"type": "text", "text": "Describe the stages in this video."},
    ]}],
    extra_body={"thinking": {"type": "disabled"}},  # fast, no <think> trace
)
print(resp.choices[0].message.content)
```

Supported in `content`: `text`, `video_url`, `image_url`. The url may be a
`data:...;base64,...` URI, an `http(s)://` url, or a local path on the server.
`thinking` defaults to **disabled** (fast direct answers); omit `extra_body` to keep
that, or pass `{"thinking": {"type": "enabled"}}` for a reasoning trace (auto-stripped
from the returned content).

### Repointing `video_auto_label.py`

Change only the client construction and model name — the message format already matches:

```python
client = OpenAI(api_key="EMPTY", base_url="http://<gpu-machine-ip>:8000/v1")
# ...and in call_llm(): model="qwen3_5_9B"   (was "kimi-k2.6")
```

## 4. Smoke test

```bash
conda activate autolabel
python test_client.py --prompt "Reply with exactly: server ok"   # text-only
python test_client.py --video /path/to/clip.mp4 --prompt "Describe this clip."
# override target: --base-url http://HOST:8000/v1 --model qwen3_5_9B
```

## Gotchas

- **Proxy hijacks localhost.** If a `SOCKS`/`http(s)` proxy is exported (this box has
  one), the `openai`/`httpx` client routes even local/intranet calls through it and
  fails with `socksio`/connection errors. Exclude the server host:
  ```bash
  export no_proxy="localhost,127.0.0.1,<gpu-machine-ip>"
  export NO_PROXY="$no_proxy"
  ```
- **Streaming is not implemented** — send `stream=False` (the default). A `stream:true`
  request returns HTTP 400.
- **Offline by design.** The server sets `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1`,
  so it loads only the local weights and never hits the HF Hub.
- **27B needs `QUANT=4bit`** on one 48 GB card — bf16 OOMs (see the table).
- **Linear-attention layers use a slow PyTorch fallback.** Optional speedup:
  `pip install causal-conv1d flash-linear-attention`.
