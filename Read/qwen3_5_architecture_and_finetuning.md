# Qwen3.5-9B (`qwen3_5_9B`) — Architecture, Training, and Fine-Tuning Guide

> Scope: this document describes the **local checkpoint** at
> `/home/kewei/YING/test_vlm/qwen3_5_9B` (HF class
> `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`). Numbers marked
> *(verified)* were measured directly from this checkpoint's `config.json` and the
> instantiated module tree; numbers marked *(model card)* come from the bundled
> `README.md` / Qwen3.5 release notes and describe the **family/training**, which is
> not always introspectable from weights.

---

## 1. Identity at a glance

| Property | Value |
|---|---|
| HF architecture | `Qwen3_5ForConditionalGeneration` *(verified)* |
| Processor | `Qwen3VLProcessor` (image proc `Qwen2VLImageProcessorFast`, video proc `Qwen3VLVideoProcessor`) *(verified)* |
| Modality | Vision-Language (image + video + text), early-fusion *(model card)* |
| Total parameters | **9.41 B** *(verified)* |
| Dtype on disk | bfloat16, 4 shards, ~19 GB *(verified)* |
| Native context | 262,144 tokens; extensible to ~1.01 M via YaRN *(model card)* |
| Thinking mode | Yes — `<think>…</think>`, toggled by chat template `enable_thinking` *(verified)* |
| Requires (this env) | `transformers 5.9.0` (has the class), torch 2.8/cu129 *(verified)* |

---

## 2. Parameter budget (verified, measured on this checkpoint)

| Component | Module path | Params | Share |
|---|---|---:|---:|
| **Vision tower** | `model.visual` | 0.456 B | 4.8 % |
| **Language model (all)** | `model.language_model` | 7.937 B | 84.3 % |
| ↳ token embeddings | `model.language_model.embed_tokens` | 1.017 B | 10.8 % |
| ↳ 32 transformer blocks | `model.language_model.layers` | 6.920 B | 73.5 % |
| **LM head** | `lm_head` (untied) | 1.017 B | 10.8 % |
| **Total** | | **9.410 B** | 100 % |

Notes:
- `tie_word_embeddings: false` → `embed_tokens` and `lm_head` are **separate** 1.0 B
  matrices (vocab 248,320 × hidden 4,096). Together they are ~21 % of the model.
- The vision tower is small (4.8 %); almost all capacity is in the LLM blocks.

---

## 3. Architecture in detail

### 3.1 Multimodal pipeline (early fusion)
```
image/video ─▶ Qwen3_5VisionModel ─▶ PatchMerger ─▶ visual tokens ┐
                                                                   ├─▶ interleave into
text ───────────────────────────────▶ token embeddings ───────────┘   the token stream
                                                                        │
                                                       Qwen3_5TextModel (32 hybrid blocks)
                                                                        │
                                                                     lm_head ─▶ logits
```
Image and video frames are turned into **visual tokens** and placed inline with text
tokens (`image_token_id 248056`, `video_token_id 248057`, vision span markers
`248053/248054`). The LLM then attends over the mixed sequence — there is no separate
cross-attention adapter; vision and text share the same transformer stack
("unified vision-language foundation, early fusion" — *model card*).

### 3.2 Vision encoder — `Qwen3_5VisionModel` (~456 M, verified)
A ViT-style encoder:
- `patch_embed` (`Qwen3_5VisionPatchEmbed`, a **Conv3d**): patch size **16**,
  temporal patch size **2** (consumes frame *pairs*), 3 input channels.
- `pos_embed` (learned, 2,304 positions) + 2-D `rotary_pos_emb`.
- **27** transformer `blocks`, hidden 1,152, 16 heads, MLP intermediate 4,304,
  `gelu_pytorch_tanh`. Each block: `norm1/norm2` (LayerNorm),
  `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`.
- `merger` (`Qwen3_5VisionPatchMerger`, ~40 M): **spatial merge size 2** (2×2 patch
  groups merged) and projects vision features to `out_hidden_size = 4096` to match
  the LLM. `deepstack_visual_indexes: []` (no deep-stack injection in this build).
- Image normalization: mean/std = 0.5/0.5; pixels are dynamic-resolution
  (the processor's `longest_edge`/`shortest_edge` cap token count — see §6.4).

### 3.3 Language model — `Qwen3_5TextModel` (hybrid, verified)
This is the defining feature. **32 layers** are split into two block types:

| Block type | Count | Token mixer | Position info |
|---|---:|---|---|
| **Linear attention** (Gated DeltaNet) | **24** | `linear_attn` (gated delta net, conv + state) | no RoPE (recurrent) |
| **Full attention** (Gated Attention) | **8** | `self_attn` (softmax MHA + output gate) | partial RoPE |

Layout (`full_attention_interval: 4`): **`8 × (3 × DeltaNet→FFN  +  1 × Attention→FFN)`**
— every 4th block is full attention, the other three are linear. This is the
"efficient hybrid architecture" — linear-attention blocks give near-linear cost over
the 262 K context, while the periodic full-attention blocks preserve exact long-range
recall.

**Gated DeltaNet block** (`layers[i].linear_attn`, the 24 linear layers):
- `in_proj_qkv`: Linear 4096 → 8192 (produces q/k/v for the delta-rule state update)
- `in_proj_a`, `in_proj_b`: Linear 4096 → 32 (per-head scalar **gates** α/β of the
  delta rule — one value per value-head)
- `in_proj_z`: Linear 4096 → 4096 (output gate)
- `conv1d`: depthwise **Conv1d** (channels 8192, kernel 4) — short causal conv that
  mixes recent tokens before the recurrence
- `norm`: `Qwen3_5RMSNormGated`; `out_proj`: Linear 4096 → 4096
- Heads (config): **16 key/query heads, 32 value heads, head dim 128**.
- *Not standard attention* — it is a chunk-parallel linear recurrence (Gated Delta
  Rule). The fast CUDA path needs `flash-linear-attention` + `causal-conv1d`;
  otherwise a slower pure-PyTorch fallback runs (see §6.1).

**Gated Attention block** (`layers[i].self_attn`, the 8 full layers):
- `q_proj` 4096→8192 (**16 Q heads × 256**), `k_proj`/`v_proj` 4096→1024
  (**4 KV heads × 256**, i.e. GQA 16:4), `o_proj` 4096→4096.
- `q_norm`, `k_norm` (RMSNorm over head_dim 256) — QK-norm for stability.
- `attn_output_gate: true` → a sigmoid gate on the attention output (this is what
  makes it "*gated* attention").
- **Partial RoPE**: `partial_rotary_factor 0.25` → only 64 of 256 head dims are
  rotated; interleaved **mRoPE** with `mrope_section [11,11,10]` (separate
  temporal/height/width frequency bands for multimodal positions); `rope_theta 1e7`.

**FFN** (every block, both types): SwiGLU — `mlp.gate_proj`/`up_proj` 4096→12288,
`mlp.down_proj` 12288→4096, SiLU. **Dense** (no Mixture-of-Experts in this 9B:
`mlp_only_layers: []`, single expert per layer — *verified*). The MoE mentioned in
the Qwen3.5 highlights applies to larger family members, not this checkpoint.

**Other**: `Qwen3_5RMSNorm` throughout (eps 1e-6); `mtp_num_hidden_layers: 1` —
a Multi-Token-Prediction head used **during training** (predicts >1 future token to
densify the loss / enable speculative decoding); it is not exposed as a separate
inference module in the HF class and does not affect normal `generate`.

---

## 4. Training method (model card / release notes)

Qwen3.5 is released as a **post-trained** checkpoint (this repo's `README.md` states
"post-trained model … in HF Transformers format"). The pipeline, per the model card:

1. **Multimodal pre-training, early fusion.** Text and visual tokens are trained
   jointly from early on (rather than bolting a vision adapter onto a frozen LLM),
   "achieving cross-generational parity with Qwen3 and outperforming Qwen3-VL." The
   **MTP** objective (multi-token prediction) is used to densify supervision.
2. **Long-context training.** Native 262 K context; YaRN RoPE scaling supported at
   inference for up to ~1 M tokens (§6.5).
3. **Post-training (alignment).**
   - **SFT** with a **dual-mode** format: a *thinking* mode (emits `<think>…</think>`
     reasoning before the answer) and a *non-thinking* (instruct) mode, switchable via
     the chat template (`enable_thinking`).
   - **Reinforcement learning at scale** — "RL scaled across million-agent
     environments with progressively complex task distributions"; asynchronous RL
     infra for agentic/tool-use behaviour.
4. **Broad coverage.** 201 languages/dialects; agentic (Qwen-Agent / Qwen-Code)
   and tool-calling abilities.

> The exact data mixtures, RL reward design, and MTP schedule are **not public** in
> this checkpoint. Treat §4 as background for *what behaviours already exist* (so you
> don't fine-tune them away), not as a recipe you can reproduce.

---

## 5. Fine-tuning: what you can tune and the impact

### 5.1 Component-by-component

| Target | Module path(s) | Effect of tuning it | Recommendation |
|---|---|---|---|
| **LLM attention/FFN projections** | `self_attn.*`, `linear_attn.*`, `mlp.*` in `model.language_model.layers` | Where reasoning, output style, task skill live. LoRA here is the highest-leverage, lowest-risk option. | **Primary LoRA target.** |
| **Vision tower** | `model.visual.*` | Adapts low-level perception (new camera domain, unusual imagery, OCR-heavy frames). Only 4.8 % of params; risky to fully unfreeze (can destabilize alignment, overfit small data). | **Freeze by default.** Unfreeze (or LoRA) only if your images differ a lot from natural images and text-only adaptation underperforms. |
| **Vision merger** | `model.visual.merger` | The vision→LLM "connector". Cheap to train; classic place to adapt when bridging a new visual distribution while keeping the LLM frozen. | Optional, cheap add-on. |
| **Token embeddings / LM head** | `embed_tokens`, `lm_head` | Needed only if you **add new tokens** (special tags, control tokens). Each is 1 B params — expensive, and tuning `lm_head` broadly can shift the whole output distribution. | Tune only the **new rows** if you add tokens; otherwise leave frozen. |
| **Norms / gates / biases** | `*_norm`, DeltaNet `in_proj_a/b`, attn output gate | BitFit-style; very few params. Small, stabilizing nudges. Rarely sufficient alone. | Optional; can include norms as trainable with LoRA for a small boost. |
| **Full model (all weights)** | everything | Maximum capacity, maximum cost and risk (catastrophic forgetting of thinking/agent/multilingual skills; needs lots of data). | Avoid unless you have large data + multi-GPU. |

### 5.2 Recommended strategy (default): **LoRA / QLoRA on the LLM, vision frozen**

This mirrors the pattern already in this repo (`verifier/model/loader.py`), which does
exactly this for Qwen2.5-VL. For Qwen3.5 the **only changes** are the model class, the
processor, and the **target module names** (which differ because of the hybrid blocks).

**Exact LoRA `target_modules` for Qwen3.5 (verified names):**

```python
# Full-attention (8 layers) + FFN (all 32 layers):
TEXT_FULL = ["q_proj", "k_proj", "v_proj", "o_proj",      # gated attention
             "gate_proj", "up_proj", "down_proj"]          # SwiGLU FFN
# Gated DeltaNet (24 linear layers) — include if you want the linear mixer to adapt:
TEXT_DELTANET = ["in_proj_qkv", "in_proj_z", "out_proj"]   # (skip tiny in_proj_a/b)
# Vision (only if unfreezing vision):
VISION = ["qkv", "proj", "linear_fc1", "linear_fc2"]

target_modules = TEXT_FULL + TEXT_DELTANET                 # LLM-only, vision frozen
```

**Impact of the choice:**
- `TEXT_FULL` only → adapts global reasoning & the 8 long-range attention layers +
  all FFNs. Cheapest, safest, usually enough for *behaviour/format/task* tuning
  (e.g. turning the base model into a structured-output verifier like this repo does).
- `+ TEXT_DELTANET` → also adapts the 24 linear-attention mixers (75 % of the layers).
  Recommended when the task depends on **temporal/streaming** dynamics (video over
  time), since most token-mixing happens in DeltaNet blocks. Slightly more params,
  slightly slower, marginally higher overfit risk.
- `+ VISION` → only when the *visual domain* shifts (e.g. depth/IR, microscopy,
  heavy on-screen text). Otherwise leave vision frozen.

### 5.3 VRAM / feasibility (single GPU, 9.4 B params)

| Mode | Trainable | Base weights | Rough peak VRAM | Fits 24 GB? | Fits 48 GB? |
|---|---|---|---:|:--:|:--:|
| **QLoRA** (4-bit base + LoRA) | adapters | ~6 GB (nf4) | ~10–16 GB | ✅ | ✅ |
| **LoRA** (bf16 base + LoRA) | adapters | ~19 GB | ~22–30 GB | ⚠️ tight | ✅ |
| **Full FT** (bf16 + Adam) | all 9.4 B | 19 GB | ~110 GB+ | ❌ | ❌ (needs ZeRO/multi-GPU or 8-bit Adam + offload) |

(Activation memory scales with frames × resolution × sequence length — cap visual
tokens via the processor, batch size 1 + gradient accumulation, and enable gradient
checkpointing, exactly as `SERVER.md` does for the 3B verifier.)

### 5.4 Caveats specific to Qwen3.5

1. **Linear-attention training path.** The Gated DeltaNet uses custom kernels. Without
   `flash-linear-attention` + `causal-conv1d` you get the **PyTorch fallback** — it
   trains but is slower and uses more memory. For serious fine-tuning, install them
   (`pip install causal-conv1d flash-linear-attention`; needs a CUDA compiler).
2. **4-bit + DeltaNet.** bitsandbytes only quantizes `nn.Linear`; the `Conv1d` and
   recurrence stay in fp16/bf16. QLoRA works, but test a few steps first — quantizing
   the delta-rule projections can be more sensitive than vanilla attention.
3. **Preserve the chat template & thinking modes.** Train with the model's chat
   template and mask the loss to the assistant turn (the repo's
   `verifier/model/collator.py` shows answer-only masking). If you fine-tune
   *non-thinking* data only, supply `enable_thinking=False` in the template and **do
   not** put `<think>` content in history (model-card best practice #4) — otherwise you
   degrade the thinking mode.
4. **Don't casually tune `lm_head`/`embed_tokens`** (1 B each, untied) unless adding
   tokens; if you add tokens, resize embeddings and train only the new rows.
5. **mRoPE / long context.** If you fine-tune for long video/long context, keep the
   mRoPE config consistent; apply YaRN at **inference** time, not by retraining RoPE.
6. **Catastrophic forgetting.** This checkpoint is already post-trained (RL, agentic,
   201 languages). Small-data full-FT will erase those. Prefer LoRA + a low LR
   (1e-4 for LoRA, ~1e-5 if any full-FT) + few epochs, and mix in some general data if
   you need to retain breadth.

---

## 6. Practical fine-tuning recipe (adapting this repo)

The repo already has a working Qwen2.5-VL LoRA SFT stack
(`verifier/model/loader.py`, `verifier/model/collator.py`,
`verifier/train/train_sft.py`). To fine-tune Qwen3.5 instead, change three things:

1. **Loader** — swap the class and processor:
   ```python
   from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
   proc  = AutoProcessor.from_pretrained("qwen3_5_9B")
   model = Qwen3_5ForConditionalGeneration.from_pretrained(
       "qwen3_5_9B", dtype=torch.bfloat16, device_map="auto",
       quantization_config=bnb_4bit_cfg)          # for QLoRA
   ```
2. **LoRA targets** — use the verified `target_modules` from §5.2 (the Qwen2.5 list
   `q_proj…down_proj` only covers the 8 full-attention layers here; add the DeltaNet
   names for the other 24).
3. **Data formatting** — build chat messages with `{"type":"video","video":[frames]}`
   / `{"type":"image","image":img}` + text, render with `apply_chat_template`, and
   call `proc(text=[...], images=..., videos=[frames])` **directly** (the
   `qwen_vl_utils.process_vision_info` helper mishandles this processor's `fps`
   typing — see `scripts/qwen3_5_camera.py`). Mask loss to the assistant span.

Keep the rest (gradient checkpointing, `use_cache=False`, batch 1 + grad-accum,
bf16) as in `train_sft.py` / `SERVER.md`.

### 6.1–6.5 quick reference
- **6.1 Kernels**: `pip install causal-conv1d flash-linear-attention` for the fast
  DeltaNet path (optional; PyTorch fallback works).
- **6.2 LR**: LoRA 1e-4 (warmup 3 %), full-FT ≤1e-5. 1–3 epochs.
- **6.3 Decoding (eval)**: non-thinking general — `temp 0.7, top_p 0.8, top_k 20,
  presence_penalty 1.5`; thinking — `temp 1.0, top_p 0.95, top_k 20`.
- **6.4 Visual tokens**: cap with processor `size`/`longest_edge` for VRAM
  (image `longest_edge` default 16,777,216; for long video the card suggests raising
  the video `longest_edge` to 469,762,048 for higher fps — but lower it for training
  to fit memory).
- **6.5 Long context**: enable YaRN via `rope_parameters` only when needed
  (static YaRN hurts short inputs).

---

## 7. References
- `qwen3_5_9B/README.md` (bundled model card: highlights, model overview, best
  practices, YaRN, long-video settings).
- `qwen3_5_9B/config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`, `chat_template.jinja` (this checkpoint).
- In-repo: `verifier/model/loader.py`, `verifier/model/collator.py`,
  `verifier/train/train_sft.py` (Qwen2.5-VL LoRA pattern to adapt),
  `scripts/qwen3_5_camera.py` (verified inference/processor usage), `SERVER.md`
  (4090 memory settings).
- Architecture facts in §2–§3 were measured from the instantiated model on
  2026-06-06 (transformers 5.9.0).
