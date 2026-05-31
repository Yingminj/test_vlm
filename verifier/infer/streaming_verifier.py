"""Streaming inference wrapper (Part B.5).

Maintains a sliding window of recent frames and, on each check, asks the model
for the verifier label conditioned on the current subtask. Two design points
from the survey are wired in as hooks:

  * memory: a simple in-process sliding window is implemented here; for long
    (15-min) tasks, back this with ReKV (KV offload + retrieval) -- see
    `ReKVMemory` stub. Reset memory at subtask boundaries via `set_subtask`.
  * latency: status is latched once `done`, constrained decoding keeps output
    tiny. Run `step` periodically / event-triggered, not every frame.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from ..schema import VerifierLabel, build_prompt, parse_label, SCHEMA_REGEX


@dataclass
class StreamingVerifier:
    model: object
    processor: object
    window: int = 32
    max_new_tokens: int = 48
    use_constrained_decoding: bool = False  # requires lm-format-enforcer/outlines

    subtask: str = ""
    _frames: Deque = field(default_factory=lambda: collections.deque())
    _latched: Optional[VerifierLabel] = None

    def set_subtask(self, subtask: str) -> None:
        """Begin a new subtask: reset memory + latch (Part B.5)."""
        self.subtask = subtask
        self._frames.clear()
        self._latched = None

    def push(self, frame) -> None:
        self._frames.append(frame)
        while len(self._frames) > self.window:
            self._frames.popleft()

    def step(self, frame=None) -> VerifierLabel:
        """Run one verification check on the current window. `frame` optional."""
        if frame is not None:
            self.push(frame)
        if self._latched is not None and self._latched.status in ("done", "failed"):
            return self._latched  # latched terminal state

        label = self._infer()
        if label.status in ("done", "failed"):
            self._latched = label
        return label

    # ---- backbone call ----
    def _infer(self) -> VerifierLabel:
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": f} for f in self._frames]
        content.append({"type": "text", "text": build_prompt(self.subtask)})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=[image_inputs],
                               padding=True, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.use_constrained_decoding:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._regex_constraint()

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        label = parse_label(decoded)
        return label or VerifierLabel("ongoing", "none", "unparsed; defaulting ongoing")

    def _regex_constraint(self):
        try:
            from lmformatenforcer import RegexParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
            parser = RegexParser(SCHEMA_REGEX)
            return build_transformers_prefix_allowed_tokens_fn(
                self.processor.tokenizer, parser
            )
        except Exception:
            return None


class ReKVMemory:
    """Placeholder for ReKV-backed bounded memory.

    Integrate https://github.com/Becomebright/ReKV here: offload per-clip KV to
    CPU/RAM, retrieve query-relevant entries on each check, and reset at subtask
    boundaries. The StreamingVerifier above uses a naive frame window; for long
    tasks replace `_frames` with retrieval over this store.
    """

    def __init__(self, max_window: int = 64):
        self.max_window = max_window
        raise NotImplementedError("Wire up ReKV; see module docstring.")
