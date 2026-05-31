"""Turn Samples into Qwen2.5-VL model inputs with answer-only loss masking.

Builds a chat with the verifier prompt + the window of frames as the user turn,
and the target JSON label as the assistant turn. Loss is computed on the
assistant answer tokens only; visual + prompt tokens are masked with -100.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..schema import VerifierLabel, build_prompt
from ..data.types import Sample

IGNORE_INDEX = -100


def sample_to_messages(sample: Sample) -> list:
    """Qwen chat-format messages for one sample (images referenced by path)."""
    content = [{"type": "image", "image": f} for f in sample.frames]
    content.append({"type": "text", "text": build_prompt(sample.subtask)})
    return [{"role": "user", "content": content}]


def target_text(sample: Sample) -> str:
    return VerifierLabel(sample.status, sample.anomaly, sample.reason).to_json()


@dataclass
class VerifierCollator:
    processor: object
    max_length: int = 8192

    def __call__(self, batch: List[Sample]) -> dict:
        from qwen_vl_utils import process_vision_info
        import torch

        texts, all_images, answer_texts = [], [], []
        for s in batch:
            messages = sample_to_messages(s)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = target_text(s) + self.processor.tokenizer.eos_token
            texts.append(prompt + answer)
            answer_texts.append(answer)
            image_inputs, _ = process_vision_info(messages)
            all_images.append(image_inputs)

        inputs = self.processor(
            text=texts, images=all_images, padding=True, return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        # Mask everything except the assistant answer span at the end of each row.
        for i, ans in enumerate(answer_texts):
            ans_ids = self.processor.tokenizer(ans, add_special_tokens=False)["input_ids"]
            n_ans = len(ans_ids)
            row = labels[i]
            # find the answer span = last n_ans non-pad tokens
            nonpad = (inputs["input_ids"][i] != self.processor.tokenizer.pad_token_id)
            last = int(nonpad.nonzero()[-1]) + 1 if nonpad.any() else row.numel()
            row[: max(0, last - n_ans)] = IGNORE_INDEX

        inputs["labels"] = labels
        return inputs
