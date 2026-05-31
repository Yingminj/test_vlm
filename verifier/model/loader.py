"""Load Qwen2.5-VL-3B as a verifier backbone, with optional 4-bit + LoRA.

Defaults target a single 24GB RTX 4090: 4-bit weights + LoRA adapters on the LLM,
vision tower frozen. Swap `model_id` for the 7B/InternVL variants if accuracy is
short and VRAM allows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def resolve_model_id(cfg: "ModelConfig") -> str:
    """Allow `VERIFIER_MODEL_ID` (e.g. a local weights dir) to override config."""
    return os.environ.get("VERIFIER_MODEL_ID", cfg.model_id)


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL
    load_in_4bit: bool = True
    bf16: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # LoRA on the language model attention/MLP projections; vision tower frozen.
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    freeze_vision: bool = True
    max_pixels: int = 256 * 28 * 28   # cap tokens/frame for latency + VRAM
    min_pixels: int = 64 * 28 * 28


def load_processor(cfg: ModelConfig):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        resolve_model_id(cfg), min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels,
    )


def load_model(cfg: ModelConfig, for_training: bool = True):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    quant = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolve_model_id(cfg),
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        quantization_config=quant,
        device_map="auto",
    )

    if cfg.freeze_vision and hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    if for_training and cfg.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()

    return model


def load_for_inference(cfg: ModelConfig, adapter_path: Optional[str] = None):
    model = load_model(cfg, for_training=False)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    proc = load_processor(cfg)
    return model, proc
