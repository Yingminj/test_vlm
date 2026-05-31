"""SFT entrypoint for the streaming task-state verifier (Part B.3, Stages A/B/C).

The three stages share this script; they differ only in data and a few knobs
(set via --config). Stage C ("streaming/timing") uses transition-upweighted
sampling so the model learns *when* to flip ongoing->done/failed; the principled
per-frame "keep-ongoing token" objective (VideoLLM-online LIVE) is described in
README and hooked via `--timing-upweight` here.

Run:
  python -m verifier.train.train_sft --config configs/train.yaml --stage b
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import yaml


@dataclass
class TrainArgs:
    train_jsonl: str
    val_jsonl: str
    output_dir: str
    epochs: float = 2.0
    lr: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    timing_upweight: float = 1.0   # >1 oversamples transition frames (Stage C)
    max_length: int = 8192
    save_steps: int = 200
    logging_steps: int = 10
    max_steps: int = 0             # >0 caps steps (smoke runs); 0 = full epochs
    seed: int = 0


def _oversample_transitions(samples, factor, seed):
    import random
    if factor <= 1.0:
        return samples
    rng = random.Random(seed)
    extra = [s for s in samples if s.is_transition]
    reps = int(factor) - 1
    out = list(samples) + extra * reps
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["a", "b", "c"], default="b")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_raw = raw.get("model", {})
    stage_raw = raw.get("stages", {}).get(args.stage, {})
    targs = TrainArgs(**{**raw.get("train", {}), **stage_raw})

    # ---- lazy heavy imports (so data/eval/tests don't need torch) ----
    import torch
    from transformers import Trainer, TrainingArguments
    from ..model.loader import ModelConfig, load_model, load_processor
    from ..model.collator import VerifierCollator
    from ..data.types import read_samples

    mcfg = ModelConfig(**model_raw)
    processor = load_processor(mcfg)
    model = load_model(mcfg, for_training=True)
    # memory: checkpointing + no kv-cache during training (see 48GB-4090 notes)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train = read_samples(targs.train_jsonl)
    train = _oversample_transitions(train, targs.timing_upweight, targs.seed)
    val = read_samples(targs.val_jsonl)

    collator = VerifierCollator(processor, max_length=targs.max_length)

    targ = TrainingArguments(
        output_dir=targs.output_dir,
        num_train_epochs=targs.epochs,
        max_steps=targs.max_steps if targs.max_steps > 0 else -1,
        per_device_train_batch_size=targs.batch_size,
        per_device_eval_batch_size=targs.batch_size,
        gradient_accumulation_steps=targs.grad_accum,
        learning_rate=targs.lr,
        warmup_ratio=targs.warmup_ratio,
        bf16=mcfg.bf16,
        logging_steps=targs.logging_steps,
        save_steps=targs.save_steps,
        eval_strategy="steps" if val else "no",
        eval_steps=targs.save_steps,
        # checkpointing enabled on the model above (use_reentrant=False)
        report_to=[],
        remove_unused_columns=False,
        seed=targs.seed,
    )

    trainer = Trainer(
        model=model,
        args=targ,
        train_dataset=train,
        eval_dataset=val or None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(targs.output_dir)
    processor.save_pretrained(targs.output_dir)
    print(f"[done] adapter saved to {targs.output_dir}")


if __name__ == "__main__":
    main()
