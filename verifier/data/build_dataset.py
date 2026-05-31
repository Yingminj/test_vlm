"""CLI: turn a demos.jsonl into train/val verifier sample shards.

Pipeline (Part B.2):
  1. read successful + failure demos
  2. synthesize failures from successes (procedural perturbation)
  3. add semantic-mismatch negatives
  4. slice everything into windowed, streaming-labeled samples
  5. class-balance (subsample the ongoing plateau)
  6. split train/val by demo (no leakage) and write jsonl

Run:
  python -m verifier.data.build_dataset --demos data/demos.jsonl --out data/processed
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

from .types import Demo, read_demos, write_samples
from .perturb import synthesize_failures, semantic_mismatch_negatives
from .streaming_format import demos_to_samples
from .balance import balance_samples, class_histogram


def build(
    demos: List[Demo],
    seed: int = 0,
    failures_per_success: int = 1,
    semantic_mismatch_per_demo: int = 1,
    window: int = 16,
    stride: int = 1,
    max_frames_per_demo: int = 64,
    ongoing_ratio: float = 2.0,
    val_fraction: float = 0.1,
):
    rng = random.Random(seed)

    successes = [d for d in demos if d.outcome == "success"]
    failures = [d for d in demos if d.outcome == "failure"]

    synth = synthesize_failures(successes, rng, per_demo=failures_per_success)
    semmis = semantic_mismatch_negatives(demos, rng, per_demo=semantic_mismatch_per_demo)
    all_demos = demos + synth + semmis

    # split by *root* demo id to avoid leakage between perturbed variants
    def root(d: Demo) -> str:
        return d.meta.get("perturbed_from") or d.meta.get("true_subtask_demo") or d.demo_id.split("__")[0]

    roots = sorted({root(d) for d in all_demos})
    rng.shuffle(roots)
    n_val = max(1, int(len(roots) * val_fraction)) if roots else 0
    val_roots = set(roots[:n_val])

    train_demos = [d for d in all_demos if root(d) not in val_roots]
    val_demos = [d for d in all_demos if root(d) in val_roots]

    train = balance_samples(
        demos_to_samples(train_demos, window=window, stride=stride,
                         max_frames_per_demo=max_frames_per_demo),
        rng, ongoing_ratio=ongoing_ratio,
    )
    # val is NOT balanced -- evaluate on the real distribution
    val = demos_to_samples(val_demos, window=window, stride=stride,
                          max_frames_per_demo=max_frames_per_demo)
    rng.shuffle(val)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--failures-per-success", type=int, default=1)
    ap.add_argument("--semantic-mismatch-per-demo", type=int, default=1)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames-per-demo", type=int, default=64)
    ap.add_argument("--ongoing-ratio", type=float, default=2.0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    args = ap.parse_args()

    demos = read_demos(args.demos)
    train, val = build(
        demos,
        seed=args.seed,
        failures_per_success=args.failures_per_success,
        semantic_mismatch_per_demo=args.semantic_mismatch_per_demo,
        window=args.window,
        stride=args.stride,
        max_frames_per_demo=args.max_frames_per_demo,
        ongoing_ratio=args.ongoing_ratio,
        val_fraction=args.val_fraction,
    )

    os.makedirs(args.out, exist_ok=True)
    write_samples(os.path.join(args.out, "train.jsonl"), train)
    write_samples(os.path.join(args.out, "val.jsonl"), val)
    stats = {"train": class_histogram(train), "val": class_histogram(val)}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
