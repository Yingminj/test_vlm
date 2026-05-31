"""Evaluate a trained verifier on a val sample shard (Part B.6).

Runs the model per sample (frame-level) and also reconstructs per-demo firing
frames for timing metrics. Writes a JSON report.

Run:
  python -m verifier.eval.evaluate --config configs/train.yaml \
      --adapter runs/stage_c --val data/processed/val.jsonl --out runs/report.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

import yaml

from ..data.types import read_samples
from .metrics import full_report, timing_error


def _first_fire(frame_status: List[tuple]) -> Optional[int]:
    """Given [(t, status)...] sorted by t, return first t with terminal status."""
    for t, s in sorted(frame_status):
        if s in ("done", "failed"):
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    from ..model.loader import ModelConfig, load_for_inference
    from ..infer.streaming_verifier import StreamingVerifier

    mcfg = ModelConfig(**raw.get("model", {}))
    model, processor = load_for_inference(mcfg, adapter_path=args.adapter)
    verifier = StreamingVerifier(model=model, processor=processor)

    samples = read_samples(args.val)
    if args.limit:
        samples = samples[: args.limit]

    pred_status, gold_status, pred_anom, gold_anom = [], [], [], []
    per_demo_pred: Dict[str, list] = defaultdict(list)
    per_demo_gold: Dict[str, list] = defaultdict(list)
    per_demo_fps: Dict[str, float] = {}
    latencies = []

    for s in samples:
        verifier.set_subtask(s.subtask)
        for fr in s.frames:
            verifier.push(fr)
        t0 = time.time()
        label = verifier.step()
        latencies.append(time.time() - t0)

        pred_status.append(label.status)
        gold_status.append(s.status)
        pred_anom.append(label.anomaly)
        gold_anom.append(s.anomaly)

        did = s.meta.get("demo_id", s.sample_id)
        t = s.meta.get("t", 0)
        per_demo_pred[did].append((t, label.status))
        per_demo_gold[did].append((t, s.status))
        per_demo_fps[did] = s.meta.get("fps", 2.0)

    demos = list(per_demo_gold)
    timing = timing_error(
        [ _first_fire(per_demo_pred[d]) for d in demos ],
        [ _first_fire(per_demo_gold[d]) for d in demos ],
        [ per_demo_fps[d] for d in demos ],
    )

    lat_sorted = sorted(latencies)
    report = {
        "classification": full_report(pred_status, gold_status, pred_anom, gold_anom),
        "timing": timing.__dict__,
        "latency_s": {
            "p50": lat_sorted[len(lat_sorted)//2] if lat_sorted else None,
            "p99": lat_sorted[int(len(lat_sorted)*0.99)-1] if lat_sorted else None,
            "mean": sum(latencies)/len(latencies) if latencies else None,
        },
        "n_samples": len(samples),
        "n_demos": len(demos),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
