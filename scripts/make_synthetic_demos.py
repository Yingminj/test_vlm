"""Generate toy demos so the whole pipeline runs end-to-end without real videos.

Creates colored placeholder frames and a demos.jsonl with successful demos
(known completion frames). Failures + semantic-mismatch negatives are then
synthesized by build_dataset. Useful for CI / smoke tests.

Run:
  python scripts/make_synthetic_demos.py --out data --n 12
"""
from __future__ import annotations

import argparse
import os
import random

SUBTASKS = [
    "pick up the red block",
    "place the cup on the shelf",
    "open the top drawer",
    "pour water into the bowl",
]


def make_frames(out_dir: str, n: int, seed: int):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for i in range(n):
        # color drifts over time so frames are not identical
        c = (40 + i * 8 % 200, rng.randint(0, 120), 200 - i * 6 % 180)
        img = Image.new("RGB", (224, 224), c)
        p = os.path.join(out_dir, f"{i:06d}.jpg")
        img.save(p, quality=80)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=12, help="number of demos")
    ap.add_argument("--frames", type=int, default=20, help="frames per demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    frames_root = os.path.join(args.out, "frames")
    demos_path = os.path.join(args.out, "demos.jsonl")
    os.makedirs(args.out, exist_ok=True)

    import json
    with open(demos_path, "w", encoding="utf-8") as f:
        for d in range(args.n):
            subtask = SUBTASKS[d % len(SUBTASKS)]
            n_fr = args.frames + rng.randint(-4, 4)
            fdir = os.path.join(frames_root, f"demo_{d:03d}")
            frames = make_frames(fdir, n_fr, seed=d)
            completion = rng.randint(n_fr // 2, n_fr - 2)
            demo = {
                "demo_id": f"demo_{d:03d}",
                "subtask": subtask,
                "frames": frames,
                "outcome": "success",
                "completion_frame": completion,
                "deviation_frame": None,
                "anomaly": "none",
                "fps": 2.0,
                "source": "synthetic",
                "meta": {},
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.n} demos -> {demos_path}")


if __name__ == "__main__":
    main()
