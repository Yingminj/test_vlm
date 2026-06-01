"""CLI: annotated segments.jsonl -> demos.jsonl (feed to build_dataset).

Validates each video's annotation, slices subtasks into Demos, and writes
demos.jsonl. Use --skip-invalid to drop unannotated/invalid videos with a
warning instead of failing.

Run:
  python -m verifier.data.segments_to_demos --segments data/segments.jsonl \
      --out data/demos.jsonl
"""
from __future__ import annotations

import argparse
from typing import List

from .segments import read_segments, segments_to_demos, validate
from .types import Demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", default="data/demos.jsonl")
    ap.add_argument("--skip-invalid", action="store_true")
    args = ap.parse_args()

    segs = read_segments(args.segments)
    demos: List[Demo] = []
    n_ok = n_skip = 0
    for seg in segs:
        problems = validate(seg)
        if problems:
            msg = f"[skip] {seg.video_id}: " + "; ".join(problems)
            if args.skip_invalid:
                print(msg)
                n_skip += 1
                continue
            raise SystemExit(msg + "\n(use --skip-invalid to drop these)")
        demos.extend(segments_to_demos(seg))
        n_ok += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for d in demos:
            f.write(d.to_json() + "\n")

    n_succ = sum(d.outcome == "success" for d in demos)
    n_fail = sum(d.outcome == "failure" for d in demos)
    print(f"[ok] {n_ok} videos -> {len(demos)} subtask demos "
          f"({n_succ} success / {n_fail} failure); {n_skip} videos skipped")
    print(f"     wrote {args.out}. Next: python -m verifier.data.build_dataset "
          f"--demos {args.out} --out data/processed")


if __name__ == "__main__":
    main()
