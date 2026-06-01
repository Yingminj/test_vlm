"""CLI: manifest of long videos -> extracted frames + segments.jsonl skeleton.

Manifest is JSONL, one line per video:
  {"video_id": "vid001", "video": "/abs/vid001.mp4",
   "subtasks": ["open the drawer", "pick up the cup", "place cup on shelf"]}

Run:
  python -m verifier.data.ingest_videos --manifest data/manifest.jsonl \
      --frames-root data/frames --out data/segments.jsonl --fps 2.0

Then annotate data/segments.jsonl (scripts/annotate_app.py), and convert with
verifier.data.segments_to_demos.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from .segments import build_skeleton, write_segments
from .video_io import extract_frames


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL: video_id, video, subtasks[]")
    ap.add_argument("--frames-root", default="data/frames")
    ap.add_argument("--out", default="data/segments.jsonl")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--no-even-guess", action="store_true",
                    help="leave start/end empty instead of an even-partition hint")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    os.makedirs(args.frames_root, exist_ok=True)
    segs = []
    for r in rows:
        vid = r["video_id"]
        fdir = os.path.join(args.frames_root, vid)
        print(f"[ingest] {vid}: extracting frames @ {args.fps} fps ...", flush=True)
        frames = extract_frames(r["video"], fdir, fps=args.fps)
        seg = build_skeleton(
            video_id=vid, frames_dir=os.path.abspath(fdir), n_frames=len(frames),
            subtasks=r["subtasks"], fps=args.fps, video=r.get("video"),
            even_guess=not args.no_even_guess,
        )
        segs.append(seg)
        print(f"           {len(frames)} frames, {len(r['subtasks'])} subtasks")

    write_segments(args.out, segs)
    print(f"[ingest] wrote skeleton -> {args.out}  ({len(segs)} videos)")
    print("Next: annotate boundaries/transition in the GUI, then "
          "`python -m verifier.data.segments_to_demos`.")


if __name__ == "__main__":
    main()
