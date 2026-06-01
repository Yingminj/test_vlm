"""Segment annotation schema + conversion to demos (multi-subtask videos).

Workflow for long videos that contain several subtasks back-to-back:

  manifest.jsonl  --ingest_videos-->  frames/ + segments.jsonl (skeleton)
  segments.jsonl  --(annotate GUI)-->  segments.jsonl (filled)
  segments.jsonl  --segments_to_demos-->  demos.jsonl  --build_dataset-->  shards

A `VideoSegments` holds, per video, the ordered subtasks with their frame
boundaries and the single transition frame (completion for success, deviation
for failure). `segments_to_demos` slices each subtask span into a `Demo` with
the transition index made relative to the span.

All functions here are pure (no video/GUI deps) so they are unit-tested.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from ..schema import ANOMALIES
from .types import Demo

OUTCOMES = ("success", "failure")


@dataclass
class SubtaskSeg:
    subtask: str
    start: Optional[int] = None       # first frame index of this subtask span
    end: Optional[int] = None         # last frame index (inclusive)
    outcome: str = "success"
    transition: Optional[int] = None  # completion (success) or deviation (failure) frame
    anomaly: str = "none"

    def is_annotated(self) -> bool:
        return None not in (self.start, self.end, self.transition)


@dataclass
class VideoSegments:
    video_id: str
    frames_dir: str
    n_frames: int
    fps: float = 2.0
    video: Optional[str] = None
    subtasks: List[SubtaskSeg] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_obj(obj: dict) -> "VideoSegments":
        subs = [SubtaskSeg(**s) for s in obj.get("subtasks", [])]
        obj = {k: v for k, v in obj.items() if k != "subtasks"}
        return VideoSegments(subtasks=subs, **obj)


# --- IO ----------------------------------------------------------------------
def read_segments(path: str) -> List[VideoSegments]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(VideoSegments.from_obj(json.loads(line)))
    return out


def write_segments(path: str, segs: List[VideoSegments]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in segs:
            f.write(s.to_json() + "\n")


# --- skeleton ----------------------------------------------------------------
def build_skeleton(video_id: str, frames_dir: str, n_frames: int,
                   subtasks: List[str], fps: float = 2.0,
                   video: Optional[str] = None,
                   even_guess: bool = True) -> VideoSegments:
    """Create an un-annotated VideoSegments for a video.

    If `even_guess`, pre-fills start/end with an even partition of the frames as a
    starting hint (transition left None so the annotator must set it).
    """
    subs: List[SubtaskSeg] = []
    k = len(subtasks)
    for i, name in enumerate(subtasks):
        if even_guess and k > 0 and n_frames > 0:
            start = (i * n_frames) // k
            end = ((i + 1) * n_frames) // k - 1
            subs.append(SubtaskSeg(subtask=name, start=start, end=max(start, end)))
        else:
            subs.append(SubtaskSeg(subtask=name))
    return VideoSegments(video_id=video_id, frames_dir=frames_dir,
                         n_frames=n_frames, fps=fps, video=video, subtasks=subs)


# --- validation --------------------------------------------------------------
def validate(seg: VideoSegments) -> List[str]:
    """Return a list of human-readable problems ([] if valid & fully annotated)."""
    errs: List[str] = []
    for i, s in enumerate(seg.subtasks):
        tag = f"[{seg.video_id} #{i} '{s.subtask}']"
        if not s.is_annotated():
            errs.append(f"{tag} not fully annotated (start/end/transition)")
            continue
        if not (0 <= s.start <= s.end < seg.n_frames):
            errs.append(f"{tag} bad span start={s.start} end={s.end} n={seg.n_frames}")
        if not (s.start <= s.transition <= s.end):
            errs.append(f"{tag} transition {s.transition} outside [{s.start},{s.end}]")
        if s.outcome not in OUTCOMES:
            errs.append(f"{tag} bad outcome {s.outcome!r}")
        if s.anomaly not in ANOMALIES:
            errs.append(f"{tag} bad anomaly {s.anomaly!r}")
        if s.outcome == "failure" and s.anomaly == "none":
            errs.append(f"{tag} failure must have a non-'none' anomaly")
        if s.outcome == "success" and s.anomaly != "none":
            errs.append(f"{tag} success must have anomaly 'none'")
    return errs


# --- conversion --------------------------------------------------------------
def _frame_paths(frames_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))


def segments_to_demos(seg: VideoSegments,
                      frame_paths: Optional[List[str]] = None) -> List[Demo]:
    """Convert one annotated video into per-subtask Demos.

    `frame_paths` may be injected (tests); otherwise read from `frames_dir`.
    Raises ValueError if the segmentation is invalid.
    """
    problems = validate(seg)
    if problems:
        raise ValueError("; ".join(problems))
    frames = frame_paths if frame_paths is not None else _frame_paths(seg.frames_dir)
    if len(frames) < seg.n_frames:
        raise ValueError(
            f"{seg.video_id}: found {len(frames)} frames < declared {seg.n_frames}")

    demos: List[Demo] = []
    for i, s in enumerate(seg.subtasks):
        span = frames[s.start: s.end + 1]
        rel = s.transition - s.start
        if not (0 <= rel < len(span)):
            raise ValueError(f"{seg.video_id} #{i}: relative transition {rel} out of span")
        demos.append(Demo(
            demo_id=f"{seg.video_id}__{i:02d}",   # video-id prefix -> no split leakage
            subtask=s.subtask,
            frames=span,
            outcome=s.outcome,
            completion_frame=rel if s.outcome == "success" else None,
            deviation_frame=rel if s.outcome == "failure" else None,
            anomaly=s.anomaly if s.outcome == "failure" else "none",
            fps=seg.fps,
            source="real",
            meta={"video_id": seg.video_id, "subtask_index": i},
        ))
    return demos
