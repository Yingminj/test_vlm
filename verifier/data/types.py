"""Core data structures for demos and training samples.

A *Demo* is the minimal human-annotated unit: one video of one subtask attempt,
plus a single key timestamp (completion frame for successes, deviation frame for
failures). Everything else in the pipeline is derived or synthesized.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Demo:
    """One subtask attempt.

    Attributes
    ----------
    demo_id: unique id.
    subtask: natural-language subtask description (the standing query).
    frames: ordered list of frame image paths (extracted at the check cadence).
    outcome: "success" or "failure".
    completion_frame: index into `frames` where the subtask becomes `done`
        (success). For frames < this index the status is `ongoing`.
    deviation_frame: index where a failure first manifests (failure). For
        frames < this index the status is `ongoing`, at/after it is `failed`.
    anomaly: anomaly label for failures (see schema.ANOMALIES); "none" otherwise.
    fps: cadence frames were sampled at (for timing-error reporting in seconds).
    source: provenance tag (e.g. "teleop", "scripted", "perturbed:slip_drop").
    """

    demo_id: str
    subtask: str
    frames: List[str]
    outcome: str = "success"
    completion_frame: Optional[int] = None
    deviation_frame: Optional[int] = None
    anomaly: str = "none"
    fps: float = 2.0
    source: str = "teleop"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Demo":
        return Demo(**json.loads(line))


@dataclass
class Sample:
    """A single training/eval example: a window of frames + the target label."""

    sample_id: str
    subtask: str
    frames: List[str]          # window of frame paths, oldest -> newest
    status: str                # target status at the newest frame
    anomaly: str
    reason: str
    is_transition: bool = False  # True if this window's newest frame is the flip point
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Sample":
        return Sample(**json.loads(line))


def read_demos(path: str) -> List[Demo]:
    out: List[Demo] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Demo.from_json(line))
    return out


def write_samples(path: str, samples: List[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json() + "\n")


def read_samples(path: str) -> List[Sample]:
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Sample.from_json(line))
    return out
