"""Failure synthesis by procedural perturbation of successful demos.

Implements the FailGen / FailCoT / FailSafe idea: instead of hand-annotating
failures per scene, take a *successful* demo and procedurally turn it into a
failure with a known anomaly label and a known deviation frame. This is what
makes the dataset scalable.

Each perturbation operates on a Demo's frame list and metadata only; it does not
edit pixels here (image-level edits such as object removal / occlusion overlays
are delegated to `image_ops`, with hooks left for a real renderer/simulator).
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional

from .types import Demo

# Each perturbation returns a new failed Demo, or None if not applicable.
Perturbation = Callable[[Demo, random.Random], Optional[Demo]]


def _fail_copy(demo: Demo, anomaly: str, deviation_frame: int, tag: str) -> Demo:
    d = copy.deepcopy(demo)
    d.demo_id = f"{demo.demo_id}__{tag}"
    d.outcome = "failure"
    d.anomaly = anomaly
    d.deviation_frame = max(0, min(deviation_frame, len(d.frames) - 1))
    d.completion_frame = None
    d.source = f"perturbed:{tag}"
    d.meta = dict(d.meta, perturbed_from=demo.demo_id)
    return d


def early_stop(demo: Demo, rng: random.Random) -> Optional[Demo]:
    """Truncate the demo before completion: the subtask never finishes."""
    if demo.completion_frame is None or demo.completion_frame < 2:
        return None
    cut = rng.randint(1, demo.completion_frame - 1)
    d = _fail_copy(demo, "precondition_violated", cut, "early_stop")
    d.frames = d.frames[: cut + 1]
    d.deviation_frame = cut  # failure recognizable once motion stalls short of goal
    return d


def slip_drop(demo: Demo, rng: random.Random) -> Optional[Demo]:
    """Object slips/drops after being grasped: failure mid-execution."""
    n = len(demo.frames)
    if n < 3:
        return None
    dev = rng.randint(1, n - 2)
    return _fail_copy(demo, "slip_drop", dev, "slip_drop")


def wrong_object(demo: Demo, rng: random.Random) -> Optional[Demo]:
    """The wrong object is manipulated; deviation visible early."""
    n = len(demo.frames)
    if n < 2:
        return None
    dev = rng.randint(0, max(0, n // 3))
    return _fail_copy(demo, "wrong_object", dev, "wrong_object")


def occlusion(demo: Demo, rng: random.Random) -> Optional[Demo]:
    """Target becomes occluded such that the task cannot be verified/completed."""
    n = len(demo.frames)
    if n < 2:
        return None
    dev = rng.randint(0, n - 1)
    return _fail_copy(demo, "occlusion", dev, "occlusion")


def pose_error(demo: Demo, rng: random.Random) -> Optional[Demo]:
    """Object placed at an incorrect pose: looks near-done but is wrong."""
    if demo.completion_frame is None:
        n = len(demo.frames)
        if n < 2:
            return None
        dev = rng.randint(n // 2, n - 1)
    else:
        dev = demo.completion_frame
    return _fail_copy(demo, "pose_error", dev, "pose_error")


DEFAULT_PERTURBATIONS: Dict[str, Perturbation] = {
    "early_stop": early_stop,
    "slip_drop": slip_drop,
    "wrong_object": wrong_object,
    "occlusion": occlusion,
    "pose_error": pose_error,
}


def synthesize_failures(
    successes: List[Demo],
    rng: random.Random,
    per_demo: int = 1,
    perturbations: Optional[Dict[str, Perturbation]] = None,
) -> List[Demo]:
    """Generate failure demos from successful ones.

    `per_demo` failures are drawn per success, each using a randomly chosen
    applicable perturbation.
    """
    perturbations = perturbations or DEFAULT_PERTURBATIONS
    names = list(perturbations)
    out: List[Demo] = []
    for demo in successes:
        if demo.outcome != "success":
            continue
        rng.shuffle(names)
        made = 0
        for name in names:
            if made >= per_demo:
                break
            failed = perturbations[name](demo, rng)
            if failed is not None:
                out.append(failed)
                made += 1
    return out


def semantic_mismatch_negatives(
    demos: List[Demo],
    rng: random.Random,
    per_demo: int = 1,
) -> List[Demo]:
    """I-FailSense trick: relabel a demo's subtask with a *different* subtask.

    The video shows subtask A while the standing query asks about subtask B, so
    the correct answer is failed / semantic_mismatch from frame 0. Pure
    relabeling of existing clips -- no new annotation.
    """
    subtasks = sorted({d.subtask for d in demos})
    if len(subtasks) < 2:
        return []
    out: List[Demo] = []
    for demo in demos:
        others = [s for s in subtasks if s != demo.subtask]
        if not others:
            continue
        for _ in range(per_demo):
            wrong = rng.choice(others)
            d = copy.deepcopy(demo)
            d.demo_id = f"{demo.demo_id}__semmis"
            d.subtask = wrong
            d.outcome = "failure"
            d.anomaly = "semantic_mismatch"
            d.deviation_frame = 0
            d.completion_frame = None
            d.source = "semantic_mismatch"
            d.meta = dict(d.meta, true_subtask=demo.subtask)
            out.append(d)
    return out
