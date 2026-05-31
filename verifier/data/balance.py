"""Class balancing for verifier samples.

`ongoing` dominates massively (long plateaus before a single transition). Left
unbalanced, the model learns to always say "ongoing" and never fires. We:
  - keep all `done` / `failed` samples,
  - keep all transition frames,
  - subsample the `ongoing` plateau to a target ratio.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List

from .types import Sample


def balance_samples(
    samples: List[Sample],
    rng: random.Random,
    ongoing_ratio: float = 2.0,
    keep_all_transitions: bool = True,
) -> List[Sample]:
    """Subsample `ongoing` so that
        n_ongoing <= ongoing_ratio * n_non_ongoing.
    Transition frames are always kept.
    """
    ongoing, other, transitions = [], [], []
    for s in samples:
        if keep_all_transitions and s.is_transition:
            transitions.append(s)
        elif s.status == "ongoing":
            ongoing.append(s)
        else:
            other.append(s)

    n_non_ongoing = len(other) + len(transitions)
    cap = int(ongoing_ratio * max(1, n_non_ongoing))
    if len(ongoing) > cap:
        ongoing = rng.sample(ongoing, cap)

    out = ongoing + other + transitions
    rng.shuffle(out)
    return out


def class_histogram(samples: List[Sample]) -> dict:
    status = Counter(s.status for s in samples)
    anomaly = Counter(s.anomaly for s in samples)
    return {
        "n": len(samples),
        "status": dict(status),
        "anomaly": dict(anomaly),
        "transitions": sum(1 for s in samples if s.is_transition),
    }
