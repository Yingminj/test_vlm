"""Verifier metrics (Part B.6).

  * success detection: precision/recall/F1 of `done` vs the rest; crucially the
    FALSE-`done` rate (premature subtask advance is the costly closed-loop error)
  * failure detection: precision/recall/F1 of `failed`; false-`failed` on clean runs
  * anomaly: per-type accuracy among true failures
  * timing: asymmetric early/late completion-time error in seconds (ARS-style)

All functions are pure (no torch) so they run anywhere and are unit-tested.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def label_metrics(pred: Sequence[str], gold: Sequence[str], positive: str) -> Dict[str, float]:
    assert len(pred) == len(gold)
    tp = sum(p == positive and g == positive for p, g in zip(pred, gold))
    fp = sum(p == positive and g != positive for p, g in zip(pred, gold))
    fn = sum(p != positive and g == positive for p, g in zip(pred, gold))
    return _prf(tp, fp, fn)


def false_rate(pred: Sequence[str], gold: Sequence[str], label: str) -> float:
    """Fraction of non-`label` gold frames predicted as `label`."""
    neg = [(p, g) for p, g in zip(pred, gold) if g != label]
    if not neg:
        return 0.0
    return sum(p == label for p, _ in neg) / len(neg)


def anomaly_accuracy(pred_anom: Sequence[str], gold_anom: Sequence[str],
                     gold_status: Sequence[str]) -> Dict[str, float]:
    """Anomaly classification accuracy among true failures, plus per-type."""
    idx = [i for i, s in enumerate(gold_status) if s == "failed"]
    if not idx:
        return {"overall": 0.0, "n": 0}
    correct = sum(pred_anom[i] == gold_anom[i] for i in idx)
    per_type_tot: Dict[str, int] = defaultdict(int)
    per_type_ok: Dict[str, int] = defaultdict(int)
    for i in idx:
        per_type_tot[gold_anom[i]] += 1
        per_type_ok[gold_anom[i]] += int(pred_anom[i] == gold_anom[i])
    per_type = {k: per_type_ok[k] / per_type_tot[k] for k in per_type_tot}
    return {"overall": correct / len(idx), "n": len(idx), "per_type": per_type}


@dataclass
class TimingResult:
    mean_abs_error_s: float
    early_error_s: float   # mean seconds fired before the true transition
    late_error_s: float    # mean seconds fired after
    ars: float             # readiness score in [0,1], asymmetric penalties
    n: int


def timing_error(
    pred_fire_frame: Sequence[Optional[int]],
    gold_fire_frame: Sequence[Optional[int]],
    fps: Sequence[float],
    early_penalty: float = 2.0,
    late_penalty: float = 1.0,
    tol_s: float = 0.5,
) -> TimingResult:
    """Per-demo timing error. `*_fire_frame` is the frame index where the model /
    ground truth first reports the terminal status (done/failed); None = never.

    ARS: 1.0 if within tol; otherwise exp decay with asymmetric penalty
    (premature firing penalized harder -- StreamReady style).
    """
    import math

    errs, earlies, lates, scores = [], [], [], []
    for pf, gf, f in zip(pred_fire_frame, gold_fire_frame, fps):
        if gf is None:
            continue
        if pf is None:
            scores.append(0.0)
            errs.append(float("nan"))
            continue
        dt = (pf - gf) / max(f, 1e-6)  # seconds; negative = early
        errs.append(abs(dt))
        if dt < 0:
            earlies.append(-dt)
        else:
            lates.append(dt)
        if abs(dt) <= tol_s:
            scores.append(1.0)
        else:
            pen = early_penalty if dt < 0 else late_penalty
            scores.append(math.exp(-pen * (abs(dt) - tol_s)))

    finite = [e for e in errs if e == e]  # drop nan
    n = len(scores)
    return TimingResult(
        mean_abs_error_s=sum(finite) / len(finite) if finite else float("nan"),
        early_error_s=sum(earlies) / len(earlies) if earlies else 0.0,
        late_error_s=sum(lates) / len(lates) if lates else 0.0,
        ars=sum(scores) / n if n else 0.0,
        n=n,
    )


def full_report(pred_status, gold_status, pred_anom, gold_anom) -> Dict:
    return {
        "done": label_metrics(pred_status, gold_status, "done"),
        "failed": label_metrics(pred_status, gold_status, "failed"),
        "false_done_rate": false_rate(pred_status, gold_status, "done"),
        "false_failed_rate": false_rate(pred_status, gold_status, "failed"),
        "anomaly": anomaly_accuracy(pred_anom, gold_anom, gold_status),
        "n": len(gold_status),
    }
