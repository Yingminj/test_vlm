"""Convert demos into windowed verifier Samples with streaming-correct labels.

Labeling rules (VideoLLM-online LIVE style temporal targets):
  success: frames [0, completion_frame)      -> ongoing
           frames [completion_frame, end]     -> done   (latched)
  failure: frames [0, deviation_frame)        -> ongoing
           frames [deviation_frame, end]       -> failed (with anomaly), latched

For each frame index t we emit a window ending at t (the last `window` frames),
and the target is the status *at t*. Transition frames (the first `done`/`failed`
frame) are flagged so the balancer / timing loss can up-weight them.
"""
from __future__ import annotations

from typing import List, Optional

from ..schema import VerifierLabel, trim_reason
from .types import Demo, Sample


def _status_at(demo: Demo, t: int) -> str:
    if demo.outcome == "success":
        if demo.completion_frame is not None and t >= demo.completion_frame:
            return "done"
        return "ongoing"
    # failure
    if demo.deviation_frame is not None and t >= demo.deviation_frame:
        return "failed"
    return "ongoing"


def _reason_for(status: str, anomaly: str, subtask: str) -> str:
    if status == "done":
        return trim_reason(f"{subtask} appears complete")
    if status == "failed":
        templates = {
            "wrong_object": "manipulating the wrong object",
            "slip_drop": "object slipped from gripper",
            "occlusion": "target occluded cannot verify",
            "pose_error": "object placed at incorrect pose",
            "precondition_violated": "stopped before completing subtask",
            "semantic_mismatch": "observed action does not match subtask",
        }
        return trim_reason(templates.get(anomaly, "execution deviated from plan"))
    return trim_reason("subtask still in progress")


def _transition_frame(demo: Demo) -> Optional[int]:
    if demo.outcome == "success":
        return demo.completion_frame
    return demo.deviation_frame


def demo_to_samples(
    demo: Demo,
    window: int = 16,
    stride: int = 1,
    max_frames_per_demo: Optional[int] = None,
) -> List[Sample]:
    """Slice a demo into windowed samples, one per check timestep."""
    samples: List[Sample] = []
    n = len(demo.frames)
    if n == 0:
        return samples
    transition = _transition_frame(demo)
    indices = list(range(0, n, stride))
    if max_frames_per_demo is not None and len(indices) > max_frames_per_demo:
        # keep transition neighborhood + uniform subsample elsewhere
        keep = set(indices[:: max(1, len(indices) // max_frames_per_demo)])
        if transition is not None:
            for d in (-1, 0, 1):
                j = transition + d
                if 0 <= j < n:
                    keep.add(j)
        indices = sorted(keep)

    for t in indices:
        lo = max(0, t - window + 1)
        win = demo.frames[lo : t + 1]
        status = _status_at(demo, t)
        anomaly = demo.anomaly if status == "failed" else "none"
        reason = _reason_for(status, anomaly, demo.subtask)
        # validate via schema
        label = VerifierLabel(status=status, anomaly=anomaly, reason=reason)
        is_trans = transition is not None and t == transition
        samples.append(
            Sample(
                sample_id=f"{demo.demo_id}@{t}",
                subtask=demo.subtask,
                frames=win,
                status=label.status,
                anomaly=label.anomaly,
                reason=label.reason,
                is_transition=is_trans,
                meta={"demo_id": demo.demo_id, "t": t, "fps": demo.fps,
                      "source": demo.source},
            )
        )
    return samples


def demos_to_samples(demos: List[Demo], **kw) -> List[Sample]:
    out: List[Sample] = []
    for d in demos:
        out.extend(demo_to_samples(d, **kw))
    return out
