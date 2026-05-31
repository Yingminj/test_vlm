"""Output schema, anomaly taxonomy, and (de)serialization for the task-state verifier.

The verifier answers, at each check, conditioned on the current subtask:

    "Has <subtask> been completed? Any anomaly? Answer:
     status in {ongoing, done, failed}, anomaly, one-line reason."

The model emits a single compact JSON object. `status` and `anomaly` are closed
sets so we can use constrained decoding and never produce an invalid label.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

# --- Closed label sets --------------------------------------------------------
STATUSES = ("ongoing", "done", "failed")

# Anomaly taxonomy. Extend here; data labels and trigger signals follow from it.
ANOMALIES = (
    "none",
    "wrong_object",
    "slip_drop",
    "occlusion",
    "pose_error",
    "precondition_violated",
    "semantic_mismatch",
)

MAX_REASON_WORDS = 12

# Prompt template. {subtask} is filled per check.
VERIFIER_PROMPT = (
    "You monitor a robot executing a subtask. Current subtask: \"{subtask}\".\n"
    "Has the subtask been completed? Is there any anomaly?\n"
    "Reply with ONE JSON object and nothing else, with keys: "
    "status (one of ongoing|done|failed), "
    "anomaly (one of " + "|".join(ANOMALIES) + "), "
    "reason (<= {max_words} words)."
).format

# Regex usable for grammar-constrained decoding (e.g. outlines / lm-format-enforcer).
_REASON = r'[^"\\]{0,80}'
STATUS_RE = "|".join(STATUSES)
ANOMALY_RE = "|".join(ANOMALIES)
SCHEMA_REGEX = (
    r'\{"status": "(?:' + STATUS_RE + r')", '
    r'"anomaly": "(?:' + ANOMALY_RE + r')", '
    r'"reason": "' + _REASON + r'"\}'
)


@dataclass
class VerifierLabel:
    status: str
    anomaly: str = "none"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}; expected one of {STATUSES}")
        if self.anomaly not in ANOMALIES:
            raise ValueError(f"bad anomaly {self.anomaly!r}; expected one of {ANOMALIES}")
        self.reason = trim_reason(self.reason)

    def to_json(self) -> str:
        # Stable key order matches SCHEMA_REGEX exactly.
        return json.dumps(
            {"status": self.status, "anomaly": self.anomaly, "reason": self.reason},
            ensure_ascii=False,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def trim_reason(reason: str, max_words: int = MAX_REASON_WORDS) -> str:
    reason = re.sub(r'["\\\n\r]', " ", (reason or "")).strip()
    words = reason.split()
    return " ".join(words[:max_words])


def build_prompt(subtask: str) -> str:
    return VERIFIER_PROMPT(subtask=subtask, max_words=MAX_REASON_WORDS)


def parse_label(text: str) -> Optional[VerifierLabel]:
    """Best-effort parse of model output into a VerifierLabel.

    Returns None if no valid object can be recovered (caller decides fallback,
    e.g. treat as 'ongoing').
    """
    if not text:
        return None
    # Grab the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    status = str(obj.get("status", "")).strip().lower()
    anomaly = str(obj.get("anomaly", "none")).strip().lower()
    if status not in STATUSES:
        return None
    if anomaly not in ANOMALIES:
        anomaly = "none"
    return VerifierLabel(status=status, anomaly=anomaly, reason=str(obj.get("reason", "")))
