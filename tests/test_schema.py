import re
from verifier.schema import (
    VerifierLabel, parse_label, build_prompt, trim_reason,
    SCHEMA_REGEX, STATUSES, ANOMALIES,
)


def test_label_roundtrip():
    lab = VerifierLabel("done", "none", "the cup is on the shelf")
    parsed = parse_label(lab.to_json())
    assert parsed.status == "done"
    assert parsed.anomaly == "none"


def test_bad_status_rejected():
    import pytest
    with pytest.raises(ValueError):
        VerifierLabel("maybe", "none", "x")


def test_reason_trim():
    long = " ".join(["w"] * 50)
    assert len(trim_reason(long).split()) <= 12
    assert '"' not in trim_reason('he said "hi"\nthen left')


def test_parse_from_noisy_output():
    txt = 'Sure! {"status": "failed", "anomaly": "slip_drop", "reason": "dropped it"} done'
    lab = parse_label(txt)
    assert lab.status == "failed" and lab.anomaly == "slip_drop"


def test_parse_unknown_anomaly_falls_back():
    lab = parse_label('{"status":"failed","anomaly":"meteor","reason":"x"}')
    assert lab.anomaly == "none"


def test_parse_garbage_returns_none():
    assert parse_label("no json here") is None


def test_schema_regex_matches_valid_output():
    for s in STATUSES:
        out = VerifierLabel(s, ANOMALIES[1] if s == "failed" else "none", "ok reason").to_json()
        assert re.fullmatch(SCHEMA_REGEX, out), out


def test_prompt_mentions_subtask():
    p = build_prompt("pick up the red block")
    assert "pick up the red block" in p
