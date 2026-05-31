import random
from verifier.data.types import Demo
from verifier.data.perturb import (
    synthesize_failures, semantic_mismatch_negatives, DEFAULT_PERTURBATIONS,
)


def _demo(i=0, n=20, comp=12, subtask="pick up the red block"):
    return Demo(
        demo_id=f"demo_{i}", subtask=subtask,
        frames=[f"f{j}.jpg" for j in range(n)],
        outcome="success", completion_frame=comp, fps=2.0,
    )


def test_synthesize_failures_labels_and_provenance():
    rng = random.Random(0)
    fails = synthesize_failures([_demo()], rng, per_demo=3)
    assert len(fails) >= 1
    for f in fails:
        assert f.outcome == "failure"
        assert f.anomaly in DEFAULT_PERTURBATIONS or f.anomaly == "precondition_violated"
        assert f.deviation_frame is not None
        assert 0 <= f.deviation_frame < len(f.frames)
        assert f.meta["perturbed_from"] == "demo_0"


def test_early_stop_truncates():
    rng = random.Random(1)
    from verifier.data.perturb import early_stop
    d = early_stop(_demo(comp=15, n=20), rng)
    assert d is not None
    assert len(d.frames) <= 15
    assert d.anomaly == "precondition_violated"


def test_semantic_mismatch_needs_two_subtasks():
    rng = random.Random(0)
    demos = [_demo(0, subtask="A"), _demo(1, subtask="B")]
    neg = semantic_mismatch_negatives(demos, rng, per_demo=1)
    assert len(neg) == 2
    for n in neg:
        assert n.anomaly == "semantic_mismatch"
        assert n.deviation_frame == 0
        assert n.subtask != n.meta["true_subtask"]


def test_semantic_mismatch_single_subtask_empty():
    rng = random.Random(0)
    assert semantic_mismatch_negatives([_demo(0, subtask="A")], rng) == []
