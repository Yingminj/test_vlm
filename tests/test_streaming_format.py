import random
from verifier.data.types import Demo
from verifier.data.streaming_format import demo_to_samples
from verifier.data.balance import balance_samples, class_histogram


def _success(n=20, comp=12):
    return Demo(demo_id="s", subtask="pick up the red block",
                frames=[f"f{j}.jpg" for j in range(n)],
                outcome="success", completion_frame=comp, fps=2.0)


def _failure(n=20, dev=8, anomaly="slip_drop"):
    return Demo(demo_id="f", subtask="pick up the red block",
                frames=[f"f{j}.jpg" for j in range(n)],
                outcome="failure", deviation_frame=dev, anomaly=anomaly, fps=2.0)


def test_success_labels_latch_to_done():
    samples = demo_to_samples(_success(n=20, comp=12), window=8, stride=1,
                              max_frames_per_demo=None)
    by_t = {s.meta["t"]: s for s in samples}
    assert by_t[0].status == "ongoing"
    assert by_t[11].status == "ongoing"
    assert by_t[12].status == "done"
    assert by_t[19].status == "done"
    assert by_t[12].is_transition is True
    assert by_t[11].is_transition is False


def test_failure_labels_and_anomaly():
    samples = demo_to_samples(_failure(n=20, dev=8), window=8, stride=1,
                              max_frames_per_demo=None)
    by_t = {s.meta["t"]: s for s in samples}
    assert by_t[7].status == "ongoing" and by_t[7].anomaly == "none"
    assert by_t[8].status == "failed" and by_t[8].anomaly == "slip_drop"
    assert by_t[8].is_transition is True


def test_window_respects_size():
    samples = demo_to_samples(_success(n=20, comp=12), window=5, stride=1,
                              max_frames_per_demo=None)
    assert all(len(s.frames) <= 5 for s in samples)


def test_balance_caps_ongoing():
    rng = random.Random(0)
    samples = demo_to_samples(_success(n=40, comp=38), window=8, stride=1,
                              max_frames_per_demo=None)
    hist0 = class_histogram(samples)
    assert hist0["status"]["ongoing"] > hist0["status"].get("done", 0)
    bal = balance_samples(samples, rng, ongoing_ratio=1.0)
    h = class_histogram(bal)
    non_ongoing = h["n"] - h["status"].get("ongoing", 0)
    assert h["status"].get("ongoing", 0) <= max(1, non_ongoing) + h["transitions"]
