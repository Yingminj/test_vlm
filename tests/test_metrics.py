from verifier.eval.metrics import (
    label_metrics, false_rate, anomaly_accuracy, timing_error, full_report,
)


def test_label_metrics_perfect():
    pred = ["done", "ongoing", "done"]
    gold = ["done", "ongoing", "done"]
    m = label_metrics(pred, gold, "done")
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_false_done_rate():
    pred = ["done", "ongoing", "done"]
    gold = ["ongoing", "ongoing", "done"]
    # among gold!=done (2 frames), one predicted done -> 0.5
    assert false_rate(pred, gold, "done") == 0.5


def test_anomaly_accuracy_among_failures():
    ps = ["failed", "failed", "ongoing"]
    gs = ["failed", "failed", "ongoing"]
    pa = ["slip_drop", "wrong_object", "none"]
    ga = ["slip_drop", "slip_drop", "none"]
    r = anomaly_accuracy(pa, ga, gs)
    assert r["n"] == 2
    assert r["overall"] == 0.5


def test_timing_on_time_scores_one():
    r = timing_error([10], [10], [2.0])
    assert r.ars == 1.0
    assert r.mean_abs_error_s == 0.0


def test_timing_early_penalized_more_than_late():
    early = timing_error([6], [10], [2.0])    # 2s early
    late = timing_error([14], [10], [2.0])     # 2s late
    assert early.ars < late.ars                # asymmetric penalty
    assert early.early_error_s == 2.0
    assert late.late_error_s == 2.0


def test_timing_never_fired_scores_zero():
    r = timing_error([None], [10], [2.0])
    assert r.ars == 0.0


def test_full_report_keys():
    rep = full_report(["done"], ["done"], ["none"], ["none"])
    assert {"done", "failed", "false_done_rate", "anomaly", "n"} <= set(rep)
