import pytest
from verifier.data.segments import (
    SubtaskSeg, VideoSegments, build_skeleton, validate, segments_to_demos,
    read_segments, write_segments,
)


def _frames(n):
    return [f"/f/{i:06d}.jpg" for i in range(n)]


def _seg():
    return VideoSegments(
        video_id="vid001", frames_dir="/f", n_frames=30, fps=2.0,
        subtasks=[
            SubtaskSeg("open the drawer", start=0, end=14, outcome="success",
                       transition=11, anomaly="none"),
            SubtaskSeg("pick up the cup", start=15, end=29, outcome="failure",
                       transition=22, anomaly="slip_drop"),
        ],
    )


def test_skeleton_even_partition():
    sk = build_skeleton("vid001", "/f", 30, ["a", "b", "c"], even_guess=True)
    assert len(sk.subtasks) == 3
    assert sk.subtasks[0].start == 0
    assert sk.subtasks[-1].end == 29
    # transition left unset -> not annotated
    assert not sk.subtasks[0].is_annotated()


def test_validate_clean():
    assert validate(_seg()) == []


def test_validate_catches_transition_out_of_span():
    seg = _seg()
    seg.subtasks[0].transition = 20  # outside [0,14]
    probs = validate(seg)
    assert any("transition" in p for p in probs)


def test_validate_failure_needs_anomaly():
    seg = _seg()
    seg.subtasks[1].anomaly = "none"
    assert any("non-'none' anomaly" in p for p in validate(seg))


def test_validate_success_must_be_none():
    seg = _seg()
    seg.subtasks[0].anomaly = "slip_drop"
    assert any("anomaly 'none'" in p for p in validate(seg))


def test_segments_to_demos_relative_transition():
    seg = _seg()
    demos = segments_to_demos(seg, frame_paths=_frames(30))
    assert len(demos) == 2
    d0, d1 = demos
    # success: completion relative to span start (11 - 0)
    assert d0.outcome == "success" and d0.completion_frame == 11
    assert d0.deviation_frame is None and d0.anomaly == "none"
    assert len(d0.frames) == 15
    assert d0.demo_id == "vid001__00"
    # failure: deviation relative to span start (22 - 15 = 7), anomaly kept
    assert d1.outcome == "failure" and d1.deviation_frame == 7
    assert d1.anomaly == "slip_drop" and d1.completion_frame is None
    assert d1.demo_id == "vid001__01"


def test_demo_id_prefix_groups_by_video():
    # both subtasks share the video-id prefix so the train/val split keeps a
    # whole video together (build_dataset roots on demo_id.split("__")[0])
    demos = segments_to_demos(_seg(), frame_paths=_frames(30))
    assert {d.demo_id.split("__")[0] for d in demos} == {"vid001"}


def test_segments_to_demos_rejects_unannotated():
    sk = build_skeleton("v", "/f", 10, ["a"], even_guess=True)
    with pytest.raises(ValueError):
        segments_to_demos(sk, frame_paths=_frames(10))


def test_roundtrip_io(tmp_path):
    p = tmp_path / "segments.jsonl"
    write_segments(str(p), [_seg()])
    got = read_segments(str(p))
    assert len(got) == 1
    assert got[0].video_id == "vid001"
    assert got[0].subtasks[1].anomaly == "slip_drop"
