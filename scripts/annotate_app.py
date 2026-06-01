"""Lightweight GUI to annotate subtask boundaries + transition frames.

Scrub frames, set START / END / TRANSITION per subtask, pick outcome + anomaly,
and save back to segments.jsonl. Reads the skeleton produced by
`verifier.data.ingest_videos`.

Run (after `pip install streamlit`):
  streamlit run scripts/annotate_app.py -- --segments data/segments.jsonl

Then convert: `python -m verifier.data.segments_to_demos --segments data/segments.jsonl`
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from verifier.schema import ANOMALIES  # noqa: E402
from verifier.data.segments import (  # noqa: E402
    read_segments, write_segments, validate, OUTCOMES,
)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="data/segments.jsonl")
    # streamlit passes its own args before `--`; parse only known
    known, _ = ap.parse_known_args()
    return known


def frame_path(frames_dir: str, idx: int) -> str:
    p = os.path.join(frames_dir, f"{idx:06d}.jpg")
    if os.path.exists(p):
        return p
    import glob
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    return files[idx] if 0 <= idx < len(files) else ""


def main():
    args = _args()
    st.set_page_config(page_title="Subtask annotator", layout="wide")
    st.title("Subtask boundary + transition annotator")

    path = st.sidebar.text_input("segments.jsonl path", args.segments)
    if "segs" not in st.session_state or st.sidebar.button("Reload from disk"):
        if os.path.exists(path):
            st.session_state.segs = read_segments(path)
        else:
            st.error(f"not found: {path}")
            st.stop()
    segs = st.session_state.segs

    # progress
    total = sum(len(s.subtasks) for s in segs)
    done = sum(1 for s in segs for t in s.subtasks if t.is_annotated())
    st.sidebar.progress(done / total if total else 0.0,
                        text=f"{done}/{total} subtasks annotated")

    vid_idx = st.sidebar.selectbox(
        "video", range(len(segs)), format_func=lambda i: segs[i].video_id)
    seg = segs[vid_idx]
    sub_idx = st.sidebar.selectbox(
        "subtask", range(len(seg.subtasks)),
        format_func=lambda i: f"#{i} {seg.subtasks[i].subtask}"
                              + (" ✓" if seg.subtasks[i].is_annotated() else ""))
    s = seg.subtasks[sub_idx]

    st.subheader(f"{seg.video_id} — #{sub_idx}: “{s.subtask}”  ({seg.n_frames} frames)")

    col_img, col_ctl = st.columns([3, 2])
    with col_img:
        cur = st.slider("frame", 0, max(0, seg.n_frames - 1),
                        value=s.transition if s.transition is not None
                        else (s.start or 0), key=f"sl_{vid_idx}_{sub_idx}")
        fp = frame_path(seg.frames_dir, cur)
        if fp:
            st.image(fp, caption=f"frame {cur}", use_container_width=True)
        else:
            st.warning(f"missing frame {cur} in {seg.frames_dir}")

    with col_ctl:
        c1, c2, c3 = st.columns(3)
        if c1.button("⏮ START = cur"):
            s.start = cur
        if c2.button("⏹ END = cur"):
            s.end = cur
        if c3.button("🎯 TRANSITION = cur"):
            s.transition = cur

        s.outcome = st.radio("outcome", OUTCOMES,
                             index=OUTCOMES.index(s.outcome) if s.outcome in OUTCOMES else 0,
                             horizontal=True)
        if s.outcome == "failure":
            opts = [a for a in ANOMALIES if a != "none"]
            s.anomaly = st.selectbox("anomaly", opts,
                                     index=opts.index(s.anomaly) if s.anomaly in opts else 0)
        else:
            s.anomaly = "none"

        st.write({"start": s.start, "end": s.end, "transition": s.transition,
                  "outcome": s.outcome, "anomaly": s.anomaly})
        kind = "completion" if s.outcome == "success" else "deviation"
        st.caption(f"TRANSITION is the **{kind}** frame for this subtask.")

        problems = validate(seg)
        mine = [p for p in problems if f"#{sub_idx} " in p]
        if mine:
            st.warning("\n".join(mine))
        elif s.is_annotated():
            st.success("this subtask is valid")

        if st.button("💾 Save segments.jsonl", type="primary"):
            write_segments(path, segs)
            st.success(f"saved {path}")

    with st.expander("whole-file validation"):
        allp = [p for sg in segs for p in validate(sg)]
        st.write("All valid & fully annotated ✅" if not allp else allp)


if __name__ == "__main__":
    main()
