#!/usr/bin/env python3
"""
Auto-labeling tool: use KIMI K2.6 to segment unlabeled videos and output
per-frame label files in the same format as existing GT labels.

Output format: one line per frame, "frame_index class_id"
  - class_id 1-11 for the 11 known states
  - class_id 0 for unrecognized/unmapped segments

Input:  data/unlabeled/*.mp4
Output: data/unlabeled/{video_name}_lable.txt
"""

import os
import json
import base64
import time
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from openai import OpenAI, RateLimitError

# Paths
UNLABELED_DIR = Path("/home/kewei/spatial_encoder/streaming-vlm/data/autolabel")
LABEL_DIR = Path("/home/kewei/spatial_encoder/streaming-vlm/data/gift_robot/json_labels")
RESULTS_DIR = Path("/home/kewei/spatial_encoder/streaming-vlm/results")

RATE_LIMIT_COOLDOWN = 5
FPS = 30  # native video fps for per-frame labels

# Few-shot example videos (use existing labeled data)
EXAMPLE_VIDEO_IDS = ["s12", "s20", "070518", "s27", "s31"]

# State name -> class_id mapping
STATE_TO_CLASS = {
    "box free and closed": 1,
    "box fixed and closed": 2,
    "box fixed and opening": 3,
    "box open and toy free": 4,
    "toy in hand and outside box": 5,
    "toy in box and box opened": 6,
    "toy in box and box closing": 7,
    "toy in box and box on table": 8,
    "right hand holding red bag, left hand holding green box": 9,
    "toy in box and box in hand": 10,
    "gift has been packaged": 11,
}


def encode_full_video(video_path: str, fps: int = 4) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-c:v", "libx264", "-preset", "fast",
            "-an", tmp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, "rb") as f:
            video_data = f.read()
        if len(video_data) == 0:
            return None
        return base64.b64encode(video_data).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"    [FFMPEG] ERROR: {e.stderr.decode()}")
        return None
    finally:
        os.unlink(tmp_path)


def get_video_info(video_path: str) -> tuple[float, int]:
    """Return (duration_sec, total_frames)."""
    cmd_dur = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    duration = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip())
    cmd_frames = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    nb_frames = int(subprocess.run(cmd_frames, capture_output=True, text=True).stdout.strip())
    return duration, nb_frames


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_gt_as_example(label_path: str, video_id: str) -> str:
    with open(label_path) as f:
        label_data = json.load(f)
    duration = label_data["duration"]
    segments = [s for s in label_data["segments"] if s["class_id"] != 0]
    lines = [f"Example: {video_id} (duration: {duration:.1f}s, {len(segments)} segments)"]
    for i, seg in enumerate(segments, 1):
        start = format_time(seg["start_time"])
        end = format_time(seg["end_time"])
        dur = seg["end_time"] - seg["start_time"]
        lines.append(
            f"  Segment {i}: {start} - {end} ({dur:.1f}s) | State: {seg['class_name']}"
        )
    return "\n".join(lines)


def build_prompt(video_duration: float, examples: list[str] | None = None) -> str:
    examples_text = "\n\n".join(examples) if examples else ""
    return f"""You are a robot manipulation video analysis assistant. You will receive a COMPLETE video of a robot performing a gift packaging task. Your job is to watch the entire video and autonomously identify all distinct manipulation stages, determine their time boundaries, and describe what happens in each stage.

========== VIDEO INFO ==========
- Total duration: {format_time(video_duration)} ({video_duration:.1f} seconds)
- Frame rate: 4 fps
- This is the FULL video, not a segment.
=================================

========== SCENE OBJECTS ==========
The scene contains the following objects:
- A green box (can be opened/closed, has a pink chain/latch)
- A small toy car
- A red bag
- Robot arm with gripper (left and right)
- A table
===================================

========== TASK DESCRIPTION ==========
The robot performs a gift packaging task. The video progresses through a sequence of distinct **object states**. Each state represents a specific configuration of the scene objects. The typical state sequence is:

1. **Box free and closed** — The green box is on the table, closed, not held by any gripper.
2. **Box fixed and closed** — One gripper holds/fixes the green box in place; the box remains closed.
3. **Box fixed and opening** — The gripper manipulates the pink chain/latch to open the box; the box is transitioning from closed to open.
4. **Box open and toy free** — The green box is open; the toy car is still on the table, not yet picked up.
5. **Toy in hand and outside box** — A gripper is holding the toy car; the toy car has not yet been placed into the box.
6. **Toy in box and box opened** — The toy car has been placed inside the green box; the box is still open.
7. **Toy in box and box closing** — The box lid is being pushed closed with the toy car inside.
8. **Toy in box and box on table** — The box is closed with the toy car inside, resting on the table; grippers may be repositioning.
9. **Right hand holding red bag, left hand holding green box** — The right gripper holds the red bag, the left gripper holds the closed green box, preparing to insert the box into the bag.
10. **Toy in box and box in hand** — The left gripper is moving the closed green box toward or into the red bag.
11. **Gift has been packaged** — The green box (with toy car inside) is inside the red bag; grippers are releasing or have fully retracted.

Not all videos follow this exact sequence. Some states may be very brief, missing, or repeated. You must determine the actual states from the video content. **Segment by object state changes, not by gripper actions**.
======================================

========== REFERENCE EXAMPLES ==========
{examples_text}

KEY OBSERVATIONS from these examples:
- "Box free and closed" (state 1) typically lasts **6-12 seconds**. It ends ONLY when a gripper physically grasps or presses against the box surface. A gripper merely entering the frame or moving toward the box is still state 1 — do NOT cut state 1 short.
- "Box fixed and closed" (state 2) typically lasts 3-23 seconds, but can last up to 50 seconds if the robot pauses. A long static scene where a gripper holds the box but nothing is being manipulated is always state 2, NOT state 3.
- "Toy in box and box closing" (state 7) is often very brief (0.5-2 seconds) but MUST be its own segment — do not merge it with state 6 or state 8.
- "Gift has been packaged" (state 11) begins the moment the green box is fully inside the red bag. State 11 MUST extend to the very last frame of the video — all frames after the grippers release the bag are still state 11.
- Every video should have close to 11 segments for the 11 states.
==========================================

========== OUTPUT FORMAT ==========
Output JSON only. Do not output Markdown, explanations, or code fences.

IMPORTANT: For each segment, you MUST include a "state" field with the exact state name from the list above (e.g., "box free and closed", "toy in box and box closing"). This is critical for automatic label generation.

{{
  "total_segments": <number>,
  "segments": [
    {{
      "id": 1,
      "start_time": "HH:MM:SS.mmm",
      "end_time": "HH:MM:SS.mmm",
      "state": "<exact state name from the 11 states above>",
      "description": "..."
    }}
  ]
}}
==================================="""


def build_messages(video_b64: str, prompt: str) -> list[dict]:
    video_url = f"data:video/mp4;base64,{video_b64}"
    return [
        {
            "role": "system",
            "content": "You are a robot manipulation video analysis assistant. You watch complete videos of robot arms performing tasks, identify distinct manipulation stages, determine their time boundaries, and generate precise descriptions for each stage.",
        },
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def call_llm(client: OpenAI, messages: list[dict]) -> tuple[str, dict]:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            t_start = time.time()
            response = client.chat.completions.create(
                model="kimi-k2.6",
                max_tokens=4096,
                messages=messages,
                extra_body={"thinking": {"type": "disabled"}},
            )
            elapsed = time.time() - t_start
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            raw_content = msg.content
            resp_text = raw_content.strip() if raw_content else ""
            print(f"    [API] {elapsed:.1f}s | tokens: in={usage['input_tokens']:,} out={usage['output_tokens']:,} | stop={finish_reason}")
            if resp_text:
                print(f"    [RAW] {resp_text[:300]}")
            return resp_text, usage
        except RateLimitError:
            wait_time = 60 * (attempt + 1)
            print(f"    [API] RATE LIMITED, waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"    [API] ERROR: {e}, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def parse_time_to_seconds(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


def match_state_name(state_str: str) -> int:
    """Map a model-output state name to class_id. Fuzzy matching."""
    state_lower = state_str.strip().lower()
    # Exact match first
    for name, cid in STATE_TO_CLASS.items():
        if name == state_lower:
            return cid
    # Substring match
    for name, cid in STATE_TO_CLASS.items():
        if name in state_lower or state_lower in name:
            return cid
    # Keyword-based fallback
    keywords = {
        1: ["free", "closed", "not held"],
        2: ["fixed", "closed", "holds", "in place"],
        3: ["opening", "chain", "latch"],
        4: ["open", "toy free", "not yet picked"],
        5: ["toy in hand", "outside box", "holding the toy"],
        6: ["toy in box", "box opened", "still open"],
        7: ["closing", "pushing", "lid"],
        8: ["box on table", "toy in box", "repositioning"],
        9: ["red bag", "holding", "both", "preparing"],
        10: ["box in hand", "moving", "toward", "into the red bag"],
        11: ["packaged", "releasing", "retracted", "inside the red bag"],
    }
    best_cid = 0
    best_score = 0
    for cid, kws in keywords.items():
        score = sum(1 for kw in kws if kw in state_lower)
        if score > best_score:
            best_score = score
            best_cid = cid
    return best_cid if best_score >= 1 else 0


def parse_response(response_text: str) -> list[dict]:
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            segments = data.get("segments", [])
            parsed = []
            for seg in segments:
                start_sec = parse_time_to_seconds(seg.get("start_time", "0"))
                end_sec = parse_time_to_seconds(seg.get("end_time", "0"))
                state_name = seg.get("state", "")
                class_id = match_state_name(state_name)
                # Fallback: try to infer from description if state field missing
                if class_id == 0 and seg.get("description"):
                    class_id = match_state_name(seg["description"])
                parsed.append({
                    "id": seg.get("id", len(parsed) + 1),
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "state": state_name,
                    "class_id": class_id,
                    "description": seg.get("description", ""),
                })
            return parsed
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"    [PARSE] Error: {e}")
    return []


def segments_to_frame_labels(segments: list[dict], total_frames: int, fps: float) -> list[int]:
    """Convert time-based segments to per-frame class_id labels."""
    labels = [0] * total_frames
    for seg in segments:
        start_frame = int(round(seg["start_time_sec"] * fps))
        end_frame = int(round(seg["end_time_sec"] * fps))
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        for f in range(start_frame, end_frame + 1):
            labels[f] = seg["class_id"]
    return labels


def apply_postprocessing(labels: list[int], total_frames: int, fps: float) -> list[int]:
    """Post-processing rules to fix known systematic errors.

    P1: Extend C11 to the last frame — the model consistently ends C11 ~4s early.
    P2: If C1 duration < 5s, extend it to ~8.5s — the model mistakes gripper-entering-
        frame for the C1→C2 boundary; the real boundary is gripper-contacts-box.
    """
    labels = list(labels)

    # P1: Find the last C11 frame and fill to video end
    last_c11 = -1
    for i in range(total_frames - 1, -1, -1):
        if labels[i] == 11:
            last_c11 = i
            break
    if last_c11 >= 0 and last_c11 < total_frames - 1:
        filled = total_frames - 1 - last_c11
        for i in range(last_c11 + 1, total_frames):
            labels[i] = 11
        print(f"  [POST-P1] Extended C11 by {filled} frames ({filled/fps:.1f}s) to video end")
    else:
        print(f"  [POST-P1] C11 already reaches video end, no change")

    # P2: Fix C1 early-cut — find C1 block at start of video
    c1_start, c1_end = -1, -1
    for i, cid in enumerate(labels):
        if cid == 1 and c1_start < 0:
            c1_start = i
        elif c1_start >= 0 and cid != 1:
            c1_end = i - 1
            break
    if c1_start >= 0 and c1_end < 0:
        c1_end = total_frames - 1  # C1 runs to end (unusual)

    if c1_start >= 0:
        c1_duration = (c1_end - c1_start + 1) / fps
        if c1_duration < 5.0:
            target_end = c1_start + int(8.5 * fps)
            # Find end of the C1+C2 block (don't extend into C3+)
            c1_c2_end = c1_end
            for i in range(c1_end + 1, total_frames):
                if labels[i] not in (1, 2):
                    c1_c2_end = i - 1
                    break
            else:
                c1_c2_end = total_frames - 1
            actual_end = min(target_end, c1_c2_end)
            if actual_end > c1_end:
                added = actual_end - c1_end
                for i in range(c1_end + 1, actual_end + 1):
                    labels[i] = 1
                print(f"  [POST-P2] C1 was {c1_duration:.1f}s (<5s), extended by {added} frames ({added/fps:.1f}s)")
            else:
                print(f"  [POST-P2] C1 was {c1_duration:.1f}s but no room to extend (C2 too short)")
        else:
            print(f"  [POST-P2] C1 is {c1_duration:.1f}s (>=5s), no adjustment needed")
    else:
        print(f"  [POST-P2] No C1 segment found, skipping")

    return labels


def write_label_txt(labels: list[int], output_path: Path):
    with open(output_path, "w") as f:
        for i, cid in enumerate(labels):
            f.write(f"{i} {cid}\n")


def main():
    client = OpenAI(
        api_key="sk-Zx2Ovom8nKpDNnsyZGzer4ny5lAof8O88wsSZdrYCmn5hmsd",
        base_url="https://api.moonshot.cn/v1",
    )

    # Create timestamped output directory: {source_folder}_{YYYYMMDD_HHMMSS}
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = UNLABELED_DIR.parent / f"{UNLABELED_DIR.name}_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Labels will be saved to: {output_dir}")

    # Collect unlabeled videos
    videos = sorted(UNLABELED_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {UNLABELED_DIR}")
        return

    # Check which already have labels in the output dir
    todo = []
    for v in videos:
        label_path = output_dir / f"{v.stem}_lable.txt"
        if label_path.exists():
            print(f"  [SKIP] {v.name} — label already exists")
        else:
            todo.append(v)

    # Load few-shot examples
    example_texts = []
    for eid in EXAMPLE_VIDEO_IDS:
        elabel_path = LABEL_DIR / f"{eid}.json"
        if elabel_path.exists():
            example_texts.append(format_gt_as_example(str(elabel_path), eid))
    print(f"[FEW-SHOT] Loaded {len(example_texts)} text examples")

    print("=" * 70)
    print("Auto-Labeling Tool — KIMI K2.6")
    print(f"Videos to process: {len(todo)} / {len(videos)}")
    print("=" * 70)

    for idx, vpath in enumerate(todo):
        duration, total_frames = get_video_info(str(vpath))
        print(f"\n{'─' * 70}")
        print(f"[{idx+1}/{len(todo)}] {vpath.name} | {duration:.1f}s | {total_frames} frames")
        print(f"{'─' * 70}")

        # Encode
        print(f"  Encoding at 4fps...")
        video_b64 = encode_full_video(str(vpath), fps=4)
        if not video_b64:
            print(f"  FAILED to encode, skipping")
            continue
        print(f"  Video size: ~{len(video_b64) * 3 / 4 / 1024:.0f}KB")

        # Build prompt & call API
        prompt = build_prompt(duration, examples=example_texts)
        messages = build_messages(video_b64, prompt)
        response_text, usage = call_llm(client, messages)

        # Parse segments
        segments = parse_response(response_text)
        print(f"  [RESULT] {len(segments)} segments")
        unmapped = 0
        for seg in segments:
            status = f"C{seg['class_id']}" if seg['class_id'] > 0 else "???"
            if seg['class_id'] == 0:
                unmapped += 1
            print(f"    seg {seg['id']}: {seg['start_time_sec']:.1f}s-{seg['end_time_sec']:.1f}s | {status} | {seg['state'][:50]}")

        if unmapped > 0:
            print(f"  [WARN] {unmapped} segments could not be mapped to a known state")

        # Convert to per-frame labels
        labels = segments_to_frame_labels(segments, total_frames, FPS)
        labels = apply_postprocessing(labels, total_frames, FPS)
        labeled_count = sum(1 for l in labels if l > 0)
        print(f"  [FRAMES] {labeled_count}/{total_frames} frames labeled ({100*labeled_count/total_frames:.1f}%)")

        # Write label file
        output_path = output_dir / f"{vpath.stem}_lable.txt"
        write_label_txt(labels, output_path)
        print(f"  [SAVED] {output_path}")

        # Also save raw JSON result for debugging
        json_path = RESULTS_DIR / f"autolabel_{vpath.stem}.json"
        with open(json_path, "w") as f:
            json.dump({
                "video": vpath.name,
                "duration": duration,
                "total_frames": total_frames,
                "segments": segments,
                "raw_response": response_text,
                "usage": usage,
            }, f, indent=2, ensure_ascii=False)

        time.sleep(RATE_LIMIT_COOLDOWN)

    print(f"\n{'=' * 70}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
