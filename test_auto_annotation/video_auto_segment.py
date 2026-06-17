#!/usr/bin/env python3
"""
Use KIMI K2.6 API to automatically segment and caption a full video.
Unlike video_state_judge.py which provides pre-defined segments, this script
sends the entire video and asks the model to infer state transitions,
time boundaries, and descriptions on its own.
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
DATA_DIR = Path("/home/kewei/spatial_encoder/streaming-vlm/data/gift_robot")
LABEL_DIR = DATA_DIR / "json_labels"
RESULTS_DIR = Path("/home/kewei/spatial_encoder/streaming-vlm/results")

# Only test s1-s11
VIDEO_IDS = [f"s{i}" for i in range(1, 12)]

RATE_LIMIT_COOLDOWN = 5  # seconds to wait between API calls

# Few-shot example videos for prompt (must NOT overlap with VIDEO_IDS test set)
EXAMPLE_VIDEO_IDS = ["s12", "s20", "070518", "s27", "s31"]

# Set to a previous results JSON path to resume from checkpoint, or None to start fresh
RESUME_CHECKPOINT = None
# Example: RESUME_CHECKPOINT = "/home/kewei/spatial_encoder/streaming-vlm/results/auto_segment_20260609_120000.json"


def encode_full_video(video_path: str, fps: int = 2) -> str | None:
    """Re-encode the full video at lower fps and return base64-encoded mp4 data."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-c:v", "libx264", "-preset", "fast",
            "-an",
            tmp_path,
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


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_gt_as_example(label_path: str, video_id: str) -> str:
    """Format a ground truth label as a text-only few-shot example."""
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
    """Build prompt for full-video auto-segmentation and captioning."""

    examples_text = "\n\n".join(examples) if examples else ""

    return f"""You are a robot manipulation video analysis assistant. You will receive a COMPLETE video of a robot performing a gift packaging task. Your job is to watch the entire video and autonomously identify all distinct manipulation stages, determine their time boundaries, and describe what happens in each stage.

========== VIDEO INFO ==========
- Total duration: {format_time(video_duration)} ({video_duration:.1f} seconds)
- Frame rate: 2 fps
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

Not all videos follow this exact sequence. Some states may be very brief, missing, or repeated. You must determine the actual states from the video content. **Segment by object state changes, not by gripper actions** — even if a gripper is performing one continuous motion, create a new segment whenever the object state changes (e.g., the moment the toy car enters the box is a state boundary, even if the gripper hasn't stopped moving).
======================================

========== REFERENCE EXAMPLES ==========
Below are correctly segmented examples from similar videos. Study the time boundaries carefully — notice how each state has a meaningful duration and no states are skipped.

{examples_text}

KEY OBSERVATIONS from these examples:
- "Box free and closed" (state 1) typically lasts 4-14 seconds. It ends ONLY when a gripper first contacts/grasps the box. Do NOT split this state into sub-states like "grippers idle" and "gripper approaching" — those are all part of "box free and closed".
- "Box fixed and closed" (state 2) typically lasts 3-23 seconds. It ends ONLY when the gripper starts actively manipulating the pink latch/chain to open the box. A second gripper entering the frame or approaching is NOT a state change — the box is still "fixed and closed".
- "Toy in box and box closing" (state 7) is often very brief (0.5-2 seconds) but MUST be its own segment. Do not merge it with "toy in box and box opened" (state 6) or "toy in box and box on table" (state 8).
- Every video should have close to 11 segments for the 11 states. If you find significantly fewer, you are likely merging states that should be separate.
==========================================

========== SEGMENTATION GUIDELINES ==========
1. **Identify state transitions**: A new segment starts when there is a meaningful change in the manipulation state — e.g., a gripper starts a new action, an object changes its state (opened/closed), or an object is picked up/placed down.

2. **Determine time boundaries**: For each segment, estimate the start and end time as precisely as possible based on when the state transition occurs. Segments should be contiguous (no gaps or overlaps).

3. **Describe each segment**: For each segment, provide:
   - A concise English description (1-2 sentences, max 32 tokens) that **must include the current object states**:
     a. **Green box state**: free/fixed/held, open/closed/opening/closing, on table/in hand/in bag
     b. **Toy car state**: free on table / in gripper hand / inside box
     c. **Red bag state**: on table / held by gripper / containing the box
     d. **Gripper states**: which gripper (left/right) is holding what, or idle
   - Describe what changed compared to the previous segment.
   - Name the main objects explicitly (e.g., "left gripper", "green box", "toy car").

4. **Be thorough**: Do not skip any state, even if it lasts only 1-2 seconds. A brief state like "box closing" (gripper pushing the lid) must be its own segment, separate from "box open" and "box closed". Similarly, "toy in hand but outside box" must be separate from "toy being placed into box".

5. **Do not guess**: Base your segmentation strictly on what you observe in the video.
================================================

========== OUTPUT FORMAT ==========
Output JSON only. Do not output Markdown, explanations, or code fences.

{{
  "total_segments": <number>,
  "segments": [
    {{
      "id": 1,
      "start_time": "HH:MM:SS.mmm",
      "end_time": "HH:MM:SS.mmm",
      "description": "..."
    }},
    {{
      "id": 2,
      "start_time": "HH:MM:SS.mmm",
      "end_time": "HH:MM:SS.mmm",
      "description": "..."
    }}
  ]
}}
==================================="""


def build_messages(video_b64: str, prompt: str) -> list[dict]:
    """Build API messages with full video."""
    video_url = f"data:video/mp4;base64,{video_b64}"
    return [
        {
            "role": "system",
            "content": "You are a robot manipulation video analysis assistant. You watch complete videos of robot arms performing tasks, identify distinct manipulation stages, determine their time boundaries, and generate precise descriptions for each stage.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]


def call_llm(client: OpenAI, messages: list[dict]) -> tuple[str, dict]:
    """Call KIMI K2.6 API with retry and rate limit handling."""
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
            reasoning = getattr(msg, 'reasoning_content', None) or ""

            print(f"    [API] {elapsed:.1f}s | tokens: in={usage['input_tokens']:,} out={usage['output_tokens']:,} | stop={finish_reason}")
            if resp_text:
                print(f"    [RAW] {resp_text[:300]}")
            else:
                print(f"    [WARN] Empty content, reasoning={len(reasoning)} chars")
            return resp_text, usage

        except RateLimitError:
            wait_time = 60 * (attempt + 1)
            print(f"    [API] RATE LIMITED, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"    [API] ERROR: {e}, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [API] FAILED after {max_retries} attempts: {e}")
                raise


def parse_time_to_seconds(time_str: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm or SS.mmm to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


def parse_response(response_text: str) -> list[dict]:
    """Parse the JSON response to extract segments."""
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            segments = data.get("segments", [])
            parsed = []
            for seg in segments:
                start_str = seg.get("start_time", "0")
                end_str = seg.get("end_time", "0")
                parsed.append({
                    "id": seg.get("id", len(parsed) + 1),
                    "start_time": start_str,
                    "end_time": end_str,
                    "start_time_sec": parse_time_to_seconds(start_str),
                    "end_time_sec": parse_time_to_seconds(end_str),
                    "description": seg.get("description", ""),
                })
            return parsed
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"    [PARSE] Error: {e}")
    return []


def main():
    client = OpenAI(
        api_key="sk-Zx2Ovom8nKpDNnsyZGzer4ny5lAof8O88wsSZdrYCmn5hmsd",
        base_url="https://api.moonshot.cn/v1",
    )
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_api_calls = 0
    total_api_time = 0.0
    script_start = time.time()

    # Load checkpoint or start fresh
    if RESUME_CHECKPOINT:
        ckpt_path = Path(RESUME_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_CHECKPOINT}")
        results_path = ckpt_path
        print(f"[RESUME] Loading checkpoint from {ckpt_path.name}")
        with open(ckpt_path) as f:
            all_results = json.load(f)
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f"auto_segment_{run_timestamp}.json"
        all_results = {"config": {
            "model": "kimi-k2.6",
            "thinking": "disabled",
            "max_tokens": 4096,
            "video_fps": 2,
            "rate_limit_cooldown": RATE_LIMIT_COOLDOWN,
            "mode": "full_video_auto_segmentation",
            "few_shot_examples": EXAMPLE_VIDEO_IDS,
            "few_shot_mode": "text_only",
            "timestamp": run_timestamp,
        }, "videos": {}}
        print(f"[NEW RUN] Results will be saved to {results_path.name}")

    # Collect videos
    videos_to_process = []
    for vid in VIDEO_IDS:
        video_path = DATA_DIR / f"{vid}.mp4"
        label_path = LABEL_DIR / f"{vid}.json"
        if not video_path.exists():
            print(f"Skipping {vid}: video not found")
            continue
        if not label_path.exists():
            print(f"Skipping {vid}: label not found")
            continue
        videos_to_process.append(vid)

    # Load few-shot examples (text only)
    example_texts = []
    for eid in EXAMPLE_VIDEO_IDS:
        elabel_path = LABEL_DIR / f"{eid}.json"
        if elabel_path.exists():
            example_texts.append(format_gt_as_example(str(elabel_path), eid))
        else:
            print(f"  [WARN] Example label not found: {eid}")
    print(f"[FEW-SHOT] Loaded {len(example_texts)} text examples: {EXAMPLE_VIDEO_IDS}")

    # Count already done
    done_vids = {v for v in videos_to_process
                 if v in all_results.get("videos", {})
                 and all_results["videos"][v].get("predicted_segments")}

    print("=" * 70)
    print("Full-Video Auto Segmentation - KIMI K2.6")
    print(f"Videos: {len(videos_to_process)} | Already done: {len(done_vids)}")
    print(f"Remaining API calls: ~{len(videos_to_process) - len(done_vids)}")
    print("=" * 70)

    for idx, vid in enumerate(videos_to_process):
        video_path = str(DATA_DIR / f"{vid}.mp4")
        label_path = LABEL_DIR / f"{vid}.json"

        # Skip if already done
        if vid in done_vids:
            n_pred = len(all_results["videos"][vid].get("predicted_segments", []))
            print(f"\n  [{idx+1}/{len(videos_to_process)}] {vid} | SKIPPED ({n_pred} segments predicted)")
            continue

        # Get video duration
        duration = get_video_duration(video_path)

        print(f"\n{'─' * 70}")
        print(f"[{idx+1}/{len(videos_to_process)}] {vid} | duration: {duration:.1f}s")
        print(f"{'─' * 70}")

        # Encode full video
        print(f"  Encoding full video at 2fps...")
        video_b64 = encode_full_video(video_path, fps=2)
        if not video_b64:
            print(f"  FAILED to encode video, skipping")
            continue

        video_size_kb = len(video_b64) * 3 / 4 / 1024
        print(f"  Video size: ~{video_size_kb:.0f}KB")

        # Build prompt and call API
        prompt = build_prompt(duration, examples=example_texts)
        messages = build_messages(video_b64, prompt)

        api_start = time.time()
        response_text, usage = call_llm(client, messages)
        api_elapsed = time.time() - api_start
        total_api_time += api_elapsed
        total_api_calls += 1
        for k in total_usage:
            total_usage[k] += usage[k]

        # Parse response
        predicted_segments = parse_response(response_text)
        print(f"  [RESULT] {len(predicted_segments)} segments predicted")
        for seg in predicted_segments:
            desc_preview = seg["description"][:80] + "..." if len(seg["description"]) > 80 else seg["description"]
            print(f"    seg {seg['id']}: {seg['start_time']}-{seg['end_time']} | {desc_preview}")

        # Load ground truth for comparison
        with open(label_path) as f:
            label_data = json.load(f)
        gt_segments = [s for s in label_data["segments"] if s["class_id"] != 0]

        # Save result
        all_results["videos"][vid] = {
            "duration": duration,
            "predicted_segments": predicted_segments,
            "ground_truth_segments": [
                {
                    "start_time_sec": s["start_time"],
                    "end_time_sec": s["end_time"],
                    "class_id": s["class_id"],
                    "class_name": s["class_name"],
                }
                for s in gt_segments
            ],
            "num_predicted": len(predicted_segments),
            "num_ground_truth": len(gt_segments),
            "llm_raw_response": response_text,
            "usage": usage,
            "api_time_seconds": api_elapsed,
        }

        # Save after each video (resume-safe)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"  [SAVED] GT segments: {len(gt_segments)} | Predicted: {len(predicted_segments)}")

        # Rate limit cooldown
        time.sleep(RATE_LIMIT_COOLDOWN)

    # Final summary
    script_elapsed = time.time() - script_start
    videos_done = all_results.get("videos", {})

    total_gt = sum(v["num_ground_truth"] for v in videos_done.values())
    total_pred = sum(v["num_predicted"] for v in videos_done.values())

    all_results["summary"] = {
        "total_videos": len(videos_done),
        "total_ground_truth_segments": total_gt,
        "total_predicted_segments": total_pred,
        "total_api_calls": total_api_calls,
        "total_input_tokens": total_usage["input_tokens"],
        "total_output_tokens": total_usage["output_tokens"],
        "total_tokens": total_usage["total_tokens"],
        "total_api_time_seconds": total_api_time,
        "total_script_time_seconds": script_elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("AUTO SEGMENTATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Videos processed:        {len(videos_done)}")
    print(f"  Total GT segments:       {total_gt}")
    print(f"  Total predicted segments:{total_pred}")
    print(f"  Total API calls:         {total_api_calls}")
    print(f"  Total input tokens:      {total_usage['input_tokens']:,}")
    print(f"  Total output tokens:     {total_usage['output_tokens']:,}")
    print(f"  Total tokens:            {total_usage['total_tokens']:,}")
    print(f"  Total API time:          {total_api_time:.1f}s")
    print(f"  Total script time:       {script_elapsed:.1f}s")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
