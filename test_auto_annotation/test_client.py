#!/usr/bin/env python3
"""Smoke test: hit the local server the same way video_auto_label.py does.

  python test_client.py --video /path/to/clip.mp4         # send a real clip
  python test_client.py                                   # text-only ping

Set BASE_URL / MODEL / API_KEY via flags or env. This mirrors the exact OpenAI
client + base64 `video_url` call shape that video_auto_label.py uses, so a green
run here means the auto-labeler will work after you repoint its client.
"""
import argparse
import base64
import os

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen3_5_9B"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    ap.add_argument("--video", default=None, help="path to an mp4 to send as base64")
    ap.add_argument("--prompt", default="Describe what happens in this video in one sentence.")
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    if args.video:
        with open(args.video, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": args.prompt},
        ]
    else:
        content = args.prompt

    resp = client.chat.completions.create(
        model=args.model,
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
        extra_body={"thinking": {"type": "disabled"}},
    )
    print("--- response ---")
    print(resp.choices[0].message.content)
    print("--- usage ---")
    print(resp.usage)


if __name__ == "__main__":
    main()
