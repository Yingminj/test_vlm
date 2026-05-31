"""Frame extraction utilities.

Extract frames from a video at a fixed cadence (fps) into a directory of JPEGs,
returning the ordered list of frame paths. Used to turn raw recordings into the
`frames` list a Demo expects. Uses decord if available, else OpenCV.
"""
from __future__ import annotations

import os
from typing import List


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0,
                   quality: int = 90) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    try:
        return _extract_decord(video_path, out_dir, fps, quality)
    except Exception:
        return _extract_opencv(video_path, out_dir, fps, quality)


def _save(img, path: str, quality: int) -> None:
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img.save(path, quality=quality)


def _extract_decord(video_path, out_dir, fps, quality) -> List[str]:
    import decord  # type: ignore
    vr = decord.VideoReader(video_path)
    native = vr.get_avg_fps() or 30.0
    step = max(1, int(round(native / fps)))
    idxs = list(range(0, len(vr), step))
    paths = []
    for i, fi in enumerate(idxs):
        frame = vr[fi].asnumpy()
        p = os.path.join(out_dir, f"{i:06d}.jpg")
        _save(frame, p, quality)
        paths.append(p)
    return paths


def _extract_opencv(video_path, out_dir, fps, quality) -> List[str]:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / fps)))
    paths, raw_idx, kept = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_idx % step == 0:
            p = os.path.join(out_dir, f"{kept:06d}.jpg")
            cv2.imwrite(p, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            paths.append(p)
            kept += 1
        raw_idx += 1
    cap.release()
    return paths
