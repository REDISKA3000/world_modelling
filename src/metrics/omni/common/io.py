from __future__ import annotations

from pathlib import Path

import cv2


def load_video_rgb_frames(video_path: str | Path) -> list:
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
