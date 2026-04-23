from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
from scipy import linalg


DEFAULT_NUM_FRAMES = 16
DEFAULT_RESIZE = (224, 224)


def normalize_resize_to(resize_to: Tuple[int, int] | Iterable[int] | None) -> tuple[int, int]:
    if resize_to is None:
        return DEFAULT_RESIZE

    resize_tuple = tuple(int(v) for v in resize_to)
    if len(resize_tuple) != 2:
        raise ValueError(f"resize_to must contain exactly two integers, got: {resize_to}")
    return resize_tuple


def resolve_num_frames(num_frames: int | None) -> int:
    resolved = int(num_frames or DEFAULT_NUM_FRAMES)
    if resolved <= 0:
        raise ValueError(f"num_frames must be positive, got: {num_frames}")
    return resolved


def sample_uniform_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got: {total_frames}")
    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def get_video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def load_video_tensor_uniform(
    video_path: str | Path,
    num_frames: int,
    resize_to: tuple[int, int],
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count: {video_path}")

    indices = sample_uniform_indices(total_frames=total_frames, num_frames=num_frames)
    target_set = set(indices.tolist())

    frames = []
    current_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if current_idx in target_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize_to)
            frames.append(frame)

        current_idx += 1
        if len(frames) == num_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    arr = np.stack(frames, axis=0)
    x = torch.from_numpy(arr).float() / 255.0
    return x.permute(3, 0, 1, 2).unsqueeze(0)


def load_video_tensor_subclip(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
    resize_to: tuple[int, int],
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count: {video_path}")

    start_frame = max(0, min(int(start_frame), max(total_frames - 1, 0)))
    end_frame = min(start_frame + num_frames, total_frames)

    frames = []
    current_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if start_frame <= current_idx < end_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize_to)
            frames.append(frame)

        current_idx += 1
        if current_idx >= end_frame:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from subclip: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    arr = np.stack(frames, axis=0)
    x = torch.from_numpy(arr).float() / 255.0
    return x.permute(3, 0, 1, 2).unsqueeze(0)


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
