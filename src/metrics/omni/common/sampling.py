from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_video_rgb_frames
from .results import OMNI_FRAME_USAGE


def sample_video_frames(
    video_path,
    sampling_mode: str = "uniform",
    max_samples: int = 8,
    custom_indices=None,
    return_metadata: bool = False,
):
    """Load a video and return all frames, sampled frames, indices, and optional sampling metadata."""
    video_path = Path(video_path)
    empty_meta = {
        "num_total_frames": 0,
        "num_sampled_frames": 0,
        "sampling_mode_used": sampling_mode,
        "sampled_to_full_index_map": {},
        "metric_frame_usage": deepcopy(OMNI_FRAME_USAGE),
    }
    all_frames = load_video_rgb_frames(video_path)
    if not all_frames:
        return ([], [], [], empty_meta) if return_metadata else ([], [], [])

    total_frames = len(all_frames)

    def _resolve_index(value: Any) -> int:
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return int(round(value * (total_frames - 1)))
        return int(value)

    if custom_indices is not None:
        indices = [_resolve_index(value) for value in custom_indices]
        mode_used = "custom_indices"
    elif sampling_mode == "first_last":
        indices = [0, total_frames - 1]
        mode_used = "first_last"
    elif sampling_mode == "uniform":
        sample_count = max(2, min(int(max_samples), total_frames))
        indices = np.linspace(0, total_frames - 1, sample_count).astype(int).tolist()
        mode_used = "uniform"
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    indices = sorted(set(max(0, min(total_frames - 1, idx)) for idx in indices))
    sampled_frames = [all_frames[idx] for idx in indices]
    metadata = {
        "num_total_frames": total_frames,
        "num_sampled_frames": len(sampled_frames),
        "sampling_mode_used": mode_used,
        "sampled_to_full_index_map": {str(i): int(idx) for i, idx in enumerate(indices)},
        "metric_frame_usage": deepcopy(OMNI_FRAME_USAGE),
    }
    return (all_frames, sampled_frames, indices, metadata) if return_metadata else (all_frames, sampled_frames, indices)
