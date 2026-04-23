from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .embeddings import omni_gray


def default_flow_backend(frame_a: np.ndarray, frame_b: np.ndarray):
    prev = omni_gray(frame_a)
    curr = omni_gray(frame_b)
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow, np.linalg.norm(flow, axis=2)


def _normalize_flow_backend_output(output: Any):
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError("flow backend must return (flow, magnitude)")
    flow, magnitude = output
    return np.asarray(flow), np.asarray(magnitude)


def flow_and_magnitude(frame_a: np.ndarray, frame_b: np.ndarray, flow_backend=None):
    backend = flow_backend or default_flow_backend
    if hasattr(backend, "flow_and_magnitude"):
        return _normalize_flow_backend_output(backend.flow_and_magnitude(frame_a=frame_a, frame_b=frame_b))
    return _normalize_flow_backend_output(backend(frame_a, frame_b))


def motion_series(frames, flow_backend=None):
    flows = []
    magnitudes = []
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        flow, magnitude = flow_and_magnitude(frame_a, frame_b, flow_backend=flow_backend)
        flows.append(flow)
        magnitudes.append(magnitude)
    return flows, magnitudes
