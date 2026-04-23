from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def omni_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def omni_ssim_gray(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = cv2.resize(omni_gray(frame_a), (256, 256)).astype(np.float32) / 255.0
    b = cv2.resize(omni_gray(frame_b), (256, 256)).astype(np.float32) / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.clip((num / (den + 1e-8)).mean(), 0.0, 1.0))


def fallback_frame_embedding(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (32, 32)).astype(np.float32) / 255.0
    gray = cv2.cvtColor((resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hist = []
    for channel in range(3):
        values, _ = np.histogram(resized[..., channel], bins=8, range=(0.0, 1.0), density=True)
        hist.append(values.astype(np.float32))
    hist.append(np.histogram(gray, bins=8, range=(0.0, 1.0), density=True)[0].astype(np.float32))
    return np.concatenate([resized.mean(axis=(0, 1)), resized.std(axis=(0, 1)), *hist], axis=0).astype(np.float32)


def _normalize_backend_output(output: Any) -> tuple[np.ndarray, str]:
    if isinstance(output, tuple) and len(output) == 2:
        vector, mode = output
    else:
        vector, mode = output, "custom_backend"
    vector = np.asarray(vector, dtype=np.float32).flatten()
    return vector, str(mode)


def _call_embedding_backend(
    embedding_backend,
    frame: np.ndarray,
    cache=None,
    cache_key=None,
) -> tuple[np.ndarray, str]:
    if hasattr(embedding_backend, "embed_frame"):
        output = embedding_backend.embed_frame(frame=frame, cache=cache, cache_key=cache_key)
        return _normalize_backend_output(output)

    if callable(embedding_backend):
        for call in (
            lambda: embedding_backend(frame=frame, cache=cache, cache_key=cache_key),
            lambda: embedding_backend(frame, cache=cache, cache_key=cache_key),
            lambda: embedding_backend(frame),
        ):
            try:
                return _normalize_backend_output(call())
            except TypeError:
                continue
    raise TypeError("Unsupported embedding_backend. Expected callable or object with embed_frame(...).")


def compute_frame_embedding(
    frame: np.ndarray,
    embedding_backend=None,
    cache=None,
    cache_key=None,
) -> tuple[np.ndarray, str]:
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    if embedding_backend is None:
        payload = (fallback_frame_embedding(frame), "fallback_histogram")
    else:
        try:
            payload = _call_embedding_backend(
                embedding_backend=embedding_backend,
                frame=frame,
                cache=cache,
                cache_key=cache_key,
            )
        except Exception:
            payload = (fallback_frame_embedding(frame), "fallback_histogram")

    if cache is not None and cache_key is not None:
        cache[cache_key] = payload
    return payload


def embedding_cosine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    embedding_backend=None,
    cache=None,
    key_a=None,
    key_b=None,
) -> tuple[float, dict[str, Any]]:
    emb_a, mode_a = compute_frame_embedding(
        frame_a,
        embedding_backend=embedding_backend,
        cache=cache,
        cache_key=key_a,
    )
    emb_b, mode_b = compute_frame_embedding(
        frame_b,
        embedding_backend=embedding_backend,
        cache=cache,
        cache_key=key_b,
    )
    denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
    cosine = float(np.dot(emb_a, emb_b) / denom)
    return float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)), {
        "embedding_mode": mode_a if mode_a == mode_b else f"{mode_a}|{mode_b}"
    }


def masked_frame(frame: np.ndarray, mask: np.ndarray | None):
    if mask is None or not mask.any():
        return frame
    ys, xs = np.where(mask)
    return frame[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def light_frame_change(frame_a: np.ndarray, frame_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    crop_a = masked_frame(frame_a, mask)
    crop_b = masked_frame(frame_b, mask)
    ssim_score = omni_ssim_gray(crop_a, crop_b)
    diff = float(np.mean(np.abs(crop_a.astype(np.float32) - crop_b.astype(np.float32))) / 255.0)
    return float(np.clip(0.55 * (1.0 - ssim_score) + 0.45 * diff, 0.0, 1.0))
