from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from .embeddings import fallback_frame_embedding as histogram_fallback_frame_embedding


_OMNI_CLIP_VISION_MODEL = None
_OMNI_CLIP_VISION_UNAVAILABLE = False
_OMNI_CLIP_VISION_LOADER = None
_OMNI_CLIP_VISION_ENCODER = None
_OMNI_CLIP_VISION_MODEL_NAME = "clip_vision_h.safetensors"


def configure_omni_clip_backend(
    loader=None,
    encoder=None,
    model_name: str | None = None,
    reset_model_cache: bool = False,
):
    """Register optional CLIP-vision loader/encoder objects for the Omni embedding backend."""
    global _OMNI_CLIP_VISION_LOADER, _OMNI_CLIP_VISION_ENCODER, _OMNI_CLIP_VISION_MODEL_NAME
    global _OMNI_CLIP_VISION_MODEL, _OMNI_CLIP_VISION_UNAVAILABLE

    _OMNI_CLIP_VISION_LOADER = loader
    _OMNI_CLIP_VISION_ENCODER = encoder
    if model_name:
        _OMNI_CLIP_VISION_MODEL_NAME = str(model_name)
    if reset_model_cache:
        _OMNI_CLIP_VISION_MODEL = None
        _OMNI_CLIP_VISION_UNAVAILABLE = False


def omni_extract_vec(cv_out: Any):
    if torch.is_tensor(cv_out):
        value = cv_out
    elif isinstance(cv_out, dict):
        if "image_embeds" in cv_out:
            value = cv_out["image_embeds"]
        elif "pooled_output" in cv_out:
            value = cv_out["pooled_output"]
        elif "last_hidden_state" in cv_out:
            hidden = cv_out["last_hidden_state"]
            value = hidden[:, 0] if hidden.ndim == 3 else hidden
        else:
            value = next((candidate for candidate in cv_out.values() if torch.is_tensor(candidate)), None)
    else:
        value = getattr(cv_out, "image_embeds", None) or getattr(cv_out, "pooled_output", None)
        if value is None and hasattr(cv_out, "last_hidden_state"):
            hidden = cv_out.last_hidden_state
            value = hidden[:, 0] if hidden.ndim == 3 else hidden
    if value is None:
        raise ValueError("Unsupported CLIP vision output type")
    value = value.detach().float()
    if value.ndim > 2:
        value = value.flatten(start_dim=1)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    return value


def get_omni_clip_vision_model(loader=None, model_name: str | None = None):
    global _OMNI_CLIP_VISION_MODEL, _OMNI_CLIP_VISION_UNAVAILABLE

    active_loader = loader or _OMNI_CLIP_VISION_LOADER
    if active_loader is None:
        return None
    if _OMNI_CLIP_VISION_UNAVAILABLE:
        return None
    if _OMNI_CLIP_VISION_MODEL is None:
        try:
            _OMNI_CLIP_VISION_MODEL = active_loader.load_clip(model_name or _OMNI_CLIP_VISION_MODEL_NAME)[0]
        except Exception:
            _OMNI_CLIP_VISION_UNAVAILABLE = True
            _OMNI_CLIP_VISION_MODEL = None
    return _OMNI_CLIP_VISION_MODEL


def fallback_frame_embedding(frame: np.ndarray) -> np.ndarray:
    return histogram_fallback_frame_embedding(frame)


@torch.no_grad()
def frame_embedding(
    frame: np.ndarray,
    cache=None,
    cache_key=None,
    loader=None,
    encoder=None,
    model_name: str | None = None,
):
    """Embed one RGB frame with optional CLIP-vision backend and histogram fallback."""
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]

    model = get_omni_clip_vision_model(loader=loader, model_name=model_name)
    active_encoder = encoder or _OMNI_CLIP_VISION_ENCODER
    if model is None or active_encoder is None:
        payload = (fallback_frame_embedding(frame), "fallback_histogram")
    else:
        image = torch.from_numpy(frame).float() / 255.0
        if image.ndim == 3:
            image = image.unsqueeze(0)
        try:
            encoded = active_encoder.encode(model, image, "none")[0]
            vec = omni_extract_vec(encoded).cpu().numpy().astype(np.float32).flatten()
            payload = (vec, "clip_vision")
        except Exception:
            payload = (fallback_frame_embedding(frame), "fallback_histogram")

    if cache is not None and cache_key is not None:
        cache[cache_key] = payload
    return payload
