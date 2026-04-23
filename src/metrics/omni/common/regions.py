from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .flow import motion_series


def center_box_mask(shape, frac: float = 0.4):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    box_h = max(1, int(height * frac))
    box_w = max(1, int(width * frac))
    top = (height - box_h) // 2
    left = (width - box_w) // 2
    mask[top:top + box_h, left:left + box_w] = True
    return mask


def outer_frame_mask(shape, inner_frac: float = 0.55):
    return ~center_box_mask(shape, frac=inner_frac)


def bbox_to_mask(shape, bbox, padding: float = 0.0):
    height, width = shape[:2]
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if max(x0, y0, x1, y1) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    pad_x = padding * width
    pad_y = padding * height
    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y
    x0, x1 = int(max(0, min(width - 1, round(x0)))), int(max(0, min(width, round(x1))))
    y0, y1 = int(max(0, min(height - 1, round(y0)))), int(max(0, min(height, round(y1))))
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask if mask.any() else None


def load_mask_hint(mask_hint, shape, padding: float = 0.0):
    if mask_hint is None:
        return None
    if isinstance(mask_hint, str) and Path(mask_hint).exists():
        mask = cv2.imread(str(mask_hint), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = cv2.resize(mask, (shape[1], shape[0]))
        return mask > 127
    if isinstance(mask_hint, dict):
        if "bbox" in mask_hint:
            return bbox_to_mask(shape, mask_hint["bbox"], padding=padding)
        if "mask_path" in mask_hint:
            return load_mask_hint(mask_hint["mask_path"], shape, padding=padding)
    if isinstance(mask_hint, (list, tuple)) and len(mask_hint) == 4:
        return bbox_to_mask(shape, mask_hint, padding=padding)
    if isinstance(mask_hint, np.ndarray):
        resized = cv2.resize(mask_hint.astype(np.uint8), (shape[1], shape[0]))
        return resized > 0
    return None


def refine_binary_mask(mask):
    if mask is None:
        return None
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    refined = mask_u8 > 0
    return refined if refined.any() else None


def analysis_sample_frames(all_frames, sampled_frames, min_frames=4):
    if sampled_frames and len(sampled_frames) >= min_frames:
        return sampled_frames
    return all_frames


def estimate_target_mask(
    all_frames,
    omni_spec,
    sampled_frames=None,
    flow_backend=None,
    region_padding: float = 0.0,
):
    if not all_frames:
        return None, {
            "target_region_source": "missing_video",
            "non_target_region_source": None,
            "segmentation_mode": "missing",
        }

    frame_shape = all_frames[0].shape
    direct_mask = load_mask_hint(omni_spec.get("motion_mask_hint"), frame_shape, padding=region_padding)
    if direct_mask is not None:
        return direct_mask.astype(bool), {
            "target_region_source": "exact_mask",
            "non_target_region_source": "inverse_target_region",
            "segmentation_mode": "exact",
        }

    region_hints = omni_spec.get("entity_region_hints") or {}
    if isinstance(region_hints, dict):
        hinted_masks = []
        for entity in omni_spec.get("affected_entities") or omni_spec.get("target_entities") or []:
            hinted = load_mask_hint(region_hints.get(entity), frame_shape, padding=region_padding)
            if hinted is not None:
                hinted_masks.append(hinted)
        if hinted_masks:
            return np.logical_or.reduce(hinted_masks).astype(bool), {
                "target_region_source": "bbox_entity_hints",
                "non_target_region_source": "inverse_target_region",
                "segmentation_mode": "bbox_hint",
            }

    analysis_frames = analysis_sample_frames(all_frames, sampled_frames, min_frames=4)
    _, magnitudes = motion_series(analysis_frames, flow_backend=flow_backend)
    if magnitudes:
        motion_map = np.mean(np.stack(magnitudes, axis=0), axis=0)
        threshold = float(np.quantile(motion_map, 0.82))
        mask = refine_binary_mask(motion_map >= threshold)
        if mask is not None and 0.02 <= mask.mean() <= 0.65:
            return mask.astype(bool), {
                "target_region_source": "inferred_target_region",
                "non_target_region_source": "inverse_target_region",
                "segmentation_mode": "inferred",
                "threshold_quantile": 0.82,
            }

    return center_box_mask(frame_shape), {
        "target_region_source": "coarse_center_prior",
        "non_target_region_source": "inverse_target_region",
        "segmentation_mode": "coarse_fallback",
    }


def resolve_target_region(
    omni_spec,
    all_frames,
    sampled_frames=None,
    flow_backend=None,
    region_padding: float = 0.0,
):
    return estimate_target_mask(
        all_frames=all_frames,
        omni_spec=omni_spec,
        sampled_frames=sampled_frames,
        flow_backend=flow_backend,
        region_padding=region_padding,
    )


def build_non_target_mask(target_mask, frame_shape):
    if target_mask is None:
        return outer_frame_mask(frame_shape), {
            "non_target_region_source": "outer_frame_fallback",
            "segmentation_mode": "coarse_fallback",
        }
    inverse_mask = ~target_mask
    if inverse_mask.any() and inverse_mask.mean() >= 0.05:
        return inverse_mask.astype(bool), {
            "non_target_region_source": "frame_minus_target_region",
            "segmentation_mode": "derived",
        }
    return outer_frame_mask(frame_shape), {
        "non_target_region_source": "outer_frame_fallback",
        "segmentation_mode": "coarse_fallback",
    }
