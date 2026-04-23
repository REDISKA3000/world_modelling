from __future__ import annotations

import numpy as np

from ..common.flow import motion_series
from ..common.regions import build_non_target_mask, resolve_target_region
from ..common.results import make_metric_result


def _parse_camera_expectation(text):
    low = (text or "").lower()
    if any(token in low for token in ["static camera", "locked camera", "fixed camera", "still camera"]):
        return {"type": "static", "direction": "static", "source_text": text}
    if any(token in low for token in ["pan left", "turn left"]):
        return {"type": "horizontal", "direction": "left", "source_text": text}
    if any(token in low for token in ["pan right", "turn right"]):
        return {"type": "horizontal", "direction": "right", "source_text": text}
    if any(token in low for token in ["zoom in", "push in", "forward"]):
        return {"type": "radial", "direction": "forward", "source_text": text}
    if any(token in low for token in ["zoom out", "pull out", "backward"]):
        return {"type": "radial", "direction": "backward", "source_text": text}
    return None


def _parse_camera_trajectory_vectors(camera_trajectory_gt, num_steps):
    if not isinstance(camera_trajectory_gt, (list, tuple)):
        return None
    vectors = []
    for item in camera_trajectory_gt:
        if isinstance(item, dict) and "dx" in item and "dy" in item:
            vectors.append([float(item["dx"]), float(item["dy"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            vectors.append([float(item[0]), float(item[1])])
        else:
            return None
    if not vectors:
        return None
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors if vectors.shape[0] == num_steps else None


class CameraControlMetric:
    """Evaluate whether observed camera motion matches available camera control metadata."""

    def __init__(
        self,
        camera_verifier=None,
        flow_backend=None,
        verbose=False,
    ):
        self.camera_verifier = camera_verifier
        self.flow_backend = flow_backend
        self.verbose = verbose

    def _resolve_motion_series(self, all_frames, shared_context=None):
        shared_context = shared_context or {}
        flows = shared_context.get("motion_flows")
        magnitudes = shared_context.get("motion_magnitudes")
        if flows is None or magnitudes is None:
            flows, magnitudes = motion_series(all_frames, flow_backend=self.flow_backend)
        return flows, magnitudes

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run trajectory-aware or prompt-hint fallback camera-control evaluation on one chunk."""
        available_camera_inputs = {
            "camera_prompt": bool(omni_spec.get("camera_prompt")),
            "camera_transform_hint": bool(omni_spec.get("camera_transform_hint")),
            "camera_trajectory_gt": bool(omni_spec.get("camera_trajectory_gt")),
            "layout_spec": bool(omni_spec.get("layout_spec")),
        }
        if len(all_frames) < 2:
            return make_metric_result(
                "camera_control",
                "camera_control",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "available_camera_inputs": available_camera_inputs,
                    "computed_path": "skipped",
                    "camera_prompt": omni_spec.get("camera_prompt"),
                    "camera_transform_hint": omni_spec.get("camera_transform_hint"),
                    "trajectory_gt_used": False,
                    "camera_score_raw": None,
                    "used_fallback_reason": None,
                    "why_skipped_or_partial": "not_enough_frames",
                },
            )

        prompt_expectation = _parse_camera_expectation(omni_spec.get("camera_prompt"))
        hint_expectation = _parse_camera_expectation(omni_spec.get("camera_transform_hint"))
        expectation = hint_expectation or prompt_expectation
        target_mask, mask_details = resolve_target_region(
            omni_spec,
            all_frames,
            sampled_frames=sampled_frames,
            flow_backend=self.flow_backend,
        )
        background_mask, _ = build_non_target_mask(target_mask, all_frames[0].shape)
        flows, magnitudes = self._resolve_motion_series(all_frames, shared_context=shared_context)
        if not flows:
            return make_metric_result(
                "camera_control",
                "camera_control",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "available_camera_inputs": available_camera_inputs,
                    "computed_path": "skipped",
                    "camera_prompt": omni_spec.get("camera_prompt"),
                    "camera_transform_hint": omni_spec.get("camera_transform_hint"),
                    "trajectory_gt_used": False,
                    "camera_score_raw": None,
                    "used_fallback_reason": None,
                    "why_skipped_or_partial": "no_optical_flow",
                },
            )

        observed_vectors = []
        observed_magnitudes = []
        radial_scores = []
        for flow, magnitude in zip(flows, magnitudes):
            flow_region = flow[background_mask] if background_mask is not None and background_mask.any() else flow.reshape(-1, 2)
            mag_region = magnitude[background_mask] if background_mask is not None and background_mask.any() else magnitude.reshape(-1)
            observed_vectors.append([float(flow_region[:, 0].mean()), float(flow_region[:, 1].mean())])
            observed_magnitudes.append(float(mag_region.mean()))
            height, width = magnitude.shape
            ys, xs = np.indices((height, width))
            cx, cy = width / 2.0, height / 2.0
            radial = np.stack([xs - cx, ys - cy], axis=-1).reshape(-1, 2)
            radial_norm = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8
            radial_unit = radial / radial_norm
            radial_scores.append(float((flow.reshape(-1, 2) * radial_unit).sum(axis=1).mean()))
        observed_vectors = np.asarray(observed_vectors, dtype=np.float32)
        observed_magnitudes = np.asarray(observed_magnitudes, dtype=np.float32)
        radial_scores = np.asarray(radial_scores, dtype=np.float32)

        gt_vectors = _parse_camera_trajectory_vectors(
            omni_spec.get("camera_trajectory_gt"),
            len(observed_vectors),
        )
        if gt_vectors is not None:
            cosine_scores = []
            translation_errors = []
            for observed, expected in zip(observed_vectors, gt_vectors):
                expected_norm = float(np.linalg.norm(expected))
                observed_norm = float(np.linalg.norm(observed))
                if expected_norm <= 1e-8 or observed_norm <= 1e-8:
                    cosine_scores.append(0.0)
                    translation_errors.append(None)
                    continue
                cosine = float(np.dot(observed, expected) / (expected_norm * observed_norm + 1e-8))
                cosine_scores.append(float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)))
                translation_errors.append(float(np.linalg.norm(observed - expected)))
            return make_metric_result(
                "camera_control",
                "camera_control",
                float(np.mean(cosine_scores)) if cosine_scores else None,
                verification_mode="trajectory_gt",
                faithfulness_status="article-faithful",
                used_fallback=False,
                status="ok",
                details={
                    "available_camera_inputs": available_camera_inputs,
                    "computed_path": "trajectory_gt",
                    "camera_prompt": omni_spec.get("camera_prompt"),
                    "camera_transform_hint": omni_spec.get("camera_transform_hint"),
                    "trajectory_gt_used": True,
                    "camera_score_raw": cosine_scores,
                    "used_fallback_reason": None,
                    "why_skipped_or_partial": None,
                    "trajectory_signals_used": {
                        "observed_vectors": observed_vectors.tolist(),
                        "expected_vectors": gt_vectors.tolist(),
                    },
                    "mask_mode": mask_details.get("target_region_source"),
                    "framewise_scores": cosine_scores,
                    "translation_error_proxy": translation_errors,
                },
            )

        if expectation is None:
            return make_metric_result(
                "camera_control",
                "camera_control",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "available_camera_inputs": available_camera_inputs,
                    "computed_path": "skipped",
                    "camera_prompt": omni_spec.get("camera_prompt"),
                    "camera_transform_hint": omni_spec.get("camera_transform_hint"),
                    "trajectory_gt_used": False,
                    "camera_score_raw": None,
                    "used_fallback_reason": None,
                    "why_skipped_or_partial": "camera_motion_not_specified",
                },
            )

        if expectation["type"] == "static":
            score = float(np.exp(-float(np.mean(observed_magnitudes)) / 0.8))
        elif expectation["type"] == "horizontal":
            expected_sign = 1.0 if expectation["direction"] == "left" else -1.0
            alignment = 0.5 * (1.0 + np.sign(expected_sign * observed_vectors[:, 0]))
            score = float(np.mean(alignment * (1.0 - np.exp(-np.abs(observed_vectors[:, 0]) / 1.0))))
        else:
            expected_sign = 1.0 if expectation["direction"] == "forward" else -1.0
            alignment = 0.5 * (1.0 + np.sign(expected_sign * radial_scores))
            score = float(np.mean(alignment * (1.0 - np.exp(-np.abs(radial_scores) / 0.5))))

        return make_metric_result(
            "camera_control",
            "camera_control",
            score,
            verification_mode="prompt_hint_fallback",
            faithfulness_status="fallback approximation",
            used_fallback=True,
            status="partial",
            details={
                "available_camera_inputs": available_camera_inputs,
                "computed_path": "prompt_hint_fallback",
                "camera_prompt": omni_spec.get("camera_prompt"),
                "camera_transform_hint": omni_spec.get("camera_transform_hint"),
                "trajectory_gt_used": False,
                "camera_score_raw": score,
                "used_fallback_reason": "trajectory_gt_unavailable",
                "why_skipped_or_partial": "trajectory_gt_unavailable",
                "expected_camera_motion": expectation,
                "mask_mode": mask_details.get("target_region_source"),
                "framewise_vectors": observed_vectors.tolist(),
                "framewise_magnitudes": observed_magnitudes.tolist(),
            },
        )

    __call__ = run
