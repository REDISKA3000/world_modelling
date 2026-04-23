from __future__ import annotations

import numpy as np

from ..common.embeddings import light_frame_change, omni_ssim_gray
from ..common.results import make_metric_result


def _smooth_curve(values, window: int | None = None):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 2:
        return values
    if window is None or int(window) <= 1:
        kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
    else:
        width = max(1, int(window))
        kernel = np.ones(width, dtype=np.float32) / float(width)
    pad = len(kernel) // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


class TransitionsDetectMetric:
    """Detect scene transitions from inter-frame visual dissimilarity and return a binary stability score."""

    def __init__(
        self,
        threshold=None,
        min_scene_length=1,
        smoothing_window=None,
        verbose=False,
    ):
        self.threshold = None if threshold is None else float(threshold)
        self.min_scene_length = max(1, int(min_scene_length))
        self.smoothing_window = smoothing_window
        self.verbose = verbose

    def _compute_threshold(self, smooth_scores):
        if self.threshold is not None:
            return float(self.threshold)
        median = float(np.median(smooth_scores))
        mad = float(np.median(np.abs(smooth_scores - median))) + 1e-6
        return max(0.40, float(median + 3.5 * 1.4826 * mad))

    def _pick_transition_indices(self, smooth_scores, threshold):
        candidates = []
        last_transition = None
        for idx, score in enumerate(smooth_scores):
            left = smooth_scores[idx - 1] if idx > 0 else score
            right = smooth_scores[idx + 1] if idx + 1 < len(smooth_scores) else score
            if score < threshold or score < left or score < right:
                continue
            boundary_idx = idx + 1
            if last_transition is not None and boundary_idx - last_transition < self.min_scene_length:
                continue
            candidates.append(boundary_idx)
            last_transition = boundary_idx
        return candidates

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run scene-boundary detection on a chunk and reduce raw boundary evidence to a binary transition score."""
        if len(all_frames) < 2:
            return make_metric_result(
                "transitions_detect",
                "transitions_detect",
                None,
                verification_mode="skipped",
                faithfulness_status="article-faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "not_enough_frames",
                    "boundary_scores": [],
                    "threshold_used": self.threshold,
                    "transition_frame_indices": [],
                    "num_transitions": 0,
                    "num_scenes": 0,
                    "binary_decision": False,
                    "smoothing_applied": False,
                    "used_fallback_reason": None,
                },
                has_transition=False,
                num_transitions=0,
            )

        boundary_scores = []
        for idx, (frame_a, frame_b) in enumerate(zip(all_frames[:-1], all_frames[1:])):
            ssim_score = omni_ssim_gray(frame_a, frame_b)
            change_score = light_frame_change(frame_a, frame_b)
            boundary_scores.append(
                {
                    "boundary_after_frame": idx,
                    "score": float(0.55 * (1.0 - ssim_score) + 0.45 * change_score),
                }
            )

        raw_scores = np.asarray([item["score"] for item in boundary_scores], dtype=np.float32)
        smooth_scores = _smooth_curve(raw_scores, window=self.smoothing_window)
        threshold = self._compute_threshold(smooth_scores)
        transition_indices = self._pick_transition_indices(smooth_scores, threshold)
        binary_score = 0.0 if transition_indices else 1.0

        return make_metric_result(
            "transitions_detect",
            "transitions_detect",
            binary_score,
            verification_mode="article_faithful",
            faithfulness_status="article-faithful",
            used_fallback=False,
            status="ok",
            details={
                "boundary_scores": boundary_scores,
                "threshold_used": threshold,
                "transition_frame_indices": transition_indices,
                "num_transitions": len(transition_indices),
                "num_scenes": len(transition_indices) + 1,
                "binary_decision": bool(transition_indices),
                "smoothing_applied": len(raw_scores) > 2,
                "used_fallback_reason": None,
            },
            has_transition=bool(transition_indices),
            num_transitions=len(transition_indices),
        )

    __call__ = run
