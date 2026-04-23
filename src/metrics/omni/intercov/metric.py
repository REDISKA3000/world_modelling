from __future__ import annotations

from typing import Any

import numpy as np

from ..common.embeddings import embedding_cosine, light_frame_change, masked_frame, omni_ssim_gray
from ..common.flow import motion_series
from ..common.regions import (
    analysis_sample_frames,
    build_non_target_mask,
    center_box_mask,
    load_mask_hint,
    outer_frame_mask,
    resolve_target_region,
)
from ..common.results import make_metric_result


def _appearance_change(
    frame_a,
    frame_b,
    mask=None,
    embedding_backend=None,
    cache=None,
    key_prefix=None,
):
    crop_a = masked_frame(frame_a, mask)
    crop_b = masked_frame(frame_b, mask)
    ssim_score = omni_ssim_gray(crop_a, crop_b)
    cosine_score, cosine_details = embedding_cosine(
        crop_a,
        crop_b,
        embedding_backend=embedding_backend,
        cache=cache,
        key_a=None if key_prefix is None else f"{key_prefix}_a",
        key_b=None if key_prefix is None else f"{key_prefix}_b",
    )
    consistency = 0.5 * (ssim_score + cosine_score)
    return 1.0 - consistency, {
        "ssim_gray": ssim_score,
        "embedding_similarity": cosine_score,
        **cosine_details,
    }


class InterCovMetric:
    """Compute object-level causal faithfulness from entity-specific motion, appearance, and position evidence."""

    def __init__(
        self,
        flow_backend=None,
        semantic_verifier=None,
        appearance_weight=1.0,
        motion_weight=1.0,
        position_weight=1.0,
        verbose=False,
        embedding_backend=None,
    ):
        self.flow_backend = flow_backend
        self.semantic_verifier = semantic_verifier
        self.appearance_weight = float(appearance_weight)
        self.motion_weight = float(motion_weight)
        self.position_weight = float(position_weight)
        self.verbose = verbose
        self.embedding_backend = embedding_backend

    def _resolve_expected_motion_for_entity(self, omni_spec, entity):
        payload = omni_spec.get("expected_motion") or {}
        if not isinstance(payload, dict):
            return {}
        if entity in payload:
            return payload.get(entity) or {}
        if len(payload) == 1 and entity in (omni_spec.get("affected_entities") or []):
            return next(iter(payload.values())) or {}
        return {}

    def _estimate_entity_mask(
        self,
        entity,
        group,
        omni_spec,
        all_frames,
        sampled_frames=None,
        aggregate_target_mask=None,
        aggregate_mask_details=None,
    ):
        if not all_frames:
            return None, {"mode": "empty_video", "segmentation_mode": "missing", "used_fallback": True}

        frame_shape = all_frames[0].shape
        region_hints = omni_spec.get("entity_region_hints") or {}
        if isinstance(region_hints, dict):
            direct_hint = load_mask_hint(region_hints.get(entity), frame_shape)
            if direct_hint is not None:
                return direct_hint.astype(bool), {
                    "mode": "entity_region_hint",
                    "segmentation_mode": "bbox_hint",
                    "used_fallback": False,
                }

        if group == "affected":
            motion_hint = load_mask_hint(omni_spec.get("motion_mask_hint"), frame_shape)
            if motion_hint is not None:
                return motion_hint.astype(bool), {
                    "mode": "motion_mask_hint",
                    "segmentation_mode": "exact",
                    "used_fallback": False,
                }

        if aggregate_target_mask is not None and aggregate_target_mask.any():
            if group == "affected":
                aggregate_mask_details = aggregate_mask_details or {}
                return aggregate_target_mask.astype(bool), {
                    "mode": aggregate_mask_details.get("target_region_source"),
                    "segmentation_mode": aggregate_mask_details.get("segmentation_mode"),
                    "used_fallback": aggregate_mask_details.get("segmentation_mode") != "exact",
                }
            inverse_mask, inverse_details = build_non_target_mask(aggregate_target_mask, frame_shape)
            return inverse_mask.astype(bool), {
                "mode": inverse_details.get("non_target_region_source"),
                "segmentation_mode": inverse_details.get("segmentation_mode"),
                "used_fallback": True,
            }

        if entity == omni_spec.get("main_subject"):
            return center_box_mask(frame_shape, frac=0.42), {
                "mode": "main_subject_center_prior",
                "segmentation_mode": "coarse_fallback",
                "used_fallback": True,
            }

        return outer_frame_mask(frame_shape, inner_frac=0.55), {
            "mode": "outer_region_prior",
            "segmentation_mode": "coarse_fallback",
            "used_fallback": True,
        }

    def _mask_motion_signature(self, frames, mask, shared_context=None):
        shared_context = shared_context or {}
        flows = shared_context.get("motion_flows")
        magnitudes = shared_context.get("motion_magnitudes")
        if flows is None or magnitudes is None:
            flows, magnitudes = motion_series(frames, flow_backend=self.flow_backend)
        if not flows:
            return None

        motion_curve = []
        dx_curve = []
        dy_curve = []
        for flow, magnitude in zip(flows, magnitudes):
            if mask is not None and mask.any():
                region_flow = flow[mask]
                region_mag = magnitude[mask]
            else:
                region_flow = flow.reshape(-1, 2)
                region_mag = magnitude.reshape(-1)
            motion_curve.append(float(region_mag.mean()))
            dx_curve.append(float(region_flow[:, 0].mean()))
            dy_curve.append(float(region_flow[:, 1].mean()))

        dx_curve = np.asarray(dx_curve, dtype=np.float32)
        dy_curve = np.asarray(dy_curve, dtype=np.float32)
        motion_curve = np.asarray(motion_curve, dtype=np.float32)
        cumulative_x = np.cumsum(dx_curve)
        cumulative_y = np.cumsum(dy_curve)
        position_shift = (
            float(np.sqrt(cumulative_x[-1] ** 2 + cumulative_y[-1] ** 2))
            if len(cumulative_x)
            else 0.0
        )
        return {
            "motion_curve": motion_curve,
            "dx_curve": dx_curve,
            "dy_curve": dy_curve,
            "mean_motion": float(motion_curve.mean()) if motion_curve.size else 0.0,
            "max_motion": float(motion_curve.max()) if motion_curve.size else 0.0,
            "mean_dx": float(dx_curve.mean()) if dx_curve.size else 0.0,
            "mean_dy": float(dy_curve.mean()) if dy_curve.size else 0.0,
            "position_shift": position_shift,
        }

    def _mask_change_signature(self, frames, mask):
        if len(frames) < 2:
            return None

        cache = {}
        pair_changes = []
        embedding_modes = []
        light_change_curve = []
        for idx, (frame_a, frame_b) in enumerate(zip(frames[:-1], frames[1:])):
            change_value, details = _appearance_change(
                frame_a,
                frame_b,
                mask,
                embedding_backend=self.embedding_backend,
                cache=cache,
                key_prefix=f"pair_{idx}",
            )
            pair_changes.append(float(change_value))
            light_change_curve.append(float(light_frame_change(frame_a, frame_b, mask)))
            embedding_modes.append(details.get("embedding_mode"))

        start_end_change, start_end_details = _appearance_change(
            frames[0],
            frames[-1],
            mask,
            embedding_backend=self.embedding_backend,
            cache=cache,
            key_prefix="start_end",
        )
        anchor_similarities = []
        for idx, frame in enumerate(frames[1:], start=1):
            change_value, _ = _appearance_change(
                frames[0],
                frame,
                mask,
                embedding_backend=self.embedding_backend,
                cache=cache,
                key_prefix=f"anchor_{idx}",
            )
            anchor_similarities.append(1.0 - change_value)

        pair_changes = np.asarray(pair_changes, dtype=np.float32)
        light_change_curve = np.asarray(light_change_curve, dtype=np.float32)
        anchor_similarities = np.asarray(anchor_similarities, dtype=np.float32)
        return {
            "pair_changes": pair_changes,
            "mean_change": float(pair_changes.mean()) if pair_changes.size else 0.0,
            "max_change": float(pair_changes.max()) if pair_changes.size else 0.0,
            "start_end_change": float(start_end_change),
            "start_end_similarity": float(1.0 - start_end_change),
            "anchor_similarity_mean": (
                float(anchor_similarities.mean())
                if anchor_similarities.size
                else float(1.0 - start_end_change)
            ),
            "anchor_similarity_min": (
                float(anchor_similarities.min())
                if anchor_similarities.size
                else float(1.0 - start_end_change)
            ),
            "light_change_mean": float(light_change_curve.mean()) if light_change_curve.size else None,
            "start_end_details": start_end_details,
            "embedding_mode": "|".join(sorted(set(mode for mode in embedding_modes if mode))),
        }

    def _direction_alignment_score(self, dx, dy, direction):
        if direction in (None, "", "unknown"):
            return None
        if direction == "static":
            return float(np.exp(-np.sqrt(dx ** 2 + dy ** 2) / 0.5))
        mapping = {
            "left": np.asarray([-1.0, 0.0], dtype=np.float32),
            "right": np.asarray([1.0, 0.0], dtype=np.float32),
            "up": np.asarray([0.0, -1.0], dtype=np.float32),
            "down": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        expected = mapping.get(direction)
        if expected is None:
            return None
        observed = np.asarray([dx, dy], dtype=np.float32)
        norm = float(np.linalg.norm(observed))
        if norm <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(observed / norm, expected), 0.0, 1.0))

    def _magnitude_expectation_score(self, mean_motion, magnitude):
        if magnitude in (None, "", "unknown"):
            return None
        targets = {"low": (0.30, 0.35), "medium": (0.75, 0.60), "high": (1.20, 0.90)}
        target, tol = targets.get(magnitude, (0.75, 0.60))
        return float(np.clip(1.0 - abs(mean_motion - target) / max(tol, 1e-6), 0.0, 1.0))

    def _semantic_entity_evidence(
        self,
        entity,
        group,
        frames,
        frame_indices,
        omni_spec,
        expected_motion=None,
    ):
        verifier = self.semantic_verifier
        if verifier is None:
            return None, None

        verifier_name = getattr(verifier, "__name__", verifier.__class__.__name__)
        try:
            if hasattr(verifier, "verify_entity"):
                payload = verifier.verify_entity(
                    metric="intercov",
                    entity=entity,
                    group=group,
                    frames=frames,
                    frame_indices=frame_indices,
                    omni_spec=omni_spec,
                    expected_motion=expected_motion,
                )
            else:
                payload = verifier(
                    metric="intercov",
                    entity=entity,
                    group=group,
                    frames=frames,
                    frame_indices=frame_indices,
                    omni_spec=omni_spec,
                    expected_motion=expected_motion,
                )
        except TypeError:
            try:
                payload = verifier(entity, group, frames, frame_indices, omni_spec, expected_motion)
            except Exception:
                return verifier_name, None
        except Exception:
            return verifier_name, None

        if isinstance(payload, dict):
            score = payload.get("score", payload.get("semantic_score"))
        else:
            score = payload
        try:
            return verifier_name, None if score is None else float(score)
        except Exception:
            return verifier_name, None

    def _weighted_decision_score(self, motion_score, appearance_score, position_score, semantic_score=None):
        weighted = []
        total_weight = 0.0
        for value, weight in (
            (motion_score, self.motion_weight),
            (appearance_score, self.appearance_weight),
            (position_score, self.position_weight),
        ):
            if value is None:
                continue
            weighted.append(float(value) * float(weight))
            total_weight += float(weight)

        heuristic_score = float(sum(weighted) / total_weight) if total_weight > 0 else None
        if semantic_score is None:
            return heuristic_score, heuristic_score
        if heuristic_score is None:
            return float(semantic_score), None
        return float(np.mean([heuristic_score, float(semantic_score)])), heuristic_score

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run InterCov on one chunk using entity-level regions plus motion, appearance, position, and optional semantic evidence."""
        affected_entities = omni_spec.get("affected_entities") or []
        unaffected_entities = omni_spec.get("unaffected_entities") or []
        if not affected_entities and not unaffected_entities:
            return make_metric_result(
                "intercov",
                "intercov",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "no_entity_metadata",
                    "entity_breakdown": {},
                    "num_entities_total": 0,
                    "num_entities_evaluated": 0,
                    "num_affected": 0,
                    "num_unaffected": 0,
                    "used_semantic_verifier": self.semantic_verifier is not None,
                    "used_fallback_reason": None,
                },
                affected_ok_ratio=None,
                unaffected_ok_ratio=None,
            )

        shared_context = shared_context or {}
        analysis_frames = analysis_sample_frames(all_frames, sampled_frames, min_frames=4)
        aggregate_mask, aggregate_details = resolve_target_region(
            omni_spec,
            all_frames,
            sampled_frames=analysis_frames,
            flow_backend=self.flow_backend,
        )
        entity_details = {}
        affected_decisions = []
        unaffected_decisions = []
        semantic_used = False
        fallback_reasons = []

        for group, entities in (("affected", affected_entities), ("unaffected", unaffected_entities)):
            for entity in entities:
                entity_mask, region_details = self._estimate_entity_mask(
                    entity,
                    group,
                    omni_spec,
                    all_frames,
                    sampled_frames=analysis_frames,
                    aggregate_target_mask=aggregate_mask,
                    aggregate_mask_details=aggregate_details,
                )
                motion_signature = self._mask_motion_signature(
                    all_frames,
                    entity_mask,
                    shared_context=shared_context,
                )
                change_signature = self._mask_change_signature(analysis_frames, entity_mask)
                expected_motion = self._resolve_expected_motion_for_entity(omni_spec, entity)
                region_source = region_details.get("mode")

                if motion_signature is None or change_signature is None:
                    entity_details[entity] = {
                        "group": group,
                        "region_source": region_source,
                        "motion_score": None,
                        "appearance_score": None,
                        "position_score": None,
                        "expected_direction": expected_motion.get("direction"),
                        "expected_magnitude": expected_motion.get("magnitude"),
                        "semantic_verifier_score": None,
                        "decision": None,
                        "verification_mode": "fallback",
                        "used_fallback": True,
                        "reason": "insufficient_frame_evidence",
                    }
                    fallback_reasons.append("insufficient_frame_evidence")
                    continue

                direction_score = self._direction_alignment_score(
                    motion_signature["mean_dx"],
                    motion_signature["mean_dy"],
                    expected_motion.get("direction"),
                )
                magnitude_score = self._magnitude_expectation_score(
                    motion_signature["mean_motion"],
                    expected_motion.get("magnitude"),
                )

                if group == "affected":
                    motion_score = float(1.0 - np.exp(-motion_signature["mean_motion"] / 0.8))
                    appearance_score = float(
                        np.clip(
                            0.6 * change_signature["start_end_change"] + 0.4 * change_signature["max_change"],
                            0.0,
                            1.0,
                        )
                    )
                    position_candidates = [
                        value for value in (direction_score, magnitude_score) if value is not None
                    ]
                    if position_candidates:
                        position_score = float(np.mean(position_candidates))
                    else:
                        position_score = float(
                            np.clip(1.0 - np.exp(-motion_signature["position_shift"] / 3.0), 0.0, 1.0)
                        )
                else:
                    motion_score = float(np.exp(-motion_signature["mean_motion"] / 0.8))
                    appearance_score = float(np.clip(1.0 - change_signature["start_end_change"], 0.0, 1.0))
                    position_score = float(np.exp(-motion_signature["position_shift"] / 3.0))

                semantic_hook_name, semantic_score = self._semantic_entity_evidence(
                    entity,
                    group,
                    analysis_frames,
                    sampled_indices,
                    omni_spec,
                    expected_motion=expected_motion,
                )
                semantic_used = semantic_used or semantic_score is not None
                decision_score, heuristic_score = self._weighted_decision_score(
                    motion_score,
                    appearance_score,
                    position_score,
                    semantic_score=semantic_score,
                )
                if group == "affected":
                    decision = bool((decision_score or 0.0) >= 0.45 and motion_score >= 0.25)
                    affected_decisions.append(float(decision))
                else:
                    decision = bool((decision_score or 0.0) >= 0.60 and motion_score >= 0.45)
                    unaffected_decisions.append(float(decision))

                expected_available = any(
                    expected_motion.get(key) not in (None, "", "unknown")
                    for key in ("direction", "magnitude")
                )
                high_quality_region = region_details.get("segmentation_mode") in {"exact", "bbox_hint"}
                entity_used_fallback = bool(region_details.get("used_fallback")) or not expected_available
                if entity_used_fallback:
                    fallback_reasons.append(region_source or "entity_region_fallback")
                if semantic_score is not None and high_quality_region and expected_available and not entity_used_fallback:
                    entity_verification_mode = "semantic_partial_faithful"
                elif high_quality_region and expected_available and not entity_used_fallback:
                    entity_verification_mode = "partial_faithful"
                else:
                    entity_verification_mode = "fallback"

                entity_details[entity] = {
                    "group": group,
                    "region_source": region_source,
                    "motion_score": motion_score,
                    "appearance_score": appearance_score,
                    "position_score": position_score,
                    "expected_direction": expected_motion.get("direction"),
                    "expected_magnitude": expected_motion.get("magnitude"),
                    "semantic_verifier_score": semantic_score,
                    "semantic_hook": semantic_hook_name,
                    "decision": decision,
                    "decision_score": decision_score,
                    "heuristic_score": heuristic_score,
                    "verification_mode": entity_verification_mode,
                    "used_fallback": entity_used_fallback,
                    "region_details": region_details,
                    "motion_signature": {
                        "mean_motion": motion_signature["mean_motion"],
                        "max_motion": motion_signature["max_motion"],
                        "mean_dx": motion_signature["mean_dx"],
                        "mean_dy": motion_signature["mean_dy"],
                        "position_shift": motion_signature["position_shift"],
                    },
                    "appearance_signature": {
                        "mean_change": change_signature["mean_change"],
                        "max_change": change_signature["max_change"],
                        "start_end_change": change_signature["start_end_change"],
                        "embedding_mode": change_signature["embedding_mode"],
                    },
                }

        affected_ok_ratio = float(np.mean(affected_decisions)) if affected_decisions else None
        unaffected_ok_ratio = float(np.mean(unaffected_decisions)) if unaffected_decisions else None
        score_components = [
            value for value in (affected_ok_ratio, unaffected_ok_ratio) if value is not None
        ]
        num_entities_total = len(affected_entities) + len(unaffected_entities)
        num_entities_evaluated = sum(
            1 for item in entity_details.values() if item.get("decision") is not None
        )
        used_fallback = any(
            item.get("used_fallback", False)
            for item in entity_details.values()
            if item.get("decision") is not None
        )

        if score_components:
            if semantic_used and not used_fallback:
                verification_mode = "semantic_partial_faithful"
                faithfulness = "partial faithful"
                status = "partial"
            elif not used_fallback:
                verification_mode = "partial_faithful"
                faithfulness = "partial faithful"
                status = "partial"
            else:
                verification_mode = "fallback"
                faithfulness = "fallback approximation"
                status = "partial"
        else:
            verification_mode = "skipped" if num_entities_evaluated == 0 else "fallback"
            faithfulness = "fallback approximation" if num_entities_total else "partial faithful"
            status = "failed" if num_entities_total else "skipped"

        return make_metric_result(
            "intercov",
            "intercov",
            float(np.mean(score_components)) if score_components else None,
            verification_mode=verification_mode,
            faithfulness_status=faithfulness,
            used_fallback=used_fallback,
            status=status,
            details={
                "entity_breakdown": entity_details,
                "num_entities_total": num_entities_total,
                "num_entities_evaluated": num_entities_evaluated,
                "num_affected": len(affected_entities),
                "num_unaffected": len(unaffected_entities),
                "used_semantic_verifier": semantic_used,
                "aggregate_target_region_source": aggregate_details.get("target_region_source"),
                "used_fallback_reason": sorted(set(reason for reason in fallback_reasons if reason)) or None,
            },
            affected_ok_ratio=affected_ok_ratio,
            unaffected_ok_ratio=unaffected_ok_ratio,
        )

    __call__ = run
