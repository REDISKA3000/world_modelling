from __future__ import annotations

import numpy as np

from ..common.embeddings import embedding_cosine, light_frame_change, masked_frame, omni_ssim_gray
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


class ObjectControlMetric:
    """Verify presence and persistence of target entities across a video chunk."""

    def __init__(
        self,
        object_verifier=None,
        embedding_backend=None,
        verbose=False,
    ):
        self.object_verifier = object_verifier
        self.embedding_backend = embedding_backend
        self.verbose = verbose

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

    def _mask_change_signature(self, frames, mask):
        if len(frames) < 2:
            return None

        cache = {}
        pair_changes = []
        embedding_modes = []
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
            "start_end_details": start_end_details,
            "embedding_mode": "|".join(sorted(set(mode for mode in embedding_modes if mode))),
        }

    def _semantic_presence_score(self, entity, frames, frame_indices, omni_spec):
        verifier = self.object_verifier
        if verifier is None:
            return None, None

        verifier_name = getattr(verifier, "__name__", verifier.__class__.__name__)
        try:
            if hasattr(verifier, "verify_object"):
                payload = verifier.verify_object(
                    metric="object_control",
                    entity=entity,
                    frames=frames,
                    frame_indices=frame_indices,
                    omni_spec=omni_spec,
                )
            else:
                payload = verifier(
                    metric="object_control",
                    entity=entity,
                    frames=frames,
                    frame_indices=frame_indices,
                    omni_spec=omni_spec,
                )
        except TypeError:
            try:
                payload = verifier(entity, frames, frame_indices, omni_spec)
            except Exception:
                return verifier_name, None
        except Exception:
            return verifier_name, None

        if isinstance(payload, dict):
            score = payload.get("score", payload.get("present_score"))
        else:
            score = payload
        try:
            return verifier_name, None if score is None else float(score)
        except Exception:
            return verifier_name, None

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run object-presence and persistence verification for target entities on one chunk."""
        target_entities = omni_spec.get("target_entities") or []
        if not target_entities or len(all_frames) < 2:
            return make_metric_result(
                "object_control",
                "object_control",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "no_targets_or_not_enough_frames",
                    "entity_results": {},
                    "num_targets": len(target_entities),
                    "num_confirmed": 0,
                    "used_semantic_verifier": self.object_verifier is not None,
                    "used_fallback_reason": None,
                    "sampled_frame_indices": sampled_indices or [],
                },
                num_targets=len(target_entities),
                num_confirmed=0,
            )

        analysis_frames = analysis_sample_frames(all_frames, sampled_frames, min_frames=4)
        aggregate_mask, aggregate_details = resolve_target_region(
            omni_spec,
            all_frames,
            sampled_frames=analysis_frames,
        )

        entity_results = {}
        confirmed = 0
        evaluable = 0
        semantic_used = False
        fallback_reasons = []
        for entity in target_entities:
            entity_mask, region_details = self._estimate_entity_mask(
                entity,
                "affected" if entity in (omni_spec.get("affected_entities") or []) else "target",
                omni_spec,
                all_frames,
                sampled_frames=analysis_frames,
                aggregate_target_mask=aggregate_mask,
                aggregate_mask_details=aggregate_details,
            )
            change_signature = self._mask_change_signature(analysis_frames, entity_mask)
            if change_signature is None:
                entity_results[entity] = {
                    "sampled_frame_indices": sampled_indices or [],
                    "region_source": region_details.get("mode"),
                    "present_score": None,
                    "preserved_score": None,
                    "semantic_presence_score": None,
                    "decision": None,
                    "verification_mode": "fallback",
                    "used_fallback": True,
                    "reason": "insufficient_visual_evidence",
                }
                fallback_reasons.append("insufficient_visual_evidence")
                continue

            present_score = float(
                np.clip(
                    0.5 * change_signature["anchor_similarity_mean"]
                    + 0.5 * change_signature["anchor_similarity_min"],
                    0.0,
                    1.0,
                )
            )
            preserved_score = float(
                np.clip(
                    0.55 * change_signature["start_end_similarity"]
                    + 0.45 * change_signature["anchor_similarity_mean"],
                    0.0,
                    1.0,
                )
            )
            verifier_name, semantic_presence_score = self._semantic_presence_score(
                entity,
                analysis_frames,
                sampled_indices,
                omni_spec,
            )
            semantic_used = semantic_used or semantic_presence_score is not None

            decision_components = [present_score, preserved_score]
            if semantic_presence_score is not None:
                decision_components.append(semantic_presence_score)
            decision_score = float(np.mean(decision_components))
            decision = bool(decision_score >= 0.45 and preserved_score >= 0.35)
            evaluable += 1
            confirmed += int(decision)

            entity_used_fallback = bool(region_details.get("used_fallback")) or semantic_presence_score is None
            if entity_used_fallback:
                fallback_reasons.append(region_details.get("mode") or "heuristic_presence_path")

            entity_results[entity] = {
                "sampled_frame_indices": sampled_indices or [],
                "region_source": region_details.get("mode"),
                "present_score": present_score,
                "preserved_score": preserved_score,
                "semantic_presence_score": semantic_presence_score,
                "decision": decision,
                "verification_mode": (
                    "semantic_partial_faithful" if semantic_presence_score is not None else "fallback"
                ),
                "used_fallback": entity_used_fallback,
                "semantic_hook": verifier_name,
                "embedding_mode": change_signature.get("embedding_mode"),
            }

        score = float(confirmed / evaluable) if evaluable else None
        if semantic_used:
            verification_mode = "semantic_partial_faithful"
            faithfulness = "partial faithful"
            used_fallback = any(
                item.get("used_fallback", False)
                for item in entity_results.values()
                if item.get("decision") is not None
            )
            status = "partial" if evaluable else "failed"
        elif evaluable:
            verification_mode = "fallback"
            faithfulness = "fallback approximation"
            used_fallback = True
            status = "partial"
        else:
            verification_mode = "fallback"
            faithfulness = "fallback approximation"
            used_fallback = True
            status = "failed"

        return make_metric_result(
            "object_control",
            "object_control",
            score,
            verification_mode=verification_mode,
            faithfulness_status=faithfulness,
            used_fallback=used_fallback,
            status=status,
            details={
                "entity_results": entity_results,
                "num_targets": len(target_entities),
                "num_confirmed": confirmed,
                "used_semantic_verifier": semantic_used,
                "used_fallback_reason": sorted(set(reason for reason in fallback_reasons if reason)) or None,
                "sampled_frame_indices": sampled_indices or [],
                "aggregate_target_region_source": aggregate_details.get("target_region_source"),
            },
            num_targets=len(target_entities),
            num_confirmed=confirmed,
        )

    __call__ = run
