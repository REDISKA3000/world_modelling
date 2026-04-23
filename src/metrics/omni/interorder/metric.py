from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..common.embeddings import light_frame_change
from ..common.flow import motion_series
from ..common.regions import analysis_sample_frames, resolve_target_region
from ..common.results import make_metric_result


def build_event_pairs(event_sequence):
    return [
        (event_sequence[left_idx], event_sequence[right_idx])
        for left_idx in range(len(event_sequence))
        for right_idx in range(left_idx + 1, len(event_sequence))
    ]


def _event_label(event):
    if isinstance(event, dict):
        return str(event.get("label") or event.get("event") or event.get("name") or "")
    return str(event)


def _infer_motion_direction(text: str | None):
    low = (text or "").lower()
    if any(token in low for token in ["left", "pan left", "move left", "turn left"]):
        return "left"
    if any(token in low for token in ["right", "pan right", "move right", "turn right"]):
        return "right"
    if any(token in low for token in ["up", "rise", "lift", "raise"]):
        return "up"
    if any(token in low for token in ["down", "fall", "drop", "lower"]):
        return "down"
    if any(token in low for token in ["zoom in", "push in", "forward", "approach", "closer"]):
        return "forward"
    if any(token in low for token in ["zoom out", "pull out", "backward", "away"]):
        return "backward"
    if any(token in low for token in ["static camera", "locked camera", "fixed camera", "still camera"]):
        return "static"
    return "unknown"


def _smooth_curve(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 2:
        return values
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.asarray([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")


def _normalize_curve(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _event_payload(event):
    if isinstance(event, dict):
        label = _event_label(event)
        return {
            "label": label,
            "kind": str(event.get("kind") or "").lower(),
            "entity": event.get("entity") or event.get("subject") or event.get("target"),
            "direction": event.get("direction") or _infer_motion_direction(label),
        }
    label = _event_label(event)
    return {"label": label, "kind": "", "entity": None, "direction": _infer_motion_direction(label)}


def _curve_mode_for_event(event_payload):
    label = event_payload["label"].lower()
    direction = event_payload.get("direction")
    kind = event_payload.get("kind", "")
    if direction == "left":
        return "dx", "min"
    if direction == "right":
        return "dx", "max"
    if direction == "up":
        return "dy", "min"
    if direction == "down":
        return "dy", "max"
    if any(token in label for token in ["start", "begin"]):
        return "target_motion", "onset"
    if any(token in label for token in ["stop", "end", "finish"]):
        return "target_motion", "offset"
    if kind == "transition":
        return "change", "max"
    if any(
        token in label
        for token in ["appear", "disappear", "pick", "place", "throw", "open", "close", "grab", "hit", "switch"]
    ):
        return "change", "max"
    return "combined", "max"


def _find_curve_position(curve, mode):
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0:
        return None
    if mode == "max":
        return int(np.argmax(curve))
    if mode == "min":
        return int(np.argmin(curve))
    if mode == "onset":
        hits = np.where(curve >= float(np.percentile(curve, 60)))[0]
        return int(hits[0]) if len(hits) else None
    if mode == "offset":
        peak = int(np.argmax(curve))
        hits = np.where(curve[peak:] <= float(np.percentile(curve, 35)))[0]
        return int(peak + hits[0]) if len(hits) else None
    return None


def _refine_curve_position(curve, mode, coarse_position=None, window_radius=4):
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0:
        return None
    if coarse_position is None:
        return _find_curve_position(curve, mode)
    left = max(0, int(coarse_position) - window_radius)
    right = min(curve.size, int(coarse_position) + window_radius + 1)
    local_curve = curve[left:right]
    local_pos = _find_curve_position(local_curve, mode)
    return int(left + local_pos) if local_pos is not None else None


def _curve_confidence(curve, position, mode):
    curve = np.asarray(curve, dtype=np.float32)
    if position is None or curve.size == 0:
        return None
    normalized = _normalize_curve(curve)
    value = float(normalized[position])
    return float(1.0 - value) if mode == "min" else value


class InterOrderMetric:
    """Compute event-order consistency from event sequence metadata and temporal evidence curves."""

    def __init__(
        self,
        event_verifier=None,
        embedding_backend=None,
        verbose=False,
        flow_backend=None,
    ):
        self.event_verifier = event_verifier
        self.embedding_backend = embedding_backend
        self.verbose = verbose
        self.flow_backend = flow_backend

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
        return {
            "motion_curve": np.asarray(motion_curve, dtype=np.float32),
            "dx_curve": np.asarray(dx_curve, dtype=np.float32),
            "dy_curve": np.asarray(dy_curve, dtype=np.float32),
        }

    def _verify_event_pair(self, pair_payload, detected_positions, frames, frame_indices, omni_spec):
        left, right = pair_payload
        left_label = _event_label(left)
        right_label = _event_label(right)

        verifier = self.event_verifier
        if verifier is not None:
            verifier_name = getattr(verifier, "__name__", verifier.__class__.__name__)
            try:
                if hasattr(verifier, "verify_pair"):
                    payload = verifier.verify_pair(
                        left_event=left,
                        right_event=right,
                        detected_event_positions=detected_positions,
                        frames=frames,
                        frame_indices=frame_indices,
                        omni_spec=omni_spec,
                    )
                else:
                    payload = verifier(
                        left_event=left,
                        right_event=right,
                        detected_event_positions=detected_positions,
                        frames=frames,
                        frame_indices=frame_indices,
                        omni_spec=omni_spec,
                    )
            except TypeError:
                try:
                    payload = verifier(left, right, detected_positions, frames, frame_indices, omni_spec)
                except Exception:
                    payload = None
            except Exception:
                payload = None

            if isinstance(payload, dict):
                score = payload.get("score", payload.get("semantic_score"))
                try:
                    score = None if score is None else float(score)
                except Exception:
                    score = None
                if payload.get("is_ordered") is not None:
                    return {
                        "left": left_label,
                        "right": right_label,
                        "is_ordered": bool(payload.get("is_ordered")),
                        "verification_mode": "semantic",
                        "score": score,
                        "hook_name": verifier_name,
                    }
            elif payload is not None:
                return {
                    "left": left_label,
                    "right": right_label,
                    "is_ordered": bool(payload),
                    "verification_mode": "semantic",
                    "score": None,
                    "hook_name": verifier_name,
                }

        left_pos = detected_positions.get(left_label)
        right_pos = detected_positions.get(right_label)
        if left_pos is None or right_pos is None:
            return {
                "left": left_label,
                "right": right_label,
                "is_ordered": None,
                "verification_mode": "heuristic",
                "score": None,
                "hook_name": None,
            }
        return {
            "left": left_label,
            "right": right_label,
            "is_ordered": bool(left_pos <= right_pos),
            "verification_mode": "heuristic",
            "score": None,
            "hook_name": None,
        }

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run InterOrder on a chunk using event-sequence pairs and semantic or proxy temporal ordering evidence."""
        event_sequence = omni_spec.get("event_sequence") or []
        if len(event_sequence) < 2 or len(all_frames) < 3:
            return make_metric_result(
                "interorder",
                "interorder",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "insufficient_event_sequence",
                    "event_sequence_used": deepcopy(event_sequence),
                    "num_expected_events": len(event_sequence),
                    "num_event_pairs": max(0, (len(event_sequence) * (len(event_sequence) - 1)) // 2),
                    "num_verified_pairs": 0,
                    "num_violated_pairs": 0,
                    "num_missing_events": len(event_sequence),
                    "detected_event_positions": [],
                    "verified_pairs": [],
                    "violated_pairs": [],
                    "missing_events": [_event_label(event) for event in event_sequence],
                    "evidence_per_event": {},
                    "pair_evidence": [],
                    "used_fallback_reason": None,
                },
                num_expected_events=len(event_sequence),
                num_verified_pairs=0,
            )

        shared_context = shared_context or {}
        analysis_frames = analysis_sample_frames(all_frames, sampled_frames, min_frames=4)
        target_mask, target_details = resolve_target_region(
            omni_spec,
            all_frames,
            sampled_frames=analysis_frames,
            flow_backend=self.flow_backend,
        )
        target_motion = self._mask_motion_signature(all_frames, target_mask, shared_context=shared_context)
        if target_motion is None:
            return make_metric_result(
                "interorder",
                "interorder",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "no_motion_field",
                    "event_sequence_used": deepcopy(event_sequence),
                    "num_expected_events": len(event_sequence),
                    "num_event_pairs": max(0, (len(event_sequence) * (len(event_sequence) - 1)) // 2),
                    "num_verified_pairs": 0,
                    "num_violated_pairs": 0,
                    "num_missing_events": len(event_sequence),
                    "detected_event_positions": [],
                    "verified_pairs": [],
                    "violated_pairs": [],
                    "missing_events": [_event_label(event) for event in event_sequence],
                    "evidence_per_event": {},
                    "pair_evidence": [],
                    "used_fallback_reason": "no_motion_field",
                },
                num_expected_events=len(event_sequence),
                num_verified_pairs=0,
            )

        full_change_curve = np.asarray(
            [
                light_frame_change(frame_a, frame_b, target_mask)
                for frame_a, frame_b in zip(all_frames[:-1], all_frames[1:])
            ],
            dtype=np.float32,
        )
        full_curves = {
            "target_motion": _smooth_curve(target_motion["motion_curve"]),
            "change": _smooth_curve(full_change_curve),
            "dx": _smooth_curve(target_motion["dx_curve"]),
            "dy": _smooth_curve(target_motion["dy_curve"]),
        }
        full_curves["combined"] = _normalize_curve(full_curves["target_motion"]) + _normalize_curve(
            full_curves["change"]
        )

        sampled_curve_map = {}
        if analysis_frames and len(analysis_frames) >= 2:
            sampled_target_motion = self._mask_motion_signature(analysis_frames, target_mask, shared_context=None)
            if sampled_target_motion is not None:
                sampled_change_curve = np.asarray(
                    [
                        light_frame_change(frame_a, frame_b, target_mask)
                        for frame_a, frame_b in zip(analysis_frames[:-1], analysis_frames[1:])
                    ],
                    dtype=np.float32,
                )
                sampled_curve_map = {
                    "target_motion": _smooth_curve(sampled_target_motion["motion_curve"]),
                    "change": _smooth_curve(sampled_change_curve),
                    "dx": _smooth_curve(sampled_target_motion["dx_curve"]),
                    "dy": _smooth_curve(sampled_target_motion["dy_curve"]),
                }
                sampled_curve_map["combined"] = _normalize_curve(
                    sampled_curve_map["target_motion"]
                ) + _normalize_curve(sampled_curve_map["change"])

        detected_event_positions = []
        evidence_per_event = {}
        missing_events = []
        for event in event_sequence:
            payload = _event_payload(event)
            curve_name, mode = _curve_mode_for_event(payload)
            sampled_curve = sampled_curve_map.get(curve_name)
            coarse_position = None
            if sampled_curve is not None and sampled_curve.size > 0 and sampled_indices and len(sampled_indices) >= 2:
                sampled_position = _find_curve_position(sampled_curve, mode)
                if sampled_position is not None:
                    sampled_position = int(min(sampled_position, len(sampled_indices) - 2))
                    coarse_position = int(
                        round(0.5 * (sampled_indices[sampled_position] + sampled_indices[sampled_position + 1]))
                    )
            refined_position = _refine_curve_position(
                full_curves[curve_name],
                mode,
                coarse_position=coarse_position,
                window_radius=max(2, len(all_frames) // 8),
            )
            confidence = _curve_confidence(full_curves[curve_name], refined_position, mode)
            evidence_per_event[payload["label"]] = {
                "curve": curve_name,
                "mode": mode,
                "coarse_position": coarse_position,
                "refined_position": refined_position,
                "confidence": confidence,
            }
            if confidence is None or confidence < 0.18:
                missing_events.append(payload["label"])
                detected_event_positions.append(
                    {
                        "label": payload["label"],
                        "detected_position": None,
                        "curve": curve_name,
                        "mode": mode,
                        "confidence": confidence,
                    }
                )
                continue
            detected_event_positions.append(
                {
                    "label": payload["label"],
                    "detected_position": int(refined_position),
                    "curve": curve_name,
                    "mode": mode,
                    "confidence": confidence,
                    "coarse_position": coarse_position,
                }
            )

        detected_map = {
            item["label"]: item["detected_position"]
            for item in detected_event_positions
            if item["detected_position"] is not None
        }
        pair_payloads = build_event_pairs(event_sequence)
        verified_pairs = []
        violated_pairs = []
        unresolved_pairs = []
        pair_evidence = []
        semantic_used = False
        for pair_payload in pair_payloads:
            pair_result = self._verify_event_pair(
                pair_payload,
                detected_map,
                analysis_frames,
                sampled_indices,
                omni_spec,
            )
            semantic_used = semantic_used or pair_result.get("verification_mode") == "semantic"
            record = {
                "left": pair_result["left"],
                "right": pair_result["right"],
                "verification_mode": pair_result["verification_mode"],
                "score": pair_result.get("score"),
                "hook_name": pair_result.get("hook_name"),
                "is_ordered": pair_result.get("is_ordered"),
            }
            pair_evidence.append(record)
            if pair_result.get("is_ordered") is None:
                unresolved_pairs.append(record)
                continue
            if pair_result["is_ordered"]:
                verified_pairs.append(record)
            else:
                violated_pairs.append(record)

        resolved_pairs = len(verified_pairs) + len(violated_pairs)
        score = float(len(verified_pairs) / resolved_pairs) if resolved_pairs else None

        if semantic_used and resolved_pairs:
            verification_mode = "semantic_partial_faithful"
            faithfulness = "partial faithful"
            used_fallback = False
            used_fallback_reason = None
        elif resolved_pairs:
            verification_mode = "fallback"
            faithfulness = "fallback approximation"
            used_fallback = True
            used_fallback_reason = "proxy_temporal_event_extraction"
        else:
            verification_mode = "fallback"
            faithfulness = "fallback approximation"
            used_fallback = True
            used_fallback_reason = "no_resolved_event_pairs"

        if resolved_pairs:
            status = "partial"
        elif missing_events:
            status = "partial"
        else:
            status = "failed"

        return make_metric_result(
            "interorder",
            "interorder",
            score,
            verification_mode=verification_mode,
            faithfulness_status=faithfulness,
            used_fallback=used_fallback,
            status=status,
            details={
                "event_sequence_used": deepcopy(event_sequence),
                "num_expected_events": len(event_sequence),
                "num_event_pairs": len(pair_payloads),
                "num_verified_pairs": len(verified_pairs),
                "num_violated_pairs": len(violated_pairs),
                "num_missing_events": len(missing_events),
                "detected_event_positions": detected_event_positions,
                "verified_pairs": verified_pairs,
                "violated_pairs": violated_pairs,
                "missing_events": missing_events,
                "evidence_per_event": evidence_per_event,
                "pair_evidence": pair_evidence,
                "target_region_source": target_details.get("target_region_source"),
                "used_fallback_reason": used_fallback_reason,
            },
            num_expected_events=len(event_sequence),
            num_verified_pairs=resolved_pairs,
        )

    __call__ = run
