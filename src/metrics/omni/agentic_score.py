from __future__ import annotations

import numpy as np

from .common.results import OMNI_VISUAL_STACK_KEYS, make_metric_result


def safe_omni_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def compute_agentic_score(metric_record, visual_metric_record=None, fixed_weights=None):
    """Fallback approximation of AgenticScore from existing Omni and VBench outputs."""
    record = {}
    if isinstance(metric_record, dict):
        record.update(metric_record)
    if isinstance(visual_metric_record, dict):
        record.update(visual_metric_record)
    fixed_weights = fixed_weights or {"A_I": 0.4, "A_C": 0.3, "A_G": 0.3}

    def _pick(*keys):
        for key in keys:
            if key in record:
                value = safe_omni_float(record.get(key))
                if value is not None:
                    return value
        return None

    interaction_values = [
        _pick("omni_interstab_l", "interstab_l"),
        _pick("omni_interstab_n", "interstab_n"),
        _pick("omni_intercov", "intercov"),
        _pick("omni_interorder", "interorder"),
    ]
    control_values = [
        _pick("omni_transitions_detect", "transitions_detect"),
        _pick("omni_object_control", "object_control"),
        _pick("omni_camera_control", "camera_control"),
    ]
    visual_values = [
        safe_omni_float(record.get(key))
        for key in OMNI_VISUAL_STACK_KEYS
        if safe_omni_float(record.get(key)) is not None
    ]
    a_i = (
        float(np.mean([value for value in interaction_values if value is not None]))
        if any(value is not None for value in interaction_values)
        else None
    )
    a_c = (
        float(np.mean([value for value in control_values if value is not None]))
        if any(value is not None for value in control_values)
        else None
    )
    a_g = float(np.mean(visual_values)) if visual_values else None

    weighted_components = {k: v for k, v in {"A_I": a_i, "A_C": a_c, "A_G": a_g}.items() if v is not None}
    if not weighted_components:
        return make_metric_result(
            "agentic_score",
            "agentic_score",
            None,
            verification_mode="skipped",
            faithfulness_status="fallback approximation",
            used_fallback=True,
            status="skipped",
            details={"reason": "no_components_available"},
            A_I=a_i,
            A_C=a_c,
            A_G=a_g,
        )
    weights_used = {key: fixed_weights[key] for key in weighted_components}
    weight_sum = float(sum(weights_used.values()))
    if weight_sum <= 1e-8:
        return make_metric_result(
            "agentic_score",
            "agentic_score",
            None,
            verification_mode="fixed_weights_fallback",
            faithfulness_status="fallback approximation",
            used_fallback=True,
            status="failed",
            details={"reason": "invalid_weights"},
            A_I=a_i,
            A_C=a_c,
            A_G=a_g,
        )
    normalized = {key: value / weight_sum for key, value in weights_used.items()}
    score = float(sum(weighted_components[key] * normalized[key] for key in weighted_components))
    return make_metric_result(
        "agentic_score",
        "agentic_score",
        score,
        verification_mode="fixed_weights_fallback",
        faithfulness_status="fallback approximation",
        used_fallback=True,
        status="partial",
        details={
            "A_I": a_i,
            "A_C": a_c,
            "A_G": a_g,
            "weighting_mode": "fixed_weights_fallback",
            "weights_used": normalized,
        },
        A_I=a_i,
        A_C=a_c,
        A_G=a_g,
        weighting_mode="fixed_weights_fallback",
        weights_used=normalized,
    )
