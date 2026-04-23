from __future__ import annotations

from typing import Any

import numpy as np


OMNI_SCORE_KEYS = [
    "interstab_l",
    "interstab_n",
    "intercov",
    "interorder",
    "transitions_detect",
    "object_control",
    "camera_control",
]

OMNI_CLIP_LOG_COLUMNS = {
    "interstab_l": "omni_interstab_l",
    "interstab_n": "omni_interstab_n",
    "intercov": "omni_intercov",
    "interorder": "omni_interorder",
    "transitions_detect": "omni_transitions_detect",
    "object_control": "omni_object_control",
    "camera_control": "omni_camera_control",
}

OMNI_IMPLEMENTATION_STATUS = {
    "interstab_l": "article-faithful",
    "interstab_n": "partial faithful",
    "intercov": "partial faithful",
    "interorder": "partial faithful",
    "transitions_detect": "article-faithful",
    "object_control": "partial faithful",
    "camera_control": "partial faithful",
    "agentic_score": "fallback approximation",
}

OMNI_FRAME_USAGE = {
    "interstab_l": "full_frames",
    "interstab_n": "both",
    "intercov": "both",
    "interorder": "both",
    "transitions_detect": "full_frames",
    "object_control": "sampled_frames",
    "camera_control": "full_frames",
    "agentic_score": "summary_only",
}

OMNI_REQUIRED_PROTOCOL_FIELDS = [
    "score",
    "verification_mode",
    "faithfulness_status",
    "used_fallback",
    "status",
    "details",
]

OMNI_VISUAL_STACK_KEYS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
]


def normalize_verification_mode(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        ordered = []
        seen = set()
        for item in value:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return "|".join(ordered)
    return str(value)


def _base_metric_details(
    metric_name: str,
    verification_mode: Any,
    faithfulness_status: str | None,
    used_fallback: bool,
    status: str,
    **kwargs,
) -> dict[str, Any]:
    payload = {
        "verification_mode": normalize_verification_mode(verification_mode),
        "faithfulness_status": faithfulness_status,
        "used_fallback": bool(used_fallback),
        "status": status,
        "frame_usage": OMNI_FRAME_USAGE.get(metric_name),
    }
    payload.update(kwargs)
    return payload


def make_metric_result(
    metric_name: str,
    score_key: str | None = None,
    score: float | None = None,
    verification_mode: Any = "skipped",
    faithfulness_status: str | None = None,
    used_fallback: bool = False,
    status: str = "skipped",
    details: dict[str, Any] | None = None,
    **extra,
) -> dict[str, Any]:
    """Build a standardized Omni metric payload while preserving metric-specific score fields."""
    score_key = score_key or metric_name
    details_payload = details or {}
    if not isinstance(details_payload, dict):
        details_payload = {"value": details_payload}
    details_payload = _base_metric_details(
        metric_name,
        verification_mode=verification_mode,
        faithfulness_status=faithfulness_status,
        used_fallback=used_fallback,
        status=status,
        **details_payload,
    )
    result = {
        score_key: score,
        "score": score,
        "verification_mode": normalize_verification_mode(verification_mode),
        "faithfulness_status": faithfulness_status,
        "used_fallback": bool(used_fallback),
        "status": status,
        "details": details_payload,
    }
    result.update(extra)
    return result


def omni_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): omni_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [omni_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def extract_metric_summary(metric_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": payload.get(metric_name, payload.get("score")),
        "status": payload.get("status"),
        "verification_mode": payload.get("verification_mode"),
        "faithfulness_status": payload.get("faithfulness_status"),
        "used_fallback": payload.get("used_fallback"),
        "details": omni_jsonable(payload.get("details", {})),
    }


def aggregate_omni_status(
    metric_statuses: dict[str, str],
    metric_faithfulness: dict[str, str],
) -> str:
    core_metrics = [
        "interstab_l",
        "interstab_n",
        "intercov",
        "interorder",
        "transitions_detect",
        "object_control",
    ]
    statuses = [metric_statuses.get(metric, "failed") for metric in core_metrics]
    if all(status == "ok" for status in statuses) and all(
        metric_faithfulness.get(metric) == "article-faithful"
        for metric in core_metrics
        if metric in metric_faithfulness
    ):
        return "ok"
    if any(status in {"ok", "partial", "skipped"} for status in statuses):
        return "partial"
    return "failed"


def empty_omni_run_result() -> dict[str, Any]:
    payload = {metric_name: None for metric_name in OMNI_SCORE_KEYS}
    payload.update(
        {
            "omni_status": "failed",
            "omni_metric_statuses": {},
            "omni_metric_faithfulness": {},
            "omni_metric_verification_modes": {},
            "omni_details": {},
        }
    )
    return payload
