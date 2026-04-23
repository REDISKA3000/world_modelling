from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .agentic_score import compute_agentic_score
from .common.results import OMNI_CLIP_LOG_COLUMNS, OMNI_SCORE_KEYS, omni_jsonable


def _parse_json_if_needed(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def add_optional_full_rollout_omni_summary(clip_log, rollout_record=None):
    """Aggregate rollout-level Omni summary fields from per-chunk clip_log rows."""
    rollout_record = rollout_record or {}
    if not clip_log:
        return rollout_record
    df = pd.DataFrame(clip_log)
    chunk_ids = pd.to_numeric(df["chunk_idx"], errors="coerce") if "chunk_idx" in df.columns else pd.Series(range(len(df)))

    for clip_col in OMNI_CLIP_LOG_COLUMNS.values():
        if clip_col not in df.columns:
            continue
        numeric_values = pd.to_numeric(df[clip_col], errors="coerce")
        trajectory = []
        for idx, value in zip(chunk_ids.tolist(), numeric_values.tolist()):
            trajectory.append({"chunk_idx": None if pd.isna(idx) else int(idx), "value": None if pd.isna(value) else float(value)})
        valid_values = numeric_values.dropna()
        rollout_record[f"{clip_col}_trajectory"] = json.dumps(omni_jsonable(trajectory), ensure_ascii=False)
        rollout_record[f"{clip_col}_valid_count"] = int(valid_values.shape[0])
        rollout_record[f"{clip_col}_missing_count"] = int(numeric_values.shape[0] - valid_values.shape[0])
        if not valid_values.empty:
            rollout_record[f"{clip_col}_mean"] = float(valid_values.mean())
            rollout_record[f"{clip_col}_min"] = float(valid_values.min())
            rollout_record[f"{clip_col}_max"] = float(valid_values.max())

    status_maps = []
    faithfulness_maps = []
    verification_maps = []
    if "omni_metric_statuses" in df.columns:
        status_maps = [_parse_json_if_needed(value) for value in df["omni_metric_statuses"].fillna("{}").tolist()]
    if "omni_metric_faithfulness" in df.columns:
        faithfulness_maps = [_parse_json_if_needed(value) for value in df["omni_metric_faithfulness"].fillna("{}").tolist()]
    if "omni_metric_verification_modes" in df.columns:
        verification_maps = [_parse_json_if_needed(value) for value in df["omni_metric_verification_modes"].fillna("{}").tolist()]
    if not status_maps or not faithfulness_maps or not verification_maps:
        fallback_maps = [_parse_json_if_needed(value).get("metric_statuses", {}) for value in df.get("omni_details", pd.Series(["{}"] * len(df))).fillna("{}").tolist()]
        fallback_faithfulness = [_parse_json_if_needed(value).get("metric_faithfulness", {}) for value in df.get("omni_details", pd.Series(["{}"] * len(df))).fillna("{}").tolist()]
        fallback_verification = [_parse_json_if_needed(value).get("metric_verification_modes", {}) for value in df.get("omni_details", pd.Series(["{}"] * len(df))).fillna("{}").tolist()]
        status_maps = status_maps or fallback_maps
        faithfulness_maps = faithfulness_maps or fallback_faithfulness
        verification_maps = verification_maps or fallback_verification

    for metric_name in OMNI_SCORE_KEYS:
        if status_maps:
            counts = pd.Series([payload.get(metric_name, "missing") for payload in status_maps]).value_counts().to_dict()
            rollout_record[f"omni_{metric_name}_status_counts"] = json.dumps(omni_jsonable(counts), ensure_ascii=False)
        if faithfulness_maps:
            counts = pd.Series([payload.get(metric_name, "missing") for payload in faithfulness_maps]).value_counts().to_dict()
            rollout_record[f"omni_{metric_name}_faithfulness_counts"] = json.dumps(omni_jsonable(counts), ensure_ascii=False)
        if verification_maps:
            counts = pd.Series([payload.get(metric_name, "missing") for payload in verification_maps]).value_counts().to_dict()
            rollout_record[f"omni_{metric_name}_verification_mode_counts"] = json.dumps(omni_jsonable(counts), ensure_ascii=False)

    if "omni_status" in df.columns:
        rollout_record["omni_status_counts"] = json.dumps(
            omni_jsonable(df["omni_status"].fillna("missing").value_counts().to_dict()),
            ensure_ascii=False,
        )

    agentic_rows = [compute_agentic_score(row.to_dict()) for _, row in df.iterrows()]
    agentic_scores = [payload.get("agentic_score") for payload in agentic_rows if payload.get("agentic_score") is not None]
    if agentic_scores:
        rollout_record["omni_agentic_score_mean"] = float(np.mean(agentic_scores))
        rollout_record["omni_agentic_score_min"] = float(np.min(agentic_scores))
        rollout_record["omni_agentic_score_max"] = float(np.max(agentic_scores))
        rollout_record["omni_agentic_score_valid_count"] = int(len(agentic_scores))
        rollout_record["omni_agentic_score_missing_count"] = int(len(agentic_rows) - len(agentic_scores))
        rollout_record["omni_agentic_score_trajectory"] = json.dumps(
            omni_jsonable(
                [
                    {"chunk_idx": None if pd.isna(idx) else int(idx), "value": payload.get("agentic_score")}
                    for idx, payload in zip(chunk_ids.tolist(), agentic_rows)
                ]
            ),
            ensure_ascii=False,
        )
    return rollout_record
