from __future__ import annotations

import pandas as pd


DEFAULT_VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def summarize_chunk_vbench_metrics(clip_log, dimensions=None, prefix: str = "vbench_"):
    """Aggregate per-chunk VBench columns into rollout-level mean/min/max fields."""
    df = pd.DataFrame(clip_log)
    summary = {}
    for dim in list(dimensions or DEFAULT_VBENCH_DIMENSIONS):
        col = f"{prefix}{dim}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else None
            summary[f"{col}_min"] = float(vals.min()) if vals.notna().any() else None
            summary[f"{col}_max"] = float(vals.max()) if vals.notna().any() else None
    return summary
