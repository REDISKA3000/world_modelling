from __future__ import annotations

import json
from pathlib import Path


def read_vbench_eval_json(eval_json_path: str | Path) -> dict:
    with open(eval_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_vbench_result(
    raw_result: dict | None,
    dimensions=None,
    prefix: str = "vbench_",
) -> dict:
    raw_result = raw_result or {}
    requested_dimensions = list(dimensions or raw_result.keys())

    dimension_scores = {}
    flat_scores = {}
    for dim in requested_dimensions:
        payload = raw_result.get(dim)
        score = None
        if isinstance(payload, list) and len(payload) >= 1:
            score = payload[0]
        dimension_scores[dim] = score
        flat_scores[f"{prefix}{dim}"] = score

    missing_dimensions = [dim for dim, score in dimension_scores.items() if score is None]
    if not requested_dimensions:
        status = "failed"
    elif len(missing_dimensions) == len(requested_dimensions):
        status = "failed"
    elif missing_dimensions:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "dimensions": dimension_scores,
        "flat_scores": flat_scores,
        "missing_dimensions": missing_dimensions,
        "raw_result": raw_result,
    }
