from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_PAIR_COLUMNS = ("real_video_path", "generated_video_path")


def load_manifest_dataframe(manifest_csv: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(manifest_csv, pd.DataFrame):
        return manifest_csv.copy()
    return pd.read_csv(manifest_csv)


def validate_pair_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PAIR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "FVD manifest is missing required columns: "
            + ", ".join(missing)
        )
    return df


def prepare_pairs_from_manifest(
    manifest_csv: str | Path | pd.DataFrame,
    max_pairs: int | None = None,
) -> list[dict[str, str]]:
    df = validate_pair_dataframe(load_manifest_dataframe(manifest_csv))
    if max_pairs is not None:
        df = df.head(int(max_pairs)).copy()

    records = df.loc[:, REQUIRED_PAIR_COLUMNS].to_dict(orient="records")
    return [
        {
            "real_video_path": str(record["real_video_path"]),
            "generated_video_path": str(record["generated_video_path"]),
        }
        for record in records
    ]


def prepare_pairs_from_lists(
    real_video_paths: Iterable[str | Path],
    generated_video_paths: Iterable[str | Path],
    max_pairs: int | None = None,
) -> list[dict[str, str]]:
    real_paths = [str(path) for path in real_video_paths]
    generated_paths = [str(path) for path in generated_video_paths]

    if len(real_paths) != len(generated_paths):
        raise ValueError(
            "real_video_paths and generated_video_paths must have the same length, "
            f"got {len(real_paths)} and {len(generated_paths)}."
        )

    pairs = [
        {
            "real_video_path": real_path,
            "generated_video_path": generated_path,
        }
        for real_path, generated_path in zip(real_paths, generated_paths)
    ]

    if max_pairs is not None:
        return pairs[: int(max_pairs)]
    return pairs
