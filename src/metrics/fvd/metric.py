from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .features import I3DFeatureExtractor
from .pairs import (
    prepare_pairs_from_lists,
    prepare_pairs_from_manifest,
    validate_pair_dataframe,
)
from .utils import (
    compute_stats,
    frechet_distance,
    get_video_frame_count,
    normalize_resize_to,
    resolve_num_frames,
)


class FVDMetric:
    def __init__(
        self,
        feature_extractor=None,
        device=None,
        batch_size=4,
        num_frames=None,
        resize_to=None,
        sampling_strategy="uniform",
        max_pairs=None,
        cache_features=False,
        cache_dir=None,
        verbose=False,
    ) -> None:
        self.device = device
        self.batch_size = int(batch_size)
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.sampling_strategy = sampling_strategy
        self.max_pairs = max_pairs
        self.cache_features = bool(cache_features)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.verbose = verbose

        self.feature_extractor = feature_extractor or I3DFeatureExtractor(
            device=device,
            verbose=verbose,
        )

        if self.cache_features and self.cache_dir is None:
            self.cache_dir = Path(".fvd_feature_cache")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def init_feature_extractor(self):
        initialize = getattr(self.feature_extractor, "initialize", None)
        if callable(initialize):
            return initialize()
        return self.feature_extractor

    def unload_feature_extractor(self) -> None:
        unload = getattr(self.feature_extractor, "unload", None)
        if callable(unload):
            unload()
        gc.collect()

    def validate_manifest(self, manifest_csv) -> pd.DataFrame:
        return validate_pair_dataframe(
            pd.read_csv(manifest_csv) if not isinstance(manifest_csv, pd.DataFrame) else manifest_csv.copy()
        )

    def prepare_pairs_from_manifest(self, manifest_csv, max_videos=None) -> list[dict[str, str]]:
        effective_max_pairs = self.max_pairs if max_videos is None else max_videos
        return prepare_pairs_from_manifest(manifest_csv, max_pairs=effective_max_pairs)

    def prepare_pairs(
        self,
        real_video_paths: Iterable[str | Path],
        generated_video_paths: Iterable[str | Path],
        max_pairs: int | None = None,
    ) -> list[dict[str, str]]:
        effective_max_pairs = self.max_pairs if max_pairs is None else max_pairs
        return prepare_pairs_from_lists(
            real_video_paths=real_video_paths,
            generated_video_paths=generated_video_paths,
            max_pairs=effective_max_pairs,
        )

    def _resolve_num_frames(self, num_frames: int | None = None) -> int:
        return resolve_num_frames(num_frames if num_frames is not None else self.num_frames)

    def _resolve_resize_to(self, resize=None) -> tuple[int, int]:
        value = resize if resize is not None else self.resize_to
        return normalize_resize_to(value)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    def _load_cached_feature(self, cache_key: str) -> np.ndarray | None:
        if not self.cache_features or self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cached_feature(self, cache_key: str, feature: np.ndarray) -> None:
        if not self.cache_features or self.cache_dir is None:
            return
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, feature)

    def _build_result(
        self,
        *,
        status: str,
        pairing_mode: str,
        pair_records: list[dict[str, str]],
        valid_pair_records: list[dict[str, str]],
        bad_examples: list[tuple[str, str, str]],
        fvd: float | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempted_real_paths = [str(pair["real_video_path"]) for pair in pair_records]
        attempted_generated_paths = [str(pair["generated_video_path"]) for pair in pair_records]
        valid_real_paths = [str(pair["real_video_path"]) for pair in valid_pair_records]
        valid_generated_paths = [str(pair["generated_video_path"]) for pair in valid_pair_records]

        return {
            "fvd": float(fvd) if fvd is not None else None,
            "n_used": len(valid_pair_records),
            "n_failed": len(bad_examples),
            "bad_examples": bad_examples[:10],
            "num_pairs": len(pair_records),
            "num_valid_pairs": len(valid_pair_records),
            "num_failed_pairs": len(bad_examples),
            "status": status,
            "error": error,
            "pairing_mode": pairing_mode,
            "used_generated_paths": attempted_generated_paths,
            "used_real_paths": attempted_real_paths,
            "details": {
                "valid_real_paths": valid_real_paths,
                "valid_generated_paths": valid_generated_paths,
                **(details or {}),
            },
        }

    def extract_video_feature(
        self,
        video_path,
        num_frames=None,
        resize=None,
        sampling_strategy=None,
    ) -> np.ndarray:
        resolved_num_frames = self._resolve_num_frames(num_frames)
        resolved_resize = self._resolve_resize_to(resize)
        resolved_strategy = sampling_strategy or self.sampling_strategy

        cache_key = self._cache_key(
            {
                "kind": "video",
                "video_path": str(video_path),
                "num_frames": resolved_num_frames,
                "resize_to": resolved_resize,
                "sampling_strategy": resolved_strategy,
            }
        )
        cached = self._load_cached_feature(cache_key)
        if cached is not None:
            return cached

        extract = getattr(self.feature_extractor, "extract_video_feature", None)
        if not callable(extract):
            raise AttributeError("feature_extractor must implement extract_video_feature(...)")

        feature = extract(
            video_path=video_path,
            num_frames=resolved_num_frames,
            resize_to=resolved_resize,
            sampling_strategy=resolved_strategy,
        )
        self._save_cached_feature(cache_key, feature)
        return feature

    def extract_subclip_feature(
        self,
        video_path,
        start_frame=0,
        num_frames=None,
        resize=None,
    ) -> np.ndarray:
        resolved_num_frames = self._resolve_num_frames(num_frames)
        resolved_resize = self._resolve_resize_to(resize)

        cache_key = self._cache_key(
            {
                "kind": "subclip",
                "video_path": str(video_path),
                "start_frame": int(start_frame),
                "num_frames": resolved_num_frames,
                "resize_to": resolved_resize,
            }
        )
        cached = self._load_cached_feature(cache_key)
        if cached is not None:
            return cached

        extract = getattr(self.feature_extractor, "extract_subclip_feature", None)
        if not callable(extract):
            raise AttributeError("feature_extractor must implement extract_subclip_feature(...)")

        feature = extract(
            video_path=video_path,
            start_frame=int(start_frame),
            num_frames=resolved_num_frames,
            resize_to=resolved_resize,
        )
        self._save_cached_feature(cache_key, feature)
        return feature

    @staticmethod
    def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return compute_stats(features)

    @staticmethod
    def frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        return frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)

    def compute_fvd_from_pairs(
        self,
        real_video_paths: Iterable[str | Path],
        generated_video_paths: Iterable[str | Path],
        num_frames=None,
        resize=None,
        max_pairs=None,
    ) -> dict[str, Any]:
        pairs = self.prepare_pairs(
            real_video_paths=real_video_paths,
            generated_video_paths=generated_video_paths,
            max_pairs=max_pairs,
        )
        return self._compute_fvd_for_pair_records(
            pair_records=pairs,
            num_frames=num_frames,
            resize=resize,
            pairing_mode="explicit_lists",
        )

    def compute_fvd_from_manifest(
        self,
        manifest_csv,
        num_frames=None,
        resize=(224, 224),
        max_videos=None,
    ) -> dict[str, Any]:
        pairs = self.prepare_pairs_from_manifest(manifest_csv, max_videos=max_videos)
        return self._compute_fvd_for_pair_records(
            pair_records=pairs,
            num_frames=num_frames,
            resize=resize,
            pairing_mode="manifest",
        )

    def run(
        self,
        manifest_csv=None,
        real_video_paths: Iterable[str | Path] | None = None,
        generated_video_paths: Iterable[str | Path] | None = None,
        num_frames=None,
        resize=None,
        max_pairs=None,
        max_videos=None,
    ) -> dict[str, Any]:
        if manifest_csv is not None:
            if real_video_paths is not None or generated_video_paths is not None:
                raise ValueError(
                    "run(...) accepts either manifest_csv or explicit video path lists, not both."
                )
            return self.compute_fvd_from_manifest(
                manifest_csv=manifest_csv,
                num_frames=num_frames,
                resize=resize,
                max_videos=max_videos,
            )

        if real_video_paths is None or generated_video_paths is None:
            raise ValueError(
                "run(...) requires either manifest_csv or both real_video_paths and generated_video_paths."
            )

        return self.compute_fvd_from_pairs(
            real_video_paths=real_video_paths,
            generated_video_paths=generated_video_paths,
            num_frames=num_frames,
            resize=resize,
            max_pairs=max_pairs,
        )

    def _compute_fvd_for_pair_records(
        self,
        pair_records: list[dict[str, str]],
        num_frames=None,
        resize=None,
        pairing_mode: str = "unknown",
    ) -> dict[str, Any]:
        real_feats = []
        gen_feats = []
        bad = []
        valid_pairs = []

        for index, pair in enumerate(pair_records):
            real_path = pair["real_video_path"]
            gen_path = pair["generated_video_path"]

            try:
                real_feat = self.extract_video_feature(
                    real_path,
                    num_frames=num_frames,
                    resize=resize,
                )
                gen_feat = self.extract_video_feature(
                    gen_path,
                    num_frames=num_frames,
                    resize=resize,
                )
                real_feats.append(real_feat)
                gen_feats.append(gen_feat)
                valid_pairs.append(pair)
            except Exception as exc:
                bad.append((real_path, gen_path, str(exc)))
                if self.verbose:
                    print(f"[bad {index}] {exc}")

        if not real_feats or not gen_feats:
            return self._build_result(
                status="failed",
                pairing_mode=pairing_mode,
                pair_records=pair_records,
                valid_pair_records=valid_pairs,
                bad_examples=bad,
                error="No valid feature pairs were extracted.",
                details={
                    "num_requested_frames": self._resolve_num_frames(num_frames),
                    "resize_to": self._resolve_resize_to(resize),
                },
            )

        real_feats_array = np.stack(real_feats, axis=0)
        gen_feats_array = np.stack(gen_feats, axis=0)

        mu_r, sigma_r = self.compute_stats(real_feats_array)
        mu_g, sigma_g = self.compute_stats(gen_feats_array)
        fvd_value = self.frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

        return self._build_result(
            status="ok",
            pairing_mode=pairing_mode,
            pair_records=pair_records,
            valid_pair_records=valid_pairs,
            bad_examples=bad,
            fvd=float(fvd_value),
            details={
                "num_requested_frames": self._resolve_num_frames(num_frames),
                "resize_to": self._resolve_resize_to(resize),
            },
        )

    def compute_full_rollout_fvd(
        self,
        real_video_path,
        generated_video_path,
        num_frames=None,
        resize=None,
    ) -> dict[str, Any]:
        return self.compute_fvd_from_pairs(
            real_video_paths=[real_video_path],
            generated_video_paths=[generated_video_path],
            num_frames=num_frames,
            resize=resize,
            max_pairs=1,
        )

    def run_on_rollout(
        self,
        real_video_path,
        generated_video_path,
        num_frames=None,
        resize=None,
    ) -> dict[str, Any]:
        return self.compute_full_rollout_fvd(
            real_video_path=real_video_path,
            generated_video_path=generated_video_path,
            num_frames=num_frames,
            resize=resize,
        )

    def build_full_rollout_callback(
        self,
        real_video_path,
        num_frames=None,
        resize=None,
    ):
        def callback(full_rollout_path):
            if full_rollout_path is None:
                raise ValueError("full_rollout_path is None")
            return self.compute_full_rollout_fvd(
                real_video_path=real_video_path,
                generated_video_path=full_rollout_path,
                num_frames=num_frames,
                resize=resize,
            )

        return callback

    def extract_chunk_feature_set(
        self,
        video_path,
        clips_per_chunk=4,
        frames_per_clip=16,
        stride_mode="uniform",
        resize=(224, 224),
    ) -> np.ndarray:
        total_frames = get_video_frame_count(video_path)
        if total_frames <= 0:
            raise ValueError(f"Cannot read frame count: {video_path}")

        if total_frames <= frames_per_clip:
            starts = [0] * clips_per_chunk
        else:
            max_start = total_frames - frames_per_clip
            if stride_mode == "uniform":
                starts = np.linspace(0, max_start, clips_per_chunk).astype(int).tolist()
            else:
                raise ValueError(f"Unknown stride_mode: {stride_mode}")

        feats = []
        for start in starts:
            feat = self.extract_subclip_feature(
                video_path=video_path,
                start_frame=int(start),
                num_frames=frames_per_clip,
                resize=resize,
            )
            feats.append(feat)
        return np.stack(feats, axis=0)

    def compute_chunk_distribution(
        self,
        video_path,
        clips_per_chunk=4,
        frames_per_clip=16,
        resize=(224, 224),
    ) -> dict[str, Any]:
        feat_set = self.extract_chunk_feature_set(
            video_path=video_path,
            clips_per_chunk=clips_per_chunk,
            frames_per_clip=frames_per_clip,
            resize=resize,
        )
        mu, sigma = self.compute_stats(feat_set)
        return {
            "video_path": str(video_path),
            "features": feat_set,
            "mu": mu,
            "sigma": sigma,
        }

    def build_generated_chunk_distributions(
        self,
        generated_chunk_paths,
        clips_per_chunk=4,
        frames_per_clip=16,
        resize=(224, 224),
    ) -> list[dict[str, Any]]:
        chunk_distributions = []
        total_chunks = len(generated_chunk_paths)

        for idx, chunk_path in enumerate(generated_chunk_paths):
            if self.verbose:
                print(f"[chunk dist] {idx + 1}/{total_chunks} -> {chunk_path}")
            dist = self.compute_chunk_distribution(
                video_path=chunk_path,
                clips_per_chunk=clips_per_chunk,
                frames_per_clip=frames_per_clip,
                resize=resize,
            )
            chunk_distributions.append(dist)

        return chunk_distributions

    def compute_adjacent_cross_chunk_frechet(self, chunk_distributions) -> pd.DataFrame:
        records = []

        for idx in range(len(chunk_distributions) - 1):
            left = chunk_distributions[idx]
            right = chunk_distributions[idx + 1]

            dist = self.frechet_distance(
                left["mu"],
                left["sigma"],
                right["mu"],
                right["sigma"],
            )

            records.append(
                {
                    "left_chunk_idx": idx,
                    "right_chunk_idx": idx + 1,
                    "adjacent_cross_chunk_frechet": float(dist),
                    "left_video_path": left["video_path"],
                    "right_video_path": right["video_path"],
                }
            )

        return pd.DataFrame(records)

    def compute_drift_from_first_chunk(self, chunk_distributions) -> pd.DataFrame:
        if len(chunk_distributions) == 0:
            return pd.DataFrame([])

        first = chunk_distributions[0]
        records = []

        for idx, current in enumerate(chunk_distributions):
            dist = self.frechet_distance(
                first["mu"],
                first["sigma"],
                current["mu"],
                current["sigma"],
            )
            records.append(
                {
                    "chunk_idx": idx,
                    "drift_from_first_chunk": float(dist),
                    "first_video_path": first["video_path"],
                    "current_video_path": current["video_path"],
                }
            )

        return pd.DataFrame(records)

    def compute_chunk_consistency_metrics(
        self,
        generated_chunk_paths,
        clips_per_chunk=4,
        frames_per_clip=16,
        resize=(224, 224),
    ) -> dict[str, Any]:
        chunk_distributions = self.build_generated_chunk_distributions(
            generated_chunk_paths=generated_chunk_paths,
            clips_per_chunk=clips_per_chunk,
            frames_per_clip=frames_per_clip,
            resize=resize,
        )

        adjacent_df = self.compute_adjacent_cross_chunk_frechet(chunk_distributions)
        drift_df = self.compute_drift_from_first_chunk(chunk_distributions)

        return {
            "chunk_distributions": chunk_distributions,
            "adjacent_df": adjacent_df,
            "drift_df": drift_df,
        }

    def save_chunk_consistency_history(self, adjacent_df, drift_df, out_dir, prefix="rollout"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        adjacent_path = out_dir / f"{prefix}_adjacent_cross_chunk_frechet.csv"
        drift_path = out_dir / f"{prefix}_drift_from_first_chunk.csv"

        adjacent_df.to_csv(adjacent_path, index=False)
        drift_df.to_csv(drift_path, index=False)

        if self.verbose:
            print("saved:", adjacent_path)
            print("saved:", drift_path)

        return adjacent_path, drift_path

    def update_chunk_consistency_history(self, chunk_distributions, new_distribution):
        updated = chunk_distributions + [new_distribution]
        idx = len(updated) - 1

        adjacent_record = None
        if idx > 0:
            prev = updated[idx - 1]
            curr = updated[idx]
            adjacent_dist = self.frechet_distance(
                prev["mu"],
                prev["sigma"],
                curr["mu"],
                curr["sigma"],
            )
            adjacent_record = {
                "left_chunk_idx": idx - 1,
                "right_chunk_idx": idx,
                "adjacent_cross_chunk_frechet": float(adjacent_dist),
                "left_video_path": prev["video_path"],
                "right_video_path": curr["video_path"],
            }

        first = updated[0]
        curr = updated[idx]
        drift_dist = self.frechet_distance(
            first["mu"],
            first["sigma"],
            curr["mu"],
            curr["sigma"],
        )
        drift_record = {
            "chunk_idx": idx,
            "drift_from_first_chunk": float(drift_dist),
            "first_video_path": first["video_path"],
            "current_video_path": curr["video_path"],
        }

        return updated, adjacent_record, drift_record

    @staticmethod
    def init_rollout_metric_history() -> dict[str, list[Any]]:
        return {
            "chunk_distributions": [],
            "adjacent_records": [],
            "drift_records": [],
        }

    def compute_distribution_for_generated_chunk(
        self,
        chunk_video_path,
        clips_per_chunk=4,
        frames_per_clip=16,
        resize=(224, 224),
    ) -> dict[str, Any]:
        return self.compute_chunk_distribution(
            video_path=chunk_video_path,
            clips_per_chunk=clips_per_chunk,
            frames_per_clip=frames_per_clip,
            resize=resize,
        )

    def save_online_chunk_metric_history(self, metric_history, out_dir, prefix="rollout"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        adjacent_df = pd.DataFrame(metric_history["adjacent_records"])
        drift_df = pd.DataFrame(metric_history["drift_records"])

        adjacent_path = out_dir / f"{prefix}_adjacent_cross_chunk_frechet.csv"
        drift_path = out_dir / f"{prefix}_drift_from_first_chunk.csv"

        adjacent_df.to_csv(adjacent_path, index=False)
        drift_df.to_csv(drift_path, index=False)

        return adjacent_path, drift_path
