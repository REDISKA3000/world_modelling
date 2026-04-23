from __future__ import annotations

import importlib
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from .parser import parse_vbench_result, read_vbench_eval_json
from .summary import DEFAULT_VBENCH_DIMENSIONS


class VBenchMetric:
    """Run and normalize VBench evaluation for chunk and rollout videos."""

    def __init__(
        self,
        dimensions=None,
        device=None,
        output_dir=None,
        cache_dir=None,
        verbose=False,
        repo_root=None,
        full_info_json=None,
        temp_input_root=None,
    ):
        self.dimensions = list(dimensions or DEFAULT_VBENCH_DIMENSIONS)
        self.device = device
        self.output_dir = output_dir or "evaluation_results"
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.repo_root = repo_root or os.environ.get("VBENCH_REPO_ROOT", "/content/VBench")
        self.full_info_json = full_info_json
        self.temp_input_root = temp_input_root or "/content/vbench_inputs"
        self._backend = None
        self._backend_output_dir = None

    def run(self, video_path, dimensions=None, extra_context=None):
        """Run VBench on a single video and return a structured result."""
        requested_dimensions = self._resolve_dimensions(dimensions)
        extra_context = dict(extra_context or {})
        output_dir = self.output_dir
        temp_input_root = self.temp_input_root
        cleanup_temp_dir = bool(extra_context.get("cleanup_temp_dir", True))
        verbose = bool(extra_context.get("verbose", self.verbose))
        eval_name_prefix = extra_context.get("eval_name_prefix", "chunk")

        temp_dir = None
        copied_video_path = None
        eval_json_path = None
        run_name = f"{eval_name_prefix}_{uuid.uuid4().hex[:8]}"

        try:
            output_dir = self._resolve_output_dir(extra_context)
            temp_input_root = self._resolve_temp_input_root(extra_context)
            backend = self._load_vbench_backend(output_dir)
            temp_dir, copied_video_path = self._make_single_video_input_dir(
                video_path,
                root_dir=temp_input_root,
            )
            if verbose:
                print(f"[VBench] running on: {copied_video_path}")
                print(f"[VBench] dims: {requested_dimensions}")

            backend.evaluate(
                videos_path=str(temp_dir),
                name=run_name,
                dimension_list=requested_dimensions,
                mode="custom_input",
            )

            eval_json_path = Path(output_dir) / f"{run_name}_eval_results.json"
            raw_result = read_vbench_eval_json(eval_json_path)
            parsed = parse_vbench_result(raw_result, dimensions=requested_dimensions)
            status = parsed["status"]
            error = None
            if status == "failed":
                error = "VBench returned no requested dimension scores."
            elif status == "partial":
                error = f"Missing dimensions: {', '.join(parsed['missing_dimensions'])}"

            return self._build_result(
                status=status,
                error=error,
                parsed=parsed,
                details={
                    "video_path": str(video_path),
                    "copied_video_path": str(copied_video_path),
                    "requested_dimensions": requested_dimensions,
                    "missing_dimensions": parsed["missing_dimensions"],
                    "eval_json_path": str(eval_json_path),
                    "run_name": run_name,
                    "output_dir": str(output_dir),
                    "temp_input_root": str(temp_input_root),
                    "raw_result": parsed["raw_result"],
                },
            )
        except Exception as exc:
            return self._build_failure_result(
                requested_dimensions=requested_dimensions,
                error=str(exc),
                details={
                    "video_path": str(video_path),
                    "copied_video_path": str(copied_video_path) if copied_video_path is not None else None,
                    "requested_dimensions": requested_dimensions,
                    "eval_json_path": str(eval_json_path) if eval_json_path is not None else None,
                    "run_name": run_name,
                    "output_dir": str(output_dir),
                    "temp_input_root": str(temp_input_root),
                    "phase": "run",
                },
            )
        finally:
            if cleanup_temp_dir and temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def run_on_chunk(self, video_path, chunk_idx=None, dimensions=None, extra_context=None):
        """Run VBench for one generated chunk."""
        extra = dict(extra_context or {})
        if chunk_idx is not None and "eval_name_prefix" not in extra:
            extra["eval_name_prefix"] = f"chunk_{int(chunk_idx):03d}"
        result = self.run(video_path=video_path, dimensions=dimensions, extra_context=extra)
        result["details"]["scope"] = "chunk"
        result["details"]["chunk_idx"] = chunk_idx
        return result

    def run_on_rollout(
        self,
        video_path=None,
        chunk_video_paths=None,
        full_rollout_output_path=None,
        dimensions=None,
        extra_context=None,
    ):
        """Run VBench on a full rollout video or on concatenated chunk videos."""
        extra = dict(extra_context or {})
        verbose = bool(extra.get("verbose", self.verbose))
        working_video_path = video_path
        generated_rollout = False

        try:
            if working_video_path is None:
                if not chunk_video_paths:
                    raise ValueError("chunk_video_paths must be provided when video_path is None.")
                if full_rollout_output_path is None:
                    raise ValueError("full_rollout_output_path is required for chunk rollout VBench.")
                working_video_path = self._concat_videos_ffmpeg(
                    chunk_video_paths,
                    output_path=full_rollout_output_path,
                    overwrite=True,
                )
                generated_rollout = True

            if "eval_name_prefix" not in extra:
                extra["eval_name_prefix"] = "full_rollout"

            result = self.run(video_path=working_video_path, dimensions=dimensions, extra_context=extra)
            result["details"]["scope"] = "rollout"
            result["details"]["source_chunk_video_paths"] = list(chunk_video_paths or [])
            result["details"]["generated_rollout"] = generated_rollout
            result["full_rollout_path"] = str(working_video_path)
            return result
        except Exception as exc:
            requested_dimensions = self._resolve_dimensions(dimensions)
            return self._build_failure_result(
                requested_dimensions=requested_dimensions,
                error=str(exc),
                details={
                    "scope": "rollout",
                    "requested_dimensions": requested_dimensions,
                    "video_path": str(video_path) if video_path is not None else None,
                    "source_chunk_video_paths": list(chunk_video_paths or []),
                    "full_rollout_output_path": str(full_rollout_output_path) if full_rollout_output_path is not None else None,
                    "phase": "run_on_rollout",
                },
                full_rollout_path=str(working_video_path) if working_video_path is not None else None,
            )

    def _resolve_dimensions(self, dimensions=None):
        return list(dimensions or self.dimensions)

    def _resolve_output_dir(self, extra_context=None):
        return self._ensure_writable_dir(
            preferred_path=(extra_context or {}).get("output_dir", self.output_dir),
            fallback_dirname="vbench_eval_outputs",
        )

    def _resolve_temp_input_root(self, extra_context=None):
        return self._ensure_writable_dir(
            preferred_path=(extra_context or {}).get("temp_input_root", self.temp_input_root),
            fallback_dirname="vbench_inputs",
        )

    def _resolve_default_device(self):
        if self.device is not None:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_writable_dir(self, preferred_path, fallback_dirname: str):
        import tempfile

        preferred = Path(preferred_path)
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except OSError:
            fallback = Path(tempfile.gettempdir()) / fallback_dirname
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _load_vbench_backend(self, output_dir: Path):
        if self._backend is not None and self._backend_output_dir == str(output_dir):
            return self._backend

        repo_root = Path(self.repo_root)
        if repo_root.exists() and str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        try:
            module = importlib.import_module("vbench")
        except Exception as exc:
            raise RuntimeError(f"Failed to import VBench backend: {exc}") from exc

        backend_cls = getattr(module, "VBench", None)
        if backend_cls is None:
            raise RuntimeError("Imported vbench module does not expose VBench.")

        full_info_json = self.full_info_json
        if full_info_json is None:
            full_info_json = repo_root / "vbench" / "VBench_full_info.json"
        full_info_json = Path(full_info_json)

        try:
            self._backend = backend_cls(
                self._resolve_default_device(),
                str(full_info_json),
                str(output_dir),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize VBench backend: {exc}") from exc

        self._backend_output_dir = str(output_dir)
        return self._backend

    def _make_single_video_input_dir(self, video_path: str, root_dir: str | Path):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        tmp_dir = root_dir / f"single_{run_id}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        copied_video_path = tmp_dir / video_path.name
        if video_path.resolve() != copied_video_path.resolve():
            shutil.copy2(video_path, copied_video_path)

        return tmp_dir, copied_video_path

    def _concat_videos_ffmpeg(self, video_paths, output_path: str | Path, overwrite: bool = True):
        import subprocess
        import tempfile

        import imageio_ffmpeg

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            concat_list_path = handle.name
            for video_path in video_paths:
                resolved = Path(video_path).resolve()
                handle.write(f"file '{resolved.as_posix()}'\n")

        cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            cmd = [
                ffmpeg_bin,
                "-y" if overwrite else "-n",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            try:
                os.remove(concat_list_path)
            except OSError:
                pass

        return str(output_path)

    def _build_result(self, status: str, error: str | None, parsed: dict, details: dict[str, Any]):
        return {
            "status": status,
            "error": error,
            "dimensions": parsed["dimensions"],
            "flat_scores": parsed["flat_scores"],
            "details": details,
        }

    def _build_failure_result(
        self,
        requested_dimensions,
        error: str,
        details: dict[str, Any],
        full_rollout_path: str | None = None,
    ):
        parsed = parse_vbench_result(
            {dimension: [None] for dimension in requested_dimensions},
            dimensions=requested_dimensions,
        )
        result = {
            "status": "failed",
            "error": error,
            "dimensions": parsed["dimensions"],
            "flat_scores": parsed["flat_scores"],
            "details": details,
        }
        if full_rollout_path is not None:
            result["full_rollout_path"] = full_rollout_path
        return result
