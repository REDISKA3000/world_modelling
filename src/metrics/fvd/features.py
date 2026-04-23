from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import (
    load_video_tensor_subclip,
    load_video_tensor_uniform,
    normalize_resize_to,
    resolve_num_frames,
)


class I3DFeatureExtractor:
    def __init__(
        self,
        device: str | None = None,
        repo_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_root = Path(
            repo_root
            or os.environ.get("I3D_REPO_ROOT")
            or "/content/pytorch-i3d"
        )
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("I3D_CKPT_PATH")
            or self.repo_root / "models" / "rgb_imagenet.pt"
        )
        self.verbose = verbose

        self._model: Any | None = None
        self._hook_handle = None
        self._features: dict[str, torch.Tensor] = {}

    def initialize(self) -> Any:
        if self._model is not None:
            return self._model

        if not self.repo_root.exists():
            raise FileNotFoundError(f"pytorch-i3d repo not found: {self.repo_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"I3D checkpoint not found: {self.checkpoint_path}")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from pytorch_i3d import InceptionI3d

        model = InceptionI3d(num_classes=400, in_channels=3).to(self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()

        self._features = {}

        def save_feat(name: str):
            def hook(module, inp, out):
                self._features[name] = out.detach()

            return hook

        self._hook_handle = model.avg_pool.register_forward_hook(save_feat("feat"))
        self._model = model

        if self.verbose:
            print(f"I3D initialized on {self.device}")

        return self._model

    @torch.no_grad()
    def extract_video_feature(
        self,
        video_path: str | Path,
        num_frames: int = 16,
        resize_to: tuple[int, int] = (224, 224),
        sampling_strategy: str = "uniform",
    ) -> np.ndarray:
        if sampling_strategy != "uniform":
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        model = self.initialize()
        self._features.pop("feat", None)

        x = load_video_tensor_uniform(
            video_path=video_path,
            num_frames=resolve_num_frames(num_frames),
            resize_to=normalize_resize_to(resize_to),
        ).to(self.device)
        _ = model(x)

        feat = self._features["feat"].flatten(1).squeeze(0).cpu().numpy()
        return feat

    @torch.no_grad()
    def extract_subclip_feature(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        num_frames: int = 16,
        resize_to: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        model = self.initialize()
        self._features.pop("feat", None)

        x = load_video_tensor_subclip(
            video_path=video_path,
            start_frame=int(start_frame),
            num_frames=resolve_num_frames(num_frames),
            resize_to=normalize_resize_to(resize_to),
        ).to(self.device)
        _ = model(x)

        feat = self._features["feat"].flatten(1).squeeze(0).cpu().numpy()
        return feat

    def unload(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

        self._model = None
        self._features = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
