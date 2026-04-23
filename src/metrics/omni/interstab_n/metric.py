from __future__ import annotations

import numpy as np

from ..common.flow import motion_series
from ..common.regions import build_non_target_mask, resolve_target_region
from ..common.results import make_metric_result


class InterStabNMetric:
    """Compute non-target stability from motion energy outside the target region."""

    def __init__(
        self,
        flow_backend=None,
        beta=None,
        region_padding=0.0,
        verbose=False,
    ):
        self.flow_backend = flow_backend
        self.beta = 1.25 if beta is None else float(beta)
        self.region_padding = float(region_padding)
        self.verbose = verbose

    def run(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None, shared_context=None):
        """Run InterStab-N on a chunk using target/non-target region separation and bounded flow-energy mapping."""
        if len(all_frames) < 2:
            return make_metric_result(
                "interstab_n",
                "interstab_n",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "not_enough_frames",
                    "num_frames_used": len(all_frames),
                },
                non_target_motion_energy=None,
            )

        shared_context = shared_context or {}
        target_mask, target_details = resolve_target_region(
            omni_spec,
            all_frames,
            sampled_frames=sampled_frames,
            flow_backend=self.flow_backend,
            region_padding=self.region_padding,
        )
        non_target_mask, non_target_details = build_non_target_mask(target_mask, all_frames[0].shape)
        magnitudes = shared_context.get("motion_magnitudes")
        if magnitudes is None:
            _, magnitudes = motion_series(all_frames, flow_backend=self.flow_backend)
        if not magnitudes:
            return make_metric_result(
                "interstab_n",
                "interstab_n",
                None,
                verification_mode="skipped",
                faithfulness_status="partial faithful",
                used_fallback=False,
                status="skipped",
                details={
                    "reason": "no_motion_field",
                    "num_frames_used": len(all_frames),
                },
                non_target_motion_energy=None,
            )

        non_target_curve = []
        target_curve = []
        for magnitude in magnitudes:
            target_curve.append(
                float(np.mean(magnitude[target_mask]))
                if target_mask is not None and target_mask.any()
                else None
            )
            non_target_curve.append(
                float(np.mean(magnitude[non_target_mask]))
                if non_target_mask is not None and non_target_mask.any()
                else None
            )

        non_target_curve = [value for value in non_target_curve if value is not None]
        target_curve = [value for value in target_curve if value is not None]
        non_target_motion_energy = float(np.mean(non_target_curve)) if non_target_curve else None
        score = float(np.exp(-non_target_motion_energy / self.beta)) if non_target_motion_energy is not None else None

        target_source = target_details.get("target_region_source")
        if target_source in {"exact_mask", "bbox_entity_hints"}:
            verification_mode = "partial_faithful"
            faithfulness = "partial faithful"
            used_fallback = False
            status = "partial"
            fallback_reason = None
        else:
            verification_mode = "fallback"
            faithfulness = "fallback approximation"
            used_fallback = True
            status = "partial"
            fallback_reason = target_source

        return make_metric_result(
            "interstab_n",
            "interstab_n",
            score,
            verification_mode=verification_mode,
            faithfulness_status=faithfulness,
            used_fallback=used_fallback,
            status=status,
            details={
                "target_region_source": target_source,
                "non_target_region_source": non_target_details.get("non_target_region_source"),
                "segmentation_mode": target_details.get("segmentation_mode"),
                "flow_energy_raw": {
                    "target_curve": target_curve,
                    "non_target_curve": non_target_curve,
                    "non_target_mean": non_target_motion_energy,
                },
                "normalization_info": {
                    "transform": "exp(-energy/beta)",
                    "beta": self.beta,
                },
                "num_frames_used": len(all_frames),
                "used_fallback_reason": fallback_reason,
            },
            non_target_motion_energy=non_target_motion_energy,
        )

    __call__ = run
