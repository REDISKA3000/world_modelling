from __future__ import annotations

from typing import Any, Callable

from .common.flow import motion_series
from .common.results import (
    OMNI_IMPLEMENTATION_STATUS,
    aggregate_omni_status,
    empty_omni_run_result,
    extract_metric_summary,
    make_metric_result,
)
from .common.sampling import sample_video_frames
from .camera_control import CameraControlMetric
from .intercov import InterCovMetric
from .interorder import InterOrderMetric
from .interstab_l import InterStabLMetric
from .interstab_n import InterStabNMetric
from .object_control import ObjectControlMetric
from .spec import build_omni_spec
from .transitions_detect import TransitionsDetectMetric


class OmniMetricRunner:
    def __init__(
        self,
        interstab_l_metric: InterStabLMetric | None = None,
        interstab_n_metric: InterStabNMetric | None = None,
        intercov_metric: InterCovMetric | None = None,
        interorder_metric: InterOrderMetric | None = None,
        transitions_detect_metric: TransitionsDetectMetric | None = None,
        object_control_metric: ObjectControlMetric | None = None,
        camera_control_metric: CameraControlMetric | None = None,
        legacy_metrics: dict[str, Callable[..., dict[str, Any]]] | None = None,
        flow_backend=None,
        semantic_verifier=None,
        event_verifier=None,
        object_verifier=None,
        camera_verifier=None,
        verbose: bool = False,
    ) -> None:
        self.interstab_l_metric = interstab_l_metric or InterStabLMetric()
        self.interstab_n_metric = interstab_n_metric or InterStabNMetric(flow_backend=flow_backend)
        self.intercov_metric = intercov_metric or InterCovMetric(
            flow_backend=flow_backend,
            semantic_verifier=semantic_verifier,
        )
        self.interorder_metric = interorder_metric or InterOrderMetric(
            event_verifier=event_verifier,
            flow_backend=flow_backend,
        )
        self.transitions_detect_metric = transitions_detect_metric or TransitionsDetectMetric()
        self.object_control_metric = object_control_metric or ObjectControlMetric(
            object_verifier=object_verifier,
        )
        self.camera_control_metric = camera_control_metric or CameraControlMetric(
            camera_verifier=camera_verifier,
            flow_backend=flow_backend,
        )
        self.legacy_metrics = legacy_metrics or {}
        self.flow_backend = flow_backend
        self.verbose = verbose

    def build_spec(self, *args, **kwargs):
        return build_omni_spec(*args, **kwargs)

    def _run_metric(
        self,
        metric_name: str,
        fn: Callable[..., dict[str, Any]] | None,
        omni_spec,
        all_frames,
        sampled_frames,
        frame_indices,
        shared_context=None,
    ) -> dict[str, Any]:
        if fn is None:
            return make_metric_result(
                metric_name,
                metric_name,
                None,
                verification_mode="skipped",
                faithfulness_status=OMNI_IMPLEMENTATION_STATUS.get(metric_name, "fallback approximation"),
                used_fallback=True,
                status="skipped",
                details={"reason": "metric_not_registered_in_runner"},
            )
        try:
            return fn(
                omni_spec,
                all_frames,
                sampled_frames=sampled_frames,
                sampled_indices=frame_indices,
                shared_context=shared_context,
            )
        except TypeError:
            try:
                return fn(
                    omni_spec,
                    all_frames,
                    sampled_frames=sampled_frames,
                    sampled_indices=frame_indices,
                )
            except Exception as exc:
                return make_metric_result(
                    metric_name,
                    metric_name,
                    None,
                    verification_mode="skipped",
                    faithfulness_status=OMNI_IMPLEMENTATION_STATUS.get(metric_name, "fallback approximation"),
                    used_fallback=True,
                    status="failed",
                    details={"error": str(exc)},
                )
        except Exception as exc:
            return make_metric_result(
                metric_name,
                metric_name,
                None,
                verification_mode="skipped",
                faithfulness_status=OMNI_IMPLEMENTATION_STATUS.get(metric_name, "fallback approximation"),
                used_fallback=True,
                status="failed",
                details={"error": str(exc)},
            )

    def _prepare_shared_inputs(self, all_frames, sampled_frames, frame_indices):
        motion_flows, motion_magnitudes = motion_series(all_frames, flow_backend=self.flow_backend)
        return {
            "all_frames": all_frames,
            "sampled_frames": sampled_frames,
            "sampled_indices": frame_indices,
            "motion_flows": motion_flows,
            "motion_magnitudes": motion_magnitudes,
        }

    def run_on_chunk(
        self,
        omni_spec,
        generated_video_path,
        last_frame_path=None,
        world_spec=None,
        sampling_mode: str = "uniform",
        max_sampled_frames: int = 8,
        custom_indices=None,
        compute_camera_control_flag: bool = True,
    ):
        """Run the Omni block on one chunk while preserving the current result format."""
        result = empty_omni_run_result()

        try:
            all_frames, sampled_frames, frame_indices, sampling_metadata = sample_video_frames(
                video_path=generated_video_path,
                sampling_mode=sampling_mode,
                max_samples=max_sampled_frames,
                custom_indices=custom_indices,
                return_metadata=True,
            )
            if not all_frames:
                result["omni_details"] = {
                    "sampling": {
                        **sampling_metadata,
                        "status": "failed",
                        "reason": "video_read_failed",
                    }
                }
                return result

            metric_payloads = {}
            metric_statuses = {}
            metric_faithfulness = {}
            metric_verification_modes = {}
            shared_context = self._prepare_shared_inputs(all_frames, sampled_frames, frame_indices)

            metric_registry: list[tuple[str, Callable[..., dict[str, Any]] | None]] = [
                ("interstab_l", self.interstab_l_metric.compute),
                ("interstab_n", self.interstab_n_metric.run),
                ("intercov", self.intercov_metric.run),
                ("interorder", self.interorder_metric.run),
                ("transitions_detect", self.transitions_detect_metric.run),
                ("object_control", self.object_control_metric.run),
            ]
            for metric_name, fn in metric_registry:
                payload = self._run_metric(
                    metric_name=metric_name,
                    fn=fn,
                    omni_spec=omni_spec,
                    all_frames=all_frames,
                    sampled_frames=sampled_frames,
                    frame_indices=frame_indices,
                    shared_context=shared_context,
                )
                result[metric_name] = payload.get(metric_name)
                metric_payloads[metric_name] = extract_metric_summary(metric_name, payload)
                metric_statuses[metric_name] = payload.get("status")
                metric_faithfulness[metric_name] = payload.get("faithfulness_status")
                metric_verification_modes[metric_name] = payload.get("verification_mode")

            if compute_camera_control_flag:
                camera_payload = self._run_metric(
                    metric_name="camera_control",
                    fn=self.camera_control_metric.run,
                    omni_spec=omni_spec,
                    all_frames=all_frames,
                    sampled_frames=sampled_frames,
                    frame_indices=frame_indices,
                    shared_context=shared_context,
                )
            else:
                camera_payload = make_metric_result(
                    "camera_control",
                    "camera_control",
                    None,
                    verification_mode="skipped",
                    faithfulness_status="partial faithful",
                    used_fallback=False,
                    status="skipped",
                    details={"reason": "disabled_by_flag"},
                )

            result["camera_control"] = camera_payload.get("camera_control")
            metric_payloads["camera_control"] = extract_metric_summary("camera_control", camera_payload)
            metric_statuses["camera_control"] = camera_payload.get("status")
            metric_faithfulness["camera_control"] = camera_payload.get("faithfulness_status")
            metric_verification_modes["camera_control"] = camera_payload.get("verification_mode")

            result["omni_metric_statuses"] = metric_statuses
            result["omni_metric_faithfulness"] = metric_faithfulness
            result["omni_metric_verification_modes"] = metric_verification_modes
            result["omni_status"] = aggregate_omni_status(metric_statuses, metric_faithfulness)
            result["omni_details"] = {
                "sampling": {
                    **sampling_metadata,
                    "sampled_frame_indices": frame_indices,
                    "last_frame_path": last_frame_path,
                },
                "metric_statuses": metric_statuses,
                "metric_faithfulness": metric_faithfulness,
                "metric_verification_modes": metric_verification_modes,
                "metrics": metric_payloads,
            }
            return result
        except Exception as exc:
            result["omni_status"] = "failed"
            result["omni_details"] = {"sampling": {"status": "failed"}, "error": str(exc)}
            return result
