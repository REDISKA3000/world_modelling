from __future__ import annotations

from copy import deepcopy


def _omni_unique(values):
    seen = set()
    ordered = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _omni_entity_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return _omni_unique([chunk.strip() for chunk in value.split(",")])
    if isinstance(value, (list, tuple, set)):
        return _omni_unique(list(value))
    return _omni_unique([value])


def _infer_motion_direction(text: str | None):
    low = (text or "").lower()
    if any(token in low for token in ["left", "pan left", "move left", "turn left"]):
        return "left"
    if any(token in low for token in ["right", "pan right", "move right", "turn right"]):
        return "right"
    if any(token in low for token in ["up", "rise", "lift", "raise"]):
        return "up"
    if any(token in low for token in ["down", "fall", "drop", "lower"]):
        return "down"
    if any(token in low for token in ["zoom in", "push in", "forward", "approach", "closer"]):
        return "forward"
    if any(token in low for token in ["zoom out", "pull out", "backward", "away"]):
        return "backward"
    if any(token in low for token in ["static camera", "locked camera", "fixed camera", "still camera"]):
        return "static"
    return "unknown"


def _infer_motion_magnitude(text: str | None):
    low = (text or "").lower()
    if any(token in low for token in ["strong", "fast", "rapid", "sudden", "hard", "all his might"]):
        return "high"
    if any(token in low for token in ["slow", "slight", "gentle", "steady", "small"]):
        return "low"
    if any(token in low for token in ["move", "motion", "walk", "run", "throw", "pick", "place", "turn"]):
        return "medium"
    return "unknown"


def _normalize_expected_motion(expected_motion, affected_entities, signal_text):
    if isinstance(expected_motion, dict) and expected_motion:
        normalized = {}
        for entity, payload in expected_motion.items():
            if isinstance(payload, dict):
                normalized[str(entity)] = {
                    "direction": payload.get("direction", "unknown"),
                    "magnitude": payload.get("magnitude", "unknown"),
                }
            else:
                normalized[str(entity)] = {
                    "direction": _infer_motion_direction(str(payload)),
                    "magnitude": _infer_motion_magnitude(str(payload)),
                }
        return normalized, "explicit"

    inferred_direction = _infer_motion_direction(signal_text)
    inferred_magnitude = _infer_motion_magnitude(signal_text)
    return (
        {
            entity: {
                "direction": inferred_direction,
                "magnitude": inferred_magnitude,
            }
            for entity in affected_entities
        },
        "inferred",
    )


def build_omni_spec(
    world_spec: dict | None = None,
    chunk_idx: int | None = None,
    input_image: str | None = None,
    generated_video_path: str | None = None,
    last_frame_path: str | None = None,
    positive_prompt: str | None = None,
    fps: int | None = None,
    num_frames: int | None = None,
    resolution=None,
    overrides: dict | None = None,
):
    """Normalize Omni inputs from world_spec and record provenance for key semantic fields."""
    world_spec = deepcopy(world_spec or {})
    provenance = {}

    main_subject = world_spec.get("main_subject")
    secondary_subjects = _omni_entity_list(world_spec.get("secondary_subjects"))
    target_objects = _omni_entity_list(world_spec.get("target_objects"))
    explicit_targets = _omni_entity_list(world_spec.get("target_entities"))
    if explicit_targets:
        target_entities = explicit_targets
        provenance["target_entities"] = "explicit"
    else:
        target_entities = _omni_unique(
            ([main_subject] if main_subject else []) + secondary_subjects + target_objects
        )
        provenance["target_entities"] = "inferred" if target_entities else "fallback default"

    affected_entities = _omni_entity_list(world_spec.get("affected_entities"))
    if affected_entities:
        provenance["affected_entities"] = "explicit"
    else:
        expected_motion_source = world_spec.get("expected_motion") or {}
        if isinstance(expected_motion_source, dict) and expected_motion_source:
            affected_entities = _omni_entity_list(expected_motion_source.keys())
            provenance["affected_entities"] = "inferred"
        elif main_subject:
            affected_entities = [main_subject]
            provenance["affected_entities"] = "inferred"
        else:
            affected_entities = target_entities[:1] if target_entities else []
            provenance["affected_entities"] = "fallback default"

    unaffected_entities = _omni_entity_list(world_spec.get("unaffected_entities"))
    if unaffected_entities:
        provenance["unaffected_entities"] = "explicit"
    else:
        affected_lower = {entity.lower() for entity in affected_entities}
        unaffected_entities = [
            entity for entity in target_entities if entity.lower() not in affected_lower
        ]
        provenance["unaffected_entities"] = "inferred" if unaffected_entities else "fallback default"

    signal_text = " ".join(
        part
        for part in [
            world_spec.get("expected_action"),
            world_spec.get("expected_scene_transition"),
            world_spec.get("camera_prompt"),
            positive_prompt,
        ]
        if part
    )

    expected_motion, motion_provenance = _normalize_expected_motion(
        world_spec.get("expected_motion"),
        affected_entities,
        signal_text,
    )
    provenance["expected_motion"] = motion_provenance if expected_motion else "fallback default"

    event_sequence = world_spec.get("event_sequence")
    if isinstance(event_sequence, list):
        event_sequence = deepcopy(event_sequence)
        provenance["event_sequence"] = "explicit" if event_sequence else "fallback default"
    else:
        event_sequence = []
        provenance["event_sequence"] = "fallback default"

    revisit_pairs = deepcopy(world_spec.get("revisit_pairs") or [])
    provenance["revisit_pairs"] = "explicit" if revisit_pairs else "fallback default"

    camera_transform_hint = deepcopy(
        world_spec.get("camera_transform_hint")
        or (world_spec.get("layout_spec") or {}).get("camera_transform_hint")
    )
    provenance["camera_transform_hint"] = "explicit" if camera_transform_hint is not None else "fallback default"

    camera_trajectory_gt = deepcopy(
        world_spec.get("camera_trajectory_gt")
        or (world_spec.get("layout_spec") or {}).get("camera_trajectory_gt")
    )
    provenance["camera_trajectory_gt"] = "explicit" if camera_trajectory_gt is not None else "fallback default"

    omni_spec = {
        "chunk_idx": world_spec.get("chunk_idx", chunk_idx if chunk_idx is not None else 0),
        "input_image": str(input_image or world_spec.get("current_scene_image") or world_spec.get("input_image") or ""),
        "generated_video_path": str(generated_video_path or world_spec.get("generated_video_path") or ""),
        "last_frame_path": str(last_frame_path or world_spec.get("last_frame_path") or world_spec.get("previous_chunk_last_frame") or ""),
        "positive_prompt": positive_prompt or world_spec.get("positive_prompt") or world_spec.get("next_scene_prompt") or "",
        "camera_prompt": world_spec.get("camera_prompt") or world_spec.get("camera_instruction_text") or "",
        "main_subject": main_subject,
        "target_entities": target_entities or [],
        "affected_entities": affected_entities or [],
        "unaffected_entities": unaffected_entities or [],
        "expected_motion": expected_motion or {},
        "event_sequence": event_sequence or [],
        "scene_context": world_spec.get("scene_context"),
        "style_tags": _omni_entity_list(world_spec.get("style_tags")),
        "revisit_pairs": revisit_pairs or [],
        "num_frames": int(num_frames or world_spec.get("frames_per_chunk") or 0),
        "fps": int(fps or world_spec.get("fps") or 0),
        "resolution": list(resolution or world_spec.get("resolution") or []),
        "entity_region_hints": deepcopy(world_spec.get("entity_region_hints") or {}),
        "motion_mask_hint": world_spec.get("motion_mask_hint"),
        "camera_trajectory_gt": camera_trajectory_gt,
        "camera_transform_hint": camera_transform_hint,
        "layout_spec": deepcopy(world_spec.get("layout_spec") or {}),
        "provenance": provenance,
    }

    if overrides:
        omni_spec.update(deepcopy(overrides))
        provenance["overrides"] = "explicit"

    return omni_spec
