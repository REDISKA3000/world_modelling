from __future__ import annotations

import numpy as np

from ..common.embeddings import embedding_cosine, light_frame_change, omni_ssim_gray
from ..common.results import make_metric_result


def _resolve_revisit_pairs(omni_spec, total_frames: int, fallback_pair_strategy: str = "long_range"):
    def _resolve(value):
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return int(round(value * (total_frames - 1)))
        return int(value)

    explicit_pairs = []
    for pair in omni_spec.get("revisit_pairs", []) or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left = max(0, min(total_frames - 1, _resolve(pair[0])))
        right = max(0, min(total_frames - 1, _resolve(pair[1])))
        if left < right:
            explicit_pairs.append((left, right))
    if explicit_pairs:
        return sorted(set(explicit_pairs)), "explicit_revisit_pairs"
    if total_frames < 2:
        return [], "insufficient_frames"
    if fallback_pair_strategy != "long_range":
        raise ValueError(f"Unsupported fallback_pair_strategy: {fallback_pair_strategy}")

    anchors = sorted(
        set(np.linspace(0, total_frames - 1, min(6, max(4, total_frames))).astype(int).tolist())
    )
    min_gap = max(2, total_frames // 4)
    fallback_pairs = []
    for left_idx, left in enumerate(anchors):
        for right in anchors[left_idx + 1:]:
            if right - left >= min_gap:
                fallback_pairs.append((left, right))
    if not fallback_pairs and len(anchors) >= 2:
        fallback_pairs.append((anchors[0], anchors[-1]))
    deduped = []
    for pair in sorted(set(fallback_pairs)):
        if pair not in deduped:
            deduped.append(pair)
    return deduped[:6], "fallback_long_range_pairs"


def _canonical_anchor_pairs(total_frames: int):
    if total_frames < 2:
        return []
    anchors = sorted(set(np.linspace(0, total_frames - 1, min(4, total_frames)).astype(int).tolist()))
    if len(anchors) < 2:
        return []
    pairs = []
    for idx in range(len(anchors) - 1):
        pairs.append((anchors[idx], anchors[idx + 1]))
    pairs.append((anchors[0], anchors[-1]))
    return sorted(set(pair for pair in pairs if pair[0] < pair[1]))


class InterStabLMetric:
    def __init__(
        self,
        embedding_backend=None,
        static_threshold=None,
        fallback_pair_strategy="long_range",
        verbose=False,
    ):
        self.embedding_backend = embedding_backend
        self.static_threshold = 0.08 if static_threshold is None else float(static_threshold)
        self.fallback_pair_strategy = fallback_pair_strategy
        self.verbose = verbose

    def compute(self, omni_spec, all_frames, sampled_frames=None, sampled_indices=None):
        """Article-faithful revisit consistency with grayscale SSIM, embedding cosine, fallback pairs, and anti-static gating."""
        if len(all_frames) < 2:
            return make_metric_result(
                "interstab_l",
                "interstab_l",
                None,
                verification_mode="skipped",
                faithfulness_status="article-faithful",
                used_fallback=False,
                status="skipped",
                details={"reason": "not_enough_frames"},
                num_pairs=0,
                used_pairs=[],
                pair_scores=[],
            )

        pairs, pair_source = _resolve_revisit_pairs(
            omni_spec,
            len(all_frames),
            fallback_pair_strategy=self.fallback_pair_strategy,
        )
        if not pairs:
            return make_metric_result(
                "interstab_l",
                "interstab_l",
                None,
                verification_mode="skipped",
                faithfulness_status="article-faithful",
                used_fallback=False,
                status="skipped",
                details={"reason": "no_valid_revisit_pairs"},
                num_pairs=0,
                used_pairs=[],
                pair_scores=[],
            )

        pair_scores = []
        pair_details = []
        embedding_modes = []
        embedding_cache = {}
        for left, right in pairs:
            ssim_score = omni_ssim_gray(all_frames[left], all_frames[right])
            cosine_score, cosine_details = embedding_cosine(
                all_frames[left],
                all_frames[right],
                embedding_backend=self.embedding_backend,
                cache=embedding_cache,
                key_a=f"frame_{left}",
                key_b=f"frame_{right}",
            )
            pair_score = float(0.5 * (ssim_score + cosine_score))
            pair_scores.append(pair_score)
            embedding_modes.append(cosine_details["embedding_mode"])
            pair_details.append(
                {
                    "pair": [left, right],
                    "ssim_gray": ssim_score,
                    "embedding_similarity": cosine_score,
                    "pair_score": pair_score,
                }
            )

        anchor_static_scores = []
        for left, right in _canonical_anchor_pairs(len(all_frames)):
            change_score = light_frame_change(all_frames[left], all_frames[right])
            anchor_static_scores.append(
                {
                    "pair": [left, right],
                    "change_score": change_score,
                    "similarity_score": float(1.0 - change_score),
                }
            )

        static_gate = 1.0
        static_gate_applied = False
        static_gate_reason = "not_applied"
        if anchor_static_scores:
            mean_anchor_change = float(np.mean([row["change_score"] for row in anchor_static_scores]))
            static_gate = float(np.clip(mean_anchor_change / max(self.static_threshold, 1e-8), 0.0, 1.0))
            if static_gate < 0.98:
                static_gate_applied = True
                static_gate_reason = "anti_static_gate_triggered"

        final_score = float(np.mean(pair_scores) * static_gate) if pair_scores else None
        used_fallback = pair_source != "explicit_revisit_pairs" or any(
            mode != "clip_vision" for mode in embedding_modes
        )
        verification_mode = "article_faithful" if not used_fallback else "partial_faithful"
        faithfulness = (
            "article-faithful"
            if pair_source == "explicit_revisit_pairs" and not any(mode != "clip_vision" for mode in embedding_modes)
            else "partial faithful"
        )
        return make_metric_result(
            "interstab_l",
            "interstab_l",
            final_score,
            verification_mode=verification_mode,
            faithfulness_status=faithfulness,
            used_fallback=used_fallback,
            status="ok",
            details={
                "pair_source": pair_source,
                "used_pairs": [list(pair) for pair in pairs],
                "pair_scores": pair_scores,
                "pair_details": pair_details,
                "anchor_static_scores": anchor_static_scores,
                "static_gate_applied": static_gate_applied,
                "static_gate_reason": static_gate_reason,
                "static_gate": static_gate,
                "embedding_mode": "|".join(sorted(set(mode for mode in embedding_modes if mode))),
            },
            num_pairs=len(pair_scores),
            used_pairs=[list(pair) for pair in pairs],
            pair_scores=pair_scores,
        )

    __call__ = compute
