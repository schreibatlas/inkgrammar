from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from skeletonpaths import (
    RefinedSkeletonGeometry,
    StrokeHypothesis,
    StrokeSearchLimits,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
    refine_graph_geometry,
    stroke_hypotheses_from_analysis,
)

from .models import (
    ContinuousPath,
    SkeletonDiagnostics,
    WritingEvent,
    WritingRouteCandidate,
    WritingRoutePairScore,
    WritingRoutePairSelection,
    WritingRouteScore,
)
from .path import (
    PathExtractionError,
    _semantic_junction_transition_indices,
    align_continuous_paths,
    refine_continuous_walk,
)

DEFAULT_ROUTE_CANDIDATE_LIMIT = 5
ROUTE_AMBIGUITY_ABSOLUTE_MARGIN = 0.04
ROUTE_AMBIGUITY_RELATIVE_MARGIN = 0.15


def _route_events(
    diagnostics: SkeletonDiagnostics,
    hypothesis: StrokeHypothesis,
    strokes: tuple[ContinuousPath, ...],
    refined_graph: RefinedSkeletonGeometry,
) -> tuple[WritingEvent, ...]:
    total_length = sum(stroke.arc_length_px for stroke in strokes)
    events = []
    occurrences: dict[int, int] = {}
    progress_before = 0.0
    for index, (walk, stroke) in enumerate(zip(hypothesis.walks, strokes, strict=True)):
        stroke_start = float(np.clip(progress_before / total_length, 0.0, 1.0))
        events.append(WritingEvent("stroke-start", stroke_start))
        transitions = _semantic_junction_transition_indices(
            diagnostics,
            walk,
            refined_graph,
            stroke.progress,
        )
        for node_id, sample_index in transitions:
            occurrences[node_id] = occurrences.get(node_id, 0) + 1
            progress = (
                progress_before + float(stroke.progress[sample_index]) * stroke.arc_length_px
            ) / total_length
            progress = float(np.clip(progress, 0.0, 1.0))
            events.append(
                WritingEvent(
                    "junction-encounter",
                    progress,
                    junction_node_id=node_id,
                    occurrence=occurrences[node_id],
                )
            )
        progress_before += stroke.arc_length_px
        stroke_end = float(np.clip(progress_before / total_length, 0.0, 1.0))
        events.append(WritingEvent("stroke-end", stroke_end))
        if index < len(strokes) - 1:
            events.append(WritingEvent("pen-lift", stroke_end))
    return tuple(events)


def _continuity_cost(strokes: tuple[ContinuousPath, ...], total_length: float) -> float:
    gaps = sum(
        float(np.linalg.norm(first.xy[-1] - second.xy[0]))
        for first, second in zip(strokes, strokes[1:], strict=False)
    )
    return gaps / max(total_length, 1e-12)


def generate_writing_route_candidates(
    diagnostics: SkeletonDiagnostics,
    *,
    decision_field: np.ndarray | None = None,
    limit: int = DEFAULT_ROUTE_CANDIDATE_LIMIT,
    limits: StrokeSearchLimits | None = None,
) -> tuple[WritingRouteCandidate, ...]:
    analysis = diagnostics.analysis
    if analysis is None:
        raise PathExtractionError("skeleton analysis is unavailable for route hypothesis search")
    if limit < 1:
        raise ValueError("route-candidate limit must be positive")
    hypotheses = stroke_hypotheses_from_analysis(
        analysis,
        limit=limit,
        limits=limits,
    )
    if not hypotheses:
        raise PathExtractionError("skeletonpaths produced no writing-route hypotheses")

    refined_graph = refine_graph_geometry(analysis, decision_field=decision_field)
    total_edge_length = sum(edge.length_px for edge in diagnostics.graph.edges)
    candidates = []
    reconstruction_failures = []
    for hypothesis in hypotheses:
        if hypothesis.uncovered_edge_ids:
            continue
        try:
            strokes = tuple(
                refine_continuous_walk(
                    diagnostics,
                    walk,
                    decision_field=decision_field,
                    refined_graph=refined_graph,
                )
                for walk in hypothesis.walks
            )
        except (
            PathExtractionError,
            SubpixelReconstructionError,
            SubpixelReconstructionLimitError,
            ValueError,
        ) as error:
            reconstruction_failures.append(error)
            continue
        if not strokes or any(stroke.arc_length_px <= 0.0 for stroke in strokes):
            continue
        score = hypothesis.score
        route_score = WritingRouteScore(
            coverage_cost=score.coverage_error,
            repeat_cost=score.repeat_length_px / max(total_edge_length, 1e-12),
            lift_cost=score.lift_cost,
            continuity_cost=_continuity_cost(strokes, sum(path.arc_length_px for path in strokes)),
            steering_complexity_cost=score.turning_cost,
            junction_pairing_cost=score.junction_pairing_cost,
        )
        candidates.append(
            WritingRouteCandidate(
                hypothesis=hypothesis,
                continuous_strokes=strokes,
                events=_route_events(diagnostics, hypothesis, strokes, refined_graph),
                steering_models=(),
                score=route_score,
                ambiguous=(
                    hypothesis.route_ambiguous
                    or hypothesis.component_order_ambiguous
                    or hypothesis.truncated
                ),
                physical_junction_count=len(diagnostics.graph.junction_regions),
                cycle_count=diagnostics.graph.cycle_count,
            )
        )
    if not candidates:
        message = "no writing-route hypothesis could be fully reconstructed"
        if reconstruction_failures:
            message += f": {reconstruction_failures[-1]}"
        raise PathExtractionError(message)
    return tuple(candidates)


def _centerline_cost(
    reference: WritingRouteCandidate,
    sample: WritingRouteCandidate,
) -> float:
    paired_count = min(len(reference.continuous_strokes), len(sample.continuous_strokes))
    weighted_error = 0.0
    total_weight = 0.0
    for reference_stroke, sample_stroke in zip(
        reference.continuous_strokes[:paired_count],
        sample.continuous_strokes[:paired_count],
        strict=True,
    ):
        alignment = align_continuous_paths(reference_stroke, sample_stroke)
        weight = 0.5 * (reference_stroke.arc_length_px + sample_stroke.arc_length_px)
        scale = max(reference_stroke.arc_length_px, sample_stroke.arc_length_px, 1e-12)
        weighted_error += np.sqrt(max(alignment.mean_squared_error, 0.0)) / scale * weight
        total_weight += weight
    cost = weighted_error / max(total_weight, 1e-12) if paired_count else 0.0
    return float(cost + abs(len(reference.continuous_strokes) - len(sample.continuous_strokes)))


def _topology_event_cost(
    reference: WritingRouteCandidate,
    sample: WritingRouteCandidate,
) -> float:
    reference_encounters = sum(event.kind == "junction-encounter" for event in reference.events)
    sample_encounters = sum(event.kind == "junction-encounter" for event in sample.events)
    return float(
        abs(reference.physical_junction_count - sample.physical_junction_count)
        + abs(reference.cycle_count - sample.cycle_count)
        + abs(reference_encounters - sample_encounters)
        + abs(reference.hypothesis.pen_lift_count - sample.hypothesis.pen_lift_count)
    )


def _near_tie(best: float, second: float) -> bool:
    margin = second - best
    relative_margin = margin / max(abs(best), 1e-9)
    return (
        margin <= ROUTE_AMBIGUITY_ABSOLUTE_MARGIN
        or relative_margin <= ROUTE_AMBIGUITY_RELATIVE_MARGIN
    )


def select_writing_route_pair(
    reference_candidates: tuple[WritingRouteCandidate, ...],
    sample_candidates: tuple[WritingRouteCandidate, ...],
    *,
    steering_distance: Callable[[WritingRouteCandidate, WritingRouteCandidate], float]
    | None = None,
) -> WritingRoutePairSelection:
    if not reference_candidates or not sample_candidates:
        raise ValueError("joint route selection requires candidates for both glyphs")
    pair_scores = []
    for reference_rank, reference in enumerate(reference_candidates, start=1):
        for sample_rank, sample in enumerate(sample_candidates, start=1):
            intrinsic = reference.score.intrinsic_cost + sample.score.intrinsic_cost
            centerline = _centerline_cost(reference, sample)
            steering = (
                float(steering_distance(reference, sample))
                if steering_distance is not None
                else 0.0
            )
            topology = _topology_event_cost(reference, sample)
            pair_scores.append(
                WritingRoutePairScore(
                    reference_rank=reference_rank,
                    sample_rank=sample_rank,
                    intrinsic_route_cost=intrinsic,
                    aligned_centerline_cost=centerline,
                    steering_profile_cost=steering,
                    topology_event_compatibility_cost=topology,
                    total_cost=intrinsic + centerline + steering + topology,
                )
            )
    pair_scores.sort(key=lambda score: (score.total_cost, score.reference_rank, score.sample_rank))
    best = pair_scores[0]
    reference = reference_candidates[best.reference_rank - 1]
    sample = sample_candidates[best.sample_rank - 1]
    near_tie = len(pair_scores) > 1 and _near_tie(best.total_cost, pair_scores[1].total_cost)
    counterpart_cost = (
        best.aligned_centerline_cost
        + best.steering_profile_cost
        + best.topology_event_compatibility_cost
    )
    reference = replace(
        reference,
        score=replace(reference.score, counterpart_similarity_cost=counterpart_cost),
    )
    sample = replace(
        sample,
        score=replace(sample.score, counterpart_similarity_cost=counterpart_cost),
    )
    return WritingRoutePairSelection(
        reference=reference,
        sample=sample,
        score=best,
        ambiguous=near_tie or reference.ambiguous or sample.ambiguous,
        alternatives=tuple(pair_scores),
    )
