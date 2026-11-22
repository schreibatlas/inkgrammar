from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from planarcurve.transforms import fit_similarity_transform
from skeletonpaths import (
    RefinedSkeletonGeometry,
    StrokeHypothesis,
    StrokeSearchLimits,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
    refine_graph_geometry,
    stroke_hypotheses_from_analysis,
)

from .descriptors import RouteDescriptor, route_descriptor, route_descriptor_distance
from .models import (
    ContinuousPath,
    SkeletonDiagnostics,
    WritingDirectionPolicy,
    WritingEvent,
    WritingPriorScore,
    WritingRouteAlignment,
    WritingRouteCandidate,
    WritingRoutePairScore,
    WritingRoutePairSelection,
    WritingRouteScore,
    WritingStructurePolicy,
)
from .path import (
    PathExtractionError,
    _semantic_junction_transition_indices,
    align_continuous_paths,
    refine_continuous_walk,
    resample_continuous_path,
    reverse_continuous_path,
    transform_continuous_path,
)

ROUTE_SEARCH_LIMIT = 32
ROUTE_REPORT_LIMIT = 5
DEFAULT_ROUTE_CANDIDATE_LIMIT = ROUTE_SEARCH_LIMIT
ROUTE_AMBIGUITY_ABSOLUTE_MARGIN = 0.04
ROUTE_AMBIGUITY_RELATIVE_MARGIN = 0.15


def _orientation_variants(
    hypothesis: StrokeHypothesis,
) -> tuple[tuple[StrokeHypothesis, str], ...]:
    variants = [(hypothesis, "native")]
    ambiguous = hypothesis.orientation_ambiguous or any(hypothesis.orientation_ambiguous_strokes)
    if ambiguous:
        reversed_hypothesis = replace(
            hypothesis,
            walks=tuple(walk.reversed() for walk in reversed(hypothesis.walks)),
            orientation_relation="global-reversal",
        )
        variants.append((reversed_hypothesis, "global-reverse"))
    return tuple(variants)


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


def _junction_continuity_cost(strokes: tuple[ContinuousPath, ...]) -> float:
    costs = []
    for stroke in strokes:
        for index in stroke.route_event_indices:
            if index <= 0 or index >= len(stroke.xy) - 1:
                continue
            incoming = stroke.xy[index] - stroke.xy[index - 1]
            outgoing = stroke.xy[index + 1] - stroke.xy[index]
            magnitude = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            if magnitude <= 1e-12:
                continue
            cosine = float(np.dot(incoming, outgoing) / magnitude)
            costs.append((1.0 - float(np.clip(cosine, -1.0, 1.0))) / 2.0)
    return float(np.mean(costs)) if costs else 0.0


def _writing_prior_score(
    candidate: WritingRouteCandidate,
    policy: WritingDirectionPolicy,
) -> WritingPriorScore:
    strokes = candidate.continuous_strokes
    start = strokes[0].xy[0]
    end = strokes[-1].xy[-1]
    all_points = np.concatenate([stroke.xy for stroke in strokes], axis=0)
    endpoints = np.asarray(candidate.glyph_endpoints, dtype=float).reshape((-1, 2))
    scale_points = np.concatenate((all_points, endpoints), axis=0) if len(endpoints) else all_points
    spans = np.ptp(scale_points, axis=0)
    scale = max(float(np.linalg.norm(spans)), 1.0)
    start_endpoint_cost = (
        float(np.min(np.linalg.norm(endpoints - start, axis=1)) / scale) if len(endpoints) else 0.0
    )
    end_endpoint_cost = (
        float(np.min(np.linalg.norm(endpoints - end, axis=1)) / scale) if len(endpoints) else 0.0
    )
    direction_cost = 0.0
    if policy.script_direction == "left-to-right":
        direction_cost = max(0.0, float(start[0] - end[0])) / max(float(spans[0]), 1.0)
    elif policy.script_direction == "right-to-left":
        direction_cost = max(0.0, float(end[0] - start[0])) / max(float(spans[0]), 1.0)
    pen_lift_cost = float(candidate.hypothesis.pen_lift_count)
    junction_cost = _junction_continuity_cost(strokes)
    total = (
        policy.pen_lift_weight * pen_lift_cost
        + policy.endpoint_bias_weight * (start_endpoint_cost + end_endpoint_cost)
        + policy.direction_weight * direction_cost
        + policy.junction_continuity_weight * junction_cost
    )
    return WritingPriorScore(
        pen_lift_cost=pen_lift_cost,
        start_endpoint_cost=start_endpoint_cost,
        end_endpoint_cost=end_endpoint_cost,
        direction_cost=direction_cost,
        junction_continuity_cost=junction_cost,
        total=total,
    )


def _writing_structure_prior_cost(
    stroke_count: int,
    expected_strokes: int,
    policy: WritingStructurePolicy,
    ambiguous_component_count: int,
) -> float:
    if ambiguous_component_count < 0:
        raise ValueError("ambiguous_component_count must be nonnegative")
    additional_strokes = max(0, stroke_count - policy.preferred_body_strokes)
    assigned_mark_strokes = max(0, expected_strokes - policy.preferred_body_strokes)
    return (
        policy.detached_mark_stroke_weight * abs(additional_strokes - assigned_mark_strokes)
        + policy.unexpected_body_lift_weight * max(0, additional_strokes - assigned_mark_strokes)
        + policy.side_component_penalty * ambiguous_component_count
    )


def generate_writing_route_candidates(
    diagnostics: SkeletonDiagnostics,
    *,
    decision_field: np.ndarray | None = None,
    limit: int = ROUTE_SEARCH_LIMIT,
    limits: StrokeSearchLimits | None = None,
    expected_strokes: int | None = None,
    structure_policy: WritingStructurePolicy | None = None,
    ambiguous_component_count: int = 0,
) -> tuple[WritingRouteCandidate, ...]:
    analysis = diagnostics.analysis
    if analysis is None:
        raise PathExtractionError("skeleton analysis is unavailable for route hypothesis search")
    if limit < 1:
        raise ValueError("route-candidate limit must be positive")
    structure_policy = structure_policy or WritingStructurePolicy()
    expected_strokes = (
        structure_policy.preferred_body_strokes if expected_strokes is None else expected_strokes
    )
    hypotheses = stroke_hypotheses_from_analysis(
        analysis,
        expected_strokes=expected_strokes,
        limit=limit,
        limits=limits,
        diversity="stroke-count",
    )
    if not hypotheses:
        raise PathExtractionError("skeletonpaths produced no writing-route hypotheses")

    refined_graph = refine_graph_geometry(analysis, decision_field=decision_field)
    total_edge_length = sum(edge.length_px for edge in diagnostics.graph.edges)
    candidates = []
    reconstruction_failures = []
    expanded_hypotheses = tuple(
        (variant, orientation_variant, orientation_group)
        for orientation_group, hypothesis in enumerate(hypotheses, start=1)
        for variant, orientation_variant in _orientation_variants(hypothesis)
    )
    for hypothesis, orientation_variant, orientation_group in expanded_hypotheses:
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
        structure_prior_cost = _writing_structure_prior_cost(
            len(strokes),
            expected_strokes,
            structure_policy,
            ambiguous_component_count,
        )
        route_score = WritingRouteScore(
            coverage_cost=score.coverage_error,
            repeat_cost=score.repeat_length_px / max(total_edge_length, 1e-12),
            lift_cost=score.lift_cost,
            continuity_cost=_continuity_cost(strokes, sum(path.arc_length_px for path in strokes)),
            steering_complexity_cost=score.turning_cost,
            junction_pairing_cost=score.junction_pairing_cost,
            structure_prior_cost=structure_prior_cost,
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
                orientation_variant=orientation_variant,
                orientation_evidence="graph-order-only",
                orientation_status=(
                    "ambiguous"
                    if hypothesis.orientation_ambiguous
                    or any(hypothesis.orientation_ambiguous_strokes)
                    else "resolved"
                ),
                orientation_group=orientation_group,
                glyph_endpoints=tuple(
                    (float(x), float(y)) for x, y in diagnostics.topology.endpoints
                ),
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
    reference_strokes = reference.continuous_strokes
    sample_strokes = sample.continuous_strokes
    paired_count = min(len(reference_strokes), len(sample_strokes))
    weighted_error = 0.0
    total_weight = 0.0
    for reference_stroke, sample_stroke in zip(
        reference_strokes[:paired_count],
        sample_strokes[:paired_count],
        strict=True,
    ):
        alignment = align_continuous_paths(reference_stroke, sample_stroke)
        weight = 0.5 * (reference_stroke.arc_length_px + sample_stroke.arc_length_px)
        scale = max(reference_stroke.arc_length_px, sample_stroke.arc_length_px, 1e-12)
        weighted_error += np.sqrt(max(alignment.mean_squared_error, 0.0)) / scale * weight
        total_weight += weight
    paired_cost = weighted_error / max(total_weight, 1e-12)
    unmatched_length = sum(
        stroke.arc_length_px for stroke in reference_strokes[paired_count:]
    ) + sum(stroke.arc_length_px for stroke in sample_strokes[paired_count:])
    total_route_length = 0.5 * (
        sum(stroke.arc_length_px for stroke in reference_strokes)
        + sum(stroke.arc_length_px for stroke in sample_strokes)
    )
    unmatched_cost = unmatched_length / max(total_route_length, 1e-12)
    stroke_count_difference = abs(len(reference_strokes) - len(sample_strokes))
    return float(paired_cost + unmatched_cost + stroke_count_difference)


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


def evaluate_writing_route_pairs(
    reference_candidates: tuple[WritingRouteCandidate, ...],
    sample_candidates: tuple[WritingRouteCandidate, ...],
    *,
    steering_distance: Callable[[WritingRouteCandidate, WritingRouteCandidate], float]
    | None = None,
    direction_policy: WritingDirectionPolicy | None = None,
    reference_descriptors: tuple[RouteDescriptor, ...] | None = None,
    sample_descriptors: tuple[RouteDescriptor, ...] | None = None,
) -> WritingRoutePairSelection:
    if not reference_candidates or not sample_candidates:
        raise ValueError("joint route selection requires candidates for both glyphs")
    policy = direction_policy or WritingDirectionPolicy()
    reference_descriptors = (
        tuple(route_descriptor(candidate) for candidate in reference_candidates)
        if reference_descriptors is None
        else tuple(reference_descriptors)
    )
    sample_descriptors = (
        tuple(route_descriptor(candidate) for candidate in sample_candidates)
        if sample_descriptors is None
        else tuple(sample_descriptors)
    )
    if len(reference_descriptors) != len(reference_candidates):
        raise ValueError("reference descriptors must align with route candidates")
    if len(sample_descriptors) != len(sample_candidates):
        raise ValueError("sample descriptors must align with route candidates")
    reference_candidates = tuple(
        replace(candidate, writing_prior=_writing_prior_score(candidate, policy))
        for candidate in reference_candidates
    )
    sample_candidates = tuple(
        replace(candidate, writing_prior=_writing_prior_score(candidate, policy))
        for candidate in sample_candidates
    )
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
            route_geometry = 0.5 * (
                reference.score.repeat_cost
                + reference.score.continuity_cost
                + reference.score.junction_pairing_cost
                + sample.score.repeat_cost
                + sample.score.continuity_cost
                + sample.score.junction_pairing_cost
            )
            geometry_cost = route_geometry + centerline + topology
            writing_prior_cost = reference.writing_prior.total + sample.writing_prior.total
            descriptor_cost = route_descriptor_distance(
                reference_descriptors[reference_rank - 1],
                sample_descriptors[sample_rank - 1],
            )
            selection_cost = sum(
                (
                    intrinsic,
                    centerline,
                    steering,
                    topology,
                    writing_prior_cost,
                    descriptor_cost,
                )
            )
            pair_scores.append(
                WritingRoutePairScore(
                    reference_rank=reference_rank,
                    sample_rank=sample_rank,
                    intrinsic_route_cost=intrinsic,
                    aligned_centerline_cost=centerline,
                    steering_profile_cost=steering,
                    topology_event_compatibility_cost=topology,
                    total_cost=selection_cost,
                    geometry_cost=geometry_cost,
                    writing_prior_cost=writing_prior_cost,
                    descriptor_matching_cost=descriptor_cost,
                    pen_lift_count=(
                        reference.hypothesis.pen_lift_count + sample.hypothesis.pen_lift_count
                    ),
                )
            )

    best_geometry = min(score.geometry_cost for score in pair_scores)
    geometry_tolerance = policy.pen_lift_geometry_tolerance
    geometry_plausible = [
        score for score in pair_scores if score.geometry_cost <= best_geometry + geometry_tolerance
    ]
    minimum_lifts = min(score.pen_lift_count for score in geometry_plausible)
    eligible_ids = {
        (score.reference_rank, score.sample_rank)
        for score in geometry_plausible
        if score.pen_lift_count == minimum_lifts
    }
    pair_scores = [
        replace(
            score,
            eligible=(score.reference_rank, score.sample_rank) in eligible_ids,
        )
        for score in pair_scores
    ]
    eligible = sorted(
        (score for score in pair_scores if score.eligible),
        key=lambda score: (score.total_cost, score.reference_rank, score.sample_rank),
    )
    remaining = sorted(
        (score for score in pair_scores if not score.eligible),
        key=lambda score: (score.total_cost, score.reference_rank, score.sample_rank),
    )
    pair_scores = [*eligible, *remaining]
    best = eligible[0]
    reference = reference_candidates[best.reference_rank - 1]
    sample = sample_candidates[best.sample_rank - 1]
    near_tie = len(eligible) > 1 and _near_tie(best.total_cost, eligible[1].total_cost)

    def orientation_resolution(
        selected: WritingRouteCandidate,
        *,
        is_reference: bool,
    ) -> tuple[str, str, float | None]:
        if not (
            selected.hypothesis.orientation_ambiguous
            or any(selected.hypothesis.orientation_ambiguous_strokes)
        ):
            return "resolved", "resolved", None
        alternatives = []
        for score in eligible:
            if is_reference:
                if score.sample_rank != best.sample_rank:
                    continue
                other = reference_candidates[score.reference_rank - 1]
            else:
                if score.reference_rank != best.reference_rank:
                    continue
                other = sample_candidates[score.sample_rank - 1]
            if (
                other.orientation_group == selected.orientation_group
                and other.orientation_variant != selected.orientation_variant
            ):
                alternatives.append((score, other))
        if not alternatives:
            return "ambiguous", "graph-order-only", None
        alternative_score, alternative = min(alternatives, key=lambda item: item[0].total_cost)
        margin = max(0.0, alternative_score.total_cost - best.total_cost)
        selected_prior = selected.writing_prior
        alternative_prior = alternative.writing_prior
        if (
            selected_prior is not None
            and alternative_prior is not None
            and selected_prior.direction_cost < alternative_prior.direction_cost
        ):
            evidence = "writing-prior"
        elif (
            abs(best.aligned_centerline_cost - alternative_score.aligned_centerline_cost) > 1e-9
            or abs(best.steering_profile_cost - alternative_score.steering_profile_cost) > 1e-9
        ):
            evidence = "pair-similarity"
        else:
            evidence = "graph-order-only"
        status = "resolved" if margin > policy.ambiguity_margin else "ambiguous"
        return status, evidence, margin

    reference_status, reference_evidence, reference_margin = orientation_resolution(
        reference, is_reference=True
    )
    sample_status, sample_evidence, sample_margin = orientation_resolution(
        sample, is_reference=False
    )
    counterpart_cost = (
        best.aligned_centerline_cost
        + best.steering_profile_cost
        + best.topology_event_compatibility_cost
        + best.descriptor_matching_cost
    )
    reference = replace(
        reference,
        score=replace(reference.score, counterpart_similarity_cost=counterpart_cost),
        orientation_status=reference_status,
        orientation_evidence=reference_evidence,
        orientation_margin=reference_margin,
        ambiguous=reference.ambiguous or reference_status == "ambiguous",
    )
    sample = replace(
        sample,
        score=replace(sample.score, counterpart_similarity_cost=counterpart_cost),
        orientation_status=sample_status,
        orientation_evidence=sample_evidence,
        orientation_margin=sample_margin,
        ambiguous=sample.ambiguous or sample_status == "ambiguous",
    )
    return WritingRoutePairSelection(
        reference=reference,
        sample=sample,
        score=best,
        ambiguous=(near_tie or reference.ambiguous or sample.ambiguous),
        alternatives=tuple(pair_scores),
    )


def align_writing_routes(
    reference: WritingRouteCandidate,
    sample: WritingRouteCandidate,
) -> WritingRouteAlignment:
    best_alignment = None
    for direction in ("forward", "reversed-for-best-match"):
        sample_strokes = sample.continuous_strokes
        if direction == "reversed-for-best-match":
            sample_strokes = tuple(
                reverse_continuous_path(stroke) for stroke in reversed(sample_strokes)
            )
        paired_count = min(len(reference.continuous_strokes), len(sample_strokes))
        reference_samples = tuple(
            resample_continuous_path(stroke, 128).xy
            for stroke in reference.continuous_strokes[:paired_count]
        )
        sample_samples = tuple(
            resample_continuous_path(stroke, 128).xy for stroke in sample_strokes[:paired_count]
        )
        try:
            transform = fit_similarity_transform(
                np.concatenate(reference_samples, axis=0),
                np.concatenate(sample_samples, axis=0),
            )
            aligned_strokes = tuple(
                transform_continuous_path(stroke, transform) for stroke in sample_strokes
            )
            stroke_errors = tuple(
                float(np.mean(np.sum((reference_xy - transform.apply(sample_xy)) ** 2, axis=1)))
                for reference_xy, sample_xy in zip(
                    reference_samples,
                    sample_samples,
                    strict=True,
                )
            )
            stroke_transforms = (transform,) * paired_count
            strategy = "shared"
        except ValueError:
            per_stroke_alignments = tuple(
                align_continuous_paths(
                    reference.continuous_strokes[index],
                    sample_strokes[index],
                )
                for index in range(paired_count)
            )
            stroke_transforms = tuple(item.transform for item in per_stroke_alignments)
            transform = stroke_transforms[0]
            paired_aligned_strokes = []
            for item in per_stroke_alignments:
                assert item.aligned_sample_path is not None
                paired_aligned_strokes.append(item.aligned_sample_path)
            aligned_strokes = (
                *paired_aligned_strokes,
                *(
                    transform_continuous_path(stroke, transform)
                    for stroke in sample_strokes[paired_count:]
                ),
            )
            stroke_errors = tuple(item.mean_squared_error for item in per_stroke_alignments)
            strategy = "per-stroke"
        weights = tuple(
            0.5
            * (
                reference.continuous_strokes[index].arc_length_px
                + sample_strokes[index].arc_length_px
            )
            for index in range(paired_count)
        )
        mean_squared_error = float(
            sum(error * weight for error, weight in zip(stroke_errors, weights, strict=True))
            / max(sum(weights), 1e-12)
        )
        alignment = WritingRouteAlignment(
            transform=transform,
            direction=direction,
            aligned_sample_strokes=aligned_strokes,
            stroke_mean_squared_errors=stroke_errors,
            mean_squared_error=mean_squared_error,
            stroke_transforms=stroke_transforms,
            strategy=strategy,
        )
        if (
            best_alignment is None
            or alignment.mean_squared_error < best_alignment.mean_squared_error
        ):
            best_alignment = alignment
    assert best_alignment is not None
    return best_alignment


def select_writing_route_pair(
    reference_candidates: tuple[WritingRouteCandidate, ...],
    sample_candidates: tuple[WritingRouteCandidate, ...],
    *,
    steering_distance: Callable[[WritingRouteCandidate, WritingRouteCandidate], float]
    | None = None,
    direction_policy: WritingDirectionPolicy | None = None,
    reference_descriptors: tuple[RouteDescriptor, ...] | None = None,
    sample_descriptors: tuple[RouteDescriptor, ...] | None = None,
) -> WritingRoutePairSelection:
    return evaluate_writing_route_pairs(
        reference_candidates,
        sample_candidates,
        steering_distance=steering_distance,
        direction_policy=direction_policy,
        reference_descriptors=reference_descriptors,
        sample_descriptors=sample_descriptors,
    )
