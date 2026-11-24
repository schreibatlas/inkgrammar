from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from inkgrammar.comparison.descriptors import (
    reverse_route_descriptor,
    route_descriptor,
    route_descriptor_distance,
)
from inkgrammar.comparison.models import (
    ContinuousPath,
    CropRect,
    WritingRouteScore,
    WritingStructurePolicy,
)
from inkgrammar.comparison.routes import (
    ROUTE_REPORT_LIMIT,
    evaluate_writing_route_pairs,
    generate_writing_route_candidates,
    select_writing_route_pair,
)
from inkgrammar.comparison.segmentation import SegmentedGlyph
from inkgrammar.comparison.skeleton import analyze_skeleton


def _candidate():
    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 5:28] = True
    diagnostics = analyze_skeleton(mask)
    candidates = generate_writing_route_candidates(diagnostics, limit=1)
    assert candidates
    return candidates[0]


def _continuous_path(points: tuple[tuple[float, float], ...]) -> ContinuousPath:
    xy = np.asarray(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    tangents = np.gradient(xy, axis=0)
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    tangents = tangents / tangent_lengths[:, None]
    return ContinuousPath(
        xy=xy,
        tangent=tangents,
        curvature=np.zeros(len(xy)),
        radius_px=np.ones(len(xy)),
        confidence=np.ones(len(xy)),
        progress=cumulative / cumulative[-1],
        arc_length_px=float(cumulative[-1]),
        source="test",
        boundary_evidence="test",
    )


def _route(candidate, path: ContinuousPath, steering_cost: float = 0.0):
    hypothesis = replace(
        candidate.hypothesis,
        orientation_ambiguous=False,
        orientation_ambiguous_strokes=(False,),
    )
    return replace(
        candidate,
        hypothesis=hypothesis,
        continuous_strokes=(path,),
        glyph_endpoints=(
            (float(path.xy[0, 0]), float(path.xy[0, 1])),
            (float(path.xy[-1, 0]), float(path.xy[-1, 1])),
        ),
        score=WritingRouteScore(
            coverage_cost=0.0,
            repeat_cost=0.0,
            lift_cost=0.0,
            continuity_cost=0.0,
            steering_complexity_cost=steering_cost,
        ),
        ambiguous=False,
    )


def test_centerline_cost_uses_each_stroke_without_joining_them() -> None:
    from inkgrammar.comparison.routes import _centerline_cost

    candidate = _candidate()
    first = _continuous_path(((0, 0), (1, 0), (2, 0), (3, 0)))
    second = _continuous_path(((100, 100), (101, 100), (102, 100), (103, 100)))
    changed_second = _continuous_path(((100, 100), (101, 101), (102, 99), (103, 100)))
    reference = replace(candidate, continuous_strokes=(first, second))
    identical = replace(candidate, continuous_strokes=(first, second))
    changed = replace(candidate, continuous_strokes=(first, changed_second))

    assert _centerline_cost(reference, identical) == 0.0
    assert _centerline_cost(reference, changed) > 0.0


def test_route_candidate_search_respects_requested_bound() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 5:28] = True
    diagnostics = analyze_skeleton(mask)

    candidates = generate_writing_route_candidates(diagnostics, limit=1)

    assert sum(candidate.orientation_variant == "native" for candidate in candidates) == 1
    assert not candidates[0].hypothesis.uncovered_edge_ids
    assert candidates[0].score.coverage_cost == 0.0


def test_route_search_uses_stroke_diversity_and_global_reversal(monkeypatch) -> None:
    from inkgrammar.comparison import routes

    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 5:28] = True
    diagnostics = analyze_skeleton(mask)
    original_search = routes.stroke_hypotheses_from_analysis
    search_options = {}

    def capture_search(analysis, **kwargs):
        search_options.update(kwargs)
        return original_search(analysis, **kwargs)

    monkeypatch.setattr(routes, "stroke_hypotheses_from_analysis", capture_search)
    candidates = routes.generate_writing_route_candidates(diagnostics, expected_strokes=2)

    assert routes.ROUTE_SEARCH_LIMIT > ROUTE_REPORT_LIMIT
    assert search_options["limit"] == routes.ROUTE_SEARCH_LIMIT
    assert search_options["diversity"] == "stroke-count"
    assert search_options["expected_strokes"] == 2
    native = next(
        candidate for candidate in candidates if candidate.orientation_variant == "native"
    )
    assert native.hypothesis.orientation_ambiguous
    reverse = next(
        candidate for candidate in candidates if candidate.orientation_variant == "global-reverse"
    )
    np.testing.assert_allclose(
        reverse.continuous_strokes[0].xy,
        native.continuous_strokes[0].xy[::-1],
    )


def test_route_shortlist_preserves_best_stroke_count_and_orientation_alternatives(monkeypatch):
    from types import SimpleNamespace

    import inkgrammar.comparison.steering as steering
    from inkgrammar.comparison.routes import shortlist_writing_route_candidates

    base = _candidate()
    candidates = []
    for index in range(10):
        hypothesis = replace(
            base.hypothesis,
            orientation_ambiguous=(index == 0),
            orientation_ambiguous_strokes=(index == 0,),
        )
        candidates.append(
            replace(
                base,
                hypothesis=hypothesis,
                continuous_strokes=(base.continuous_strokes[0],) * (2 if index == 2 else 1),
                score=replace(base.score, steering_complexity_cost=float(index)),
                orientation_variant="global-reverse" if index == 1 else "native",
                orientation_group=1 if index in (0, 1) else index + 1,
            )
        )

    shortlist = shortlist_writing_route_candidates(tuple(candidates), limit=4)

    assert {id(item) for item in shortlist} == {id(candidates[index]) for index in (0, 1, 2, 3)}

    fit_calls = []

    def count_fit(path):
        fit_calls.append(path)
        return SimpleNamespace(selected_change_count=0)

    monkeypatch.setattr(steering, "fit_heading_rate_model", count_fit)
    fitted = steering.fit_writing_route_candidates(shortlist)

    assert len(fitted) == len(shortlist)
    assert len(fit_calls) == sum(len(candidate.continuous_strokes) for candidate in shortlist)
    assert len(fit_calls) < sum(len(candidate.continuous_strokes) for candidate in candidates)


def test_joint_route_selection_is_deterministic_and_marks_near_ties() -> None:
    candidate = _candidate()
    straight = _continuous_path(((0, 0), (1, 0), (2, 0), (3, 0)))
    curved = _continuous_path(((0, 0), (1, 1), (2, -1), (3, 0)))
    reference_routes = (_route(candidate, straight), _route(candidate, curved, 0.2))
    sample_routes = (_route(candidate, straight), _route(candidate, curved, 0.2))

    selected = select_writing_route_pair(reference_routes, sample_routes)

    assert (selected.score.reference_rank, selected.score.sample_rank) == (1, 1)
    assert not selected.ambiguous
    assert selected.score.total_cost < selected.alternatives[1].total_cost
    assert selected.score.total_cost == pytest.approx(sum(selected.score.cost_breakdown.values()))

    tied = select_writing_route_pair(
        (_route(candidate, straight), _route(candidate, straight)),
        (_route(candidate, straight),),
    )
    assert (tied.score.reference_rank, tied.score.sample_rank) == (1, 1)
    assert tied.ambiguous


def test_route_pair_evaluation_includes_descriptor_cost_in_additive_breakdown() -> None:
    candidate = _candidate()
    straight = _route(
        candidate,
        _continuous_path(((0, 0), (1, 0), (2, 0), (3, 0))),
    )
    curved = _route(
        candidate,
        _continuous_path(((0, 0), (1, 1), (2, -1), (3, 0))),
    )
    reference_descriptors = (route_descriptor(straight), route_descriptor(curved))
    sample_descriptors = (route_descriptor(curved),)

    result = evaluate_writing_route_pairs(
        (straight, curved),
        (curved,),
        reference_descriptors=reference_descriptors,
        sample_descriptors=sample_descriptors,
    )

    assert result.score.reference_rank == 2
    assert len(result.alternatives) == 2
    assert result.score.descriptor_matching_cost == pytest.approx(0.0)
    assert result.alternatives[1].descriptor_matching_cost > 0.0
    assert result.score.total_cost == pytest.approx(sum(result.score.cost_breakdown.values()))


def test_planarcurve_fit_summary_and_width_normalization() -> None:
    from inkgrammar.comparison.metrics import _width_summary
    from inkgrammar.comparison.steering import (
        fit_writing_route_candidate,
        steering_model_summary,
        steering_profile_distance,
    )

    candidate = _candidate()
    fitted = fit_writing_route_candidate(candidate)
    summary = steering_model_summary(fitted.steering_models[0])

    assert summary["fit"]["success"]
    assert summary["knot_progress"][0] == 0.0
    assert summary["knot_progress"][-1] == 1.0
    assert steering_profile_distance(fitted, fitted) == 0.0

    path = _continuous_path(((0, 0), (1, 0), (2, 0), (3, 0)))
    narrow = _route(candidate, replace(path, radius_px=np.full(len(path.xy), 2.0)))
    wide = _route(candidate, replace(path, radius_px=np.full(len(path.xy), 4.0)))

    narrow_width = _width_summary(narrow, glyph_height=20.0)
    wide_width = _width_summary(wide, glyph_height=40.0)

    assert (
        narrow_width["median_width_over_glyph_height"]
        == wide_width["median_width_over_glyph_height"]
    )


def test_internal_graph_edge_boundary_is_not_a_writing_event() -> None:
    from types import SimpleNamespace

    from skeletonpaths import SkeletonEdge, SkeletonGraph, SkeletonNode, SkeletonWalk

    from inkgrammar.comparison.path import _semantic_junction_transition_indices

    nodes = (
        SkeletonNode(0, (0, 0), 1, 1.0, 1.0),
        SkeletonNode(1, (1, 0), 2, 1.0, 1.0),
        SkeletonNode(2, (2, 0), 1, 1.0, 1.0),
    )
    edges = (
        SkeletonEdge(0, 1, np.asarray(((0, 0), (1, 0))), 1.0, 1.0, 1.0),
        SkeletonEdge(1, 2, np.asarray(((1, 0), (2, 0))), 1.0, 1.0, 1.0),
    )
    graph = SkeletonGraph(nodes, edges)
    walk = SkeletonWalk(
        xy=np.asarray(((0, 0), (1, 0), (1, 0), (2, 0))),
        edge_ids=(0, 1),
        edge_directions=("forward", "forward"),
        start_node=0,
        end_node=2,
        covered_edge_ids=(0, 1),
        covered_edge_count=2,
        total_edge_count=2,
        repeated_edge_count=0,
        length_px=2.0,
    )
    diagnostics = SimpleNamespace(graph=graph)
    refined_graph = SimpleNamespace(
        edges=(
            SimpleNamespace(edge_id=0, xy=np.asarray(((0, 0), (1, 0)))),
            SimpleNamespace(edge_id=1, xy=np.asarray(((1, 0), (2, 0)))),
        )
    )

    transitions = _semantic_junction_transition_indices(
        diagnostics,
        walk,
        refined_graph,
        np.asarray((0.0, 0.5, 0.5, 1.0)),
    )

    assert transitions == ()


def test_writing_structure_prior_is_soft_and_uses_assigned_mark_count() -> None:
    from inkgrammar.comparison.routes import _writing_structure_prior_cost

    policy = WritingStructurePolicy()
    assert _writing_structure_prior_cost(1, 1, policy, 0) == 0.0
    assert _writing_structure_prior_cost(2, 1, policy, 0) > 0.0
    assert _writing_structure_prior_cost(2, 2, policy, 0) == 0.0
    assert _writing_structure_prior_cost(1, 2, policy, 0) == policy.detached_mark_stroke_weight
    assert _writing_structure_prior_cost(1, 1, policy, 1) == policy.side_component_penalty


def test_route_descriptor_uses_planarcurve_and_carries_component_roles() -> None:
    candidate = _candidate()
    glyph = SegmentedGlyph(
        glyph_id="glyph",
        bbox=CropRect(x=0, y=0, width=32, height=32),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=Image.new("RGB", (32, 32), "white"),
        mask=np.ones((32, 32), dtype=bool),
        component_indices=(0, 1, 2),
        primary_component_indices=(0,),
        secondary_component_indices=(1,),
        ambiguous_component_indices=(2,),
        secondary_mark_centroids=((0.8, 0.2),),
    )
    descriptor = route_descriptor(candidate, glyph)
    reversed_descriptor = reverse_route_descriptor(descriptor)

    assert descriptor.stroke_count == 1
    assert descriptor.pen_lift_count == 0
    assert descriptor.secondary_mark_count == 1
    assert descriptor.component_role_sequence == ("primary", "secondary", "ambiguous")
    assert descriptor.relative_mark_centroids == ((0.8, 0.2),)
    assert route_descriptor_distance(descriptor, descriptor) == pytest.approx(0.0)
    assert route_descriptor_distance(descriptor, reversed_descriptor) == pytest.approx(0.0)
