from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from rasterregions import BBox, Region
from skeletonpaths import PathSearchLimits

from inkgrammar.gestures import (
    GestureSpanAnnotation,
    MovementAnnotation,
    MovementAnnotations,
    RasterRegionCandidate,
    build_glyph_observation_candidates,
)


def _region(region_id: str = "region-a") -> Region:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 4] = True
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return Region(
        region_id=region_id,
        bbox=BBox(0, 0, 9, 9),
        row_index=0,
        column_index=0,
        reading_order=0,
        confidence=1.0,
        component_indices=(0,),
        image=image,
        mask=mask,
    )


def _region_candidate(region_id: str = "region-a") -> RasterRegionCandidate:
    return RasterRegionCandidate(
        segmentation_hypothesis_id="segmentation-0",
        region_hypothesis_id=region_id,
        region=_region(region_id),
    )


def test_inferred_observations_preserve_bounded_skeleton_alternatives() -> None:
    candidates = build_glyph_observation_candidates(
        "obs-a",
        [_region_candidate("region-a"), _region_candidate("region-b")],
        max_candidates=2,
        path_search_limits=PathSearchLimits(max_paths=8),
    )

    assert len(candidates) == 2
    assert all(candidate.source_observation_id == "obs-a" for candidate in candidates)
    assert all(candidate.segmentation_hypothesis_id == "segmentation-0" for candidate in candidates)
    assert all(len(candidate.stroke_hypotheses) <= 2 for candidate in candidates)
    assert all(
        candidate.provenance.evidence_level == "inferred-ambiguous"
        for candidate in candidates
    )
    assert all(
        stroke.evidence.skeleton_stroke_hypothesis_id == stroke.id
        and stroke.evidence.direction_status == "ambiguous"
        and stroke.evidence.order_status == "ambiguous"
        for candidate in candidates
        for stroke in candidate.stroke_hypotheses
    )
    assert all(
        stroke.provenance.source_ids
        and candidate.region_hypothesis_id in stroke.provenance.source_ids
        for candidate in candidates
        for stroke in candidate.stroke_hypotheses
    )


def test_reviewed_annotations_bypass_static_skeleton_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = MovementAnnotation(
        id="movement-main",
        source_observation_id="obs-a",
        annotation_source_id="review-a",
        stroke_id="main",
        stroke_order=0,
        stroke_order_status="reviewed",
        direction="reverse",
        direction_status="reviewed",
        ordered_landmarks=[(0.0, 8.0), (4.0, 0.0), (8.0, 8.0)],
        gesture_spans=[
            GestureSpanAnnotation(id="underturn", start_index=0, end_index=2),
        ],
    )

    def fail_if_called(_mask: np.ndarray) -> None:
        pytest.fail("reviewed annotations must take precedence over static route inference")

    monkeypatch.setattr("inkgrammar.gestures.training.analyze_mask", fail_if_called)
    candidates = build_glyph_observation_candidates(
        "obs-a",
        [_region_candidate()],
        reviewed_annotations=MovementAnnotations(movements=[annotation]),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provenance.evidence_level == "reviewed-manual"
    assert candidate.provenance.orientation_status == "reviewed"
    assert candidate.provenance.stroke_order_status == "reviewed"
    assert len(candidate.stroke_hypotheses) == 1
    stroke = candidate.stroke_hypotheses[0]
    assert stroke.annotation is annotation
    assert stroke.direction == "reverse"
    assert stroke.stroke_order == 0
    assert stroke.gesture_boundaries[0].id == "boundary-movement-main-underturn"
    assert stroke.gesture_boundaries[0].status == "reviewed"
    assert stroke.evidence.evidence_level == "reviewed-manual"
    assert stroke.evidence.skeleton_stroke_hypothesis_id is None



def test_observed_annotations_remain_observed_in_candidate_provenance() -> None:
    annotation = MovementAnnotation(
        id="movement-observed",
        source_observation_id="obs-a",
        stroke_id="main",
        stroke_order=0,
        stroke_order_status="observed",
        direction="forward",
        direction_status="observed",
        ordered_landmarks=[(0.0, 0.0), (1.0, 1.0)],
    )

    candidates = build_glyph_observation_candidates(
        "obs-a",
        [_region_candidate()],
        reviewed_annotations=MovementAnnotations(movements=[annotation]),
    )

    assert candidates[0].provenance.orientation_status == "observed"
    assert candidates[0].provenance.stroke_order_status == "observed"


def test_reviewed_annotations_are_selected_by_source_observation_id() -> None:
    annotation = MovementAnnotation(
        id="movement-other",
        source_observation_id="obs-other",
        stroke_id="main",
        ordered_landmarks=[(0.0, 0.0), (1.0, 1.0)],
    )

    candidates = build_glyph_observation_candidates(
        "obs-a",
        [_region_candidate()],
        reviewed_annotations=MovementAnnotations(movements=[annotation]),
        max_candidates=1,
        path_search_limits=PathSearchLimits(max_paths=4),
    )

    assert candidates[0].stroke_hypotheses
    assert all(stroke.annotation is None for stroke in candidates[0].stroke_hypotheses)
    assert candidates[0].provenance.evidence_level == "inferred-ambiguous"
