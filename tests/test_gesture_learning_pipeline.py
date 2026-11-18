from __future__ import annotations

import numpy as np
from PIL import Image
from planarcurve.fitting import TrajectoryFitOptions, TrajectoryModelSelectionOptions
from rasterregions import BBox, Region

from inkgrammar.fitting.models import (
    CurvatureProfile,
    CurvatureStrokeGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
)
from inkgrammar.gestures import (
    DirectedCurveOccurrence,
    GestureGlyphProgram,
    GestureLearningResult,
    GestureOccurrence,
    GesturePrimitiveSlot,
    GestureStroke,
    GestureTemplate,
    MovementAnnotation,
    MovementAnnotations,
    OccurrenceEvidence,
    ProgramGesture,
    RasterRegionCandidate,
    build_glyph_observation_candidates,
    run_gesture_learning,
)
from inkgrammar.programs.models import ProgramLandmark


def _region() -> Region:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 4] = True
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return Region(
        region_id="region-a",
        bbox=BBox(0, 0, 9, 9),
        row_index=0,
        column_index=0,
        reading_order=0,
        confidence=1.0,
        component_indices=(0,),
        image=image,
        mask=mask,
    )


def _template() -> GestureTemplate:
    return GestureTemplate(
        id="stem-template",
        role="stem",
        primitives=[GesturePrimitiveSlot(id="main", primitive_family="curve-family-001")],
    )


def _program(template: GestureTemplate) -> GestureGlyphProgram:
    gesture = ProgramGesture(
        id="main-instance",
        gesture=template.id,
        from_landmark="start",
        to_landmark="end",
    )
    return GestureGlyphProgram(
        id="glyph-a",
        character="a",
        variant="default",
        gestures={gesture.id: gesture},
        strokes=[GestureStroke(id="stroke-main", gestures=[gesture.id])],
        landmarks={
            "start": ProgramLandmark(id="start", kind="start", x=0.0, y=0.0),
            "end": ProgramLandmark(id="end", kind="end", x=0.0, y=1.0),
        },
        entry_landmark="start",
        exit_landmark="end",
    )


def _primitive() -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id="curve-family-001",
        kind="curvature-profile",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(0.0, 1.0)),
        curvature_stroke=CurvatureStrokeGeometry(
            length=1.0,
            start_angle_deg=90.0,
            curvature=CurvatureProfile(start_rate_deg=0.0, end_rate_deg=0.0),
        ),
    )


def _geometry_occurrences(points: np.ndarray) -> dict[str, list[DirectedCurveOccurrence]]:
    return {
        "curve-family-001": [
            DirectedCurveOccurrence(
                id="family-training",
                source_observation_id="obs-a",
                points=points,
            ),
            DirectedCurveOccurrence(
                id="family-held-out",
                source_observation_id="obs-held-out",
                points=points * 1.2 + np.array([2.0, 3.0]),
                split="held-out",
            ),
        ]
    }


def _gesture_occurrences(
    points: np.ndarray,
    *,
    skeleton_id: str | None,
    evidence_ids: tuple[str, ...] = (),
) -> list[GestureOccurrence]:
    return [
        GestureOccurrence(
            id="gesture-training",
            primitive_curves={"main": points},
            evidence=OccurrenceEvidence(
                source_observation_id="obs-a",
                segmentation_hypothesis_id="seg-a",
                skeleton_stroke_hypothesis_id=skeleton_id,
                evidence_level="inferred-ambiguous" if skeleton_id else "reviewed-manual",
            ),
            split="training",
            topology_compatibility="unknown",
            evidence_ids=evidence_ids,
        ),
        GestureOccurrence(
            id="gesture-held-out",
            primitive_curves={"main": points * 1.2 + np.array([2.0, 3.0])},
            evidence=OccurrenceEvidence(
                source_observation_id="obs-held-out",
                segmentation_hypothesis_id="seg-held-out",
                skeleton_stroke_hypothesis_id="skeleton-held-out",
                evidence_level="source-backed",
            ),
            split="held-out",
            topology_compatibility="compatible",
        ),
    ]


def _run(
    *,
    reviewed: bool,
    region_candidate: RasterRegionCandidate,
    points: np.ndarray,
    stroke_id: str | None,
) -> GestureLearningResult:
    template = _template()
    annotation = MovementAnnotation(
        id="reviewed-main",
        source_observation_id="obs-a",
        annotation_source_id="review-source-a",
        stroke_id="main",
        stroke_order=0,
        stroke_order_status="reviewed",
        direction="forward",
        direction_status="reviewed",
        ordered_landmarks=[(float(x), float(y)) for x, y in points],
    )
    gesture_occurrences = _gesture_occurrences(
        points,
        skeleton_id=stroke_id,
        evidence_ids=(annotation.id,) if reviewed else (),
    )
    return run_gesture_learning(
        "obs-a",
        [region_candidate],
        _geometry_occurrences(points),
        [template],
        gesture_occurrences,
        [_program(template)],
        {"curve-family-001": _primitive()},
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=12,
            restarts=0,
            max_nfev=60,
        ),
        selection_options=TrajectoryModelSelectionOptions(
            change_counts=(0,),
            validation_fraction=0.2,
        ),
        reviewed_annotations=(MovementAnnotations(movements=[annotation]) if reviewed else None),
        max_candidates=2,
    )


def test_unreviewed_learning_pipeline_is_bounded_and_preserves_joint_ambiguity() -> None:
    region_candidate = RasterRegionCandidate(
        segmentation_hypothesis_id="seg-a",
        region_hypothesis_id="region-a",
        region=_region(),
    )
    # First construct the deterministic top-ranked route ID consumed by the dictionary stage.

    observation = build_glyph_observation_candidates(
        "obs-a",
        [region_candidate],
        max_candidates=2,
    )[0]
    stroke = observation.stroke_hypotheses[0]
    points = np.asarray(stroke.points)

    result = _run(
        reviewed=False,
        region_candidate=region_candidate,
        points=points,
        stroke_id=stroke.id,
    )

    assert len(result.observation_candidates) == 1
    assert len(result.observation_candidates[0].stroke_hypotheses) == 2
    assert result.joint_alternatives[0].ambiguity_status == "ambiguous"
    assert result.joint_alternatives[0].stroke_hypothesis_ids == tuple(
        item.id for item in result.observation_candidates[0].stroke_hypotheses
    )
    assert len(result.joint_alternatives[0].matches) == 1
    assert result.geometry_family_fits[0].family.held_out_metrics
    assert result.dictionary_fit.template_fits[0].held_out_metrics
    assert len(result.compiled_programs) == 1
    assert result.metrics["skeleton_stroke_hypothesis_count"] == 2.0
    assert "mean_geometry_held_out_directed_rmse" in result.metrics
    assert "mean_gesture_held_out_geometry_fit" in result.metrics
    assert "total_score" not in result.metrics

    repeated = _run(
        reviewed=False,
        region_candidate=region_candidate,
        points=points,
        stroke_id=stroke.id,
    )
    assert result.metrics == repeated.metrics
    assert result.joint_alternatives == repeated.joint_alternatives


def test_reviewed_learning_pipeline_uses_annotations_and_keeps_status() -> None:
    region_candidate = RasterRegionCandidate(
        segmentation_hypothesis_id="seg-a",
        region_hypothesis_id="region-a",
        region=_region(),
    )
    points = np.asarray([(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)])
    result = _run(
        reviewed=True,
        region_candidate=region_candidate,
        points=points,
        stroke_id=None,
    )

    assert len(result.observation_candidates[0].stroke_hypotheses) == 1
    assert result.observation_candidates[0].provenance.evidence_level == "reviewed-manual"
    assert result.joint_alternatives[0].ambiguity_status == "reviewed"
    assert result.joint_alternatives[0].matches[0].breakdown.evidence_source_ids
    assert len(result.compiled_programs) == 1
