from __future__ import annotations

import numpy as np
import pytest
from planarcurve import HeadingRateProfile, Trajectory2D, evaluate_trajectory
from planarcurve.fitting import TrajectoryFitOptions, TrajectoryModelSelectionOptions

from inkgrammar.gestures import (
    DirectedCurveOccurrence,
    GeometryPrimitiveFamily,
    GeometryVariationPayload,
    GestureConstraint,
    GestureOccurrence,
    GestureParameter,
    GesturePrimitiveSlot,
    GestureTemplate,
    OccurrenceEvidence,
    fit_geometry_family,
    fit_gesture_template,
    learn_gesture_dictionary,
    match_gesture_candidates,
    rank_gesture_matches,
    trajectory_to_payload,
)


def _trajectory() -> Trajectory2D:
    return Trajectory2D(
        length=1.0,
        start_heading_deg=0.0,
        heading_rate=HeadingRateProfile(90.0, 90.0),
    )


def _sampled_curve(scale: float = 1.0, dx: float = 0.0) -> np.ndarray:
    points = evaluate_trajectory(_trajectory(), count=24).xy
    return points * scale + np.array([dx, -dx / 2.0])


def _family() -> GeometryPrimitiveFamily:
    return GeometryPrimitiveFamily(
        id="curve-family-001",
        trajectory=trajectory_to_payload(_trajectory()),
        selected_change_count=0,
        variation=GeometryVariationPayload(
            parameter_dispersion={},
            outlier_rate=0.0,
            occurrence_count=3,
        ),
    )


def _template(template_id: str) -> GestureTemplate:
    return GestureTemplate(
        id=template_id,
        role="oval",
        primitives=[
            GesturePrimitiveSlot(id="round", primitive_family="curve-family-001"),
            GesturePrimitiveSlot(id="exit", primitive_family="curve-family-001"),
        ],
        constraints=[
            GestureConstraint(
                id="join",
                kind="attachment",
                first="round",
                second="exit",
            ),
        ],
        parameters=[
            GestureParameter(
                id="exit-length",
                target="exit.relative-length",
                category="relative-length",
                initial=0.5,
                lower=0.2,
                upper=0.8,
                shared_group="exit-length",
            ),
        ],
    )


def _occurrence(
    occurrence_id: str,
    *,
    split: str = "training",
    parameter_value: float = 0.5,
) -> GestureOccurrence:
    evidence = OccurrenceEvidence(
        source_observation_id=f"source-{occurrence_id}",
        segmentation_hypothesis_id=f"seg-{occurrence_id}",
        skeleton_stroke_hypothesis_id=f"skeleton-{occurrence_id}",
        evidence_level="inferred-ambiguous",
    )
    curve = _sampled_curve()
    return GestureOccurrence(
        id=occurrence_id,
        primitive_curves={"round": curve, "exit": curve},
        evidence=evidence,
        split=split,
        parameter_values={"exit-length": parameter_value},
        constraint_residuals={"join": 0.02},
        topology_compatibility="compatible",
        evidence_ids=(f"annotation-{occurrence_id}",),
    )


def test_geometry_family_fit_delegates_to_planarcurve_and_reports_holdout() -> None:
    occurrences = [
        DirectedCurveOccurrence(
            id=f"occurrence-{index}",
            source_observation_id=f"observation-{index}",
            points=_sampled_curve(scale=1.0 + index * 0.1, dx=float(index)),
            split="training",
            evidence_ids=(f"source-record-{index}",),
        )
        for index in range(3)
    ]
    occurrences.append(
        DirectedCurveOccurrence(
            id="occurrence-held-out",
            source_observation_id="observation-held-out",
            points=_sampled_curve(scale=1.4, dx=4.0),
            split="held-out",
            evidence_ids=("source-record-held-out",),
        )
    )

    result = fit_geometry_family(
        "curve-family-004",
        occurrences,
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=24,
            restarts=0,
            max_nfev=100,
        ),
        selection_options=TrajectoryModelSelectionOptions(
            change_counts=(0,),
            validation_fraction=0.25,
        ),
    )

    assert result.family.selected_change_count == 0
    assert result.family.variation.occurrence_count == 4
    assert result.family.training_metrics["occurrence_count"] == 3.0
    assert result.family.held_out_metrics["occurrence_count"] == 1.0
    assert result.family.held_out_metrics["mean_normalized_directed_rmse"] < 0.02
    assert result.family.held_out_metrics["mean_normalized_symmetric_distance"] < 0.02
    assert [match.occurrence_id for match in result.occurrence_fits] == [
        "occurrence-0",
        "occurrence-1",
        "occurrence-2",
        "occurrence-held-out",
    ]


def test_gesture_dictionary_fits_bounded_parameters_and_keeps_match_breakdown() -> None:
    family = _family()
    template = _template("oval-exit")
    occurrences = [
        _occurrence("train-a", parameter_value=0.4),
        _occurrence("train-b", parameter_value=0.6),
        _occurrence("held-out", split="held-out", parameter_value=0.7),
    ]

    fit = fit_gesture_template(
        template,
        {family.id: family},
        occurrences,
    )
    held_out = fit.matches[-1]

    assert fit.fitted_parameter_values["exit-length"] == pytest.approx(0.5)
    assert fit.template.parameters[0].initial == pytest.approx(0.5)
    assert fit.primitive_reuse_counts == {family.id: 2}
    assert fit.held_out_metrics["occurrence_count"] == 1.0
    assert held_out.breakdown.geometry_fit < 1e-3
    assert held_out.breakdown.constraint_fit == pytest.approx(0.02)
    assert held_out.breakdown.parameter_deviation == pytest.approx(1 / 3)
    assert held_out.breakdown.topology_compatibility == "compatible"
    assert "annotation-held-out" in held_out.breakdown.evidence_source_ids


def test_dictionary_ranks_ties_as_ambiguous_and_reports_shared_family_reuse() -> None:
    family = _family()
    templates = [_template("template-a"), _template("template-b")]
    occurrences = [_occurrence("candidate-a"), _occurrence("candidate-b")]

    matches = match_gesture_candidates(
        templates,
        {family.id: family},
        occurrences,
        max_candidates=2,
    )
    ranked = rank_gesture_matches(matches, max_candidates=2)
    learned = learn_gesture_dictionary(
        templates,
        {family.id: family},
        occurrences,
        max_candidates=2,
    )

    assert len(ranked) == 4
    assert all(match.ambiguous for match in ranked)
    assert all(match.rank in {1, 2} for match in ranked)
    assert learned.family_reuse_counts == {family.id: 4}
    assert len(learned.ranked_matches) == 4


def test_gesture_parameter_observations_must_respect_template_bounds() -> None:
    family = _family()
    occurrence = _occurrence("out-of-bounds", parameter_value=0.9)

    with pytest.raises(ValueError, match="outside bounds"):
        fit_gesture_template(
            _template("bounded-template"),
            {family.id: family},
            [occurrence],
        )
