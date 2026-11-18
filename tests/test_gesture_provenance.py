from __future__ import annotations

import pytest
from pydantic import ValidationError

from inkgrammar.gestures import CandidateProvenance, OccurrenceEvidence, PrimitiveProvenance


def test_occurrence_evidence_preserves_separate_hypotheses_and_annotation_status() -> None:
    evidence = OccurrenceEvidence(
        source_observation_id="glyph-sample-1",
        segmentation_hypothesis_id="region-route-2",
        skeleton_stroke_hypothesis_id="stroke-route-3",
        order_status="reviewed",
        direction_status="observed",
        gesture_annotation_status="reviewed",
        primitive_assignment_status="inferred",
        evidence_level="reviewed-manual",
        fit_residuals={"centerline": 0.04, "topology": 0.0},
    )

    assert evidence.model_dump()["direction_status"] == "observed"
    assert evidence.model_dump()["order_status"] == "reviewed"
    assert evidence.segmentation_hypothesis_id != evidence.skeleton_stroke_hypothesis_id
    with pytest.raises(ValidationError, match="nonnegative"):
        OccurrenceEvidence(source_observation_id="sample", fit_residuals={"geometry": -0.1})
    with pytest.raises(ValidationError):
        OccurrenceEvidence(source_observation_id="sample", confidence=0.9)


def test_candidate_and_primitive_provenance_do_not_overstate_inferred_routes() -> None:
    candidate = CandidateProvenance(
        route_status="ambiguous",
        orientation_status="ambiguous",
        stroke_order_status="not-recorded",
        evidence_level="inferred-ambiguous",
        source_ids=["observation-1"],
    )
    primitive = PrimitiveProvenance(
        geometry_status="inferred",
        construction_status="reviewed",
        trajectory_status="inferred",
        source_ids=["observation-1"],
    )

    assert candidate.route_status == "ambiguous"
    assert candidate.stroke_order_status == "not-recorded"
    assert primitive.geometry_status == "inferred"
    with pytest.raises(ValidationError):
        CandidateProvenance(route_status="verified")
    with pytest.raises(ValidationError, match="unique"):
        PrimitiveProvenance(source_ids=["scan-1", "scan-1"])
