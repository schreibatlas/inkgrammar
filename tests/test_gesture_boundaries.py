from __future__ import annotations

import pytest
from pydantic import ValidationError

from inkgrammar.gestures import (
    BoundaryEvent,
    GestureBoundaryCandidate,
    GestureSpanAnnotation,
    MovementAnnotation,
    propose_gesture_boundaries,
)


def test_reviewed_annotation_boundaries_override_inferred_events() -> None:
    annotation = MovementAnnotation(
        id="move-main",
        source_observation_id="obs-a",
        annotation_source_id="review-a",
        stroke_id="main",
        ordered_landmarks=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0)],
        gesture_spans=[
            GestureSpanAnnotation(id="oval", start_index=0, end_index=2),
        ],
    )
    events = [BoundaryEvent(id="curvature-1", index=1, kind="curvature-event")]

    candidates = propose_gesture_boundaries(
        4,
        events,
        reviewed_annotation=annotation,
    )

    assert len(candidates) == 1
    assert candidates[0].start_index == 0
    assert candidates[0].end_index == 2
    assert candidates[0].status == "reviewed"
    assert candidates[0].evidence_level == "reviewed-manual"
    assert candidates[0].evidence_ids == ["move-main", "review-a", "oval"]


def test_inferred_boundary_alternatives_are_bounded_and_retain_evidence() -> None:
    events = [
        BoundaryEvent(id="turn-a", index=2, kind="curvature-event"),
        BoundaryEvent(id="contact-b", index=5, kind="baseline-contact"),
        BoundaryEvent(id="crossing-c", index=7, kind="crossing"),
    ]

    candidates = propose_gesture_boundaries(10, events, max_candidates=4)

    assert len(candidates) == 4
    assert all(candidate.status == "ambiguous" for candidate in candidates)
    assert all(candidate.evidence_level == "inferred-ambiguous" for candidate in candidates)
    assert {"turn-a", "contact-b", "crossing-c"} <= {
        evidence_id for candidate in candidates for evidence_id in candidate.evidence_ids
    }
    assert candidates == propose_gesture_boundaries(10, events, max_candidates=4)


def test_boundary_models_do_not_accept_unvalidated_confidence() -> None:
    with pytest.raises(ValidationError):
        BoundaryEvent(
            id="event-a",
            index=1,
            kind="topology-landmark",
            confidence=0.95,
        )


def test_boundary_candidate_rejects_empty_or_repeated_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        GestureBoundaryCandidate(
            id="boundary-a",
            start_index=0,
            end_index=2,
            evidence_ids=["event-a", "event-a"],
            status="ambiguous",
            evidence_level="inferred-ambiguous",
        )

    with pytest.raises(ValidationError):
        GestureBoundaryCandidate(
            id="boundary-b",
            start_index=0,
            end_index=2,
            evidence_ids=[" "],
            status="ambiguous",
            evidence_level="inferred-ambiguous",
        )
