from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from ..models import StrictModel
from .evidence import EvidenceLevel, EvidenceStatus

if TYPE_CHECKING:
    from .training import MovementAnnotation


BoundaryKind = Literal[
    "curvature-event",
    "topology-landmark",
    "baseline-contact",
    "loop-closure",
    "crossing",
    "template-endpoint",
    "contact",
]


class BoundaryEvent(StrictModel):
    id: str = Field(min_length=1)
    index: int = Field(ge=0)
    kind: BoundaryKind
    status: EvidenceStatus = "inferred"


class GestureBoundaryCandidate(StrictModel):
    id: str = Field(min_length=1)
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=1)
    evidence_ids: list[str] = Field(default_factory=list)
    status: EvidenceStatus
    evidence_level: EvidenceLevel

    @model_validator(mode="after")
    def validate_boundary(self) -> GestureBoundaryCandidate:
        if self.start_index >= self.end_index:
            raise ValueError("gesture boundary end_index must be after start_index")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("boundary evidence IDs must not be blank")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("boundary evidence IDs must be unique")
        return self


def propose_gesture_boundaries(
    landmark_count: int,
    events: tuple[BoundaryEvent, ...] | list[BoundaryEvent] = (),
    *,
    reviewed_annotation: MovementAnnotation | None = None,
    max_candidates: int = 8,
) -> tuple[GestureBoundaryCandidate, ...]:
    """Propose bounded gesture spans without treating inferred boundaries as verified."""
    if (
        isinstance(landmark_count, bool)
        or not isinstance(landmark_count, int)
        or landmark_count < 2
    ):
        raise ValueError("landmark_count must be an integer of at least two")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer")

    if reviewed_annotation is not None:
        if len(reviewed_annotation.ordered_landmarks) != landmark_count:
            raise ValueError("reviewed annotation landmark count does not match the proposal path")
        spans = reviewed_annotation.gesture_spans
        if len(spans) > max_candidates:
            raise ValueError("reviewed gesture spans exceed max_candidates")
        evidence_ids = [reviewed_annotation.id]
        if reviewed_annotation.annotation_source_id is not None:
            evidence_ids.append(reviewed_annotation.annotation_source_id)
        return tuple(
            GestureBoundaryCandidate(
                id=f"boundary-{reviewed_annotation.id}-{span.id}",
                start_index=span.start_index,
                end_index=span.end_index,
                evidence_ids=list(dict.fromkeys([*evidence_ids, span.id])),
                status="reviewed",
                evidence_level="reviewed-manual",
            )
            for span in spans
        )

    boundary_events = tuple(events)
    event_ids = [event.id for event in boundary_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("boundary event IDs must be unique")
    if any(event.index >= landmark_count for event in boundary_events):
        raise ValueError("boundary event index exceeds the proposal path")

    events_by_index: dict[int, list[str]] = {}
    for event in boundary_events:
        events_by_index.setdefault(event.index, []).append(event.id)
    indices = sorted({0, landmark_count - 1, *events_by_index})
    pairs: list[tuple[int, int]] = [(0, landmark_count - 1)]
    pairs.extend((indices[index], indices[index + 1]) for index in range(len(indices) - 1))
    pairs.extend(
        (indices[index], indices[index + 2]) for index in range(len(indices) - 2)
    )
    unique_pairs = list(dict.fromkeys(pairs))[:max_candidates]

    return tuple(
        GestureBoundaryCandidate(
            id=f"boundary-{start}-{end}",
            start_index=start,
            end_index=end,
            evidence_ids=list(
                dict.fromkeys([*events_by_index.get(start, []), *events_by_index.get(end, [])])
            ),
            status="ambiguous",
            evidence_level="inferred-ambiguous",
        )
        for start, end in unique_pairs
    )
