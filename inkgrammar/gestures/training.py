from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from string import ascii_lowercase
from typing import Literal

from pydantic import Field, field_validator, model_validator
from rasterregions import Region, RegionHypothesis, SpacedSegmentationHypotheses
from skeletonpaths import (
    PathSearchLimits,
    SkeletonPath,
    analyze_mask,
    prepare_topology_paths,
    rank_paths,
)

from ..models import StrictModel
from .boundaries import (
    BoundaryEvent,
    GestureBoundaryCandidate,
    propose_gesture_boundaries,
)
from .evidence import CandidateProvenance, EvidenceStatus, OccurrenceEvidence
from .models import GestureRole

Point = tuple[float, float]


class GestureSpanAnnotation(StrictModel):
    id: str = Field(min_length=1)
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=1)
    role: GestureRole | None = None

    @model_validator(mode="after")
    def validate_span(self) -> GestureSpanAnnotation:
        if self.start_index >= self.end_index:
            raise ValueError("gesture span end_index must be after start_index")
        return self


class MovementAnnotation(StrictModel):
    id: str = Field(min_length=1)
    source_observation_id: str = Field(min_length=1)
    annotation_source_id: str | None = Field(default=None, min_length=1)
    stroke_id: str = Field(min_length=1)
    stroke_order: int | None = Field(default=None, ge=0)
    stroke_order_status: EvidenceStatus = "not-recorded"
    direction: Literal["forward", "reverse"] | None = None
    direction_status: EvidenceStatus = "not-recorded"
    ordered_landmarks: list[Point] = Field(min_length=2)
    gesture_spans: list[GestureSpanAnnotation] = Field(default_factory=list)

    @field_validator("ordered_landmarks")
    @classmethod
    def validate_landmarks(cls, values: list[Point]) -> list[Point]:
        if any(not all(isfinite(coordinate) for coordinate in point) for point in values):
            raise ValueError("movement annotation landmarks must be finite")
        return values

    @model_validator(mode="after")
    def validate_annotation(self) -> MovementAnnotation:
        if self.stroke_order_status in {"observed", "reviewed"} and self.stroke_order is None:
            raise ValueError("observed or reviewed stroke order requires stroke_order")
        if self.direction_status in {"observed", "reviewed"} and self.direction is None:
            raise ValueError("observed or reviewed direction requires direction")
        span_ids = [span.id for span in self.gesture_spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("gesture span IDs must be unique")
        previous_end = -1
        for span in sorted(self.gesture_spans, key=lambda item: item.start_index):
            if span.end_index >= len(self.ordered_landmarks):
                raise ValueError(f"gesture span exceeds movement landmarks: {span.id}")
            if span.start_index < previous_end:
                raise ValueError("gesture spans must not overlap")
            previous_end = span.end_index
        return self


class MovementAnnotations(StrictModel):
    format: Literal["inkgrammar-movement-annotations"] = "inkgrammar-movement-annotations"
    format_version: Literal[1] = 1
    movements: list[MovementAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_movement_ids(self) -> MovementAnnotations:
        ids = [movement.id for movement in self.movements]
        if len(ids) != len(set(ids)):
            raise ValueError("movement annotation IDs must be unique")
        return self




@dataclass(frozen=True)
class RasterRegionCandidate:
    segmentation_hypothesis_id: str
    region_hypothesis_id: str
    region: Region | RegionHypothesis

    def __post_init__(self) -> None:
        if not self.segmentation_hypothesis_id.strip() or not self.region_hypothesis_id.strip():
            raise ValueError("region candidate hypothesis IDs must not be blank")


@dataclass(frozen=True)
class StrokePathHypothesis:
    id: str
    path_kind: Literal["open", "loop", "open-with-loop"] | None
    points: tuple[Point, ...]
    edge_ids: tuple[int, ...]
    stroke_order: int | None
    direction: Literal["forward", "reverse"] | None
    gesture_boundaries: tuple[GestureBoundaryCandidate, ...]
    provenance: CandidateProvenance
    evidence: OccurrenceEvidence
    annotation: MovementAnnotation | None = None


@dataclass(frozen=True)
class GlyphObservationCandidate:
    id: str
    source_observation_id: str
    segmentation_hypothesis_id: str
    region_hypothesis_id: str
    region: Region | RegionHypothesis
    stroke_hypotheses: tuple[StrokePathHypothesis, ...]
    provenance: CandidateProvenance


def region_candidates_from_segmentation(
    segmentation: SpacedSegmentationHypotheses,
    *,
    max_candidates: int = 8,
) -> tuple[RasterRegionCandidate, ...]:
    """Keep bounded RasterRegions alternatives with deterministic source IDs."""
    _validate_candidate_limit(max_candidates)
    candidates: list[RasterRegionCandidate] = []
    if segmentation.page_hypotheses:
        for page_index, page in enumerate(segmentation.page_hypotheses):
            for region_index, region in enumerate(page.regions):
                candidates.append(
                    RasterRegionCandidate(
                        segmentation_hypothesis_id=f"segmentation-{page_index:03d}",
                        region_hypothesis_id=f"region-{page_index:03d}-{region_index:03d}",
                        region=region,
                    )
                )
                if len(candidates) == max_candidates:
                    return tuple(candidates)
        return tuple(candidates)

    for region_index, region in enumerate(segmentation.selected.regions):
        candidates.append(
            RasterRegionCandidate(
                segmentation_hypothesis_id="segmentation-selected",
                region_hypothesis_id=f"region-selected-{region_index:03d}",
                region=region,
            )
        )
        if len(candidates) == max_candidates:
            break
    return tuple(candidates)


def build_glyph_observation_candidates(
    source_observation_id: str,
    region_candidates: Sequence[RasterRegionCandidate],
    *,
    reviewed_annotations: MovementAnnotations | None = None,
    boundary_events: Mapping[str, Sequence[BoundaryEvent]] | None = None,
    max_candidates: int = 8,
    path_search_limits: PathSearchLimits | None = None,
) -> tuple[GlyphObservationCandidate, ...]:
    """Build evidence-aware observations, retaining bounded segmentation and path alternatives."""
    _validate_candidate_limit(max_candidates)
    if not source_observation_id.strip():
        raise ValueError("source_observation_id must not be blank")

    regions = tuple(region_candidates[:max_candidates])
    region_ids = [candidate.region_hypothesis_id for candidate in regions]
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("region hypothesis IDs must be unique")
    annotations = (
        tuple(
            movement
            for movement in reviewed_annotations.movements
            if movement.source_observation_id == source_observation_id
        )
        if reviewed_annotations is not None
        else ()
    )
    search_limits = path_search_limits or PathSearchLimits(max_paths=max_candidates * 4)
    candidates: list[GlyphObservationCandidate] = []

    for region_candidate in regions:
        if annotations:
            stroke_hypotheses = tuple(
                _reviewed_stroke_hypothesis(
                    source_observation_id,
                    region_candidate,
                    annotation,
                    max_candidates=max_candidates,
                )
                for annotation in annotations
            )
            candidate_sources = _unique_ids(
                [
                    source_observation_id,
                    region_candidate.segmentation_hypothesis_id,
                    region_candidate.region_hypothesis_id,
                    *(
                        source_id
                        for annotation in annotations
                        for source_id in (annotation.id, annotation.annotation_source_id)
                        if source_id is not None
                    ),
                ]
            )
            provenance = CandidateProvenance(
                route_status="reviewed",
                orientation_status=_aggregate_review_status(
                    annotation.direction_status for annotation in annotations
                ),
                stroke_order_status=_aggregate_review_status(
                    annotation.stroke_order_status for annotation in annotations
                ),
                evidence_level="reviewed-manual",
                source_ids=candidate_sources,
            )
        else:
            stroke_hypotheses = _inferred_stroke_hypotheses(
                source_observation_id,
                region_candidate,
                boundary_events=boundary_events or {},
                max_candidates=max_candidates,
                path_search_limits=search_limits,
            )
            provenance = CandidateProvenance(
                route_status="ambiguous" if len(stroke_hypotheses) > 1 else "inferred",
                orientation_status="ambiguous",
                stroke_order_status="ambiguous",
                evidence_level="inferred-ambiguous",
                source_ids=_unique_ids(
                    [
                        source_observation_id,
                        region_candidate.segmentation_hypothesis_id,
                        region_candidate.region_hypothesis_id,
                    ]
                ),
            )
        candidates.append(
            GlyphObservationCandidate(
                id=(
                    f"{source_observation_id}--{region_candidate.segmentation_hypothesis_id}"
                    f"--{region_candidate.region_hypothesis_id}"
                ),
                source_observation_id=source_observation_id,
                segmentation_hypothesis_id=region_candidate.segmentation_hypothesis_id,
                region_hypothesis_id=region_candidate.region_hypothesis_id,
                region=region_candidate.region,
                stroke_hypotheses=stroke_hypotheses,
                provenance=provenance,
            )
        )
    return tuple(candidates)


def _reviewed_stroke_hypothesis(
    source_observation_id: str,
    region_candidate: RasterRegionCandidate,
    annotation: MovementAnnotation,
    *,
    max_candidates: int,
) -> StrokePathHypothesis:
    hypothesis_id = f"reviewed-{annotation.id}"
    source_ids = _unique_ids(
        [
            source_observation_id,
            region_candidate.segmentation_hypothesis_id,
            region_candidate.region_hypothesis_id,
            annotation.id,
            annotation.annotation_source_id,
        ]
    )
    points = tuple((float(x), float(y)) for x, y in annotation.ordered_landmarks)
    boundaries = propose_gesture_boundaries(
        len(points),
        reviewed_annotation=annotation,
        max_candidates=max_candidates,
    )
    evidence = OccurrenceEvidence(
        source_observation_id=source_observation_id,
        segmentation_hypothesis_id=region_candidate.segmentation_hypothesis_id,
        skeleton_stroke_hypothesis_id=None,
        order_status=annotation.stroke_order_status,
        direction_status=annotation.direction_status,
        gesture_annotation_status="reviewed" if annotation.gesture_spans else "not-recorded",
        primitive_assignment_status="not-recorded",
        evidence_level="reviewed-manual",
    )
    return StrokePathHypothesis(
        id=hypothesis_id,
        path_kind=None,
        points=points,
        edge_ids=(),
        stroke_order=annotation.stroke_order,
        direction=annotation.direction,
        gesture_boundaries=boundaries,
        provenance=CandidateProvenance(
            route_status="reviewed",
            orientation_status=annotation.direction_status,
            stroke_order_status=annotation.stroke_order_status,
            evidence_level="reviewed-manual",
            source_ids=source_ids,
        ),
        evidence=evidence,
        annotation=annotation,
    )


def _inferred_stroke_hypotheses(
    source_observation_id: str,
    region_candidate: RasterRegionCandidate,
    *,
    boundary_events: Mapping[str, Sequence[BoundaryEvent]],
    max_candidates: int,
    path_search_limits: PathSearchLimits,
) -> tuple[StrokePathHypothesis, ...]:
    analysis = analyze_mask(region_candidate.region.mask)
    path_kinds: tuple[Literal["open", "loop", "open-with-loop"], ...] = (
        "open",
        "loop",
        "open-with-loop",
    )
    inventory = prepare_topology_paths(
        analysis.graph,
        path_kinds=path_kinds,
        limits=path_search_limits,
    )
    ranked = {
        path_kind: rank_paths(
            analysis.graph,
            path_kind=path_kind,
            path_inventory=inventory,
            limit=max_candidates,
        )
        for path_kind in path_kinds
    }
    selected: list[tuple[Literal["open", "loop", "open-with-loop"], SkeletonPath, int]] = []
    seen_paths: set[tuple[str, tuple[int, ...], int, int]] = set()
    for rank_index in range(max_candidates):
        for path_kind in path_kinds:
            paths = ranked[path_kind]
            if rank_index >= len(paths):
                continue
            path = paths[rank_index]
            identity = (path_kind, path.edge_ids, path.start_node, path.end_node)
            if identity in seen_paths:
                continue
            seen_paths.add(identity)
            selected.append((path_kind, path, rank_index))
            if len(selected) >= max_candidates:
                break
        if len(selected) >= max_candidates:
            break

    hypotheses: list[StrokePathHypothesis] = []
    for path_kind, path, rank_index in selected:
        hypothesis_id = (
            f"skeleton-{region_candidate.region_hypothesis_id}-{path_kind}-{rank_index:03d}"
        )
        points = tuple((float(x), float(y)) for x, y in path.xy)
        boundaries = propose_gesture_boundaries(
            len(points),
            boundary_events.get(hypothesis_id, ()),
            max_candidates=max_candidates,
        )
        source_ids = _unique_ids(
            [
                source_observation_id,
                region_candidate.segmentation_hypothesis_id,
                region_candidate.region_hypothesis_id,
                hypothesis_id,
                *(event.id for event in boundary_events.get(hypothesis_id, ())),
            ]
        )
        evidence = OccurrenceEvidence(
            source_observation_id=source_observation_id,
            segmentation_hypothesis_id=region_candidate.segmentation_hypothesis_id,
            skeleton_stroke_hypothesis_id=hypothesis_id,
            order_status="ambiguous",
            direction_status="ambiguous",
            gesture_annotation_status="ambiguous" if boundaries else "not-recorded",
            primitive_assignment_status="not-recorded",
            evidence_level="inferred-ambiguous",
        )
        hypotheses.append(
            StrokePathHypothesis(
                id=hypothesis_id,
                path_kind=path_kind,
                points=points,
                edge_ids=tuple(path.edge_ids),
                stroke_order=None,
                direction=None,
                gesture_boundaries=boundaries,
                provenance=CandidateProvenance(
                    route_status="ambiguous" if len(selected) > 1 else "inferred",
                    orientation_status="ambiguous",
                    stroke_order_status="ambiguous",
                    evidence_level="inferred-ambiguous",
                    source_ids=source_ids,
                ),
                evidence=evidence,
            )
        )
    return tuple(hypotheses)


def _aggregate_review_status(statuses: Iterable[EvidenceStatus]) -> EvidenceStatus:
    values = tuple(statuses)
    if values and all(status in {"observed", "reviewed"} for status in values):
        return "reviewed" if "reviewed" in values else "observed"
    if values and all(status == "inferred" for status in values):
        return "inferred"
    if any(status == "ambiguous" for status in values):
        return "ambiguous"
    return "not-recorded"


def _unique_ids(values: Sequence[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value is not None))


def _validate_candidate_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_candidates must be a positive integer")

class GeometryLearningConfig(StrictModel):
    allowed_change_counts: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    held_out_fraction: float = Field(default=0.2, ge=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_change_counts(self) -> GeometryLearningConfig:
        if not self.allowed_change_counts or any(value < 0 for value in self.allowed_change_counts):
            raise ValueError("allowed_change_counts must contain nonnegative values")
        if len(self.allowed_change_counts) != len(set(self.allowed_change_counts)):
            raise ValueError("allowed_change_counts must be unique")
        return self


class StrokeHypothesesConfig(StrictModel):
    max_candidates: int = Field(default=8, ge=1)
    max_repeated_edges: int = Field(default=2, ge=0)


class GestureDictionaryConfig(StrictModel):
    seeded_roles: list[GestureRole] = Field(
        default_factory=lambda: [
            "entry",
            "oval",
            "ascender-loop",
            "stem",
            "hump",
            "underturn",
            "exit",
            "connector",
        ]
    )

    @field_validator("seeded_roles")
    @classmethod
    def validate_roles(cls, values: list[GestureRole]) -> list[GestureRole]:
        if len(values) != len(set(values)):
            raise ValueError("seeded gesture roles must be unique")
        return values


class TrainingValidationConfig(StrictModel):
    letters: str = ascii_lowercase

    @field_validator("letters")
    @classmethod
    def validate_letters(cls, value: str) -> str:
        if not value or any(letter not in ascii_lowercase for letter in value):
            raise ValueError("validation letters must be lowercase ASCII letters")
        if len(value) != len(set(value)):
            raise ValueError("validation letters must be unique")
        return value


class GestureTrainingConfig(StrictModel):
    format: Literal["inkgrammar-gesture-training"] = "inkgrammar-gesture-training"
    format_version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    style: str = Field(min_length=1)
    geometry_learning: GeometryLearningConfig = Field(default_factory=GeometryLearningConfig)
    stroke_hypotheses: StrokeHypothesesConfig = Field(default_factory=StrokeHypothesesConfig)
    gesture_dictionary: GestureDictionaryConfig = Field(default_factory=GestureDictionaryConfig)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
