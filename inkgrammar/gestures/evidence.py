from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ..models import StrictModel

EvidenceLevel = Literal[
    "observed-online",
    "reviewed-manual",
    "source-backed",
    "inferred-high-support",
    "inferred-ambiguous",
    "synthetic",
]
EvidenceStatus = Literal["observed", "reviewed", "inferred", "ambiguous", "not-recorded"]


def _validate_ids(values: list[str]) -> list[str]:
    if any(not value.strip() for value in values):
        raise ValueError("evidence IDs must not be blank")
    if len(values) != len(set(values)):
        raise ValueError("evidence IDs must be unique")
    return values


class PrimitiveProvenance(StrictModel):
    geometry_status: Literal["observed", "inferred"] = "inferred"
    construction_status: EvidenceStatus = "inferred"
    trajectory_status: EvidenceStatus = "inferred"
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("source_ids", "evidence_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        return _validate_ids(values)


class GestureProvenance(StrictModel):
    level: EvidenceLevel = "inferred-ambiguous"
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("source_ids", "evidence_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        return _validate_ids(values)


class CandidateProvenance(StrictModel):
    route_status: EvidenceStatus = "not-recorded"
    orientation_status: EvidenceStatus = "not-recorded"
    stroke_order_status: EvidenceStatus = "not-recorded"
    evidence_level: EvidenceLevel = "inferred-ambiguous"
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: list[str]) -> list[str]:
        return _validate_ids(values)


class OccurrenceEvidence(StrictModel):
    source_observation_id: str = Field(min_length=1)
    segmentation_hypothesis_id: str | None = Field(default=None, min_length=1)
    skeleton_stroke_hypothesis_id: str | None = Field(default=None, min_length=1)
    order_status: EvidenceStatus = "not-recorded"
    direction_status: EvidenceStatus = "not-recorded"
    gesture_annotation_status: EvidenceStatus = "not-recorded"
    primitive_assignment_status: EvidenceStatus = "not-recorded"
    evidence_level: EvidenceLevel = "inferred-ambiguous"
    fit_residuals: dict[str, float] = Field(default_factory=dict)

    @field_validator("fit_residuals")
    @classmethod
    def validate_residuals(cls, values: dict[str, float]) -> dict[str, float]:
        from math import isfinite

        if any(not name.strip() for name in values):
            raise ValueError("fit residual names must not be blank")
        if any(not isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("fit residuals must be finite and nonnegative")
        return values
