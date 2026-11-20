from __future__ import annotations

from math import isfinite
from typing import Literal

from planarcurve import HeadingRateChange, HeadingRateProfile, Trajectory2D
from pydantic import Field, field_validator, model_validator

from ..models import StrictModel
from .evidence import PrimitiveProvenance


class HeadingRateChangePayload(StrictModel):
    at: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    heading_rate_deg_per_u: float = Field(allow_inf_nan=False)


class HeadingRateProfilePayload(StrictModel):
    start_rate_deg_per_u: float = Field(allow_inf_nan=False)
    end_rate_deg_per_u: float = Field(allow_inf_nan=False)
    changes: list[HeadingRateChangePayload] = Field(default_factory=list)
    interpolation: Literal["linear", "cubic-hermite"] = "cubic-hermite"

    @model_validator(mode="after")
    def validate_change_order(self) -> HeadingRateProfilePayload:
        positions = [change.at for change in self.changes]
        if any(
            current <= previous for previous, current in zip(positions, positions[1:], strict=False)
        ):
            raise ValueError("heading-rate changes must be strictly ordered")
        return self


class PlanarTrajectoryPayload(StrictModel):
    length: float = Field(gt=0.0, allow_inf_nan=False)
    start_heading_deg: float = Field(allow_inf_nan=False)
    heading_rate: HeadingRateProfilePayload


def trajectory_to_payload(trajectory: Trajectory2D) -> PlanarTrajectoryPayload:
    profile = trajectory.heading_rate
    return PlanarTrajectoryPayload(
        length=trajectory.length,
        start_heading_deg=trajectory.start_heading_deg,
        heading_rate=HeadingRateProfilePayload(
            start_rate_deg_per_u=profile.start_rate_deg_per_u,
            end_rate_deg_per_u=profile.end_rate_deg_per_u,
            changes=[
                HeadingRateChangePayload(
                    at=change.at,
                    heading_rate_deg_per_u=change.heading_rate_deg_per_u,
                )
                for change in profile.changes
            ],
            interpolation=profile.interpolation,
        ),
    )


def trajectory_from_payload(payload: PlanarTrajectoryPayload) -> Trajectory2D:
    profile = payload.heading_rate
    return Trajectory2D(
        length=payload.length,
        start_heading_deg=payload.start_heading_deg,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=profile.start_rate_deg_per_u,
            end_rate_deg_per_u=profile.end_rate_deg_per_u,
            changes=tuple(
                HeadingRateChange(change.at, change.heading_rate_deg_per_u)
                for change in profile.changes
            ),
            interpolation=profile.interpolation,
        ),
    )


class GeometryVariationPayload(StrictModel):
    parameter_dispersion: dict[str, float] = Field(default_factory=dict)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    occurrence_count: int = Field(ge=1)

    @field_validator("parameter_dispersion")
    @classmethod
    def validate_dispersion(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in values):
            raise ValueError("dispersion parameter names must not be blank")
        if any(not isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("parameter dispersion must be finite and nonnegative")
        return values


class GeometryPrimitiveFamily(StrictModel):
    id: str = Field(pattern=r"^curve-family-[0-9]+$")
    geometry_kind: Literal["heading-rate-trajectory"] = "heading-rate-trajectory"
    trajectory: PlanarTrajectoryPayload
    selected_change_count: int = Field(ge=0)
    variation: GeometryVariationPayload
    provenance: PrimitiveProvenance = Field(default_factory=PrimitiveProvenance)
    training_metrics: dict[str, float] = Field(default_factory=dict)
    held_out_metrics: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("training_metrics", "held_out_metrics")
    @classmethod
    def validate_metrics(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in values):
            raise ValueError("metric names must not be blank")
        if any(not isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("fit metrics must be finite and nonnegative")
        return values

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence IDs must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("evidence IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_complexity(self) -> GeometryPrimitiveFamily:
        if self.selected_change_count != len(self.trajectory.heading_rate.changes):
            raise ValueError("selected_change_count must match the trajectory heading-rate changes")
        return self
