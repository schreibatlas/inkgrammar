from __future__ import annotations

import numpy as np
import pytest
from planarcurve import HeadingRateChange, HeadingRateProfile, Trajectory2D, evaluate_trajectory
from pydantic import ValidationError

from inkgrammar.gestures import (
    GeometryPrimitiveFamily,
    GeometryVariationPayload,
    PrimitiveProvenance,
    trajectory_from_payload,
    trajectory_to_payload,
)


def test_planar_trajectory_payload_round_trip_preserves_direction_and_geometry() -> None:
    trajectory = Trajectory2D(
        length=1.4,
        start_heading_deg=55.0,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=60.0,
            changes=(HeadingRateChange(0.3, 320.0), HeadingRateChange(0.6, -80.0)),
            end_rate_deg_per_u=20.0,
            interpolation="cubic-hermite",
        ),
    )

    payload = trajectory_to_payload(trajectory)
    restored = trajectory_from_payload(type(payload).model_validate_json(payload.model_dump_json()))

    assert restored == trajectory
    np.testing.assert_allclose(
        evaluate_trajectory(restored, count=65).xy,
        evaluate_trajectory(trajectory, count=65).xy,
        atol=0.0,
    )
    assert restored.heading_rate.changes[0].heading_rate_deg_per_u == 320.0


def test_geometry_family_retains_fit_evidence_complexity_and_variation() -> None:
    trajectory = Trajectory2D(
        length=1.0,
        start_heading_deg=15.0,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=0.0,
            changes=(HeadingRateChange(0.5, 120.0),),
            end_rate_deg_per_u=0.0,
        ),
    )
    family = GeometryPrimitiveFamily(
        id="curve-family-017",
        trajectory=trajectory_to_payload(trajectory),
        selected_change_count=1,
        variation=GeometryVariationPayload(
            parameter_dispersion={"length": 0.08},
            outlier_rate=0.1,
            occurrence_count=12,
        ),
        provenance=PrimitiveProvenance(
            geometry_status="inferred",
            construction_status="reviewed",
            trajectory_status="reviewed",
            source_ids=["scan-1"],
            evidence_ids=["annotation-4"],
        ),
        training_metrics={"mean_residual": 0.03},
        held_out_metrics={"mean_residual": 0.05},
        evidence=["scan-1", "annotation-4"],
    )
    restored = GeometryPrimitiveFamily.model_validate_json(family.model_dump_json())

    assert restored == family
    assert restored.provenance.geometry_status == "inferred"
    assert restored.provenance.trajectory_status == "reviewed"
    assert restored.training_metrics["mean_residual"] < restored.held_out_metrics["mean_residual"]

    with pytest.raises(ValidationError, match="pattern"):
        GeometryPrimitiveFamily(
            id="oval",
            trajectory=trajectory_to_payload(trajectory),
            selected_change_count=1,
            variation={"occurrence_count": 1},
        )
    with pytest.raises(ValidationError, match="must match"):
        GeometryPrimitiveFamily(
            id="curve-family-018",
            trajectory=trajectory_to_payload(trajectory),
            selected_change_count=0,
            variation={"occurrence_count": 1},
        )
