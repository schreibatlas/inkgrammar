from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike
from planarcurve import Trajectory2D, evaluate_trajectory, fit_similarity_transform
from planarcurve.polyline import (
    polyline_length,
    resample_polyline_by_arclength,
    symmetric_polyline_distance,
)

if TYPE_CHECKING:
    from planarcurve.fitting import (
        TrajectoryFamilyFitResult,
        TrajectoryFitOptions,
        TrajectoryModelSelectionOptions,
        TrajectoryVariationPolicy,
    )

from .evidence import PrimitiveProvenance
from .geometry import (
    GeometryPrimitiveFamily,
    GeometryVariationPayload,
    trajectory_from_payload,
    trajectory_to_payload,
)


@dataclass(frozen=True)
class DirectedCurveOccurrence:
    id: str
    source_observation_id: str
    points: ArrayLike
    split: Literal["training", "held-out"] = "training"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.source_observation_id.strip():
            raise ValueError("geometry occurrence IDs must not be blank")
        if self.split not in {"training", "held-out"}:
            raise ValueError("geometry occurrence split must be training or held-out")
        points = np.array(self.points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("directed geometry points must have shape (N, 2), N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("directed geometry points must be finite")
        points.setflags(write=False)
        evidence_ids = tuple(self.evidence_ids)
        if any(not value.strip() for value in evidence_ids):
            raise ValueError("geometry evidence IDs must not be blank")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("geometry evidence IDs must be unique")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class GeometryOccurrenceFit:
    occurrence_id: str
    split: Literal["training", "held-out"]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class FittedGeometryFamily:
    family: GeometryPrimitiveFamily
    planarcurve_fit: TrajectoryFamilyFitResult
    occurrence_fits: tuple[GeometryOccurrenceFit, ...]


def fit_geometry_family(
    family_id: str,
    occurrences: Sequence[DirectedCurveOccurrence],
    *,
    fit_options: TrajectoryFitOptions,
    selection_options: TrajectoryModelSelectionOptions,
    variation_policy: TrajectoryVariationPolicy | None = None,
) -> FittedGeometryFamily:
    """Fit a semantic-free family through PlanarCurve's shared trajectory APIs."""
    from planarcurve.fitting import fit_trajectory_family, select_shared_trajectory_model
    occurrence_values = tuple(occurrences)
    if not occurrence_values:
        raise ValueError("geometry family fitting requires occurrences")
    if len({item.id for item in occurrence_values}) != len(occurrence_values):
        raise ValueError("geometry occurrence IDs must be unique")
    training = tuple(item for item in occurrence_values if item.split == "training")
    held_out = tuple(item for item in occurrence_values if item.split == "held-out")
    if not training:
        raise ValueError("geometry family fitting requires a training occurrence")

    selection = select_shared_trajectory_model(
        [item.points for item in training],
        fit_options=fit_options,
        selection_options=replace(selection_options, validation_fraction=0.0),
    )
    selected_options = replace(
        fit_options,
        change_count=selection.model.change_count,
        seed=selection_options.seed,
    )
    family_fit = (
        fit_trajectory_family(
            [item.points for item in training],
            fit_options=selected_options,
            variation_policy=variation_policy,
        )
        if variation_policy is not None
        else selection
    )
    trajectory = family_fit.model.trajectory
    occurrence_fits: list[GeometryOccurrenceFit] = []
    for occurrence, fit in zip(training, family_fit.occurrences, strict=True):
        occurrence_fits.append(
            GeometryOccurrenceFit(
                occurrence_id=occurrence.id,
                split="training",
                metrics={
                    "position_rmse": fit.position_rmse,
                    "heading_rmse_deg": fit.heading_rmse_deg,
                    "endpoint_error": fit.endpoint_error,
                    "turn_error_deg": fit.turn_error_deg,
                    "robust_cost": fit.robust_cost,
                },
            )
        )

    held_out_results = [
        _fixed_family_geometry_metrics(trajectory, item.points, fit_options.samples)
        for item in held_out
    ]
    for occurrence, metrics in zip(held_out, held_out_results, strict=True):
        occurrence_fits.append(
            GeometryOccurrenceFit(
                occurrence_id=occurrence.id,
                split="held-out",
                metrics=metrics,
            )
        )

    accepted_count = sum(item.accepted for item in family_fit.occurrences)
    training_metrics = {
        **dict(family_fit.model.fit_metrics),
        "robust_cost": family_fit.training_cost,
        "complexity_cost": selection.complexity_cost,
        "score": selection.score,
        "outlier_rate": 1.0 - accepted_count / len(training),
        "occurrence_count": float(len(training)),
    }
    held_out_metrics = (
        {
            "mean_normalized_directed_rmse": float(
                np.mean([item["normalized_directed_rmse"] for item in held_out_results])
            ),
            "mean_normalized_symmetric_distance": float(
                np.mean([item["normalized_symmetric_distance"] for item in held_out_results])
            ),
            "occurrence_count": float(len(held_out)),
        }
        if held_out_results
        else {}
    )
    source_ids = list(
        dict.fromkeys(item.source_observation_id for item in occurrence_values)
    )
    evidence = list(
        dict.fromkeys(
            evidence_id
            for occurrence in occurrence_values
            for evidence_id in occurrence.evidence_ids
        )
    )
    family = GeometryPrimitiveFamily(
        id=family_id,
        trajectory=trajectory_to_payload(trajectory),
        selected_change_count=family_fit.model.change_count,
        variation=GeometryVariationPayload(
            parameter_dispersion=dict(family_fit.model.parameter_scale),
            outlier_rate=training_metrics["outlier_rate"],
            occurrence_count=len(occurrence_values),
        ),
        provenance=PrimitiveProvenance(
            geometry_status="inferred",
            construction_status="inferred",
            trajectory_status="inferred",
            source_ids=source_ids,
            evidence_ids=evidence,
        ),
        training_metrics=training_metrics,
        held_out_metrics=held_out_metrics,
        evidence=evidence,
    )
    return FittedGeometryFamily(
        family=family,
        planarcurve_fit=family_fit,
        occurrence_fits=tuple(occurrence_fits),
    )


def normalized_family_geometry_error(
    family: GeometryPrimitiveFamily,
    observed_points: ArrayLike,
    *,
    samples: int = 64,
) -> float:
    """Align and compare a directed curve with a family using PlanarCurve primitives."""
    trajectory = trajectory_from_payload(family.trajectory)
    return _fixed_family_geometry_metrics(trajectory, observed_points, samples)[
        "normalized_directed_rmse"
    ]


def _fixed_family_geometry_metrics(
    trajectory: Trajectory2D,
    observed_points: ArrayLike,
    samples: int,
) -> dict[str, float]:
    observed = np.asarray(observed_points, dtype=np.float64)
    target = resample_polyline_by_arclength(observed, samples)
    source = evaluate_trajectory(trajectory, count=samples).xy
    transform = fit_similarity_transform(target, source)
    aligned = transform.apply(source)
    target_length = polyline_length(target)
    directed_errors = np.linalg.norm(aligned - target, axis=1)
    return {
        "normalized_directed_rmse": float(
            np.sqrt(np.mean(directed_errors**2)) / target_length
        ),
        "normalized_symmetric_distance": (
            symmetric_polyline_distance(aligned, target, samples=samples) / target_length
        ),
    }
