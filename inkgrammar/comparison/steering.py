from __future__ import annotations

from dataclasses import replace

import numpy as np
from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryModelSelectionOptions,
    TrajectoryModelSelectionResult,
    select_trajectory_model,
)
from planarcurve.trajectory import evaluate_heading_rate, evaluate_trajectory, trajectory_total_turn

from .models import ContinuousPath, WritingRouteCandidate

_TRAJECTORY_FIT_OPTIONS = TrajectoryFitOptions(
    change_count=0,
    samples=32,
    restarts=0,
    smoothing_window=7,
    max_nfev=20,
)
_TRAJECTORY_SELECTION_OPTIONS = TrajectoryModelSelectionOptions(
    change_counts=(0, 1, 2),
    validation_fraction=0.0,
    seed=0,
)


def fit_heading_rate_model(path: ContinuousPath) -> TrajectoryModelSelectionResult:
    return select_trajectory_model(
        path.xy,
        fit_options=_TRAJECTORY_FIT_OPTIONS,
        selection_options=_TRAJECTORY_SELECTION_OPTIONS,
    )


def fitted_steering_distance(
    reference: ContinuousPath,
    sample: ContinuousPath,
) -> float:
    return _profile_pair_distance(
        fit_heading_rate_model(reference),
        fit_heading_rate_model(sample),
    )


def fit_writing_route_candidate(
    candidate: WritingRouteCandidate,
) -> WritingRouteCandidate:
    if len(candidate.steering_models) == len(candidate.continuous_strokes):
        return candidate
    models = tuple(fit_heading_rate_model(stroke) for stroke in candidate.continuous_strokes)
    complexity = sum(model.selected_change_count for model in models)
    score = replace(
        candidate.score,
        steering_complexity_cost=candidate.hypothesis.score.turning_cost + complexity,
    )
    return replace(candidate, steering_models=models, score=score)


def fit_writing_route_candidates(
    candidates: tuple[WritingRouteCandidate, ...],
) -> tuple[WritingRouteCandidate, ...]:
    return tuple(fit_writing_route_candidate(candidate) for candidate in candidates)


def _model_profile(
    model: TrajectoryModelSelectionResult,
) -> tuple[np.ndarray, float]:
    trajectory = model.selected.trajectory
    progress = np.linspace(0.0, 1.0, 101)
    rates = evaluate_heading_rate(trajectory.heading_rate, progress)
    return rates, trajectory_total_turn(trajectory)


def _profile_pair_distance(
    reference_model: TrajectoryModelSelectionResult,
    sample_model: TrajectoryModelSelectionResult,
) -> float:
    reference_rates, reference_turn = _model_profile(reference_model)
    sample_rates, sample_turn = _model_profile(sample_model)
    reversed_rates = -sample_rates[::-1]
    distances = []
    for rates, turn in ((sample_rates, sample_turn), (reversed_rates, -sample_turn)):
        scale = max(
            360.0,
            float(np.sqrt(np.mean(reference_rates**2))),
            float(np.sqrt(np.mean(rates**2))),
        )
        rate_distance = float(np.sqrt(np.mean((reference_rates - rates) ** 2)) / scale)
        turn_distance = abs(reference_turn - turn) / 360.0
        distances.append(rate_distance + turn_distance)
    return min(distances)


def steering_profile_distance(
    reference: WritingRouteCandidate,
    sample: WritingRouteCandidate,
) -> float:
    paired_count = min(len(reference.steering_models), len(sample.steering_models))
    if not paired_count:
        return 1.0
    weighted_distance = 0.0
    total_weight = 0.0
    for index in range(paired_count):
        weight = 0.5 * (
            reference.continuous_strokes[index].arc_length_px
            + sample.continuous_strokes[index].arc_length_px
        )
        weighted_distance += (
            _profile_pair_distance(
                reference.steering_models[index],
                sample.steering_models[index],
            )
            * weight
        )
        total_weight += weight
    return float(weighted_distance / max(total_weight, 1e-12))


def steering_model_summary(
    model: TrajectoryModelSelectionResult,
) -> dict[str, object]:
    fit = model.selected
    profile = fit.trajectory.heading_rate
    knots = (
        (0.0, profile.start_rate_deg_per_u),
        *((change.at, change.heading_rate_deg_per_u) for change in profile.changes),
        (1.0, profile.end_rate_deg_per_u),
    )
    evaluated = evaluate_trajectory(fit.trajectory, count=101)
    dominant_index = int(np.argmax(np.abs(evaluated.heading_rate_deg_per_u)))
    return {
        "selected_change_count": model.selected_change_count,
        "knot_progress": [float(progress) for progress, _ in knots],
        "knot_heading_rate_deg_per_u": [float(rate) for _, rate in knots],
        "total_signed_turn_deg": float(trajectory_total_turn(fit.trajectory)),
        "canonical_endpoint_displacement": evaluated.xy[-1].tolist(),
        "dominant_turn_progress": float(evaluated.u[dominant_index]),
        "fit": {
            "success": bool(fit.success),
            "cost": float(fit.cost),
            "metrics": {name: float(value) for name, value in fit.metrics.items()},
        },
    }
