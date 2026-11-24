from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import WritingRouteCandidate


@dataclass(frozen=True)
class StrokeVariationPolicy:
    heading_sigma_deg: float = 12.0
    total_turn_sigma_deg: float = 25.0
    steering_rate_sigma_deg_per_u: float = 25.0
    steering_change_progress_sigma: float = 0.04
    junction_progress_sigma: float = 0.05
    relative_width_sigma: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.heading_sigma_deg,
            self.total_turn_sigma_deg,
            self.steering_rate_sigma_deg_per_u,
            self.steering_change_progress_sigma,
            self.junction_progress_sigma,
            self.relative_width_sigma,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("stroke variation scales must be finite and positive")


@dataclass(frozen=True)
class DirectedStrokePattern:
    arc_length_fraction: float
    start_heading_deg: float
    end_heading_deg: float
    total_signed_turn_deg: float
    steering_change_count: int
    steering_change_progress: tuple[float, ...]
    steering_rates_deg_per_u: tuple[float, ...]
    junction_event_progress: tuple[float, ...]
    relative_width_profile: tuple[float, ...]
    junction_event_types: tuple[str, ...]


@dataclass(frozen=True)
class StrokePatternDescriptor:
    stroke_count: int
    pen_lift_count: int
    cycle_count: int
    physical_junction_count: int
    strokes: tuple[DirectedStrokePattern, ...]
    component_role_sequence: tuple[str, ...] = ()


def _heading_deg(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def stroke_pattern_descriptor(
    route: WritingRouteCandidate,
    *,
    component_role_sequence: tuple[str, ...] = (),
    profile_samples: int = 32,
) -> StrokePatternDescriptor:
    if profile_samples < 2:
        raise ValueError("profile_samples must be at least two")
    strokes = route.continuous_strokes
    total_length = sum(stroke.arc_length_px for stroke in strokes)
    patterns = []

    for index, stroke in enumerate(strokes):
        headings = np.unwrap(np.arctan2(stroke.tangent[:, 1], stroke.tangent[:, 0]))
        widths = np.maximum(2.0 * stroke.radius_px, 1e-9)
        relative_width = widths / float(np.median(widths))
        sampled_width = np.interp(
            np.linspace(0.0, 1.0, profile_samples),
            stroke.progress,
            relative_width,
        )
        model = route.steering_models[index] if index < len(route.steering_models) else None
        if model is None:
            rates: tuple[float, ...] = ()
            changes: tuple[float, ...] = ()
        else:
            trajectory = model.selected.trajectory
            rates = tuple(
                float(value)
                for value in (
                    trajectory.heading_rate.start_rate_deg_per_u,
                    *(change.heading_rate_deg_per_u for change in trajectory.heading_rate.changes),
                    trajectory.heading_rate.end_rate_deg_per_u,
                )
            )
            changes = tuple(float(change.at) for change in trajectory.heading_rate.changes)

        events = tuple(
            event
            for event in route.events
            if event.kind == "junction-encounter"
            and sum(candidate.arc_length_px for candidate in strokes[:index])
            <= event.progress * total_length
            <= sum(candidate.arc_length_px for candidate in strokes[: index + 1])
        )
        event_progress = tuple(float(event.progress) for event in events)
        event_types = tuple(f"junction:{event.occurrence or 0}" for event in events)
        sampled_width = np.convolve(
            np.pad(sampled_width, (1, 1), mode="edge"), np.full(3, 1.0 / 3.0), mode="valid"
        )
        patterns.append(
            DirectedStrokePattern(
                arc_length_fraction=stroke.arc_length_px / max(total_length, 1e-12),
                start_heading_deg=_heading_deg(stroke.tangent[0]),
                end_heading_deg=_heading_deg(stroke.tangent[-1]),
                total_signed_turn_deg=float(np.degrees(headings[-1] - headings[0])),
                steering_change_count=len(changes),
                steering_change_progress=changes,
                steering_rates_deg_per_u=rates,
                junction_event_progress=event_progress,
                junction_event_types=event_types,
                relative_width_profile=tuple(float(value) for value in sampled_width),
            )
        )

    return StrokePatternDescriptor(
        stroke_count=len(strokes),
        pen_lift_count=route.hypothesis.pen_lift_count,
        cycle_count=route.cycle_count,
        physical_junction_count=route.physical_junction_count,
        strokes=tuple(patterns),
        component_role_sequence=component_role_sequence,
    )


def _wrapped_angle_delta(first: float, second: float) -> float:
    return float((first - second + 180.0) % 360.0 - 180.0)


def _categorical_sequence_cost(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    count = max(len(first), len(second))
    if count == 0:
        return 0.0
    paired = min(len(first), len(second))
    mismatches = sum(first[index] != second[index] for index in range(paired))
    return (mismatches + count - paired) / count


def _sequence_cost(first: tuple[float, ...], second: tuple[float, ...], sigma: float) -> float:
    count = max(len(first), len(second))
    if count == 0:
        return 0.0
    paired = min(len(first), len(second))
    residuals = [abs(first[index] - second[index]) / sigma for index in range(paired)]
    residuals.extend([1.0] * (count - paired))
    return float(np.mean(np.minimum(residuals, 3.0)))


def directed_stroke_pattern_distance(
    reference: StrokePatternDescriptor,
    sample: StrokePatternDescriptor,
    policy: StrokeVariationPolicy | None = None,
) -> float:
    policy = policy or StrokeVariationPolicy()
    count = max(reference.stroke_count, sample.stroke_count)
    if count == 0:
        return 0.0
    costs = [
        abs(reference.stroke_count - sample.stroke_count) / count,
        abs(reference.pen_lift_count - sample.pen_lift_count)
        / max(reference.pen_lift_count, sample.pen_lift_count, 1),
        abs(reference.cycle_count - sample.cycle_count)
        / max(reference.cycle_count, sample.cycle_count, 1),
        abs(reference.physical_junction_count - sample.physical_junction_count)
        / max(reference.physical_junction_count, sample.physical_junction_count, 1),
    ]
    costs.append(
        _categorical_sequence_cost(
            reference.component_role_sequence,
            sample.component_role_sequence,
        )
    )
    for first, second in zip(reference.strokes, sample.strokes, strict=False):
        costs.extend(
            (
                abs(first.arc_length_fraction - second.arc_length_fraction),
                abs(_wrapped_angle_delta(first.start_heading_deg, second.start_heading_deg))
                / policy.heading_sigma_deg,
                abs(_wrapped_angle_delta(first.end_heading_deg, second.end_heading_deg))
                / policy.heading_sigma_deg,
                abs(first.total_signed_turn_deg - second.total_signed_turn_deg)
                / policy.total_turn_sigma_deg,
                abs(first.steering_change_count - second.steering_change_count)
                / max(first.steering_change_count, second.steering_change_count, 1),
                _sequence_cost(
                    first.steering_change_progress,
                    second.steering_change_progress,
                    policy.steering_change_progress_sigma,
                ),
                _sequence_cost(
                    first.steering_rates_deg_per_u,
                    second.steering_rates_deg_per_u,
                    policy.steering_rate_sigma_deg_per_u,
                ),
                _categorical_sequence_cost(
                    first.junction_event_types,
                    second.junction_event_types,
                ),
                _sequence_cost(
                    first.junction_event_progress,
                    second.junction_event_progress,
                    policy.junction_progress_sigma,
                ),
                _sequence_cost(
                    first.relative_width_profile,
                    second.relative_width_profile,
                    policy.relative_width_sigma,
                ),
            )
        )
    return float(np.mean(np.minimum(costs, 3.0)))
