from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from planarcurve.polyline import polyline_length
from rasterregions import foreground_evidence

from .geometry import (
    DifferentialGeometry,
    continuous_differential_geometry,
    legacy_differential_geometry_from_polyline,
)
from .models import ComparisonResult
from .path import resample_continuous_path, reverse_continuous_path
from .reconstruction import reconstruct_route_ink
from .steering import steering_model_summary


@dataclass(frozen=True)
class ComparisonAnalysis:
    progress: np.ndarray
    reference_geometry: DifferentialGeometry
    sample_geometry: DifferentialGeometry
    reference_width_px: np.ndarray
    sample_width_px: np.ndarray
    series: dict[str, np.ndarray]
    metrics: dict[str, Any]


def _angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = second - first
    return np.arctan2(np.sin(difference), np.cos(difference))


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _sample_width(radius: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    height, width = radius.shape
    x = np.clip(points_xy[:, 0], 0.0, width - 1.0)
    y = np.clip(points_xy[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    sampled = (
        radius[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + radius[y0, x1] * tx * (1.0 - ty)
        + radius[y1, x0] * (1.0 - tx) * ty
        + radius[y1, x1] * tx * ty
    )
    return 2.0 * sampled


def _continuous_inputs(result: ComparisonResult):
    if (
        result.reference.continuous_path is None
        or result.sample.continuous_path is None
        or result.alignment.aligned_sample_path is None
    ):
        return None
    reference = resample_continuous_path(result.reference.continuous_path)
    sample_source = resample_continuous_path(
        result.sample.continuous_path,
        len(reference.xy),
    )
    if result.alignment.direction == "reversed-for-best-match":
        sample_source = reverse_continuous_path(sample_source)
    return reference, result.alignment.aligned_sample_path, sample_source


def _glyph_height(extracted) -> float:
    mask = extracted.mask_result.mask
    rows = np.flatnonzero(mask.any(axis=1))
    return float(rows[-1] - rows[0] + 1) if rows.size else float(mask.shape[0])


def _width_profile(candidate, glyph_height: float) -> tuple[np.ndarray, np.ndarray]:
    total_length = sum(stroke.arc_length_px for stroke in candidate.continuous_strokes)
    progress_parts = []
    width_parts = []
    length_before = 0.0
    for stroke in candidate.continuous_strokes:
        progress_parts.append(
            (length_before + stroke.progress * stroke.arc_length_px) / total_length
        )
        width_parts.append(2.0 * stroke.radius_px / glyph_height)
        length_before += stroke.arc_length_px
    return np.concatenate(progress_parts), np.concatenate(width_parts)


def _width_summary(candidate, glyph_height: float) -> dict[str, object]:
    progress, widths = _width_profile(candidate, glyph_height)
    phases = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0))
    phase_means = []
    for index, (start, end) in enumerate(phases):
        selected = (progress >= start) & (
            (progress <= end) if index == len(phases) - 1 else (progress < end)
        )
        phase_means.append(float(np.mean(widths[selected])) if np.any(selected) else None)
    median = float(np.median(widths))
    return {
        "median_width_over_glyph_height": median,
        "coefficient_of_variation": float(np.std(widths) / median) if median > 0.0 else None,
        "phase_mean_width_over_glyph_height": {
            "entry": phase_means[0],
            "middle": phase_means[1],
            "exit": phase_means[2],
        },
    }


def _progress_allocation(candidate) -> dict[str, float | None]:
    encounters = [
        event.progress for event in candidate.events if event.kind == "junction-encounter"
    ]
    if not encounters:
        return {
            "fraction_before_first_junction_encounter": None,
            "fraction_between_first_and_last_junction_encounter": None,
            "fraction_after_last_junction_encounter": None,
        }
    return {
        "fraction_before_first_junction_encounter": float(encounters[0]),
        "fraction_between_first_and_last_junction_encounter": float(encounters[-1] - encounters[0]),
        "fraction_after_last_junction_encounter": float(1.0 - encounters[-1]),
    }


def _reconstruction_metrics(
    result: ComparisonResult,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    selection = result.route_selection
    if selection is None:
        return (
            {
                "available": False,
                "unavailable_reason": "joint writing-route selection is unavailable",
            },
            {},
        )

    metrics: dict[str, object] = {
        "available": True,
        "coverage_comparison": "observed binary mask versus anti-aliased rendered coverage",
        "reference": {},
        "sample": {},
    }
    series: dict[str, np.ndarray] = {}
    for side, extracted, candidate in (
        ("reference", result.reference, selection.reference),
        ("sample", result.sample, selection.sample),
    ):
        evidence = extracted.foreground_evidence
        if evidence is None:
            evidence = foreground_evidence(extracted.image, mode=extracted.mask_result.method)
        reconstruction = reconstruct_route_ink(candidate, evidence)
        metrics[side] = {
            **reconstruction.metrics,
            "stroke_width_profiles": [
                {
                    "progress": [progress for progress, _ in profile],
                    "width_px": [width for _, width in profile],
                }
                for profile in reconstruction.width_profiles
            ],
        }
        series[f"{side}_observed_ink"] = reconstruction.observed_mask
        series[f"{side}_rendered_coverage"] = reconstruction.rendered_coverage
        series[f"{side}_under_coverage"] = reconstruction.under_coverage_mask
        series[f"{side}_over_coverage"] = reconstruction.over_coverage_mask
    return metrics, series


def _writing_metrics(result: ComparisonResult) -> dict[str, object]:
    selection = result.route_selection
    if selection is None:
        return {
            "available": False,
            "unavailable_reason": "joint writing-route selection is unavailable",
            "direction_source": "inferred_from_static_image_geometry",
        }
    reference = selection.reference
    sample = selection.sample
    reference_topology = result.reference.diagnostics.topology
    sample_topology = result.sample.diagnostics.topology
    junction_mismatch = reference.physical_junction_count != sample.physical_junction_count
    cycle_mismatch = reference.cycle_count != sample.cycle_count
    component_mismatch = reference_topology.component_count != sample_topology.component_count
    reference_models = [steering_model_summary(model) for model in reference.steering_models]
    sample_models = [steering_model_summary(model) for model in sample.steering_models]
    reference_turn = sum(model["total_signed_turn_deg"] for model in reference_models)
    sample_turn = sum(model["total_signed_turn_deg"] for model in sample_models)
    dominant_phase_differences = [
        abs(reference_model["dominant_turn_progress"] - sample_model["dominant_turn_progress"])
        for reference_model, sample_model in zip(reference_models, sample_models, strict=False)
    ]
    reference_width = _width_summary(reference, _glyph_height(result.reference))
    sample_width = _width_summary(sample, _glyph_height(result.sample))
    reference_median_width = reference_width["median_width_over_glyph_height"]
    sample_median_width = sample_width["median_width_over_glyph_height"]
    reference_variability = reference_width["coefficient_of_variation"]
    sample_variability = sample_width["coefficient_of_variation"]
    phase_width_ratios = {
        phase: (
            sample_width["phase_mean_width_over_glyph_height"][phase]
            / reference_width["phase_mean_width_over_glyph_height"][phase]
            if reference_width["phase_mean_width_over_glyph_height"][phase] not in {None, 0.0}
            and sample_width["phase_mean_width_over_glyph_height"][phase] is not None
            else None
        )
        for phase in ("entry", "middle", "exit")
    }
    if result.route_alignment is not None:
        sample_source_strokes = sample.continuous_strokes
        if result.route_alignment.direction == "reversed-for-best-match":
            sample_source_strokes = tuple(
                reverse_continuous_path(stroke) for stroke in reversed(sample_source_strokes)
            )
        stroke_metrics = []
        stroke_weights = []
        for reference_path, aligned_sample_path, sample_source_path in zip(
            reference.continuous_strokes,
            result.route_alignment.aligned_sample_strokes,
            sample_source_strokes,
            strict=False,
        ):
            reference_heading = np.degrees(
                np.arctan2(
                    reference_path.tangent[[0, -1], 1],
                    reference_path.tangent[[0, -1], 0],
                )
            )
            sample_heading = np.degrees(
                np.arctan2(
                    aligned_sample_path.tangent[[0, -1], 1],
                    aligned_sample_path.tangent[[0, -1], 0],
                )
            )
            heading_difference = (sample_heading - reference_heading + 180.0) % 360.0 - 180.0
            vertical_delta = (
                aligned_sample_path.xy[[0, -1], 1] - reference_path.xy[[0, -1], 1]
            ) / max(_glyph_height(result.reference), 1.0)
            stroke_metrics.append(
                {
                    "start_heading_difference_deg": float(heading_difference[0]),
                    "end_heading_difference_deg": float(heading_difference[1]),
                    "start_vertical_placement_delta_over_glyph_height": float(vertical_delta[0]),
                    "end_vertical_placement_delta_over_glyph_height": float(vertical_delta[1]),
                }
            )
            stroke_weights.append(
                0.5 * (reference_path.arc_length_px + sample_source_path.arc_length_px)
            )
        total_stroke_weight = max(sum(stroke_weights), 1e-12)
        entry_exit_metrics = {
            "direction_source": "inferred_from_static_image_geometry",
            "per_stroke": stroke_metrics,
            **{
                key: float(
                    sum(
                        item[key] * weight
                        for item, weight in zip(
                            stroke_metrics,
                            stroke_weights,
                            strict=True,
                        )
                    )
                    / total_stroke_weight
                )
                for key in (
                    "start_heading_difference_deg",
                    "end_heading_difference_deg",
                    "start_vertical_placement_delta_over_glyph_height",
                    "end_vertical_placement_delta_over_glyph_height",
                )
            },
        }
    else:
        reference_path = reference.continuous_strokes[0]
        sample_path = (
            result.alignment.aligned_sample_path
            if result.alignment is not None and result.alignment.aligned_sample_path is not None
            else sample.continuous_strokes[0]
        )
        reference_heading = np.degrees(
            np.arctan2(reference_path.tangent[[0, -1], 1], reference_path.tangent[[0, -1], 0])
        )
        sample_heading = np.degrees(
            np.arctan2(sample_path.tangent[[0, -1], 1], sample_path.tangent[[0, -1], 0])
        )
        heading_difference = (sample_heading - reference_heading + 180.0) % 360.0 - 180.0
        glyph_height = _glyph_height(result.reference)
        vertical_delta = (sample_path.xy[[0, -1], 1] - reference_path.xy[[0, -1], 1]) / max(
            glyph_height, 1.0
        )
        entry_exit_metrics = {
            "direction_source": "inferred_from_static_image_geometry",
            "start_heading_difference_deg": float(heading_difference[0]),
            "end_heading_difference_deg": float(heading_difference[1]),
            "start_vertical_placement_delta_over_glyph_height": float(vertical_delta[0]),
            "end_vertical_placement_delta_over_glyph_height": float(vertical_delta[1]),
        }
    return {
        "available": True,
        "direction_source": "inferred_from_static_image_geometry",
        "stroke_structure": {
            "reference_stroke_count": len(reference.continuous_strokes),
            "sample_stroke_count": len(sample.continuous_strokes),
            "reference_pen_lift_count": reference.hypothesis.pen_lift_count,
            "sample_pen_lift_count": sample.hypothesis.pen_lift_count,
            "reference_repeated_length_ratio": reference.score.repeat_cost,
            "sample_repeated_length_ratio": sample.score.repeat_cost,
            "reference_full_coverage": not reference.hypothesis.uncovered_edge_ids,
            "sample_full_coverage": not sample.hypothesis.uncovered_edge_ids,
            "route_ambiguous": selection.ambiguous,
        },
        "topology": {
            "reference_physical_junction_count": reference.physical_junction_count,
            "sample_physical_junction_count": sample.physical_junction_count,
            "reference_junction_encounter_count": sum(
                event.kind == "junction-encounter" for event in reference.events
            ),
            "sample_junction_encounter_count": sum(
                event.kind == "junction-encounter" for event in sample.events
            ),
            "reference_cycle_count": reference.cycle_count,
            "sample_cycle_count": sample.cycle_count,
            "reference_component_count": reference_topology.component_count,
            "sample_component_count": sample_topology.component_count,
            "mismatch_flags": {
                "physical_junction_count": junction_mismatch,
                "cycle_count": cycle_mismatch,
                "component_count": component_mismatch,
            },
        },
        "entry_exit": entry_exit_metrics,
        "steering": {
            "reference_strokes": reference_models,
            "sample_strokes": sample_models,
            "reference_total_signed_turn_deg": float(reference_turn),
            "sample_total_signed_turn_deg": float(sample_turn),
            "total_signed_turn_difference_deg": float(sample_turn - reference_turn),
            "profile_distance": selection.score.steering_profile_cost,
            "dominant_turn_phase_difference": (
                float(np.mean(dominant_phase_differences)) if dominant_phase_differences else None
            ),
        },
        "path_allocation": {
            "reference": _progress_allocation(reference),
            "sample": _progress_allocation(sample),
            "units": "normalized path progress, not elapsed time",
        },
        "width": {
            "normalization": "robust glyph height",
            "reference": reference_width,
            "sample": sample_width,
            "sample_to_reference_median_ratio": (
                float(sample_median_width / reference_median_width)
                if reference_median_width > 0.0
                else None
            ),
            "sample_to_reference_variability_ratio": (
                float(sample_variability / reference_variability)
                if (
                    sample_variability is not None
                    and reference_variability is not None
                    and reference_variability > 0.0
                )
                else None
            ),
            "sample_to_reference_phase_width_ratios": phase_width_ratios,
        },
        "route_pair_score": {
            "intrinsic_route_cost": selection.score.intrinsic_route_cost,
            "aligned_centerline_cost": selection.score.aligned_centerline_cost,
            "steering_profile_cost": selection.score.steering_profile_cost,
            "topology_event_compatibility_cost": (
                selection.score.topology_event_compatibility_cost
            ),
            "descriptor_matching_cost": selection.score.descriptor_matching_cost,
            "cost_breakdown": selection.score.cost_breakdown,
            "total_cost": selection.score.total_cost,
            "alignment_direction": (
                result.route_alignment.direction if result.route_alignment is not None else None
            ),
            "alignment_strategy": (
                result.route_alignment.strategy
                if result.route_alignment is not None
                else "single-path"
            ),
            "alignment_mean_squared_error_px2": (
                result.route_alignment.mean_squared_error
                if result.route_alignment is not None
                else None
            ),
            "per_stroke_alignment_mean_squared_error_px2": (
                list(result.route_alignment.stroke_mean_squared_errors)
                if result.route_alignment is not None
                else []
            ),
        },
    }


def _analyze_multistroke_comparison(
    result: ComparisonResult,
    route_event_guard_progress: float,
) -> ComparisonAnalysis:
    selection = result.route_selection
    route_alignment = result.route_alignment
    assert selection is not None and route_alignment is not None
    reference_strokes = selection.reference.continuous_strokes
    sample_source_strokes = selection.sample.continuous_strokes
    if route_alignment.direction == "reversed-for-best-match":
        sample_source_strokes = tuple(
            reverse_continuous_path(stroke) for stroke in reversed(sample_source_strokes)
        )
    paired_count = min(len(reference_strokes), len(sample_source_strokes))
    if paired_count == 0:
        raise ValueError("route alignment must contain at least one paired stroke")
    paired_reference = reference_strokes[:paired_count]
    paired_sample = route_alignment.aligned_sample_strokes[:paired_count]
    paired_sample_source = sample_source_strokes[:paired_count]
    pair_weights = tuple(
        0.5 * (reference.arc_length_px + sample.arc_length_px)
        for reference, sample in zip(paired_reference, paired_sample_source, strict=True)
    )
    total_pair_weight = sum(pair_weights)
    sample_budget = 256
    minimum_per_stroke = 8
    remaining_samples = sample_budget - minimum_per_stroke * paired_count
    proportional_samples = remaining_samples * np.asarray(pair_weights) / total_pair_weight
    sample_counts_array = np.floor(proportional_samples).astype(int) + minimum_per_stroke
    remainder = sample_budget - int(sample_counts_array.sum())
    if remainder:
        largest_remainders = np.argsort(-(proportional_samples % 1.0))[:remainder]
        sample_counts_array[largest_remainders] += 1
    sample_counts = tuple(int(count) for count in sample_counts_array)
    reference_paths = tuple(
        resample_continuous_path(path, count)
        for path, count in zip(paired_reference, sample_counts, strict=True)
    )
    sample_paths = tuple(
        resample_continuous_path(path, count)
        for path, count in zip(paired_sample, sample_counts, strict=True)
    )
    sample_source_paths = tuple(
        resample_continuous_path(path, count)
        for path, count in zip(paired_sample_source, sample_counts, strict=True)
    )
    reference_geometries = tuple(continuous_differential_geometry(path) for path in reference_paths)
    sample_geometries = tuple(continuous_differential_geometry(path) for path in sample_paths)

    reference_length = sum(path.arc_length_px for path in paired_reference)
    sample_length = sum(path.arc_length_px for path in selection.sample.continuous_strokes)
    aligned_sample_length = sum(path.arc_length_px for path in paired_sample)
    cursor = 0.0
    progress_parts = []
    reference_event_progress = []
    sample_event_progress = []
    for reference_path, sample_path in zip(reference_paths, sample_paths, strict=True):
        local_fraction = reference_path.arc_length_px / reference_length
        progress_parts.append(cursor + reference_path.progress * local_fraction)
        reference_event_progress.extend(
            cursor + np.asarray(reference_path.route_event_progress) * local_fraction
        )
        sample_event_progress.extend(
            cursor + np.asarray(sample_path.route_event_progress) * local_fraction
        )
        if cursor > 0.0:
            reference_event_progress.append(cursor)
            sample_event_progress.append(cursor)
        cursor += local_fraction
    progress = np.concatenate(progress_parts)
    reference_xy = np.concatenate([path.xy for path in reference_paths], axis=0)
    sample_xy = np.concatenate([path.xy for path in sample_paths], axis=0)
    sample_source_xy = np.concatenate([path.xy for path in sample_source_paths], axis=0)

    def combine_geometry(geometries: tuple[DifferentialGeometry, ...]) -> DifferentialGeometry:
        heading = np.concatenate([geometry.heading_rad for geometry in geometries])
        turn_rate = np.concatenate([geometry.turn_rate_rad_per_progress for geometry in geometries])
        cumulative_turn_parts = []
        turn_offset = 0.0
        for geometry in geometries:
            cumulative_turn_parts.append(geometry.cumulative_turn_rad + turn_offset)
            turn_offset += float(geometry.cumulative_turn_rad[-1])
        return DifferentialGeometry(
            np.concatenate([geometry.xy for geometry in geometries], axis=0),
            heading,
            turn_rate,
            np.concatenate(cumulative_turn_parts),
        )

    reference_geometry = combine_geometry(reference_geometries)
    sample_geometry = combine_geometry(sample_geometries)
    reference_width = np.concatenate([2.0 * path.radius_px for path in reference_paths])
    sample_width = np.concatenate([2.0 * path.radius_px for path in sample_source_paths])
    position_error = np.linalg.norm(sample_xy - reference_xy, axis=1) / reference_length
    tangent_error_deg = np.degrees(
        _angle_difference(reference_geometry.heading_rad, sample_geometry.heading_rad)
    )
    turn_rate_error = (
        sample_geometry.turn_rate_rad_per_progress - reference_geometry.turn_rate_rad_per_progress
    )
    route_events = np.asarray([*reference_event_progress, *sample_event_progress], dtype=float)
    turn_rate_summary_mask = np.ones(len(progress), dtype=bool)
    for event_progress in route_events:
        turn_rate_summary_mask &= np.abs(progress - event_progress) > route_event_guard_progress
    scored_turn_rate_error = turn_rate_error[turn_rate_summary_mask]
    scored_absolute_turn_rate_error = np.abs(scored_turn_rate_error)
    position_summary = _summary(position_error)
    normalized_origin = reference_xy[0]
    reference_xy_normalized = (reference_xy - normalized_origin) / reference_length
    sample_xy_normalized = (sample_xy - normalized_origin) / reference_length
    reference_start_end = [path.xy[[0, -1]] for path in reference_paths]
    sample_start_end = [path.xy[[0, -1]] for path in sample_paths]
    endpoint_position_pairs = [
        (
            float(np.linalg.norm(sample_points[0] - reference_points[0]) / reference_length),
            float(np.linalg.norm(sample_points[-1] - reference_points[-1]) / reference_length),
        )
        for reference_points, sample_points in zip(
            reference_start_end,
            sample_start_end,
            strict=True,
        )
    ]
    endpoint_heading_errors = [
        np.degrees(
            np.abs(_angle_difference(reference.heading_rad[[0, -1]], sample.heading_rad[[0, -1]]))
        )
        for reference, sample in zip(reference_geometries, sample_geometries, strict=True)
    ]
    reference_turn_total = sum(
        float(geometry.cumulative_turn_rad[-1]) for geometry in reference_geometries
    )
    sample_turn_total = sum(
        float(geometry.cumulative_turn_rad[-1]) for geometry in sample_geometries
    )
    reference_topology = result.reference.diagnostics.topology
    sample_topology = result.sample.diagnostics.topology
    reference_offsets = np.cumsum([0, *[len(path.xy) for path in reference_strokes]])
    sample_offsets = np.cumsum(
        [0, *[len(path.xy) for path in route_alignment.aligned_sample_strokes]]
    )
    resampled_offsets = np.cumsum([0, *sample_counts])
    reference_raw = np.concatenate(
        [walk.xy for walk in selection.reference.hypothesis.walks], axis=0
    )
    sample_raw = np.concatenate([walk.xy for walk in selection.sample.hypothesis.walks], axis=0)

    def route_event_points(paths: tuple[Any, ...]) -> np.ndarray:
        event_arrays = [
            path.xy[list(path.route_event_indices)] for path in paths if path.route_event_indices
        ]
        return np.concatenate(event_arrays, axis=0) if event_arrays else np.empty((0, 2))

    reference_route_event_xy = route_event_points(reference_strokes)
    sample_route_event_xy = route_event_points(route_alignment.aligned_sample_strokes)
    metrics: dict[str, Any] = {
        "path": {
            "reference_arc_length_px": reference_length,
            "sample_arc_length_px": sample_length,
            "raw_sample_to_reference_length_ratio": sample_length / reference_length,
            "reference_arc_length_normalization_px": reference_length,
            "reference_normalized_arc_length": 1.0,
            "aligned_sample_normalized_arc_length": aligned_sample_length / reference_length,
            "stroke_count_difference": abs(
                len(selection.reference.continuous_strokes)
                - len(selection.sample.continuous_strokes)
            ),
            "mean_error": position_summary["mean"],
            "median_error": position_summary["median"],
            "p90_error": position_summary["p90"],
            "p95_error": position_summary["p95"],
            "max_error": position_summary["maximum"],
            "x_rmse": float(
                np.sqrt(np.mean((sample_xy_normalized[:, 0] - reference_xy_normalized[:, 0]) ** 2))
            ),
            "y_rmse": float(
                np.sqrt(np.mean((sample_xy_normalized[:, 1] - reference_xy_normalized[:, 1]) ** 2))
            ),
        },
        "heading": {
            "mean_abs_error_deg": float(np.mean(np.abs(tangent_error_deg))),
            "median_abs_error_deg": float(np.median(np.abs(tangent_error_deg))),
            "p95_abs_error_deg": float(np.percentile(np.abs(tangent_error_deg), 95.0)),
            "maximum_abs_error_deg": float(np.max(np.abs(tangent_error_deg))),
        },
        "turn_rate": {
            "units": "radians per normalized stroke progress",
            "mae": float(np.mean(scored_absolute_turn_rate_error)),
            "rmse": float(np.sqrt(np.mean(scored_turn_rate_error**2))),
            "p95_abs": float(np.percentile(scored_absolute_turn_rate_error, 95.0)),
            "maximum_abs": float(np.max(scored_absolute_turn_rate_error)),
            "summary_sample_count": int(turn_rate_summary_mask.sum()),
            "route_event_guard_progress": route_event_guard_progress,
        },
        "cumulative_turn": {
            "reference_total_rad": reference_turn_total,
            "sample_total_rad": sample_turn_total,
            "difference_rad": sample_turn_total - reference_turn_total,
        },
        "endpoints": {
            "direction_source": "inferred",
            "per_stroke": [
                {
                    "start_error": start_error,
                    "end_error": end_error,
                    "start_tangent_error_deg": float(tangent_errors[0]),
                    "end_tangent_error_deg": float(tangent_errors[1]),
                    "reference_start_xy_px": reference_points[0].tolist(),
                    "reference_end_xy_px": reference_points[-1].tolist(),
                    "sample_start_xy_aligned_px": sample_points[0].tolist(),
                    "sample_end_xy_aligned_px": sample_points[-1].tolist(),
                }
                for (
                    (start_error, end_error),
                    tangent_errors,
                    reference_points,
                    sample_points,
                ) in zip(
                    endpoint_position_pairs,
                    endpoint_heading_errors,
                    reference_start_end,
                    sample_start_end,
                    strict=True,
                )
            ],
            "start_error": float(np.mean([pair[0] for pair in endpoint_position_pairs])),
            "end_error": float(np.mean([pair[1] for pair in endpoint_position_pairs])),
            "start_tangent_error_deg": float(
                np.mean([pair[0] for pair in endpoint_heading_errors])
            ),
            "end_tangent_error_deg": float(np.mean([pair[1] for pair in endpoint_heading_errors])),
        },
        "topology": {
            "reference": reference_topology.model_dump(mode="json"),
            "sample": sample_topology.model_dump(mode="json"),
            "reference_path_coverage": not selection.reference.hypothesis.uncovered_edge_ids,
            "sample_path_coverage": not selection.sample.hypothesis.uncovered_edge_ids,
        },
        "width": {
            "scored": False,
            "source": "subpixel-cross-section",
            "units": "pixels in each original crop",
            "reference_median_px": float(np.median(reference_width)),
            "sample_median_px": float(np.median(sample_width)),
        },
    }
    metrics["writing"] = _writing_metrics(result)
    metrics["reconstruction"], reconstruction_series = _reconstruction_metrics(result)
    series = {
        "progress": progress,
        "position_error": position_error,
        "reference_xy_normalized": reference_xy_normalized,
        "sample_xy_normalized": sample_xy_normalized,
        "reference_heading_deg": np.degrees(reference_geometry.heading_rad),
        "sample_heading_deg": np.degrees(sample_geometry.heading_rad),
        "tangent_error_deg": tangent_error_deg,
        "reference_turn_rate": reference_geometry.turn_rate_rad_per_progress,
        "sample_turn_rate": sample_geometry.turn_rate_rad_per_progress,
        "turn_rate_error": turn_rate_error,
        "reference_cumulative_turn": reference_geometry.cumulative_turn_rad,
        "sample_cumulative_turn": sample_geometry.cumulative_turn_rad,
        "reference_width_px": reference_width,
        "sample_width_px": sample_width,
        "reference_raw_xy": reference_raw,
        "sample_raw_xy": sample_raw,
        "reference_raw_skeleton_xy": np.column_stack(
            np.nonzero(result.reference.diagnostics.skeleton)[::-1]
        ).astype(float),
        "sample_raw_skeleton_xy": np.column_stack(
            np.nonzero(result.sample.diagnostics.skeleton)[::-1]
        ).astype(float),
        "reference_subpixel_xy": np.concatenate([path.xy for path in reference_strokes], axis=0),
        "sample_subpixel_xy": np.concatenate(
            [path.xy for path in route_alignment.aligned_sample_strokes], axis=0
        ),
        "reference_centerline_confidence": np.concatenate(
            [path.confidence for path in reference_strokes]
        ),
        "sample_centerline_confidence": np.concatenate(
            [path.confidence for path in route_alignment.aligned_sample_strokes]
        ),
        "reference_route_event_progress": np.asarray(reference_event_progress, dtype=float),
        "sample_route_event_progress": np.asarray(sample_event_progress, dtype=float),
        "reference_route_event_xy": reference_route_event_xy,
        "sample_route_event_xy": sample_route_event_xy,
        "reference_resampled_xy": reference_xy,
        "sample_resampled_xy": sample_xy,
        "sample_source_resampled_xy": sample_source_xy,
        "reference_stroke_offsets": reference_offsets,
        "sample_stroke_offsets": sample_offsets,
        "reference_resampled_stroke_offsets": resampled_offsets,
        "sample_resampled_stroke_offsets": resampled_offsets,
    }
    series.update(reconstruction_series)
    return ComparisonAnalysis(
        progress,
        reference_geometry,
        sample_geometry,
        reference_width,
        sample_width,
        series,
        metrics,
    )


def analyze_comparison(
    result: ComparisonResult,
    *,
    route_event_guard_progress: float = 0.005,
) -> ComparisonAnalysis:
    if result.route_alignment is not None:
        return _analyze_multistroke_comparison(result, route_event_guard_progress)
    continuous_inputs = _continuous_inputs(result)
    if continuous_inputs is None:
        reference_xy = result.reference.resampled_xy
        sample_xy = result.alignment.aligned_sample_xy
        sample_source_xy = result.sample.resampled_xy
        if result.alignment.direction == "reversed-for-best-match":
            sample_source_xy = sample_source_xy[::-1]
        reference_geometry = legacy_differential_geometry_from_polyline(
            reference_xy,
            window_length=result.smoothing_window,
            polynomial_order=result.smoothing_polynomial_order,
        )
        sample_geometry = legacy_differential_geometry_from_polyline(
            sample_xy,
            window_length=result.smoothing_window,
            polynomial_order=result.smoothing_polynomial_order,
        )
        reference_length = float(polyline_length(result.reference.ordered_path.xy))
        sample_length = float(polyline_length(result.sample.ordered_path.xy))
        reference_width = _sample_width(result.reference.diagnostics.radius, reference_xy)
        sample_width = _sample_width(result.sample.diagnostics.radius, sample_source_xy)
        progress = np.linspace(0.0, 1.0, len(reference_xy))
        aligned_sample_length = float(polyline_length(sample_xy))
    else:
        reference_path, aligned_sample_path, sample_source_path = continuous_inputs
        reference_xy = reference_path.xy
        sample_xy = aligned_sample_path.xy
        sample_source_xy = sample_source_path.xy
        reference_geometry = continuous_differential_geometry(reference_path)
        sample_geometry = continuous_differential_geometry(aligned_sample_path)
        reference_length = result.reference.continuous_path.arc_length_px
        sample_length = result.sample.continuous_path.arc_length_px
        reference_width = 2.0 * reference_path.radius_px
        sample_width = 2.0 * sample_source_path.radius_px
        progress = reference_path.progress
        aligned_sample_length = aligned_sample_path.arc_length_px

    if reference_length <= 0.0:
        raise ValueError("reference path must have positive arc length")
    position_delta = sample_xy - reference_xy
    position_error = np.linalg.norm(position_delta, axis=1) / reference_length
    tangent_error_rad = _angle_difference(
        reference_geometry.heading_rad,
        sample_geometry.heading_rad,
    )
    tangent_error_deg = np.degrees(tangent_error_rad)
    heading_error_deg = np.abs(tangent_error_deg)
    turn_rate_error = (
        sample_geometry.turn_rate_rad_per_progress - reference_geometry.turn_rate_rad_per_progress
    )
    absolute_turn_rate_error = np.abs(turn_rate_error)
    turn_rate_summary_mask = np.ones(len(turn_rate_error), dtype=bool)
    if continuous_inputs is not None:
        for path in (reference_path, aligned_sample_path):
            for event_progress in path.route_event_progress:
                turn_rate_summary_mask &= (
                    np.abs(progress - event_progress) > route_event_guard_progress
                )
    normalized_origin = reference_xy[0]
    reference_xy_normalized = (reference_xy - normalized_origin) / reference_length
    sample_xy_normalized = (sample_xy - normalized_origin) / reference_length

    reference_start, reference_end = reference_xy[0], reference_xy[-1]
    sample_start, sample_end = sample_xy[0], sample_xy[-1]
    endpoint_position_error = {
        "start": float(np.linalg.norm(sample_start - reference_start) / reference_length),
        "end": float(np.linalg.norm(sample_end - reference_end) / reference_length),
    }
    endpoint_tangent_error = {
        "start": float(abs(tangent_error_deg[0])),
        "end": float(abs(tangent_error_deg[-1])),
    }
    scored_turn_rate_error = turn_rate_error[turn_rate_summary_mask]
    scored_absolute_turn_rate_error = absolute_turn_rate_error[turn_rate_summary_mask]
    turn_summary = {
        "mae": float(np.mean(scored_absolute_turn_rate_error)),
        "rmse": float(np.sqrt(np.mean(scored_turn_rate_error**2))),
        "p95_abs": float(np.percentile(scored_absolute_turn_rate_error, 95.0)),
        "maximum_abs": float(np.max(scored_absolute_turn_rate_error)),
        "summary_sample_count": int(turn_rate_summary_mask.sum()),
        "route_event_guard_progress": route_event_guard_progress,
    }
    reference_topology = result.reference.diagnostics.topology
    sample_topology = result.sample.diagnostics.topology
    position_summary = _summary(position_error)
    metrics = {
        "path": {
            "reference_arc_length_px": reference_length,
            "sample_arc_length_px": sample_length,
            "raw_sample_to_reference_length_ratio": sample_length / reference_length,
            "reference_arc_length_normalization_px": reference_length,
            "reference_normalized_arc_length": 1.0,
            "aligned_sample_normalized_arc_length": aligned_sample_length / reference_length,
            "mean_error": position_summary["mean"],
            "median_error": position_summary["median"],
            "p90_error": position_summary["p90"],
            "p95_error": position_summary["p95"],
            "max_error": position_summary["maximum"],
            "x_rmse": float(
                np.sqrt(np.mean((sample_xy_normalized[:, 0] - reference_xy_normalized[:, 0]) ** 2))
            ),
            "y_rmse": float(
                np.sqrt(np.mean((sample_xy_normalized[:, 1] - reference_xy_normalized[:, 1]) ** 2))
            ),
        },
        "heading": {
            "mean_abs_error_deg": float(np.mean(heading_error_deg)),
            "median_abs_error_deg": float(np.median(heading_error_deg)),
            "p95_abs_error_deg": float(np.percentile(heading_error_deg, 95.0)),
            "maximum_abs_error_deg": float(np.max(heading_error_deg)),
        },
        "turn_rate": {
            "units": "radians per normalized stroke progress",
            **turn_summary,
        },
        "cumulative_turn": {
            "reference_total_rad": float(reference_geometry.cumulative_turn_rad[-1]),
            "sample_total_rad": float(sample_geometry.cumulative_turn_rad[-1]),
            "difference_rad": float(
                sample_geometry.cumulative_turn_rad[-1] - reference_geometry.cumulative_turn_rad[-1]
            ),
        },
        "endpoints": {
            "direction_source": "inferred",
            "start_error": endpoint_position_error["start"],
            "end_error": endpoint_position_error["end"],
            "start_tangent_error_deg": endpoint_tangent_error["start"],
            "end_tangent_error_deg": endpoint_tangent_error["end"],
            "reference_start_xy_px": reference_start.tolist(),
            "reference_end_xy_px": reference_end.tolist(),
            "sample_start_xy_aligned_px": sample_start.tolist(),
            "sample_end_xy_aligned_px": sample_end.tolist(),
        },
        "topology": {
            "reference": reference_topology.model_dump(mode="json"),
            "sample": sample_topology.model_dump(mode="json"),
            "reference_path_coverage": result.reference.ordered_path.coverage,
            "sample_path_coverage": result.sample.ordered_path.coverage,
        },
        "width": {
            "scored": False,
            "source": (
                "subpixel-cross-section"
                if continuous_inputs is not None
                else "edt-bilinear-fallback"
            ),
            "units": "pixels in each original crop",
            "reference_median_px": float(np.median(reference_width)),
            "sample_median_px": float(np.median(sample_width)),
        },
    }
    metrics["writing"] = _writing_metrics(result)
    metrics["reconstruction"], reconstruction_series = _reconstruction_metrics(result)
    reference_continuous = result.reference.continuous_path
    sample_continuous = result.sample.continuous_path
    empty_points = np.empty((0, 2), dtype=float)
    series = {
        "progress": progress,
        "position_error": position_error,
        "reference_xy_normalized": reference_xy_normalized,
        "sample_xy_normalized": sample_xy_normalized,
        "reference_heading_deg": np.degrees(reference_geometry.heading_rad),
        "sample_heading_deg": np.degrees(sample_geometry.heading_rad),
        "tangent_error_deg": tangent_error_deg,
        "reference_turn_rate": reference_geometry.turn_rate_rad_per_progress,
        "sample_turn_rate": sample_geometry.turn_rate_rad_per_progress,
        "turn_rate_error": turn_rate_error,
        "reference_cumulative_turn": reference_geometry.cumulative_turn_rad,
        "sample_cumulative_turn": sample_geometry.cumulative_turn_rad,
        "reference_width_px": reference_width,
        "sample_width_px": sample_width,
        "reference_raw_xy": result.reference.ordered_path.xy,
        "sample_raw_xy": result.sample.ordered_path.xy,
        "reference_raw_skeleton_xy": np.column_stack(
            np.nonzero(result.reference.diagnostics.skeleton)[::-1]
        ).astype(float),
        "sample_raw_skeleton_xy": np.column_stack(
            np.nonzero(result.sample.diagnostics.skeleton)[::-1]
        ).astype(float),
        "reference_subpixel_xy": (
            reference_continuous.xy if reference_continuous is not None else empty_points
        ),
        "sample_subpixel_xy": (
            sample_continuous.xy if sample_continuous is not None else empty_points
        ),
        "reference_centerline_confidence": (
            reference_continuous.confidence
            if reference_continuous is not None
            else np.empty(0, dtype=float)
        ),
        "sample_centerline_confidence": (
            sample_continuous.confidence
            if sample_continuous is not None
            else np.empty(0, dtype=float)
        ),
        "reference_route_event_progress": np.asarray(
            reference_continuous.route_event_progress if reference_continuous is not None else (),
            dtype=float,
        ),
        "sample_route_event_progress": np.asarray(
            sample_continuous.route_event_progress if sample_continuous is not None else (),
            dtype=float,
        ),
        "reference_resampled_xy": reference_xy,
        "sample_resampled_xy": sample_xy,
        "sample_source_resampled_xy": sample_source_xy,
    }
    series.update(reconstruction_series)
    return ComparisonAnalysis(
        progress,
        reference_geometry,
        sample_geometry,
        reference_width,
        sample_width,
        series,
        metrics,
    )
