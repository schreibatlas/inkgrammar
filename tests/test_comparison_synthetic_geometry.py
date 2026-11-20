from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from inkgrammar.comparison import ComparisonResult, extract_letter
from inkgrammar.comparison.geometry import continuous_differential_geometry, differential_geometry
from inkgrammar.comparison.metrics import analyze_comparison
from inkgrammar.comparison.path import (
    align_continuous_paths,
    resample_continuous_path,
    resample_path,
)

_IMAGE_SIZE = 128
_SUPERSAMPLE = 2
_STROKE_WIDTH = 5


def _render_and_extract(
    tmp_path: Path,
    name: str,
    points: np.ndarray,
    *,
    width: int = _STROKE_WIDTH,
):
    image = Image.new(
        "L",
        (_IMAGE_SIZE * _SUPERSAMPLE, _IMAGE_SIZE * _SUPERSAMPLE),
        255,
    )
    scaled_points = [(round(x * _SUPERSAMPLE), round(y * _SUPERSAMPLE)) for x, y in points]
    ImageDraw.Draw(image).line(
        scaled_points,
        fill=0,
        width=width * _SUPERSAMPLE,
        joint="curve",
    )
    image = image.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.LANCZOS)
    image_path = tmp_path / f"{name}.png"
    image.save(image_path)
    return extract_letter(
        image_path,
        (0, 0, _IMAGE_SIZE, _IMAGE_SIZE),
        allow_path_failure=True,
    )


def _analytic_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    line_t = np.linspace(0.0, 1.0, 1600)
    line = np.column_stack((15.0 + 98.0 * line_t, 30.25 + 4.9 * line_t))
    line_tangent = np.tile((98.0, 4.9), (len(line_t), 1))

    s_t = np.linspace(0.0, 1.0, 1600)
    s_curve = np.column_stack((14.0 + 100.0 * s_t, 64.0 + 4.0 * np.sin(2.0 * np.pi * s_t)))
    s_tangent = np.column_stack((np.full(len(s_t), 100.0), 8.0 * np.pi * np.cos(2.0 * np.pi * s_t)))

    arc_t = np.linspace(0.0, 1.0, 1400)
    sweep = 100.0
    angle = np.deg2rad(-sweep / 2.0 + sweep * arc_t)
    arc = np.column_stack((64.0 + 24.0 * np.cos(angle), 64.0 + 24.0 * np.sin(angle)))
    arc_tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
    return {
        "rotated-line": (line, line_tangent),
        "s-curve": (s_curve, s_tangent),
        "circular-arc": (arc, arc_tangent),
    }


def _heading_energy(heading: np.ndarray) -> float:
    return float(np.mean(np.diff(heading, n=2) ** 2))


@pytest.mark.parametrize("name", ("rotated-line", "s-curve", "circular-arc"))
def test_synthetic_centerline_heading_and_smoothness_gates(tmp_path, name: str) -> None:
    points, truth_tangents = _analytic_cases()[name]
    extracted = _render_and_extract(tmp_path, name, points)
    path = extracted.continuous_path
    assert path is not None
    assert extracted.diagnostics.topology.endpoint_count == 2
    assert extracted.diagnostics.topology.junction_count == 0
    assert extracted.diagnostics.topology.cycle_count == 0
    assert path.edge_ids == extracted.ordered_path.walk.edge_ids
    assert path.edge_directions == extracted.ordered_path.walk.edge_directions

    distances, nearest = cKDTree(points).query(path.xy)
    truth = truth_tangents[nearest]
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    valid = (path.progress >= 0.03) & (path.progress <= 0.97)
    direction = 1.0 if np.mean(np.sum(path.tangent[valid] * truth[valid], axis=1)) >= 0 else -1.0
    truth *= direction
    angular_error = np.degrees(
        np.arctan2(
            path.tangent[:, 1] * truth[:, 0] - path.tangent[:, 0] * truth[:, 1],
            np.sum(path.tangent * truth, axis=1),
        )
    )

    assert np.median(distances[valid]) <= 0.25
    assert np.percentile(distances[valid], 95.0) <= 0.60
    assert np.sqrt(np.mean(angular_error[valid] ** 2)) <= 3.0
    assert np.percentile(np.abs(angular_error[valid]), 95.0) <= 7.0

    continuous = continuous_differential_geometry(resample_continuous_path(path, 256))
    legacy = differential_geometry(resample_path(extracted.ordered_path.xy, 256))
    energy_reduction = 1.0 - _heading_energy(continuous.heading_rad) / _heading_energy(
        legacy.heading_rad
    )
    assert energy_reduction >= 0.70

    if name == "circular-arc":
        expected_turn_rate = direction * np.deg2rad(100.0)
        progress = np.linspace(0.0, 1.0, 256)
        metric_valid = (progress >= 0.03) & (progress <= 0.97)
        continuous_rmse = np.sqrt(
            np.mean((continuous.turn_rate_rad_per_progress[metric_valid] - expected_turn_rate) ** 2)
        )
        legacy_rmse = np.sqrt(
            np.mean((legacy.turn_rate_rad_per_progress[metric_valid] - expected_turn_rate) ** 2)
        )
        assert continuous_rmse < 1.0
        assert 1.0 - continuous_rmse / legacy_rmse >= 0.60


def _loop_route(turns: int) -> np.ndarray:
    center_x, center_y, radius = 52.0, 56.0, 14.0
    entry = np.column_stack((np.linspace(10.0, center_x, 250), np.full(250, center_y - radius)))
    angle = np.linspace(-np.pi / 2.0, 3.0 * np.pi / 2.0, 1200)
    loop = np.column_stack((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))
    if turns == 2:
        loop = np.vstack((loop, loop[1:]))
    return np.vstack((entry, loop, entry[::-1]))


def test_loop_entry_and_retrace_preserve_topology_route_and_events(tmp_path) -> None:
    entry = _render_and_extract(tmp_path, "loop-entry", _loop_route(1), width=6)
    retrace = _render_and_extract(tmp_path, "loop-retrace", _loop_route(2), width=6)

    np.testing.assert_array_equal(entry.mask_result.mask, retrace.mask_result.mask)
    for extracted in (entry, retrace):
        topology = extracted.diagnostics.topology
        path = extracted.continuous_path
        assert path is not None
        assert (topology.endpoint_count, topology.junction_count, topology.cycle_count) == (1, 1, 1)
        assert extracted.ordered_path.coverage == 1.0
        assert path.edge_ids == extracted.ordered_path.walk.edge_ids
        assert path.edge_directions == extracted.ordered_path.walk.edge_directions
        assert len(path.route_event_progress) == 2
        assert len(path.route_event_indices) == 2

        distances, _ = cKDTree(_loop_route(1)).query(path.xy)
        valid = (path.progress >= 0.03) & (path.progress <= 0.97)
        for event in path.route_event_progress:
            valid &= np.abs(path.progress - event) > 0.025
        assert np.median(distances[valid]) <= 0.25
        assert np.percentile(distances[valid], 95.0) <= 0.60

        sampled = resample_continuous_path(path, 256)
        assert len(sampled.route_event_indices) == 2
        assert sampled.route_event_progress == pytest.approx(path.route_event_progress)

    assert entry.continuous_path.edge_ids == retrace.continuous_path.edge_ids


def test_turn_rate_summary_excludes_configured_route_event_guard(tmp_path) -> None:
    reference = _render_and_extract(tmp_path, "guard-reference", _loop_route(1), width=6)
    sample = _render_and_extract(tmp_path, "guard-sample", _loop_route(1), width=6)
    alignment = align_continuous_paths(
        reference.continuous_path,
        sample.continuous_path,
    )
    aligned_sample_path = alignment.aligned_sample_path
    assert aligned_sample_path is not None

    curvature = np.array(aligned_sample_path.curvature, copy=True)
    event_index = aligned_sample_path.route_event_indices[0]
    curvature[event_index] += 10.0
    alignment = replace(
        alignment,
        aligned_sample_path=replace(aligned_sample_path, curvature=curvature),
    )
    analysis = analyze_comparison(
        ComparisonResult("e", reference, sample, alignment),
        route_event_guard_progress=0.01,
    )

    assert abs(analysis.series["turn_rate_error"][event_index]) > 1.0
    assert analysis.metrics["turn_rate"]["rmse"] < 1e-8
    assert analysis.metrics["turn_rate"]["summary_sample_count"] < 256
    assert analysis.metrics["turn_rate"]["route_event_guard_progress"] == 0.01
