from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inkgrammar.comparison.models import ContinuousPath
from inkgrammar.comparison.patterns import (
    directed_stroke_pattern_distance,
    stroke_pattern_descriptor,
)
from inkgrammar.comparison.routes import generate_writing_route_candidates
from inkgrammar.comparison.skeleton import analyze_skeleton


def _path(points: np.ndarray, radii: np.ndarray | None = None) -> ContinuousPath:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return ContinuousPath(
        xy=points,
        tangent=tangents,
        curvature=np.zeros(len(points)),
        radius_px=np.ones(len(points)) if radii is None else radii,
        confidence=np.ones(len(points)),
        progress=cumulative / cumulative[-1],
        arc_length_px=float(cumulative[-1]),
        source="test",
        boundary_evidence="test",
    )


def _route(path: ContinuousPath):
    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 5:28] = True
    candidate = generate_writing_route_candidates(analyze_skeleton(mask), limit=1)[0]
    return replace(candidate, continuous_strokes=(path,))


def test_relative_width_pattern_ignores_absolute_width_and_detects_profile_change() -> None:
    points = np.column_stack((np.linspace(0.0, 10.0, 16), np.sin(np.linspace(0.0, 2.0, 16))))
    radii = np.linspace(1.0, 2.0, len(points))
    first = stroke_pattern_descriptor(_route(_path(points, radii)))
    doubled = stroke_pattern_descriptor(_route(_path(points, radii * 2.0)))
    changed = stroke_pattern_descriptor(_route(_path(points, radii[::-1])))

    assert directed_stroke_pattern_distance(first, doubled) == pytest.approx(0.0, abs=1e-10)
    assert directed_stroke_pattern_distance(first, changed) > 0.1


def test_directed_pattern_preserves_signed_turn_and_endpoint_headings() -> None:
    forward_points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (2.0, 2.0)))
    reverse_points = forward_points[::-1]
    forward = stroke_pattern_descriptor(_route(_path(forward_points)))
    reverse = stroke_pattern_descriptor(_route(_path(reverse_points)))

    assert directed_stroke_pattern_distance(forward, reverse) > 0.1
    assert forward.strokes[0].total_signed_turn_deg != pytest.approx(
        reverse.strokes[0].total_signed_turn_deg
    )
