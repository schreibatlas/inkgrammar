from __future__ import annotations

import numpy as np
import pytest

from inkgrammar.comparison.models import ContinuousPath, IdentityAlignmentPolicy
from inkgrammar.comparison.path import align_bounded_continuous_paths


def _path(points: np.ndarray) -> ContinuousPath:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return ContinuousPath(
        xy=points,
        tangent=tangent,
        curvature=np.zeros(len(points)),
        radius_px=np.ones(len(points)),
        confidence=np.ones(len(points)),
        progress=cumulative / cumulative[-1],
        arc_length_px=float(cumulative[-1]),
        source="test",
        boundary_evidence="test",
    )


def test_small_page_rotation_is_eligible_without_reversal() -> None:
    reference = _path(np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 2.0))))
    angle = np.deg2rad(8.0)
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    sample = _path(reference.xy @ rotation.T)

    result = align_bounded_continuous_paths(reference, sample)

    assert result.orientation_eligible
    assert result.direction == "forward"
    assert abs(result.transform.rotation_deg) == pytest.approx(8.0, abs=0.5)
    assert result.mean_squared_error < 1e-8


def test_excessive_and_upside_down_rotations_are_ineligible() -> None:
    reference = _path(np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 2.0))))

    def rotated(angle_deg: float) -> ContinuousPath:
        angle = np.deg2rad(angle_deg)
        rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        return _path(reference.xy @ rotation.T)

    policy = IdentityAlignmentPolicy(max_rotation_deg=15.0)
    for angle in (30.0, 174.0):
        result = align_bounded_continuous_paths(reference, rotated(angle), policy)
        assert not result.orientation_eligible
        assert abs(result.required_rotation_deg) > policy.max_rotation_deg
        assert result.rejection_reason == "required_rotation_exceeds_identity_limit"
        assert abs(result.transform.rotation_deg) <= policy.max_rotation_deg
