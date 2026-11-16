from __future__ import annotations

from inspect import signature
from typing import get_args


def test_planarcurve_minimum_api_contract() -> None:
    from planarcurve.polyline import polyline_self_intersections
    from planarcurve.trajectory_chain import (
        DirectedTrajectory,
        TrajectoryChain,
        evaluate_trajectory_chain,
        trajectory_chain_to_cubic_path,
    )
    from planarcurve.transforms import fit_similarity_transform

    assert callable(fit_similarity_transform)
    assert callable(polyline_self_intersections)
    assert all(
        value is not None
        for value in (
            DirectedTrajectory,
            TrajectoryChain,
            evaluate_trajectory_chain,
            trajectory_chain_to_cubic_path,
        )
    )


def test_skeletonpaths_minimum_api_contract() -> None:
    from skeletonpaths import SkeletonAnalysis, SkeletonWalk, analyze_mask, covering_walk

    assert all(
        value is not None for value in (SkeletonAnalysis, SkeletonWalk, analyze_mask, covering_walk)
    )


def test_strokeoutline_minimum_api_contract() -> None:
    from strokeoutline import WidthParameterization, outline

    assert set(get_args(WidthParameterization)) == {"segment", "arclength"}
    assert "width_parameterization" in signature(outline).parameters
