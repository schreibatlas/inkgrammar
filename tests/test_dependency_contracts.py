from __future__ import annotations

from inspect import signature
from typing import get_args


def test_planarcurve_minimum_api_contract() -> None:
    from planarcurve.fitting import select_trajectory_model
    from planarcurve.polyline import polyline_self_intersections
    from planarcurve.trajectory import HeadingRateProfile
    from planarcurve.trajectory_chain import (
        DirectedTrajectory,
        TrajectoryChain,
        evaluate_trajectory_chain,
        trajectory_chain_to_cubic_path,
    )
    from planarcurve.transforms import fit_similarity_transform

    assert callable(fit_similarity_transform)
    assert callable(polyline_self_intersections)
    assert callable(select_trajectory_model)
    assert all(
        value is not None
        for value in (
            HeadingRateProfile,
            DirectedTrajectory,
            TrajectoryChain,
            evaluate_trajectory_chain,
            trajectory_chain_to_cubic_path,
        )
    )


def test_skeletonpaths_minimum_api_contract() -> None:
    from skeletonpaths import (
        JunctionRegion,
        JunctionRegionOptions,
        SkeletonAnalysis,
        SkeletonWalk,
        analyze_mask,
        covering_walk,
        graph_from_skeleton,
        stroke_hypotheses_from_analysis,
    )

    assert all(
        value is not None
        for value in (
            JunctionRegion,
            JunctionRegionOptions,
            SkeletonAnalysis,
            SkeletonWalk,
            analyze_mask,
            covering_walk,
            graph_from_skeleton,
            stroke_hypotheses_from_analysis,
        )
    )


def test_strokeoutline_minimum_api_contract() -> None:
    from strokeoutline import WidthParameterization, outline

    assert set(get_args(WidthParameterization)) == {"segment", "arclength"}
    assert "width_parameterization" in signature(outline).parameters
