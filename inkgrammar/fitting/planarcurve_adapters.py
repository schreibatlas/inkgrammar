from __future__ import annotations

from planarcurve.spline import BSplineCurve, spline_to_cubic_segments
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
)
from planarcurve.transforms import Affine2D, Similarity2D

from .models import (
    BSplineGeometry,
    CubicSegment,
    CurvatureChange,
    CurvatureProfile,
    CurvatureStrokeGeometry,
    TopologyTransform,
    TrajectoryTransform,
)


def to_planar_bspline(geometry: BSplineGeometry) -> BSplineCurve:
    return BSplineCurve(
        knots=tuple(geometry.knots),
        control_points=tuple(geometry.control_points),
        degree=geometry.degree,
    )


def topology_transform_to_affine(transform: TopologyTransform) -> Affine2D:
    return Affine2D.from_scale_shear_translation(
        scale_x=transform.scale_x,
        scale_y=transform.scale_y,
        shear_x=transform.shear_x,
        dx=transform.dx,
        dy=transform.dy,
    )


def bspline_geometry_to_cubic_segments(
    geometry: BSplineGeometry,
    *,
    tolerance: float = 1e-8,
) -> list[CubicSegment]:
    segments = spline_to_cubic_segments(to_planar_bspline(geometry), tolerance=tolerance)
    return [CubicSegment(c1=segment.c1, c2=segment.c2, end=segment.end) for segment in segments]


def to_planar_heading_rate(profile: CurvatureProfile) -> HeadingRateProfile:
    return HeadingRateProfile(
        start_rate_deg_per_u=profile.start_rate_deg,
        end_rate_deg_per_u=profile.end_rate_deg,
        changes=tuple(HeadingRateChange(change.at, change.rate_deg) for change in profile.changes),
        interpolation=profile.interpolation,
    )


def to_planar_trajectory(geometry: CurvatureStrokeGeometry) -> Trajectory2D:
    profile = geometry.curvature
    return Trajectory2D(
        length=geometry.length,
        start_heading_deg=geometry.start_angle_deg,
        heading_rate=to_planar_heading_rate(profile),
    )


def from_planar_trajectory(trajectory: Trajectory2D) -> CurvatureStrokeGeometry:
    profile = trajectory.heading_rate
    return CurvatureStrokeGeometry(
        length=trajectory.length,
        start_angle_deg=trajectory.start_heading_deg,
        curvature=CurvatureProfile(
            interpolation=profile.interpolation,
            start_rate_deg=profile.start_rate_deg_per_u,
            changes=[
                CurvatureChange(at=change.at, rate_deg=change.heading_rate_deg_per_u)
                for change in profile.changes
            ],
            end_rate_deg=profile.end_rate_deg_per_u,
        ),
    )


def to_planar_similarity(transform: TrajectoryTransform) -> Similarity2D:
    return Similarity2D(
        scale=transform.scale,
        rotation_deg=transform.rotation_deg,
        dx=transform.dx,
        dy=transform.dy,
    )


def from_planar_similarity(transform: Similarity2D) -> TrajectoryTransform:
    return TrajectoryTransform(
        scale=transform.scale,
        rotation_deg=transform.rotation_deg,
        dx=transform.dx,
        dy=transform.dy,
    )
