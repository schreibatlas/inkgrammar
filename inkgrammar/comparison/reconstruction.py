from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from planarcurve import Affine2D
from rasterregions import ForegroundEvidence
from strokeoutline import (
    OutlineConfig,
    RasterFrame,
    StrokeInput,
    path_from_polyline,
    render_strokes,
)

from .models import ContinuousPath, WritingRouteCandidate


@dataclass(frozen=True)
class InkReconstruction:
    observed_mask: np.ndarray
    rendered_coverage: np.ndarray
    under_coverage_mask: np.ndarray
    over_coverage_mask: np.ndarray
    metrics: dict[str, float | int | str]
    width_profiles: tuple[tuple[tuple[float, float], ...], ...]

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_mask, dtype=bool)
        coverage = np.asarray(self.rendered_coverage, dtype=np.float32)
        under = np.asarray(self.under_coverage_mask, dtype=bool)
        over = np.asarray(self.over_coverage_mask, dtype=bool)
        if observed.ndim != 2 or any(
            array.shape != observed.shape for array in (coverage, under, over)
        ):
            raise ValueError("reconstruction fields must be matching two-dimensional arrays")
        if not np.isfinite(coverage).all() or np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError("rendered coverage must be finite and within [0, 1]")
        for name, array in (
            ("observed_mask", observed),
            ("rendered_coverage", coverage),
            ("under_coverage_mask", under),
            ("over_coverage_mask", over),
        ):
            frozen = np.array(array, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "width_profiles", tuple(self.width_profiles))


def _distinct_points(stroke: ContinuousPath) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
    keep = np.concatenate(([True], distances > 1e-9))
    points = stroke.xy[keep]
    radius = stroke.radius_px[keep]
    if len(points) < 2:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    return points, radius


def radius_derived_width_profile(stroke: ContinuousPath) -> tuple[tuple[float, float], ...]:
    points, radius = _distinct_points(stroke)
    if len(points) < 2:
        raise ValueError("a rendered stroke must contain at least two distinct centerline points")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(distances)))
    progress /= progress[-1]
    count = min(33, max(9, len(points)))
    knots = np.linspace(0.0, 1.0, count)
    widths = np.maximum(2.0 * np.interp(knots, progress, radius), 1.0)
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0)) / 9.0
    smooth_widths = np.convolve(np.pad(widths, (2, 2), mode="edge"), kernel, mode="valid")
    return tuple(
        (float(position), float(width))
        for position, width in zip(knots, smooth_widths, strict=True)
    )


def _interpolate_width_profile(
    profile: tuple[tuple[float, float], ...],
) -> Callable[[float], float]:
    positions = np.asarray([position for position, _ in profile], dtype=float)
    widths = np.asarray([width for _, width in profile], dtype=float)

    def width_at(progress: float) -> float:
        return float(np.interp(progress, positions, widths))

    return width_at


def render_route_coverage(
    candidate: WritingRouteCandidate,
    *,
    frame_width: int,
    frame_height: int,
    supersampling: int = 4,
) -> tuple[np.ndarray, tuple[tuple[tuple[float, float], ...], ...]]:
    frame = RasterFrame(frame_width, frame_height, Affine2D.identity())
    strokes = []
    width_profiles = []
    for index, stroke in enumerate(candidate.continuous_strokes):
        points, _ = _distinct_points(stroke)
        if len(points) < 2:
            continue
        profile = radius_derived_width_profile(stroke)
        width_profiles.append(profile)
        strokes.append(
            StrokeInput(
                id=f"stroke-{index + 1}",
                path=path_from_polyline(points),
                width=_interpolate_width_profile(profile),
                outline_config=OutlineConfig(width_parameterization="arclength"),
            )
        )
    rendered = render_strokes(strokes, raster_frame=frame, supersampling=supersampling)
    assert rendered.combined_coverage is not None
    return rendered.combined_coverage, tuple(width_profiles)


def compare_rendered_coverage(
    rendered_coverage: np.ndarray,
    evidence: ForegroundEvidence,
    *,
    width_profiles: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> InkReconstruction:
    coverage = np.asarray(rendered_coverage, dtype=np.float32)
    observed = evidence.mask
    if coverage.shape != observed.shape:
        raise ValueError("rendered coverage and foreground evidence must have matching dimensions")
    rendered_mask = coverage >= 0.5
    under = observed & ~rendered_mask
    over = rendered_mask & ~observed
    observed_values = observed.astype(np.float32)
    soft_under = float(np.maximum(observed_values - coverage, 0.0).sum())
    soft_over = float(np.maximum(coverage - observed_values, 0.0).sum())
    soft_intersection = float(np.minimum(observed_values, coverage).sum())
    soft_union = float(np.maximum(observed_values, coverage).sum())
    binary_intersection = int((observed & rendered_mask).sum())
    binary_union = int((observed | rendered_mask).sum())
    observed_pixels = int(observed.sum())
    rendered_pixels = int(rendered_mask.sum())
    coverage_pixels = float(coverage.sum())
    metrics: dict[str, float | int | str] = {
        "renderer": "strokeoutline.render_strokes",
        "evidence_source": "rasterregions.foreground_evidence",
        "evidence_method": evidence.method,
        "rendered_mask_threshold": 0.5,
        "observed_ink_pixel_count": observed_pixels,
        "rendered_ink_pixel_count": rendered_pixels,
        "rendered_coverage_pixel_equivalents": coverage_pixels,
        "under_coverage_pixel_count": int(under.sum()),
        "over_coverage_pixel_count": int(over.sum()),
        "intersection_over_union": (binary_intersection / binary_union if binary_union else 1.0),
        "under_coverage_fraction": soft_under / observed_pixels if observed_pixels else 0.0,
        "over_coverage_fraction": soft_over / coverage_pixels if coverage_pixels else 0.0,
        "soft_under_coverage_pixel_equivalents": soft_under,
        "soft_over_coverage_pixel_equivalents": soft_over,
        "soft_intersection_over_union": (soft_intersection / soft_union if soft_union else 1.0),
    }
    return InkReconstruction(
        observed_mask=observed,
        rendered_coverage=coverage,
        under_coverage_mask=under,
        over_coverage_mask=over,
        metrics=metrics,
        width_profiles=width_profiles,
    )


def reconstruct_route_ink(
    candidate: WritingRouteCandidate,
    evidence: ForegroundEvidence,
    *,
    supersampling: int = 4,
) -> InkReconstruction:
    height, width = evidence.mask.shape
    coverage, profiles = render_route_coverage(
        candidate,
        frame_width=width,
        frame_height=height,
        supersampling=supersampling,
    )
    return compare_rendered_coverage(coverage, evidence, width_profiles=profiles)
