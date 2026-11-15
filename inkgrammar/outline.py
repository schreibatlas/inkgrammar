from __future__ import annotations

from planarcurve import CubicPath
from planarcurve.types import Point2D as Point
from strokeoutline import (
    OutlineConfig,
    OutlineResult,
    WidthProfile,
    combine_outlines,
    outline,
    outline_disc,
)
from strokeoutline.svg import outline_svg_path

from .shaping import ShapedText

DEFAULT_OUTLINE_CONFIG = OutlineConfig()


def expand_path(
    d: str | CubicPath,
    width: WidthProfile,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> OutlineResult:
    """Expand an SVG path or typed PlanarCurve path through StrokeOutline."""
    if isinstance(d, CubicPath):
        return outline(d, width, config=config)
    return outline_svg_path(d, width, config=config)


def expand_tap(
    point: Point,
    width: WidthProfile,
    *,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> OutlineResult:
    """Expand a point stroke through StrokeOutline's disc implementation."""
    return outline_disc(point, width, config=config)


def expand_shaped_text(
    shaped: ShapedText,
    default_width: float,
    *,
    width_profiles: dict[str, WidthProfile] | None = None,
    config: OutlineConfig = DEFAULT_OUTLINE_CONFIG,
) -> OutlineResult:
    """Apply InkGrammar's per-stroke width policy using StrokeOutline geometry."""
    profiles = width_profiles or {}
    outlines = []
    for stroke in shaped.strokes:
        width = profiles.get(stroke.id, default_width)
        if stroke.d is not None:
            outlines.append(expand_path(stroke.d, width, config=config))
        elif stroke.tap is not None:
            outlines.append(expand_tap(stroke.tap, width, config=config))
    return combine_outlines(outlines)
