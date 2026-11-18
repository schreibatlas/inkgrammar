from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from planarcurve.transforms import Affine2D
from strokeoutline import RasterFrame, WidthProfile, rasterize_outline
from strokeoutline.svg import outline_svg_path

from .trace import CompiledPartTrace


@dataclass(frozen=True)
class FittingRender:
    coverage: np.ndarray
    part_layers: dict[str, np.ndarray]
    stroke_layers: dict[str, np.ndarray]


def render_paths(
    paths: Mapping[str, str],
    *,
    canvas_size: tuple[int, int],
    origin: tuple[float, float],
    pixels_per_unit: float,
    stroke_width: WidthProfile,
    supersampling: int = 4,
    stroke_ids: Mapping[str, str] | None = None,
) -> FittingRender:
    """Render SVG centerlines through StrokeOutline into a fixed pixel frame."""
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if pixels_per_unit <= 0:
        raise ValueError("scale must be positive")
    if supersampling < 1:
        raise ValueError("supersampling must be at least one")
    frame = RasterFrame(
        width,
        height,
        Affine2D(m00=pixels_per_unit, m11=-pixels_per_unit, dx=origin[0], dy=origin[1]),
    )
    part_layers: dict[str, np.ndarray] = {}
    stroke_layers: dict[str, np.ndarray] = {}
    coverage = np.zeros((height, width), dtype=np.float32)

    for part_id, path in paths.items():
        outline_result = outline_svg_path(path, stroke_width)
        part = rasterize_outline(outline_result, frame, supersampling=supersampling).coverage
        part_layers[part_id] = part
        stroke_id = stroke_ids.get(part_id, part_id) if stroke_ids else part_id
        if stroke_id in stroke_layers:
            stroke_layers[stroke_id] = np.maximum(stroke_layers[stroke_id], part)
        else:
            stroke_layers[stroke_id] = part.copy()
        coverage = np.maximum(coverage, part)

    return FittingRender(coverage, part_layers, stroke_layers)


def render_traces(
    traces: Sequence[CompiledPartTrace],
    *,
    canvas_size: tuple[int, int],
    origin: tuple[float, float],
    pixels_per_unit: float,
    stroke_width: WidthProfile,
    supersampling: int = 4,
) -> FittingRender:
    paths = {trace.instance_id: trace.d for trace in traces}
    stroke_ids = {trace.instance_id: trace.stroke_id for trace in traces}
    return render_paths(
        paths,
        canvas_size=canvas_size,
        origin=origin,
        pixels_per_unit=pixels_per_unit,
        stroke_width=stroke_width,
        supersampling=supersampling,
        stroke_ids=stroke_ids,
    )
