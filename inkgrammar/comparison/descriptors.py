from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterregions as rr
from PIL import Image
from planarcurve import (
    TrajectoryDescriptor,
    descriptor_distance,
    reverse_descriptor,
    trajectory_descriptor,
)
from skeletonpaths import SkeletonAnalysis

from .models import ExtractedLetter
from .segmentation import SegmentedGlyph
from .skeleton import analyze_skeleton


@dataclass(frozen=True)
class GlyphDescriptor:
    glyph_id: str
    normalized_mask: np.ndarray
    aspect_ratio: float
    ink_fill: float
    component_count: int
    endpoint_count: int | None = None
    junction_count: int | None = None
    cycle_count: int | None = None
    path_available: bool | None = None
    analysis: SkeletonAnalysis | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        mask = np.asarray(self.normalized_mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("normalized glyph mask must be a nonempty 2D array")
        if self.aspect_ratio <= 0.0 or not 0.0 <= self.ink_fill <= 1.0:
            raise ValueError("glyph descriptor geometry is invalid")
        if self.component_count < 1:
            raise ValueError("glyph component count must be positive")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "normalized_mask", mask)


def canonicalize_mask(
    mask: np.ndarray,
    *,
    size: int = 96,
    padding: int = 8,
) -> np.ndarray:
    """Fit a tight binary glyph mask into a centered, aspect-preserving square."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("glyph mask must be two-dimensional")
    if size < 1 or padding < 0 or padding * 2 >= size:
        raise ValueError("size must be positive and padding must leave a nonempty canvas")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("cannot normalize an empty glyph mask")
    tight = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = size - padding * 2
    scale = min(target / tight.shape[1], target / tight.shape[0])
    width = max(1, round(tight.shape[1] * scale))
    height = max(1, round(tight.shape[0] * scale))
    resized = Image.fromarray(tight.astype(np.uint8) * 255).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    raster = np.asarray(resized, dtype=np.uint8) > 0
    canvas = np.zeros((size, size), dtype=bool)
    x0 = (size - width) // 2
    y0 = (size - height) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = raster
    return canvas


def glyph_descriptor(
    candidate: SegmentedGlyph,
    *,
    size: int = 96,
    analysis: SkeletonAnalysis | None = None,
) -> GlyphDescriptor:
    """Create a coarse raster and semantic-topology descriptor for retrieval."""
    ys, xs = np.nonzero(candidate.mask)
    if not len(xs):
        raise ValueError(f"glyph candidate {candidate.glyph_id} has an empty mask")
    tight = candidate.mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if analysis is None:
        analysis = analyze_skeleton(candidate.mask).analysis
    assert analysis is not None
    graph = analysis.graph
    return GlyphDescriptor(
        glyph_id=candidate.glyph_id,
        normalized_mask=canonicalize_mask(tight, size=size),
        aspect_ratio=float(tight.shape[1] / tight.shape[0]),
        ink_fill=float(tight.mean()),
        component_count=len(rr.components(tight)),
        endpoint_count=sum(node.degree == 1 for node in graph.nodes),
        junction_count=len(graph.junction_regions),
        cycle_count=graph.cycle_count,
        path_available=bool(graph.edges) and graph.component_count == 1,
        analysis=analysis,
    )


@dataclass(frozen=True)
class RouteDescriptor:
    stroke_descriptors: tuple[TrajectoryDescriptor | None, ...]
    stroke_count: int
    pen_lift_count: int
    cycle_count: int
    junction_count: int
    secondary_mark_count: int
    component_role_sequence: tuple[str, ...]
    relative_mark_centroids: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stroke_descriptors", tuple(self.stroke_descriptors))
        object.__setattr__(self, "component_role_sequence", tuple(self.component_role_sequence))
        object.__setattr__(self, "relative_mark_centroids", tuple(self.relative_mark_centroids))
        if self.stroke_count != len(self.stroke_descriptors):
            raise ValueError("stroke_count must match the route descriptors")
        if (
            min(
                self.stroke_count,
                self.pen_lift_count,
                self.cycle_count,
                self.junction_count,
                self.secondary_mark_count,
            )
            < 0
        ):
            raise ValueError("route descriptor counts must be nonnegative")
        if any(
            len(centroid) != 2 or not np.isfinite(centroid).all()
            for centroid in self.relative_mark_centroids
        ):
            raise ValueError("relative mark centroids must be finite coordinate pairs")


def _stroke_descriptor(points: np.ndarray) -> TrajectoryDescriptor | None:
    try:
        return trajectory_descriptor(points)
    except ValueError:
        return None


def route_descriptor(
    route,
    glyph: SegmentedGlyph | ExtractedLetter | None = None,
) -> RouteDescriptor:
    strokes = route.continuous_strokes
    return RouteDescriptor(
        stroke_descriptors=tuple(_stroke_descriptor(stroke.xy) for stroke in strokes),
        stroke_count=len(strokes),
        pen_lift_count=sum(event.kind == "pen-lift" for event in route.events),
        cycle_count=route.cycle_count,
        junction_count=route.physical_junction_count,
        secondary_mark_count=(len(glyph.secondary_mark_centroids) if glyph is not None else 0),
        component_role_sequence=(glyph.component_role_sequence if glyph is not None else ()),
        relative_mark_centroids=(glyph.secondary_mark_centroids if glyph is not None else ()),
    )


def reverse_route_descriptor(descriptor: RouteDescriptor) -> RouteDescriptor:
    return RouteDescriptor(
        stroke_descriptors=tuple(
            reverse_descriptor(stroke) if stroke is not None else None
            for stroke in reversed(descriptor.stroke_descriptors)
        ),
        stroke_count=descriptor.stroke_count,
        pen_lift_count=descriptor.pen_lift_count,
        cycle_count=descriptor.cycle_count,
        junction_count=descriptor.junction_count,
        secondary_mark_count=descriptor.secondary_mark_count,
        component_role_sequence=descriptor.component_role_sequence,
        relative_mark_centroids=descriptor.relative_mark_centroids,
    )


def route_descriptor_distance(reference: RouteDescriptor, sample: RouteDescriptor) -> float:
    """Compare route shape and structural evidence in either traversal direction."""

    def trajectory_cost(sample_strokes: tuple[TrajectoryDescriptor | None, ...]) -> float:
        reference_strokes = reference.stroke_descriptors
        paired_count = min(len(reference_strokes), len(sample_strokes))
        if not paired_count:
            shape_cost = 0.0 if not reference_strokes and not sample_strokes else 1.0
        else:
            shape_cost = float(
                np.mean(
                    [
                        (
                            descriptor_distance(reference_strokes[index], sample_strokes[index])
                            if reference_strokes[index] is not None
                            and sample_strokes[index] is not None
                            else 1.0
                        )
                        for index in range(paired_count)
                    ]
                )
            )
        count_cost = abs(len(reference_strokes) - len(sample_strokes)) / max(
            len(reference_strokes), len(sample_strokes), 1
        )
        return float((shape_cost + count_cost) / 2.0)

    def count_cost(first: int, second: int) -> float:
        return abs(first - second) / max(first, second, 1)

    def sequence_cost(first: tuple[str, ...], second: tuple[str, ...]) -> float:
        paired_count = min(len(first), len(second))
        mismatches = sum(first[index] != second[index] for index in range(paired_count))
        return float((mismatches + abs(len(first) - len(second))) / max(len(first), len(second), 1))

    def mark_cost() -> float:
        reference_marks = reference.relative_mark_centroids
        sample_marks = sample.relative_mark_centroids
        paired_count = min(len(reference_marks), len(sample_marks))
        position_cost = (
            float(
                np.mean(
                    [
                        np.linalg.norm(
                            np.asarray(reference_marks[index]) - np.asarray(sample_marks[index])
                        )
                        / np.sqrt(2.0)
                        for index in range(paired_count)
                    ]
                )
            )
            if paired_count
            else 0.0
        )
        return float((position_cost + count_cost(len(reference_marks), len(sample_marks))) / 2.0)

    direct = trajectory_cost(sample.stroke_descriptors)
    reversed_strokes = tuple(
        reverse_descriptor(stroke) if stroke is not None else None
        for stroke in reversed(sample.stroke_descriptors)
    )
    shape_cost = min(direct, trajectory_cost(reversed_strokes))
    structural_costs = (
        count_cost(reference.pen_lift_count, sample.pen_lift_count),
        count_cost(reference.cycle_count, sample.cycle_count),
        count_cost(reference.junction_count, sample.junction_count),
        sequence_cost(reference.component_role_sequence, sample.component_role_sequence),
        mark_cost(),
        count_cost(reference.secondary_mark_count, sample.secondary_mark_count),
    )
    return float(np.mean((shape_cost, *structural_costs)))
