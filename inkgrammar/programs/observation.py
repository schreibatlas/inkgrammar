from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from rasterregions import BBox, MaskMode, MaskResult, foreground_mask
from skeletonpaths import SkeletonAnalysis, analyze_mask

from ..dataset.transforms import (
    ObservationFrame,
    glyph_units_to_source_px,
    source_px_to_glyph_units,
    standard_frame,
)

__all__ = [
    "GlyphObservation",
    "ObservationFrame",
    "TopologyDiagnostics",
    "glyph_units_to_source_px",
    "prepare_glyph_observation",
    "source_px_to_glyph_units",
    "standard_frame",
]


@dataclass(frozen=True)
class TopologyDiagnostics:
    """Raster-derived topology counts, kept separate from program topology."""

    component_count: int
    endpoint_count: int
    junction_count: int
    cycle_count: int
    node_count: int
    edge_count: int

    @classmethod
    def from_analysis(cls, analysis: SkeletonAnalysis) -> TopologyDiagnostics:
        graph = analysis.graph
        return cls(
            component_count=graph.component_count,
            endpoint_count=graph.endpoint_count,
            junction_count=graph.junction_count,
            cycle_count=graph.cycle_count,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
        )


@dataclass(frozen=True)
class GlyphObservation:
    """Prepared raster evidence and candidate-independent cached measurements."""

    image: Image.Image
    mask: np.ndarray
    bbox: BBox
    frame: ObservationFrame
    distance_field: np.ndarray
    mask_result: MaskResult | None = None
    skeleton_analysis: SkeletonAnalysis | None = None

    def __post_init__(self) -> None:
        image = self.image.copy()
        mask = np.asarray(self.mask)
        distance_field = np.asarray(self.distance_field, dtype=np.float32)
        if mask.ndim != 2 or mask.dtype != np.dtype(bool):
            raise ValueError("observation mask must be a two-dimensional boolean array")
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError("observation image and mask dimensions must match")
        if self.bbox != _tight_bbox(mask):
            raise ValueError("observation bbox must be the tight foreground bounds")
        if (
            distance_field.shape != mask.shape
            or not np.isfinite(distance_field).all()
            or np.any(distance_field < 0.0)
        ):
            raise ValueError("distance field must be finite, nonnegative, and match the mask")
        if self.mask_result is not None and not np.array_equal(self.mask_result.mask, mask):
            raise ValueError("mask result must match the observation mask")
        if self.skeleton_analysis is not None and not np.array_equal(
            self.skeleton_analysis.mask, mask
        ):
            raise ValueError("skeleton analysis must match the observation mask")
        mask = np.array(mask, dtype=bool, copy=True)
        distance_field = np.array(distance_field, dtype=np.float32, copy=True)
        mask.setflags(write=False)
        distance_field.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "distance_field", distance_field)

    @property
    def topology_diagnostics(self) -> TopologyDiagnostics | None:
        if self.skeleton_analysis is None:
            return None
        return TopologyDiagnostics.from_analysis(self.skeleton_analysis)


def _binary_mask(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("observation mask must be two-dimensional")
    if array.dtype != np.dtype(bool) and (
        not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
        or np.any((array != 0) & (array != 1))
    ):
        raise ValueError("observation mask must contain only boolean or zero/one values")
    return np.asarray(array, dtype=bool)


def _tight_bbox(mask: np.ndarray) -> BBox:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("observation mask contains no foreground pixels")
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _distance_field(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:
        raise ImportError(
            "distance-field preparation requires SciPy; install inkgrammar[fit]"
        ) from error
    return distance_transform_edt(~mask).astype(np.float32)


def _display_image(source: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    array = np.asarray(source)
    if array.dtype == np.dtype(bool):
        return Image.fromarray(array.astype(np.uint8) * 255)
    try:
        return Image.fromarray(array)
    except (TypeError, ValueError) as error:
        raise ValueError("observation array cannot be represented as an image") from error


def prepare_glyph_observation(
    source: Image.Image | np.ndarray | GlyphObservation,
    *,
    mask: ArrayLike | None = None,
    frame: ObservationFrame | None = None,
    mask_mode: MaskMode = "auto",
    analyze_skeleton: bool = False,
) -> GlyphObservation:
    """Prepare a mask, frame, cached distance field, and optional skeleton evidence."""
    if isinstance(source, GlyphObservation):
        if mask is not None or frame is not None:
            raise ValueError("a prepared observation cannot be combined with a new mask or frame")
        if not analyze_skeleton or source.skeleton_analysis is not None:
            return source
        return GlyphObservation(
            image=source.image,
            mask=source.mask,
            bbox=source.bbox,
            frame=source.frame,
            distance_field=source.distance_field,
            mask_result=source.mask_result,
            skeleton_analysis=analyze_mask(source.mask),
        )

    image = _display_image(source)
    if mask is None:
        if isinstance(source, np.ndarray) and source.dtype == np.dtype(bool):
            observed_mask = _binary_mask(source)
            mask_result = None
        else:
            mask_result = foreground_mask(source, mode=mask_mode)
            observed_mask = np.asarray(mask_result.mask, dtype=bool)
    else:
        observed_mask = _binary_mask(mask)
        mask_result = None
    if image.size != (observed_mask.shape[1], observed_mask.shape[0]):
        raise ValueError("observation image and mask dimensions must match")
    bbox = _tight_bbox(observed_mask)
    observation_frame = frame or standard_frame(
        origin_source_px=(float(bbox.x0), float(bbox.y1)),
        x_height_px=float(bbox.height),
    )
    skeleton = analyze_mask(observed_mask) if analyze_skeleton else None
    return GlyphObservation(
        image=image,
        mask=observed_mask,
        bbox=bbox,
        frame=observation_frame,
        distance_field=_distance_field(observed_mask),
        mask_result=mask_result,
        skeleton_analysis=skeleton,
    )
