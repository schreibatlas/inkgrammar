"""Deterministic, parameterized, versioned image preprocessing.

Every operation records its policy in :class:`PreprocessParams`; behavior
changes bump :data:`PREPROCESS_VERSION`. Raw binary interpretations are never
overwritten: :func:`clean_mask` returns the raw mask untouched plus a cleanup
record of removed speck candidates. Automatic outputs are candidates for
review, never truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import rasterregions as rr
from PIL import Image
from pydantic import Field

from ..models import StrictModel
from .models import ObservationMetrics

PREPROCESS_VERSION = 2

GlyphClass = Literal["lowercase", "uppercase", "mixed"]


class PreprocessParams(StrictModel):
    """Parameters for all preprocessing operations (versioned as a set)."""

    version: Literal[2] = 2
    threshold_policy: Literal["rasterregions-grayscale-v2"] = "rasterregions-grayscale-v2"
    background_percentile: float | None = Field(default=None, gt=0.0, le=100.0)
    speck_min_area: int = Field(default=4, ge=1)
    crop_pad: int = Field(default=2, ge=0)


class RemovedComponent(StrictModel):
    bbox: tuple[int, int, int, int]
    area: int
    reason: str = "automatic-speck-candidate"
    reviewed: bool = False


class CleanupRecord(StrictModel):
    removed_components: list[RemovedComponent] = Field(default_factory=list)


class IsolationComponent(StrictModel):
    bbox: tuple[int, int, int, int]
    area: int
    kept: bool
    reason: str


class IsolationProposal(StrictModel):
    """Automatic target-ink isolation candidate; review decides acceptance."""

    center: tuple[float, float]
    components: list[IsolationComponent] = Field(default_factory=list)


def isolate_target_ink(
    mask: np.ndarray,
    *,
    center: tuple[float, float],
    neighbor_centers: tuple[tuple[float, float], ...] = (),
) -> tuple[np.ndarray, IsolationProposal]:
    """Adapt generic component isolation to the reviewed dataset proposal model."""
    result = rr.isolate_component(
        mask,
        center=center,
        competing_centers=neighbor_centers,
    )
    reason_map = {
        "nearest-to-center": "nearest-to-reviewed-center",
        "nearest-to-competing-center": "nearest-to-neighbor-center",
        "not-nearest-component": "not-nearest-component",
    }
    records = [
        IsolationComponent(
            bbox=candidate.bbox.as_tuple(),
            area=candidate.area,
            kept=candidate.kept,
            reason=reason_map[candidate.reason],
        )
        for candidate in result.candidates
    ]
    return result.mask.copy(), IsolationProposal(center=center, components=records)


@dataclass(frozen=True)
class PreprocessResult:
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    cleanup: CleanupRecord
    metrics: ObservationMetrics
    params: PreprocessParams


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert to 8-bit grayscale intensity."""
    return np.asarray(image.convert("L"), dtype=np.uint8)


def normalize_background(gray: np.ndarray, *, percentile: float = 95.0) -> np.ndarray:
    """Scale intensities so the background percentile maps to white."""
    background = float(np.percentile(gray, percentile))
    if background <= 0:
        return gray
    scaled = gray.astype(float) * (255.0 / background)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Binarize grayscale pixels with RasterRegions' versioned grayscale policy."""
    return rr.foreground_mask(gray, mode="grayscale").mask


def detect_speck_candidates(mask: np.ndarray, *, min_area: int) -> list[RemovedComponent]:
    """Propose small components as dust/speck candidates (review decides)."""
    return [
        RemovedComponent(bbox=component.bbox.as_tuple(), area=component.area)
        for component in rr.components(mask)
        if component.area < min_area
    ]


def clean_mask(mask: np.ndarray, params: PreprocessParams) -> tuple[np.ndarray, CleanupRecord]:
    """Remove speck candidates from a copy of the raw mask; never mutate raw."""
    cleaned = mask.copy()
    removed = []
    for component in rr.components(mask):
        if component.area < params.speck_min_area:
            removed.append(
                RemovedComponent(
                    bbox=component.bbox.as_tuple(),
                    area=component.area,
                )
            )
            cleaned[component.ys, component.xs] = False
    return cleaned, CleanupRecord(removed_components=removed)


def pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Pad a pixel-space bbox and clamp it to the image shape (h, w)."""
    x0, y0, x1, y1 = bbox
    height, width = shape
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1]))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def estimate_metric_guides(
    mask: np.ndarray, *, glyph_class: GlyphClass = "mixed"
) -> ObservationMetrics:
    """Propose metric guides from trimmed ink row extents.

    All values are crop-normalized (y downward) ``automatic-candidate``
    estimates. For lowercase crops the trimmed top edge is the x-height line
    candidate; for uppercase crops it is the cap height candidate. Mixed
    content only yields baseline and ascender-edge candidates.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("cannot estimate metric guides from an empty mask")
    weights = mask.sum(axis=1)[rows].astype(float)
    top = _weighted_quantile(rows, weights, 0.02) / mask.shape[0]
    bottom = _weighted_quantile(rows, weights, 0.98) / mask.shape[0]
    if glyph_class == "lowercase":
        return ObservationMetrics(
            baseline_y=bottom, x_height_y=top, metric_status="automatic-candidate"
        )
    if glyph_class == "uppercase":
        return ObservationMetrics(
            baseline_y=bottom, cap_height_y=top, metric_status="automatic-candidate"
        )
    return ObservationMetrics(
        baseline_y=bottom, ascender_y=top, metric_status="automatic-candidate"
    )


def preprocess_crop(
    image: Image.Image,
    params: PreprocessParams,
    *,
    glyph_class: GlyphClass = "mixed",
) -> PreprocessResult:
    """Run the full pipeline: gray, background, threshold, clean, metrics."""
    gray = to_grayscale(image)
    if params.background_percentile is not None:
        gray = normalize_background(gray, percentile=params.background_percentile)
    raw = threshold_mask(gray)
    cleaned, cleanup = clean_mask(raw, params)
    metrics = estimate_metric_guides(cleaned, glyph_class=glyph_class)
    return PreprocessResult(
        raw_mask=raw,
        clean_mask=cleaned,
        cleanup=cleanup,
        metrics=metrics,
        params=params,
    )
