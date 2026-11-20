from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from PIL import Image
from skeletonpaths import (
    SkeletonAnalysis,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
)

from .descriptors import GlyphDescriptor, glyph_descriptor
from .geometry import continuous_differential_geometry
from .models import ContinuousPath, ExtractedLetter
from .path import (
    PathExtractionError,
    align_continuous_paths,
    recover_ordered_path,
    refine_continuous_path,
    resample_continuous_path,
)
from .segmentation import ReferenceGallery, SegmentedGlyph
from .skeleton import analyze_skeleton
from .steering import fitted_steering_distance

RASTER_WEIGHT = 1.0
ASPECT_WEIGHT = 0.15
FILL_WEIGHT = 0.10
TOPOLOGY_WEIGHT = 0.15
ROTATION_ANGLES_DEG = (-8, -4, 0, 4, 8)
DEFAULT_TOP_K = 5
MINIMUM_ALTERNATIVES = 3
CENTERLINE_WEIGHT = 0.45
HEADING_WEIGHT = 0.15
FITTED_STEERING_WEIGHT = 0.15
ROUTE_STRUCTURE_WEIGHT = 0.10
AMBIGUITY_ABSOLUTE_MARGIN = 0.04
AMBIGUITY_RELATIVE_MARGIN = 0.15


@dataclass(frozen=True)
class ReferenceIndex:
    gallery: ReferenceGallery
    descriptors: tuple[GlyphDescriptor, ...]

    def __post_init__(self) -> None:
        if len(self.gallery.candidates) != len(self.descriptors):
            raise ValueError("reference candidates and descriptors must have matching lengths")


@dataclass(frozen=True)
class MatchScore:
    reference_id: str
    rank: int
    coarse_cost: float
    raster_cost: float
    aspect_cost: float
    fill_cost: float
    topology_cost: float
    refined_cost: float | None = None
    centerline_cost: float | None = None
    heading_cost: float | None = None
    fitted_steering_cost: float | None = None
    route_structure_cost: float | None = None

    @property
    def cost(self) -> float:
        return self.refined_cost if self.refined_cost is not None else self.coarse_cost


def build_reference_index(gallery: ReferenceGallery) -> ReferenceIndex:
    """Compute reusable coarse descriptors for every reference candidate."""
    return ReferenceIndex(
        gallery=gallery,
        descriptors=tuple(glyph_descriptor(candidate) for candidate in gallery.candidates),
    )


def _normalized_count_difference(first: int | None, second: int | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(first - second) / max(first, second, 1)


def _raster_cost(first: np.ndarray, second: np.ndarray) -> float:
    best_iou = 0.0
    for angle in ROTATION_ANGLES_DEG:
        if angle:
            rotated = Image.fromarray(second.astype(np.uint8) * 255).rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                fillcolor=0,
            )
            compared = np.asarray(rotated, dtype=np.uint8) > 0
        else:
            compared = second
        intersection = int(np.logical_and(first, compared).sum())
        union = int(np.logical_or(first, compared).sum())
        best_iou = max(best_iou, intersection / union if union else 1.0)
    return 1.0 - best_iou


def coarse_match_cost(sample: GlyphDescriptor, reference: GlyphDescriptor) -> MatchScore:
    """Compute an internal retrieval rank cost, not a handwriting quality score."""
    raster = _raster_cost(sample.normalized_mask, reference.normalized_mask)
    aspect = min(1.0, abs(np.log(sample.aspect_ratio / reference.aspect_ratio)))
    fill = abs(sample.ink_fill - reference.ink_fill) / max(
        sample.ink_fill, reference.ink_fill, 1e-9
    )
    topology_terms = [
        _normalized_count_difference(sample.component_count, reference.component_count),
        _normalized_count_difference(sample.endpoint_count, reference.endpoint_count),
        _normalized_count_difference(sample.junction_count, reference.junction_count),
        _normalized_count_difference(sample.cycle_count, reference.cycle_count),
    ]
    available_topology = [term for term in topology_terms if term is not None]
    topology = float(np.mean(available_topology)) if available_topology else 0.0
    cost = (
        RASTER_WEIGHT * raster
        + ASPECT_WEIGHT * aspect
        + FILL_WEIGHT * fill
        + TOPOLOGY_WEIGHT * topology
    )
    return MatchScore(
        reference_id=reference.glyph_id,
        rank=0,
        coarse_cost=float(cost),
        raster_cost=float(raster),
        aspect_cost=float(aspect),
        fill_cost=float(fill),
        topology_cost=topology,
    )


def rank_reference_candidates(
    sample: SegmentedGlyph,
    gallery: ReferenceGallery | ReferenceIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
    sample_analysis: SkeletonAnalysis | None = None,
) -> list[MatchScore]:
    """Return the best coarse shape matches, independently for this sample."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    index = gallery if isinstance(gallery, ReferenceIndex) else build_reference_index(gallery)
    sample_descriptor = glyph_descriptor(sample, analysis=sample_analysis)
    ranked = sorted(
        (coarse_match_cost(sample_descriptor, descriptor) for descriptor in index.descriptors),
        key=lambda score: (score.coarse_cost, score.reference_id),
    )
    return [replace(score, rank=rank) for rank, score in enumerate(ranked[:top_k], start=1)]


def _continuous_path_for_match(
    glyph: SegmentedGlyph,
    extracted: ExtractedLetter | None,
) -> ContinuousPath | None:
    if extracted is not None:
        if extracted.continuous_path is not None:
            return extracted.continuous_path
        if extracted.diagnostics.analysis is not None:
            return None
    diagnostics = analyze_skeleton(glyph.mask)
    ordered_path = recover_ordered_path(diagnostics)
    return refine_continuous_path(diagnostics, ordered_path)


def _route_structure_cost(
    sample_extracted: ExtractedLetter | None,
    reference_extracted: ExtractedLetter | None,
) -> float | None:
    if (
        sample_extracted is None
        or reference_extracted is None
        or not sample_extracted.route_candidates
        or not reference_extracted.route_candidates
    ):
        return None
    sample_route = min(
        sample_extracted.route_candidates,
        key=lambda candidate: candidate.score.intrinsic_cost,
    )
    reference_route = min(
        reference_extracted.route_candidates,
        key=lambda candidate: candidate.score.intrinsic_cost,
    )
    sample_lifts = sum(event.kind == "pen-lift" for event in sample_route.events)
    reference_lifts = sum(event.kind == "pen-lift" for event in reference_route.events)
    terms = (
        _normalized_count_difference(
            len(sample_route.continuous_strokes), len(reference_route.continuous_strokes)
        ),
        _normalized_count_difference(sample_lifts, reference_lifts),
        abs(sample_route.score.repeat_cost - reference_route.score.repeat_cost),
    )
    return float(np.mean(terms))


def refine_match(
    sample: SegmentedGlyph,
    reference: SegmentedGlyph,
    coarse_score: MatchScore,
    *,
    sample_extracted: ExtractedLetter | None = None,
    reference_extracted: ExtractedLetter | None = None,
) -> MatchScore:
    """Add continuous path evidence to raster ranking without rejecting candidates."""
    route_structure = _route_structure_cost(sample_extracted, reference_extracted)
    try:
        reference_path = _continuous_path_for_match(reference, reference_extracted)
        sample_path = _continuous_path_for_match(sample, sample_extracted)
        if reference_path is None or sample_path is None:
            raise PathExtractionError("continuous matching paths are unavailable")
        alignment = align_continuous_paths(reference_path, sample_path)
        aligned_sample_path = alignment.aligned_sample_path
        if aligned_sample_path is None:
            raise PathExtractionError("aligned continuous sample path is unavailable")
        reference_samples = resample_continuous_path(reference_path)
        reference_geometry = continuous_differential_geometry(reference_samples)
        sample_geometry = continuous_differential_geometry(aligned_sample_path)
        fitted_steering = fitted_steering_distance(reference_path, sample_path)
    except (
        PathExtractionError,
        SubpixelReconstructionError,
        SubpixelReconstructionLimitError,
        ValueError,
    ):
        route_increment = ROUTE_STRUCTURE_WEIGHT * (route_structure or 0.0)
        return replace(
            coarse_score,
            refined_cost=coarse_score.coarse_cost + route_increment,
            route_structure_cost=route_structure,
        )

    centerline = float(
        np.mean(np.linalg.norm(alignment.aligned_sample_xy - reference_samples.xy, axis=1))
        / reference_path.arc_length_px
    )
    heading_delta = sample_geometry.heading_rad - reference_geometry.heading_rad
    heading = float(
        np.mean(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))) / np.pi
    )
    refined = (
        coarse_score.coarse_cost
        + CENTERLINE_WEIGHT * centerline
        + HEADING_WEIGHT * heading
        + FITTED_STEERING_WEIGHT * fitted_steering
        + ROUTE_STRUCTURE_WEIGHT * (route_structure or 0.0)
    )
    return replace(
        coarse_score,
        refined_cost=float(refined),
        centerline_cost=centerline,
        heading_cost=heading,
        fitted_steering_cost=fitted_steering,
        route_structure_cost=route_structure,
    )


def refine_top_k_matches(
    sample: SegmentedGlyph,
    gallery: ReferenceGallery,
    ranked: list[MatchScore],
    *,
    sample_extracted: ExtractedLetter | None = None,
    reference_extractions: Mapping[str, ExtractedLetter] | None = None,
) -> list[MatchScore]:
    """Refine coarse top-K candidates with prepared sample and reference paths."""
    candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    refined = []
    for score in ranked:
        candidate = candidates.get(score.reference_id)
        if candidate is None:
            continue
        extracted = (
            reference_extractions.get(score.reference_id)
            if reference_extractions is not None
            else None
        )
        refined.append(
            refine_match(
                sample,
                candidate,
                score,
                sample_extracted=sample_extracted,
                reference_extracted=extracted,
            )
        )
    refined.sort(key=lambda score: (score.cost, score.reference_id))
    return [replace(score, rank=rank) for rank, score in enumerate(refined, start=1)]


def match_status(scores: list[MatchScore]) -> Literal["clear", "ambiguous"]:
    """Classify best-match ambiguity from the gap to the second-ranked candidate."""
    if len(scores) < 2:
        return "clear"
    ranked = sorted(scores, key=lambda score: (score.cost, score.reference_id))
    best, second = ranked[:2]
    margin = second.cost - best.cost
    relative_margin = margin / max(abs(best.cost), 1e-9)
    if margin <= AMBIGUITY_ABSOLUTE_MARGIN or relative_margin <= AMBIGUITY_RELATIVE_MARGIN:
        return "ambiguous"
    return "clear"
