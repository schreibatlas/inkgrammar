from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from PIL import Image
from skeletonpaths import (
    SkeletonAnalysis,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
)

from .descriptors import GlyphDescriptor, glyph_descriptor, route_descriptor
from .geometry import continuous_differential_geometry
from .models import ContinuousPath, ExtractedLetter, WritingRoutePairSelection
from .path import (
    PathExtractionError,
    align_continuous_paths,
    recover_ordered_path,
    refine_continuous_path,
    resample_continuous_path,
)
from .routes import evaluate_writing_route_pairs
from .segmentation import ReferenceGallery, SegmentedGlyph
from .skeleton import analyze_skeleton
from .steering import (
    fit_writing_route_candidates,
    fitted_steering_distance,
    steering_profile_distance,
)

RASTER_WEIGHT = 1.0
ASPECT_WEIGHT = 0.15
FILL_WEIGHT = 0.10
TOPOLOGY_WEIGHT = 0.15
ROTATION_ANGLES_DEG = (-8, -4, 0, 4, 8)
DEFAULT_TOP_K = 5
DEFAULT_REFINEMENT_LIMIT = 32
MINIMUM_ALTERNATIVES = 3
CENTERLINE_WEIGHT = 0.45
HEADING_WEIGHT = 0.15
FITTED_STEERING_WEIGHT = 0.15
ROUTE_STRUCTURE_WEIGHT = 0.10
AMBIGUITY_ABSOLUTE_MARGIN = 0.04
ROUTE_DESCRIPTOR_WEIGHT = 0.25
ROUTE_TOPOLOGY_WEIGHT = 0.10
AMBIGUITY_RELATIVE_MARGIN = 0.15


@dataclass(frozen=True)
class ReferenceIndex:
    gallery: ReferenceGallery
    descriptors: tuple[GlyphDescriptor, ...]

    def __post_init__(self) -> None:
        if len(self.gallery.candidates) != len(self.descriptors):
            raise ValueError("reference candidates and descriptors must have matching lengths")


@dataclass(frozen=True)
class MatchEvidence:
    raster_cost: float
    aspect_cost: float
    fill_cost: float
    topology_cost: float
    route_descriptor_cost: float | None = None
    route_centerline_cost: float | None = None
    route_heading_cost: float | None = None
    route_steering_cost: float | None = None
    route_topology_cost: float | None = None
    route_structure_cost: float | None = None
    missing_evidence_penalty: float = 0.0

    def __post_init__(self) -> None:
        costs = (
            self.raster_cost,
            self.aspect_cost,
            self.fill_cost,
            self.topology_cost,
            self.route_descriptor_cost,
            self.route_centerline_cost,
            self.route_heading_cost,
            self.route_steering_cost,
            self.route_topology_cost,
            self.route_structure_cost,
            self.missing_evidence_penalty,
        )
        if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in costs):
            raise ValueError("match evidence costs and penalties must be finite and nonnegative")

    @property
    def available_features(self) -> tuple[str, ...]:
        features = ["raster", "aspect", "fill", "topology"]
        route_features = (
            ("route_descriptor", self.route_descriptor_cost),
            ("route_centerline", self.route_centerline_cost),
            ("route_heading", self.route_heading_cost),
            ("route_steering", self.route_steering_cost),
            ("route_topology", self.route_topology_cost),
            ("route_structure", self.route_structure_cost),
        )
        features.extend(name for name, value in route_features if value is not None)
        return tuple(features)

    @property
    def unavailable_features(self) -> tuple[str, ...]:
        route_features = (
            ("route_descriptor", self.route_descriptor_cost),
            ("route_centerline", self.route_centerline_cost),
            ("route_heading", self.route_heading_cost),
            ("route_steering", self.route_steering_cost),
            ("route_topology", self.route_topology_cost),
            ("route_structure", self.route_structure_cost),
        )
        return tuple(name for name, value in route_features if value is None)


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
    route_descriptor_cost: float | None = None
    route_topology_cost: float | None = None
    missing_evidence_penalty: float = 0.0
    route_pair_selection: WritingRoutePairSelection | None = None

    @property
    def cost(self) -> float:
        return self.refined_cost if self.refined_cost is not None else self.coarse_cost

    @property
    def evidence(self) -> MatchEvidence:
        return MatchEvidence(
            raster_cost=self.raster_cost,
            aspect_cost=self.aspect_cost,
            fill_cost=self.fill_cost,
            topology_cost=self.topology_cost,
            route_descriptor_cost=self.route_descriptor_cost,
            route_centerline_cost=self.centerline_cost,
            route_heading_cost=self.heading_cost,
            route_steering_cost=self.fitted_steering_cost,
            route_topology_cost=self.route_topology_cost,
            route_structure_cost=self.route_structure_cost,
            missing_evidence_penalty=self.missing_evidence_penalty,
        )


@dataclass(frozen=True)
class ProgressiveRefinementResult:
    scores: tuple[MatchScore, ...]
    coarse_candidate_count: int
    refined_candidate_count: int
    stop_reason: Literal[
        "all_candidates_refined", "coarse_lower_bound", "candidate_refinement_limit"
    ]
    search_truncated: bool

    @property
    def best(self) -> MatchScore | None:
        return self.scores[0] if self.scores else None

    @property
    def ambiguous(self) -> bool:
        return self.search_truncated or match_status(list(self.scores)) == "ambiguous"


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
    route_pair_selection: WritingRoutePairSelection | None = None,
) -> float | None:
    if sample_extracted is None or reference_extracted is None:
        return None
    if route_pair_selection is None and (
        not sample_extracted.route_candidates or not reference_extracted.route_candidates
    ):
        return None
    sample_route = (
        route_pair_selection.sample
        if route_pair_selection is not None
        else min(
            sample_extracted.route_candidates,
            key=lambda candidate: candidate.score.intrinsic_cost,
        )
    )
    reference_route = (
        route_pair_selection.reference
        if route_pair_selection is not None
        else min(
            reference_extracted.route_candidates,
            key=lambda candidate: candidate.score.intrinsic_cost,
        )
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


def _missing_evidence_penalty(
    *,
    centerline_cost: float | None,
    heading_cost: float | None,
    fitted_steering_cost: float | None,
    route_structure_cost: float | None,
    route_descriptor_cost: float | None,
    route_topology_cost: float | None,
) -> float:
    missing_features = (
        (centerline_cost, CENTERLINE_WEIGHT),
        (heading_cost, HEADING_WEIGHT),
        (fitted_steering_cost, FITTED_STEERING_WEIGHT),
        (route_structure_cost, ROUTE_STRUCTURE_WEIGHT),
        (route_descriptor_cost, ROUTE_DESCRIPTOR_WEIGHT),
        (route_topology_cost, ROUTE_TOPOLOGY_WEIGHT),
    )
    return sum(weight for value, weight in missing_features if value is None)


def _route_pair_selection(
    sample_extracted: ExtractedLetter | None,
    reference_extracted: ExtractedLetter | None,
) -> WritingRoutePairSelection | None:
    if (
        sample_extracted is None
        or reference_extracted is None
        or not sample_extracted.route_candidates
        or not reference_extracted.route_candidates
    ):
        return None
    sample_candidates = fit_writing_route_candidates(sample_extracted.route_candidates)
    reference_candidates = fit_writing_route_candidates(reference_extracted.route_candidates)
    sample_descriptors = sample_extracted.route_descriptors or tuple(
        route_descriptor(candidate, sample_extracted) for candidate in sample_candidates
    )
    reference_descriptors = reference_extracted.route_descriptors or tuple(
        route_descriptor(candidate, reference_extracted) for candidate in reference_candidates
    )
    return evaluate_writing_route_pairs(
        reference_candidates,
        sample_candidates,
        steering_distance=steering_profile_distance,
        reference_descriptors=reference_descriptors,
        sample_descriptors=sample_descriptors,
    )


def refine_match(
    sample: SegmentedGlyph,
    reference: SegmentedGlyph,
    coarse_score: MatchScore,
    *,
    sample_extracted: ExtractedLetter | None = None,
    reference_extracted: ExtractedLetter | None = None,
) -> MatchScore:
    """Add continuous path evidence to raster ranking without rejecting candidates."""
    pair_selection = _route_pair_selection(sample_extracted, reference_extracted)
    if pair_selection is not None:
        route_structure = _route_structure_cost(
            sample_extracted,
            reference_extracted,
            pair_selection,
        )
        if route_structure is not None:
            route_structure += (
                pair_selection.score.intrinsic_route_cost + pair_selection.score.writing_prior_cost
            )
        centerline = pair_selection.score.aligned_centerline_cost
        steering = pair_selection.score.steering_profile_cost
        descriptor_cost = pair_selection.score.descriptor_matching_cost
        topology_cost = pair_selection.score.topology_event_compatibility_cost
        missing_evidence_penalty = _missing_evidence_penalty(
            centerline_cost=centerline,
            heading_cost=None,
            fitted_steering_cost=steering,
            route_structure_cost=route_structure,
            route_descriptor_cost=descriptor_cost,
            route_topology_cost=topology_cost,
        )
        refined = (
            coarse_score.coarse_cost
            + CENTERLINE_WEIGHT * centerline
            + FITTED_STEERING_WEIGHT * steering
            + ROUTE_STRUCTURE_WEIGHT * (route_structure or 0.0)
            + ROUTE_DESCRIPTOR_WEIGHT * descriptor_cost
            + ROUTE_TOPOLOGY_WEIGHT * topology_cost
            + missing_evidence_penalty
        )
        return replace(
            coarse_score,
            refined_cost=float(refined),
            centerline_cost=centerline,
            fitted_steering_cost=steering,
            route_structure_cost=route_structure,
            route_descriptor_cost=descriptor_cost,
            route_topology_cost=topology_cost,
            missing_evidence_penalty=missing_evidence_penalty,
            route_pair_selection=pair_selection,
        )
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
        missing_evidence_penalty = _missing_evidence_penalty(
            centerline_cost=None,
            heading_cost=None,
            fitted_steering_cost=None,
            route_structure_cost=route_structure,
            route_descriptor_cost=None,
            route_topology_cost=None,
        )
        route_increment = ROUTE_STRUCTURE_WEIGHT * (route_structure or 0.0)
        return replace(
            coarse_score,
            refined_cost=(coarse_score.coarse_cost + route_increment + missing_evidence_penalty),
            route_structure_cost=route_structure,
            missing_evidence_penalty=missing_evidence_penalty,
        )

    centerline = float(
        np.mean(np.linalg.norm(alignment.aligned_sample_xy - reference_samples.xy, axis=1))
        / reference_path.arc_length_px
    )
    heading_delta = sample_geometry.heading_rad - reference_geometry.heading_rad
    heading = float(
        np.mean(np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))) / np.pi
    )
    missing_evidence_penalty = _missing_evidence_penalty(
        centerline_cost=centerline,
        heading_cost=heading,
        fitted_steering_cost=fitted_steering,
        route_structure_cost=route_structure,
        route_descriptor_cost=None,
        route_topology_cost=None,
    )
    refined = (
        coarse_score.coarse_cost
        + CENTERLINE_WEIGHT * centerline
        + HEADING_WEIGHT * heading
        + FITTED_STEERING_WEIGHT * fitted_steering
        + ROUTE_STRUCTURE_WEIGHT * (route_structure or 0.0)
        + ROUTE_DESCRIPTOR_WEIGHT * (coarse_score.route_descriptor_cost or 0.0)
        + ROUTE_TOPOLOGY_WEIGHT * (coarse_score.route_topology_cost or 0.0)
        + missing_evidence_penalty
    )
    return replace(
        coarse_score,
        refined_cost=float(refined),
        centerline_cost=centerline,
        heading_cost=heading,
        fitted_steering_cost=fitted_steering,
        route_structure_cost=route_structure,
        missing_evidence_penalty=missing_evidence_penalty,
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


def refine_matches_progressively(
    sample: SegmentedGlyph,
    gallery: ReferenceGallery,
    coarse_scores: list[MatchScore],
    *,
    sample_extracted: ExtractedLetter | None = None,
    reference_extractions: Mapping[str, ExtractedLetter] | None = None,
    reference_extractor: Callable[[SegmentedGlyph], ExtractedLetter] | None = None,
    max_refinements: int = DEFAULT_REFINEMENT_LIMIT,
) -> ProgressiveRefinementResult:
    """Refine coarse candidates while their lower bounds can affect the decision."""
    if max_refinements < 1:
        raise ValueError("max_refinements must be positive")

    candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    ordered = sorted(
        (score for score in coarse_scores if score.reference_id in candidates),
        key=lambda score: (score.coarse_cost, score.reference_id),
    )
    refined: list[MatchScore] = []
    stop_reason: Literal[
        "all_candidates_refined", "coarse_lower_bound", "candidate_refinement_limit"
    ] = "all_candidates_refined"
    search_truncated = False

    for score in ordered:
        if refined:
            best_cost = min(candidate.cost for candidate in refined)
            ambiguity_envelope = max(
                AMBIGUITY_ABSOLUTE_MARGIN,
                AMBIGUITY_RELATIVE_MARGIN * abs(best_cost),
            )
            if score.coarse_cost > best_cost + ambiguity_envelope:
                stop_reason = "coarse_lower_bound"
                break
        if len(refined) >= max_refinements:
            stop_reason = "candidate_refinement_limit"
            search_truncated = True
            break

        candidate = candidates[score.reference_id]
        reference_extracted = (
            reference_extractions.get(score.reference_id)
            if reference_extractions is not None
            else None
        )
        if reference_extracted is None and reference_extractor is not None:
            reference_extracted = reference_extractor(candidate)
        refined.append(
            refine_match(
                sample,
                candidate,
                score,
                sample_extracted=sample_extracted,
                reference_extracted=reference_extracted,
            )
        )

    refined.sort(key=lambda candidate: (candidate.cost, candidate.reference_id))
    ranked = tuple(replace(score, rank=rank) for rank, score in enumerate(refined, start=1))
    return ProgressiveRefinementResult(
        scores=ranked,
        coarse_candidate_count=len(ordered),
        refined_candidate_count=len(ranked),
        stop_reason=stop_reason,
        search_truncated=search_truncated,
    )


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
