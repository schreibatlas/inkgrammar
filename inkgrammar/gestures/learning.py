from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from rasterregions import SpacedSegmentationHypotheses

from ..fitting.models import PrimitiveTemplate
from ..programs.compiler import CompiledGlyphProgram
from .compiler import compile_gesture_program
from .dictionary import (
    GestureDictionaryFit,
    GestureMatch,
    GestureOccurrence,
    learn_gesture_dictionary,
)
from .evidence import CandidateProvenance
from .geometry_fit import (
    DirectedCurveOccurrence,
    FittedGeometryFamily,
    fit_geometry_family,
)
from .models import GestureGlyphProgram, GestureTemplate
from .training import (
    GlyphObservationCandidate,
    MovementAnnotations,
    RasterRegionCandidate,
    build_glyph_observation_candidates,
    region_candidates_from_segmentation,
)

if TYPE_CHECKING:
    from planarcurve.fitting import (
        TrajectoryFitOptions,
        TrajectoryModelSelectionOptions,
        TrajectoryVariationPolicy,
    )


AmbiguityStatus = Literal["reviewed", "inferred", "ambiguous", "not-recorded"]


@dataclass(frozen=True)
class JointGestureAlternative:
    observation_candidate_id: str
    source_observation_id: str
    segmentation_hypothesis_id: str
    stroke_hypothesis_ids: tuple[str, ...]
    matches: tuple[GestureMatch, ...]
    ambiguity_status: AmbiguityStatus
    provenance: CandidateProvenance


@dataclass(frozen=True)
class GestureLearningResult:
    observation_candidates: tuple[GlyphObservationCandidate, ...]
    geometry_family_fits: tuple[FittedGeometryFamily, ...]
    dictionary_fit: GestureDictionaryFit
    compiled_programs: tuple[CompiledGlyphProgram, ...]
    joint_alternatives: tuple[JointGestureAlternative, ...]
    metrics: Mapping[str, float]


def run_gesture_learning(
    source_observation_id: str,
    raster_candidates: SpacedSegmentationHypotheses | Sequence[RasterRegionCandidate],
    geometry_occurrences_by_family: Mapping[str, Sequence[DirectedCurveOccurrence]],
    gesture_templates: Sequence[GestureTemplate],
    gesture_occurrences: Sequence[GestureOccurrence],
    gesture_programs: Sequence[GestureGlyphProgram],
    primitive_templates: Mapping[str, PrimitiveTemplate],
    *,
    fit_options: TrajectoryFitOptions,
    selection_options: TrajectoryModelSelectionOptions,
    reviewed_annotations: MovementAnnotations | None = None,
    variation_policy: TrajectoryVariationPolicy | None = None,
    max_candidates: int = 8,
) -> GestureLearningResult:
    """Run bounded raster, skeleton, family, dictionary, and glyph-compile stages.

    Family assignments and gesture occurrences are explicit training inputs. Pixel
    geometry alone is not used to invent semantic roles or verified directions.
    """
    _validate_limit(max_candidates)
    region_candidates = (
        region_candidates_from_segmentation(raster_candidates, max_candidates=max_candidates)
        if isinstance(raster_candidates, SpacedSegmentationHypotheses)
        else tuple(raster_candidates[:max_candidates])
    )
    observations = build_glyph_observation_candidates(
        source_observation_id,
        region_candidates,
        reviewed_annotations=reviewed_annotations,
        max_candidates=max_candidates,
    )

    geometry_fits = tuple(
        fit_geometry_family(
            family_id,
            tuple(sorted(occurrences, key=lambda item: item.id)),
            fit_options=fit_options,
            selection_options=selection_options,
            variation_policy=variation_policy,
        )
        for family_id, occurrences in sorted(geometry_occurrences_by_family.items())
    )
    primitive_families = {fit.family.id: fit.family for fit in geometry_fits}
    templates = tuple(sorted(gesture_templates, key=lambda item: item.id))
    occurrences = tuple(sorted(gesture_occurrences, key=lambda item: item.id))
    dictionary_fit = learn_gesture_dictionary(
        templates,
        primitive_families,
        occurrences,
        max_candidates=max_candidates,
    )
    template_map = {template.id: template for template in templates}
    compiled_programs = tuple(
        compile_gesture_program(
            program,
            template_map,
            primitive_templates,
        )
        for program in sorted(gesture_programs, key=lambda item: item.id)
    )
    joint_alternatives = _joint_alternatives(
        observations,
        occurrences,
        dictionary_fit.ranked_matches,
    )
    metrics = _pipeline_metrics(
        region_candidates,
        observations,
        geometry_fits,
        dictionary_fit,
        compiled_programs,
        joint_alternatives,
    )
    return GestureLearningResult(
        observation_candidates=observations,
        geometry_family_fits=geometry_fits,
        dictionary_fit=dictionary_fit,
        compiled_programs=compiled_programs,
        joint_alternatives=joint_alternatives,
        metrics=MappingProxyType(metrics),
    )


def _joint_alternatives(
    observations: Sequence[GlyphObservationCandidate],
    occurrences: Sequence[GestureOccurrence],
    matches: Sequence[GestureMatch],
) -> tuple[JointGestureAlternative, ...]:
    matches_by_occurrence: dict[str, list[GestureMatch]] = {}
    for match in matches:
        matches_by_occurrence.setdefault(match.occurrence_id, []).append(match)

    alternatives: list[JointGestureAlternative] = []
    for observation in observations:
        stroke_ids = tuple(stroke.id for stroke in observation.stroke_hypotheses)
        compatible_occurrence_ids = {
            occurrence.id
            for occurrence in occurrences
            if occurrence.evidence.source_observation_id == observation.source_observation_id
            and occurrence.evidence.segmentation_hypothesis_id
            == observation.segmentation_hypothesis_id
            and (
                occurrence.evidence.skeleton_stroke_hypothesis_id in stroke_ids
                or any(
                    stroke.annotation is not None
                    and stroke.annotation.id in occurrence.evidence_ids
                    for stroke in observation.stroke_hypotheses
                )
            )
        }
        candidate_matches = tuple(
            match
            for occurrence_id in sorted(compatible_occurrence_ids)
            for match in matches_by_occurrence.get(occurrence_id, ())
        )
        ambiguous = (
            observation.provenance.evidence_level == "inferred-ambiguous"
            or len(stroke_ids) > 1
            or len(candidate_matches) > 1
            or any(match.ambiguous for match in candidate_matches)
        )
        if ambiguous:
            status: AmbiguityStatus = "ambiguous"
        elif candidate_matches:
            status = (
                "reviewed"
                if observation.provenance.evidence_level == "reviewed-manual"
                else "inferred"
            )
        else:
            status = "not-recorded"
        alternatives.append(
            JointGestureAlternative(
                observation_candidate_id=observation.id,
                source_observation_id=observation.source_observation_id,
                segmentation_hypothesis_id=observation.segmentation_hypothesis_id,
                stroke_hypothesis_ids=stroke_ids,
                matches=candidate_matches,
                ambiguity_status=status,
                provenance=observation.provenance,
            )
        )
    return tuple(alternatives)


def _pipeline_metrics(
    raster_candidates: Sequence[RasterRegionCandidate],
    observations: Sequence[GlyphObservationCandidate],
    geometry_fits: Sequence[FittedGeometryFamily],
    dictionary_fit: GestureDictionaryFit,
    compiled_programs: Sequence[CompiledGlyphProgram],
    joint_alternatives: Sequence[JointGestureAlternative],
) -> dict[str, float]:
    metrics = {
        "raster_region_candidate_count": float(len(raster_candidates)),
        "observation_candidate_count": float(len(observations)),
        "skeleton_stroke_hypothesis_count": float(
            sum(len(candidate.stroke_hypotheses) for candidate in observations)
        ),
        "geometry_family_count": float(len(geometry_fits)),
        "gesture_template_count": float(len(dictionary_fit.template_fits)),
        "compiled_glyph_program_count": float(len(compiled_programs)),
        "ambiguous_observation_count": float(
            sum(item.ambiguity_status == "ambiguous" for item in joint_alternatives)
        ),
    }
    training_fits = [
        fit.family.training_metrics["mean_position_rmse"]
        for fit in geometry_fits
        if "mean_position_rmse" in fit.family.training_metrics
    ]
    held_out_fits = [
        fit.family.held_out_metrics["mean_normalized_directed_rmse"]
        for fit in geometry_fits
        if "mean_normalized_directed_rmse" in fit.family.held_out_metrics
    ]
    gesture_training_fits = [
        fit.training_metrics["mean_geometry_fit"]
        for fit in dictionary_fit.template_fits
        if "mean_geometry_fit" in fit.training_metrics
    ]
    gesture_held_out_fits = [
        fit.held_out_metrics["mean_geometry_fit"]
        for fit in dictionary_fit.template_fits
        if "mean_geometry_fit" in fit.held_out_metrics
    ]
    if training_fits:
        metrics["mean_geometry_training_position_rmse"] = mean(training_fits)
    if held_out_fits:
        metrics["mean_geometry_held_out_directed_rmse"] = mean(held_out_fits)
    if gesture_training_fits:
        metrics["mean_gesture_training_geometry_fit"] = mean(gesture_training_fits)
    if gesture_held_out_fits:
        metrics["mean_gesture_held_out_geometry_fit"] = mean(gesture_held_out_fits)
    return metrics


def _validate_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_candidates must be a positive integer")
