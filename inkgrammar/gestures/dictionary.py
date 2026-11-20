from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from statistics import median
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .evidence import EvidenceLevel, OccurrenceEvidence
from .geometry import GeometryPrimitiveFamily
from .geometry_fit import normalized_family_geometry_error
from .models import GestureParameter, GestureTemplate

Point = tuple[float, float]
TopologyCompatibility = Literal["compatible", "incompatible", "unknown"]


@dataclass(frozen=True)
class GestureOccurrence:
    id: str
    primitive_curves: Mapping[str, ArrayLike]
    evidence: OccurrenceEvidence
    split: Literal["training", "held-out"] = "training"
    parameter_values: Mapping[str, float] | None = None
    constraint_residuals: Mapping[str, float] | None = None
    topology_compatibility: TopologyCompatibility = "unknown"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("gesture occurrence ID must not be blank")
        if self.split not in {"training", "held-out"}:
            raise ValueError("gesture occurrence split must be training or held-out")
        if self.topology_compatibility not in {"compatible", "incompatible", "unknown"}:
            raise ValueError("invalid topology compatibility status")
        curves: dict[str, tuple[Point, ...]] = {}
        for slot_id, points in self.primitive_curves.items():
            values = np.asarray(points, dtype=np.float64)
            if not slot_id.strip() or values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
                raise ValueError("primitive curves require nonblank slot IDs and shape (N, 2)")
            if not np.isfinite(values).all():
                raise ValueError("primitive curve points must be finite")
            curves[slot_id] = tuple((float(x), float(y)) for x, y in values)
        parameters = {key: float(value) for key, value in (self.parameter_values or {}).items()}
        residuals = {key: float(value) for key, value in (self.constraint_residuals or {}).items()}
        if any(not key.strip() for key in (*parameters, *residuals)):
            raise ValueError("gesture parameter and constraint IDs must not be blank")
        if any(not isfinite(value) for value in parameters.values()):
            raise ValueError("gesture parameter values must be finite")
        if any(not isfinite(value) or value < 0.0 for value in residuals.values()):
            raise ValueError("gesture constraint residuals must be finite and nonnegative")
        evidence_ids = tuple(self.evidence_ids)
        if any(not value.strip() for value in evidence_ids):
            raise ValueError("gesture occurrence evidence IDs must not be blank")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("gesture occurrence evidence IDs must be unique")
        object.__setattr__(self, "primitive_curves", MappingProxyType(curves))
        object.__setattr__(self, "parameter_values", MappingProxyType(parameters))
        object.__setattr__(self, "constraint_residuals", MappingProxyType(residuals))
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class GestureMatchBreakdown:
    geometry_fit: float
    constraint_fit: float | None
    topology_compatibility: TopologyCompatibility
    parameter_deviation: float | None
    complexity: int
    evidence_level: EvidenceLevel
    evidence_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = [self.geometry_fit, self.complexity]
        if self.constraint_fit is not None:
            values.append(self.constraint_fit)
        if self.parameter_deviation is not None:
            values.append(self.parameter_deviation)
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("gesture match metrics must be finite and nonnegative")
        if self.complexity < 0:
            raise ValueError("gesture match complexity must be nonnegative")
        if any(not value.strip() for value in self.evidence_source_ids):
            raise ValueError("gesture match evidence IDs must not be blank")
        if len(self.evidence_source_ids) != len(set(self.evidence_source_ids)):
            raise ValueError("gesture match evidence IDs must be unique")


@dataclass(frozen=True)
class GestureMatch:
    template_id: str
    occurrence_id: str
    breakdown: GestureMatchBreakdown
    rank: int = 0
    ambiguous: bool = False


@dataclass(frozen=True)
class GestureTemplateFit:
    template: GestureTemplate
    fitted_parameter_values: Mapping[str, float]
    matches: tuple[GestureMatch, ...]
    primitive_reuse_counts: Mapping[str, int]
    training_metrics: Mapping[str, float]
    held_out_metrics: Mapping[str, float]


@dataclass(frozen=True)
class GestureDictionaryFit:
    template_fits: tuple[GestureTemplateFit, ...]
    ranked_matches: tuple[GestureMatch, ...]
    family_reuse_counts: Mapping[str, int]


def fit_gesture_template(
    template: GestureTemplate,
    primitive_families: Mapping[str, GeometryPrimitiveFamily],
    occurrences: Sequence[GestureOccurrence],
    *,
    samples: int = 64,
) -> GestureTemplateFit:
    """Fit bounded template parameters and score directed occurrences by family reuse."""
    occurrence_values = tuple(occurrences)
    if not occurrence_values:
        raise ValueError("gesture template fitting requires occurrences")
    if len({item.id for item in occurrence_values}) != len(occurrence_values):
        raise ValueError("gesture occurrence IDs must be unique")
    training = tuple(item for item in occurrence_values if item.split == "training")
    if not training:
        raise ValueError("gesture template fitting requires a training occurrence")
    for slot in template.primitives:
        if slot.primitive_family not in primitive_families:
            raise ValueError(f"unknown primitive family: {slot.primitive_family}")

    fitted_values = _fit_bounded_parameters(template.parameters, training)
    fitted_parameters = [
        parameter.model_copy(update={"initial": fitted_values[parameter.id]})
        for parameter in template.parameters
    ]
    fitted_template = template.model_copy(update={"parameters": fitted_parameters})
    matches = tuple(
        _score_occurrence(
            fitted_template,
            primitive_families,
            occurrence,
            fitted_values,
            samples=samples,
        )
        for occurrence in occurrence_values
    )
    training_matches = [
        match
        for match, occurrence in zip(matches, occurrence_values, strict=True)
        if occurrence.split == "training"
    ]
    held_out_matches = [
        match
        for match, occurrence in zip(matches, occurrence_values, strict=True)
        if occurrence.split == "held-out"
    ]
    training_metrics = _aggregate_match_metrics(training_matches)
    held_out_metrics = _aggregate_match_metrics(held_out_matches)
    reuse = Counter(slot.primitive_family for slot in template.primitives)
    return GestureTemplateFit(
        template=fitted_template,
        fitted_parameter_values=MappingProxyType(fitted_values),
        matches=matches,
        primitive_reuse_counts=MappingProxyType(dict(sorted(reuse.items()))),
        training_metrics=MappingProxyType(training_metrics),
        held_out_metrics=MappingProxyType(held_out_metrics),
    )


def match_gesture_candidates(
    templates: Sequence[GestureTemplate],
    primitive_families: Mapping[str, GeometryPrimitiveFamily],
    occurrences: Sequence[GestureOccurrence],
    *,
    max_candidates: int = 8,
    samples: int = 64,
) -> tuple[GestureMatch, ...]:
    """Return deterministic top-K template matches for each occurrence."""
    _validate_limit(max_candidates)
    template_values = tuple(templates)
    template_ids = [template.id for template in template_values]
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("gesture template IDs must be unique")
    matches = [
        match
        for template in template_values
        for match in fit_gesture_template(
            template,
            primitive_families,
            occurrences,
            samples=samples,
        ).matches
    ]
    return rank_gesture_matches(matches, max_candidates=max_candidates)


def rank_gesture_matches(
    matches: Sequence[GestureMatch],
    *,
    max_candidates: int = 8,
) -> tuple[GestureMatch, ...]:
    """Rank matches lexicographically while retaining ties as explicit ambiguity."""
    _validate_limit(max_candidates)
    grouped: dict[str, list[GestureMatch]] = {}
    for match in matches:
        grouped.setdefault(match.occurrence_id, []).append(match)
    match_ids = [(match.occurrence_id, match.template_id) for match in matches]
    if len(match_ids) != len(set(match_ids)):
        raise ValueError("gesture occurrence/template match pairs must be unique")

    ranked: list[GestureMatch] = []
    for occurrence_id in sorted(grouped):
        options = sorted(grouped[occurrence_id], key=_ranking_key)
        retained = options[:max_candidates]
        for rank, match in enumerate(retained, start=1):
            comparison_key = _comparison_key(match)
            ambiguous = any(
                other is not match and _comparison_key(other) == comparison_key
                for other in retained
            )
            ranked.append(replace(match, rank=rank, ambiguous=ambiguous))
    return tuple(ranked)


def learn_gesture_dictionary(
    templates: Sequence[GestureTemplate],
    primitive_families: Mapping[str, GeometryPrimitiveFamily],
    occurrences: Sequence[GestureOccurrence],
    *,
    max_candidates: int = 8,
    samples: int = 64,
) -> GestureDictionaryFit:
    """Fit seeded semantic templates and retain bounded alternatives per occurrence."""
    template_values = tuple(templates)
    template_ids = [template.id for template in template_values]
    if not template_values or len(template_ids) != len(set(template_ids)):
        raise ValueError("gesture dictionary requires uniquely identified templates")
    fits = tuple(
        fit_gesture_template(
            template,
            primitive_families,
            occurrences,
            samples=samples,
        )
        for template in template_values
    )
    all_matches = tuple(match for fit in fits for match in fit.matches)
    reuse = Counter(slot.primitive_family for fit in fits for slot in fit.template.primitives)
    return GestureDictionaryFit(
        template_fits=fits,
        ranked_matches=rank_gesture_matches(
            all_matches,
            max_candidates=max_candidates,
        ),
        family_reuse_counts=MappingProxyType(dict(sorted(reuse.items()))),
    )


def _score_occurrence(
    template: GestureTemplate,
    primitive_families: Mapping[str, GeometryPrimitiveFamily],
    occurrence: GestureOccurrence,
    fitted_values: Mapping[str, float],
    *,
    samples: int,
) -> GestureMatch:
    parameter_ids = {parameter.id for parameter in template.parameters}
    constraint_ids = {constraint.id for constraint in template.constraints}
    unknown_parameters = set(occurrence.parameter_values) - parameter_ids
    unknown_constraints = set(occurrence.constraint_residuals) - constraint_ids
    if unknown_parameters:
        raise ValueError(
            f"occurrence reports unknown gesture parameters: {sorted(unknown_parameters)}"
        )
    if unknown_constraints:
        raise ValueError(
            f"occurrence reports unknown gesture constraints: {sorted(unknown_constraints)}"
        )

    geometry_errors = []
    for slot in template.primitives:
        points = occurrence.primitive_curves.get(slot.id)
        if points is None:
            raise ValueError(f"occurrence is missing primitive curve for slot: {slot.id}")
        geometry_errors.append(
            normalized_family_geometry_error(
                primitive_families[slot.primitive_family],
                points,
                samples=samples,
            )
        )
    constraint_fit = (
        float(np.mean(list(occurrence.constraint_residuals.values())))
        if occurrence.constraint_residuals
        else None
    )
    parameter_deviations = []
    parameters = {parameter.id: parameter for parameter in template.parameters}
    for parameter_id, value in occurrence.parameter_values.items():
        parameter = parameters[parameter_id]
        if not parameter.lower <= value <= parameter.upper:
            raise ValueError(f"occurrence parameter is outside bounds: {parameter_id}")
        parameter_deviations.append(
            abs(value - fitted_values[parameter_id]) / (parameter.upper - parameter.lower)
        )
    parameter_deviation = float(np.mean(parameter_deviations)) if parameter_deviations else None
    evidence = occurrence.evidence
    evidence_ids = tuple(
        dict.fromkeys(
            source_id
            for source_id in (
                evidence.source_observation_id,
                evidence.segmentation_hypothesis_id,
                evidence.skeleton_stroke_hypothesis_id,
                *occurrence.evidence_ids,
            )
            if source_id is not None
        )
    )
    return GestureMatch(
        template_id=template.id,
        occurrence_id=occurrence.id,
        breakdown=GestureMatchBreakdown(
            geometry_fit=float(np.mean(geometry_errors)),
            constraint_fit=constraint_fit,
            topology_compatibility=occurrence.topology_compatibility,
            parameter_deviation=parameter_deviation,
            complexity=len(template.primitives) + len(template.parameters),
            evidence_level=evidence.evidence_level,
            evidence_source_ids=evidence_ids,
        ),
    )


def _fit_bounded_parameters(
    parameters: Sequence[GestureParameter],
    training: Sequence[GestureOccurrence],
) -> dict[str, float]:
    parameter_ids = {parameter.id for parameter in parameters}
    for occurrence in training:
        unknown = set(occurrence.parameter_values) - parameter_ids
        if unknown:
            raise ValueError(f"occurrence reports unknown gesture parameters: {sorted(unknown)}")
        for parameter in parameters:
            value = occurrence.parameter_values.get(parameter.id)
            if value is not None and not parameter.lower <= value <= parameter.upper:
                raise ValueError(f"occurrence parameter is outside bounds: {parameter.id}")

    groups: dict[str, list[GestureParameter]] = {}
    for parameter in parameters:
        groups.setdefault(parameter.shared_group or parameter.id, []).append(parameter)
    fitted: dict[str, float] = {}
    for group in groups.values():
        lower = max(parameter.lower for parameter in group)
        upper = min(parameter.upper for parameter in group)
        if lower > upper:
            raise ValueError("shared gesture parameter bounds do not overlap")
        observed = [
            occurrence.parameter_values[parameter.id]
            for occurrence in training
            for parameter in group
            if parameter.id in occurrence.parameter_values
        ]
        center = median(observed) if observed else median(parameter.initial for parameter in group)
        value = min(upper, max(lower, center))
        fitted.update({parameter.id: value for parameter in group})
    return fitted


def _aggregate_match_metrics(matches: Sequence[GestureMatch]) -> dict[str, float]:
    if not matches:
        return {}
    metrics = {
        "mean_geometry_fit": float(np.mean([item.breakdown.geometry_fit for item in matches])),
        "occurrence_count": float(len(matches)),
    }
    constraint_values = [
        item.breakdown.constraint_fit
        for item in matches
        if item.breakdown.constraint_fit is not None
    ]
    parameter_values = [
        item.breakdown.parameter_deviation
        for item in matches
        if item.breakdown.parameter_deviation is not None
    ]
    if constraint_values:
        metrics["mean_constraint_fit"] = float(np.mean(constraint_values))
    if parameter_values:
        metrics["mean_parameter_deviation"] = float(np.mean(parameter_values))
    return metrics


def _ranking_key(match: GestureMatch) -> tuple[object, ...]:
    compatibility = {"compatible": 0, "unknown": 1, "incompatible": 2}
    return (
        compatibility[match.breakdown.topology_compatibility],
        match.breakdown.geometry_fit,
        float("inf") if match.breakdown.constraint_fit is None else match.breakdown.constraint_fit,
        float("inf")
        if match.breakdown.parameter_deviation is None
        else match.breakdown.parameter_deviation,
        match.breakdown.complexity,
        match.template_id,
    )


def _comparison_key(match: GestureMatch) -> tuple[object, ...]:
    return _ranking_key(match)[:-1]


def _validate_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_candidates must be a positive integer")
