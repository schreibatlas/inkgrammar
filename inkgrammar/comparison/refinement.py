from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryFitSeed,
    TrajectoryRefinementOptions,
    fit_trajectory,
    refine_trajectory_model,
)
from planarcurve.trajectory import Trajectory2D, evaluate_trajectory
from planarcurve.transforms import Similarity2D
from pydantic import Field, field_validator, model_validator
from rasterregions import (
    ForegroundEvidence,
    ForegroundPyramid,
    ForegroundPyramidLevel,
    foreground_pyramid,
)

from ..fitting.objective import soft_mask_loss, symmetric_distance_loss
from ..models import StrictModel
from .models import ContinuousPath, WritingRouteCandidate

_MINIMUM_STROKE_WIDTH_PX = 1.0


@dataclass(frozen=True)
class SparseWidthChange:
    at: float
    width_px: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.at) or not 0.0 < self.at < 1.0:
            raise ValueError("width changes must be at an interior normalized position")
        if not np.isfinite(self.width_px) or self.width_px <= 0.0:
            raise ValueError("width change values must be finite and positive")


@dataclass(frozen=True)
class SparseWidthProfile:
    start_width_px: float
    end_width_px: float
    changes: tuple[SparseWidthChange, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(width) or width <= 0.0
            for width in (self.start_width_px, self.end_width_px)
        ):
            raise ValueError("width profile endpoints must be finite and positive")
        changes = tuple(self.changes)
        positions = tuple(change.at for change in changes)
        if any(right <= left for left, right in zip(positions, positions[1:], strict=False)):
            raise ValueError("width changes must be ordered by normalized position")
        object.__setattr__(self, "changes", changes)

    @classmethod
    def constant(cls, width_px: float) -> SparseWidthProfile:
        return cls(start_width_px=width_px, end_width_px=width_px)

    @property
    def width_change_count(self) -> int:
        return len(self.changes)

    @property
    def parameter_count(self) -> int:
        if self.start_width_px == self.end_width_px and not self.changes:
            return 1
        return 2 + len(self.changes)

    def width_at(self, progress: float) -> float:
        if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("width progress must be finite and between zero and one")
        positions = (0.0, *(change.at for change in self.changes), 1.0)
        widths = (
            self.start_width_px,
            *(change.width_px for change in self.changes),
            self.end_width_px,
        )
        return float(np.interp(progress, positions, widths))


@dataclass(frozen=True)
class ReconstructedStrokeModel:
    trajectory: Trajectory2D
    transform: Similarity2D
    width: SparseWidthProfile


@dataclass(frozen=True)
class ReconstructionLevelResult:
    level_index: int
    scale_x: float
    scale_y: float
    objective_before: float
    objective_after: float
    iterations: int
    accepted_geometry_changes: int
    accepted_width_changes: int

    def __post_init__(self) -> None:
        if self.level_index < 0:
            raise ValueError("reconstruction level index must be nonnegative")
        if any(not np.isfinite(value) or value <= 0.0 for value in (self.scale_x, self.scale_y)):
            raise ValueError("reconstruction level scales must be finite and positive")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (self.objective_before, self.objective_after)
        ):
            raise ValueError("reconstruction objectives must be finite and nonnegative")
        if (
            min(
                self.iterations,
                self.accepted_geometry_changes,
                self.accepted_width_changes,
            )
            < 0
        ):
            raise ValueError("reconstruction counts must be nonnegative")


@dataclass(frozen=True)
class MinimumComplexityReconstruction:
    strokes: tuple[ReconstructedStrokeModel, ...]
    levels: tuple[ReconstructionLevelResult, ...]
    rendered_coverage: np.ndarray
    energy: float
    topology_preserved: bool
    ambiguous: bool
    fit_terms: tuple[tuple[str, float], ...] = ()
    complexity_terms: tuple[tuple[str, float], ...] = ()
    objective_evaluations: int = 0

    def __post_init__(self) -> None:
        strokes = tuple(self.strokes)
        levels = tuple(self.levels)
        coverage = np.asarray(self.rendered_coverage, dtype=np.float32)
        if coverage.ndim != 2:
            raise ValueError("rendered coverage must be a two-dimensional array")
        if not np.isfinite(coverage).all() or np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError("rendered coverage must be finite and within [0, 1]")
        if not np.isfinite(self.energy) or self.energy < 0.0:
            raise ValueError("reconstruction energy must be finite and nonnegative")
        if self.objective_evaluations < 0:
            raise ValueError("objective evaluation count must be nonnegative")
        fit_terms = tuple((name, float(value)) for name, value in self.fit_terms)
        complexity_terms = tuple((name, float(value)) for name, value in self.complexity_terms)
        if any(
            not name or not np.isfinite(value) for name, value in (*fit_terms, *complexity_terms)
        ):
            raise ValueError("reconstruction score terms must have names and finite values")
        object.__setattr__(self, "fit_terms", fit_terms)
        object.__setattr__(self, "complexity_terms", complexity_terms)
        coverage = np.array(coverage, copy=True)
        coverage.setflags(write=False)
        object.__setattr__(self, "strokes", strokes)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "rendered_coverage", coverage)


class ReconstructionRefinementConfig(StrictModel):
    scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    max_route_candidates: int = Field(default=5, ge=1)
    max_geometry_changes: int = Field(default=4, ge=0)
    max_width_changes: int = Field(default=2, ge=0)
    supersampling_coarse: int = Field(default=2, ge=1)
    supersampling_fine: int = Field(default=4, ge=1)
    max_iterations_per_level: int = Field(default=80, ge=1)
    minimum_absolute_gain: float = Field(default=1e-3, ge=0.0)
    minimum_relative_gain: float = Field(default=0.01, ge=0.0)
    complexity_weight: float = Field(default=0.02, ge=0.0)
    seed_prior_weight_coarse: float = Field(default=0.1, ge=0.0)
    seed_prior_weight_fine: float = Field(default=0.01, ge=0.0)

    @field_validator("scales")
    @classmethod
    def validate_scales(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in values):
            raise ValueError("reconstruction scales must be nonempty and within (0, 1]")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("reconstruction scales must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_level_weights(self) -> ReconstructionRefinementConfig:
        if self.seed_prior_weight_fine > self.seed_prior_weight_coarse:
            raise ValueError("fine-level seed prior weight cannot exceed the coarse-level weight")
        if self.supersampling_fine < self.supersampling_coarse:
            raise ValueError(
                "fine-level supersampling cannot be lower than coarse-level supersampling"
            )
        return self


def seed_reconstructed_strokes(
    candidate: WritingRouteCandidate,
) -> tuple[ReconstructedStrokeModel, ...]:
    if len(candidate.steering_models) != len(candidate.continuous_strokes):
        raise ValueError("route candidate must have a fitted trajectory model for every stroke")

    strokes = []
    for path, model in zip(candidate.continuous_strokes, candidate.steering_models, strict=True):
        width_px = max(float(np.median(2.0 * path.radius_px)), _MINIMUM_STROKE_WIDTH_PX)
        selected = model.selected
        strokes.append(
            ReconstructedStrokeModel(
                trajectory=selected.trajectory,
                transform=selected.transform,
                width=SparseWidthProfile.constant(width_px),
            )
        )
    return tuple(strokes)


RenderAtLevel = Callable[
    [tuple[ReconstructedStrokeModel, ...], ForegroundPyramidLevel, int], np.ndarray
]


def _seed_deviation(
    strokes: tuple[ReconstructedStrokeModel, ...],
    seeds: tuple[ReconstructedStrokeModel, ...],
    normalization: float,
) -> float:
    progress = np.linspace(0.0, 1.0, 17)
    deviations = []
    for stroke, seed in zip(strokes, seeds, strict=True):
        points = evaluate_trajectory(stroke.trajectory, count=len(progress)).xy
        seed_points = evaluate_trajectory(seed.trajectory, count=len(progress)).xy
        geometry_error = np.linalg.norm(
            stroke.transform.apply(points) - seed.transform.apply(seed_points), axis=1
        ).mean()
        width_error = np.mean(
            [abs(stroke.width.width_at(u) - seed.width.width_at(u)) for u in progress]
        )
        deviations.append(float((geometry_error + width_error) / normalization))
    return float(np.mean(deviations)) if deviations else 0.0


class _RasterObjective:
    def __init__(
        self,
        candidate: WritingRouteCandidate,
        seeds: tuple[ReconstructedStrokeModel, ...],
        evidence: ForegroundEvidence,
        level_count: int,
        render_at_level: RenderAtLevel,
        config: ReconstructionRefinementConfig,
    ) -> None:
        self.candidate = candidate
        self.seeds = seeds
        self.evidence = evidence
        self.level_count = level_count
        self.render_at_level = render_at_level
        self.config = config
        self.pen_lift_count = sum(event.kind == "pen-lift" for event in candidate.events)
        self.normalization = max(float(np.hypot(*evidence.mask.shape)), 1.0)
        self.evaluations = 0

    def evaluate(
        self,
        strokes: tuple[ReconstructedStrokeModel, ...],
        level: ForegroundPyramidLevel,
    ) -> tuple[float, dict[str, float], np.ndarray]:
        self.evaluations += 1
        generated = np.asarray(
            self.render_at_level(
                strokes,
                level,
                self.config.supersampling_coarse
                if level.index == 0
                else self.config.supersampling_fine,
            ),
            dtype=np.float32,
        )
        if generated.shape != level.shape:
            raise ValueError("rendered level coverage must match the foreground pyramid shape")
        if not np.isfinite(generated).all() or np.any((generated < 0.0) | (generated > 1.0)):
            raise ValueError("rendered level coverage must be finite and within [0, 1]")

        soft = soft_mask_loss(level.coverage, generated)
        distance = symmetric_distance_loss(level.coverage, generated) if level.index > 0 else 0.0
        seed_weight = (
            self.config.seed_prior_weight_fine
            if self.level_count == 1
            else self.config.seed_prior_weight_coarse
            + (self.config.seed_prior_weight_fine - self.config.seed_prior_weight_coarse)
            * level.index
            / (self.level_count - 1)
        )
        seed_deviation = _seed_deviation(strokes, self.seeds, self.normalization)
        geometry_changes = sum(len(stroke.trajectory.heading_rate.changes) for stroke in strokes)
        width_parameters = sum(stroke.width.parameter_count - 1 for stroke in strokes)
        terms = {
            "soft_coverage": float(soft),
            "distance": 0.25 * float(distance),
            "seed_deviation": seed_weight * seed_deviation,
            "geometry_complexity": self.config.complexity_weight * geometry_changes,
            "width_complexity": self.config.complexity_weight * width_parameters,
            "pen_lift_complexity": self.config.complexity_weight * self.pen_lift_count,
        }
        return float(sum(terms.values())), terms, generated

    def score(
        self,
        strokes: tuple[ReconstructedStrokeModel, ...],
        level: ForegroundPyramidLevel,
    ) -> float:
        return self.evaluate(strokes, level)[0]


def _gain_is_sufficient(
    before: float,
    after: float,
    config: ReconstructionRefinementConfig,
) -> bool:
    absolute_gain = before - after
    relative_gain = absolute_gain / max(abs(before), np.finfo(float).eps)
    return (
        absolute_gain >= config.minimum_absolute_gain
        and relative_gain >= config.minimum_relative_gain
    )


def _fit_smoothing_window(point_count: int) -> int:
    window = min(7, point_count if point_count % 2 else point_count - 1)
    return max(window, 1)


def _refit_existing_trajectories(
    candidate: WritingRouteCandidate,
    strokes: tuple[ReconstructedStrokeModel, ...],
    level: ForegroundPyramidLevel,
    objective: _RasterObjective,
    config: ReconstructionRefinementConfig,
) -> tuple[ReconstructedStrokeModel, ...]:
    refined = strokes
    current_score = objective.score(refined, level)
    for index, path in enumerate(candidate.continuous_strokes):
        stroke = refined[index]
        change_count = len(stroke.trajectory.heading_rate.changes)
        options = TrajectoryFitOptions(
            change_count=change_count,
            samples=max(32, len(path.xy)),
            restarts=0,
            smoothing_window=_fit_smoothing_window(len(path.xy)),
            max_nfev=config.max_iterations_per_level,
        )
        fit = fit_trajectory(
            path.xy,
            options,
            initial=TrajectoryFitSeed(stroke.trajectory, stroke.transform),
        )
        if not fit.success:
            continue
        proposed = list(refined)
        proposed[index] = replace(stroke, trajectory=fit.trajectory, transform=fit.transform)
        proposed_tuple = tuple(proposed)
        proposed_score = objective.score(proposed_tuple, level)
        if proposed_score < current_score - 1e-12:
            refined = proposed_tuple
            current_score = proposed_score
    return refined


def _fit_placement_and_width(
    strokes: tuple[ReconstructedStrokeModel, ...],
    level: ForegroundPyramidLevel,
    objective: _RasterObjective,
    config: ReconstructionRefinementConfig,
) -> tuple[tuple[ReconstructedStrokeModel, ...], int, int]:
    def centroid(values: np.ndarray) -> tuple[float, float] | None:
        mass = float(values.sum())
        if mass <= 1e-12:
            return None
        yy, xx = np.indices(values.shape, dtype=float)
        return float((xx * values).sum() / mass), float((yy * values).sum() / mass)

    before_score, _, rendered = objective.evaluate(strokes, level)
    proposals: list[tuple[ReconstructedStrokeModel, ...]] = []
    observed_center = centroid(level.coverage)
    rendered_center = centroid(rendered)
    if observed_center is not None and rendered_center is not None:
        source_height, source_width = level.source_shape
        shift_x = float(
            np.clip(
                (observed_center[0] - rendered_center[0]) / level.scale_x,
                -0.25 * source_width,
                0.25 * source_width,
            )
        )
        shift_y = float(
            np.clip(
                (observed_center[1] - rendered_center[1]) / level.scale_y,
                -0.25 * source_height,
                0.25 * source_height,
            )
        )
        if abs(shift_x) > 0.25 or abs(shift_y) > 0.25:
            proposals.append(
                tuple(
                    replace(
                        stroke,
                        transform=replace(
                            stroke.transform,
                            dx=stroke.transform.dx + shift_x,
                            dy=stroke.transform.dy + shift_y,
                        ),
                    )
                    for stroke in strokes
                )
            )

    rendered_mass = float(rendered.sum())
    observed_mass = float(level.coverage.sum())
    if rendered_mass > 1e-12 and observed_mass > 0.0:
        width_factor = float(np.clip(np.sqrt(observed_mass / rendered_mass), 0.5, 2.0))
        if abs(width_factor - 1.0) > 0.02:
            proposals.append(
                tuple(
                    replace(
                        stroke,
                        width=replace(
                            stroke.width,
                            start_width_px=stroke.width.start_width_px * width_factor,
                            end_width_px=stroke.width.end_width_px * width_factor,
                            changes=tuple(
                                replace(change, width_px=change.width_px * width_factor)
                                for change in stroke.width.changes
                            ),
                        ),
                    )
                    for stroke in strokes
                )
            )

    best = strokes
    best_score = before_score
    for proposal in proposals:
        proposal_score = objective.score(proposal, level)
        if proposal_score < best_score:
            best = proposal
            best_score = proposal_score

    before_width_parameters = sum(stroke.width.parameter_count - 1 for stroke in strokes)
    after_width_parameters = sum(stroke.width.parameter_count - 1 for stroke in best)
    added_width_parameters = max(0, after_width_parameters - before_width_parameters)
    accepted = (
        _gain_is_sufficient(before_score, best_score, config)
        if added_width_parameters
        else best_score <= before_score + 1e-12
    )
    if not accepted:
        return strokes, len(proposals), 0
    return best, len(proposals), added_width_parameters


def _promote_trajectory_changes(
    candidate: WritingRouteCandidate,
    strokes: tuple[ReconstructedStrokeModel, ...],
    level: ForegroundPyramidLevel,
    objective: _RasterObjective,
    config: ReconstructionRefinementConfig,
) -> tuple[tuple[ReconstructedStrokeModel, ...], int]:
    refined = strokes
    accepted_changes = 0
    while (
        sum(len(stroke.trajectory.heading_rate.changes) for stroke in refined)
        < config.max_geometry_changes
    ):
        before_score = objective.score(refined, level)
        best_proposal: tuple[float, tuple[ReconstructedStrokeModel, ...]] | None = None
        for index, path in enumerate(candidate.continuous_strokes):
            current_count = len(refined[index].trajectory.heading_rate.changes)
            if current_count >= config.max_geometry_changes:
                continue
            fit_options = TrajectoryFitOptions(
                change_count=0,
                samples=max(32, len(path.xy)),
                restarts=0,
                smoothing_window=_fit_smoothing_window(len(path.xy)),
                max_nfev=config.max_iterations_per_level,
            )
            refinement_options = TrajectoryRefinementOptions(
                max_change_count=current_count + 1,
                candidate_insertions=3,
                residual_samples=max(32, min(96, len(path.xy))),
                minimum_absolute_gain=0.0,
                minimum_relative_gain=0.0,
                complexity_weight=config.complexity_weight,
            )
            refinement = refine_trajectory_model(
                path.xy,
                fit_options=fit_options,
                refinement_options=refinement_options,
            )
            fit = refinement.selected
            if not fit.success or refinement.selected_change_count <= current_count:
                continue
            proposed = list(refined)
            proposed[index] = replace(
                refined[index], trajectory=fit.trajectory, transform=fit.transform
            )
            proposed_tuple = tuple(proposed)
            proposed_score = objective.score(proposed_tuple, level)
            if not _gain_is_sufficient(before_score, proposed_score, config):
                continue
            if best_proposal is None or proposed_score < best_proposal[0]:
                best_proposal = proposed_score, proposed_tuple
        if best_proposal is None:
            break
        refined = best_proposal[1]
        accepted_changes += 1
    return refined, accepted_changes


def _propose_width_change(
    path: ContinuousPath,
    profile: SparseWidthProfile,
) -> SparseWidthChange | None:
    progress = path.progress
    if len(progress) < 2:
        return None
    measured_width = np.maximum(2.0 * path.radius_px, _MINIMUM_STROKE_WIDTH_PX)
    residual = measured_width - np.asarray([profile.width_at(float(u)) for u in progress])
    window = min(5, len(residual) if len(residual) % 2 else len(residual) - 1)
    if window > 1:
        kernel = np.full(window, 1.0 / window)
        half_window = window // 2
        residual = np.convolve(
            np.pad(residual, (half_window, half_window), mode="edge"),
            kernel,
            mode="valid",
        )
    else:
        half_window = 0
    eligible = (progress > 0.05) & (progress < 0.95)
    for change in profile.changes:
        eligible &= np.abs(progress - change.at) >= 0.05
    if not np.any(eligible):
        return None
    scores = np.where(eligible, np.abs(residual), -np.inf)
    index = int(np.argmax(scores))
    if not np.isfinite(scores[index]) or scores[index] <= 1e-6:
        return None
    start = max(0, index - half_window)
    end = min(len(measured_width), index + half_window + 1)
    width_px = max(float(np.median(measured_width[start:end])), _MINIMUM_STROKE_WIDTH_PX)
    return SparseWidthChange(float(progress[index]), width_px)


def _promote_width_changes(
    candidate: WritingRouteCandidate,
    strokes: tuple[ReconstructedStrokeModel, ...],
    level: ForegroundPyramidLevel,
    objective: _RasterObjective,
    config: ReconstructionRefinementConfig,
) -> tuple[tuple[ReconstructedStrokeModel, ...], int]:
    refined = strokes
    accepted_changes = 0
    while sum(stroke.width.width_change_count for stroke in refined) < config.max_width_changes:
        before_score = objective.score(refined, level)
        best_proposal: tuple[float, tuple[ReconstructedStrokeModel, ...]] | None = None
        for index, path in enumerate(candidate.continuous_strokes):
            profile = refined[index].width
            if profile.width_change_count >= config.max_width_changes:
                continue
            change = _propose_width_change(path, profile)
            if change is None:
                continue
            changes = tuple(sorted((*profile.changes, change), key=lambda item: item.at))
            proposed = list(refined)
            proposed[index] = replace(refined[index], width=replace(profile, changes=changes))
            proposed_tuple = tuple(proposed)
            proposed_score = objective.score(proposed_tuple, level)
            if not _gain_is_sufficient(before_score, proposed_score, config):
                continue
            if best_proposal is None or proposed_score < best_proposal[0]:
                best_proposal = proposed_score, proposed_tuple
        if best_proposal is None:
            break
        refined = best_proposal[1]
        accepted_changes += 1
    return refined, accepted_changes


def fit_minimum_complexity_reconstruction(
    candidate: WritingRouteCandidate,
    evidence: ForegroundEvidence,
    render_at_level: RenderAtLevel,
    *,
    config: ReconstructionRefinementConfig | None = None,
    pyramid: ForegroundPyramid | None = None,
    progress_callback: Callable[..., None] | None = None,
    candidate_index: int = 1,
    candidate_count: int = 1,
) -> MinimumComplexityReconstruction:
    config = config or ReconstructionRefinementConfig()
    seeds = seed_reconstructed_strokes(candidate)
    if not seeds:
        raise ValueError("route candidate must contain at least one stroke")
    if pyramid is None:
        if progress_callback is not None:
            progress_callback(
                "pyramid_construction",
                completed=1,
                total=1,
                details={"message": "Build foreground pyramid"},
            )
        pyramid = foreground_pyramid(evidence, scales=config.scales)
    elif pyramid.evidence is not evidence:
        raise ValueError("cached foreground pyramid must use the reconstruction evidence object")
    objective = _RasterObjective(
        candidate,
        seeds,
        evidence,
        len(pyramid.levels),
        render_at_level,
        config,
    )
    strokes = seeds
    level_results = []
    final_terms: dict[str, float] = {}
    final_coverage = np.zeros_like(evidence.mask, dtype=np.float32)
    final_energy = 0.0

    for level in pyramid.levels:
        if progress_callback is not None:
            progress_callback(
                "reconstruction_level",
                completed=level.index + 1,
                total=len(pyramid.levels),
                details={
                    "message": (
                        f"Candidate {candidate_index}/{candidate_count}: "
                        f"fit pyramid level {level.index + 1}/{len(pyramid.levels)}"
                    ),
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "level_index": level.index,
                    "objective_evaluations": objective.evaluations,
                },
            )
        objective_before = objective.score(strokes, level)
        accepted_geometry_changes = 0
        accepted_width_changes = 0
        if level.index > 0:
            strokes = _refit_existing_trajectories(candidate, strokes, level, objective, config)
        strokes, iterations, width_parameters = _fit_placement_and_width(
            strokes, level, objective, config
        )
        accepted_width_changes += width_parameters
        if level.index == len(pyramid.levels) - 1:
            strokes, accepted_geometry_changes = _promote_trajectory_changes(
                candidate, strokes, level, objective, config
            )
            strokes, sparse_width_changes = _promote_width_changes(
                candidate, strokes, level, objective, config
            )
            accepted_width_changes += sparse_width_changes
        final_energy, final_terms, final_coverage = objective.evaluate(strokes, level)
        if final_energy > objective_before + 1e-9:
            raise RuntimeError("accepted reconstruction changes regressed the level objective")
        level_results.append(
            ReconstructionLevelResult(
                level_index=level.index,
                scale_x=level.scale_x,
                scale_y=level.scale_y,
                objective_before=objective_before,
                objective_after=final_energy,
                iterations=iterations,
                accepted_geometry_changes=accepted_geometry_changes,
                accepted_width_changes=accepted_width_changes,
            )
        )
        if progress_callback is not None:
            progress_callback(
                "reconstruction_level",
                completed=level.index + 1,
                total=len(pyramid.levels),
                details={
                    "message": (
                        f"Completed candidate {candidate_index}/{candidate_count}, "
                        f"pyramid level {level.index + 1}/{len(pyramid.levels)}"
                    ),
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "level_index": level.index,
                    "objective": final_energy,
                    "objective_evaluations": objective.evaluations,
                },
            )

    fit_terms = tuple(
        (name, value)
        for name, value in final_terms.items()
        if name in {"soft_coverage", "distance", "seed_deviation"}
    )
    complexity_terms = tuple(
        (name, value) for name, value in final_terms.items() if name.endswith("_complexity")
    )
    return MinimumComplexityReconstruction(
        strokes=strokes,
        levels=tuple(level_results),
        rendered_coverage=final_coverage,
        energy=final_energy,
        topology_preserved=True,
        ambiguous=candidate.ambiguous or candidate.orientation_status == "ambiguous",
        fit_terms=fit_terms,
        complexity_terms=complexity_terms,
        objective_evaluations=objective.evaluations,
    )
