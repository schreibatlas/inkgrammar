from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator

from ..models import StrictModel
from .models import DirectedGlyphProgram, ProgramProvenance
from .objective import ProgramFitEnergy, ProgramObjective, ProgramObjectiveWeights
from .observation import GlyphObservation, prepare_glyph_observation
from .parameters import resolve_program_parameter_values

FitStage = Literal["alignment", "geometry", "curvature", "width", "joint"]


class DirectedFitConfig(StrictModel):
    max_iterations_per_stage: int = Field(default=20, ge=1, le=500)
    seeds: tuple[int, ...] = Field(default=(0, 1), min_length=1)
    seed_perturbation: float = Field(default=0.12, ge=0.0, le=0.5)
    samples_per_movement: int = Field(default=48, ge=2)
    supersampling: int = Field(default=3, ge=1)
    tolerance: float = Field(default=1e-6, gt=0.0, allow_inf_nan=False)
    analyze_skeleton: bool = False
    objective_weights: ProgramObjectiveWeights = Field(default_factory=ProgramObjectiveWeights)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(seed < 0 for seed in value):
            raise ValueError("fit seeds must be nonnegative integers")
        if len(value) != len(set(value)):
            raise ValueError("fit seeds must be unique")
        return value


class ProgramFitStageResult(StrictModel):
    stage: FitStage
    objective: float = Field(ge=0.0, allow_inf_nan=False)
    iterations: int = Field(ge=0)
    success: bool


class DirectedGlyphFitResult(StrictModel):
    program_id: str
    character: str
    success: bool
    parameters: dict[str, float]
    energy: ProgramFitEnergy
    initial_energy: float = Field(ge=0.0, allow_inf_nan=False)
    stages: list[ProgramFitStageResult]
    topology_preserved: bool
    route_ambiguous: bool
    direction_ambiguous: bool
    direction_uncertainty_note: str
    provenance: ProgramProvenance


def _parameter_values(
    program: DirectedGlyphProgram,
    initial_values: Mapping[str, float] | None,
) -> dict[str, float]:
    return resolve_program_parameter_values(program, initial_values)


def _synchronize_shared(program: DirectedGlyphProgram, vector: np.ndarray) -> np.ndarray:
    indexes_by_group: dict[str, list[int]] = {}
    for index, parameter in enumerate(program.parameters):
        if parameter.shared_group is not None:
            indexes_by_group.setdefault(parameter.shared_group, []).append(index)
    for indexes in indexes_by_group.values():
        vector[indexes] = vector[indexes[0]]
    return vector


def _topology_signature(program: DirectedGlyphProgram) -> tuple[object, ...]:
    return (
        tuple((stroke.id, tuple(stroke.movements)) for stroke in program.strokes),
        tuple(landmark.model_dump_json() for landmark in program.landmarks.values()),
        tuple(
            (
                movement.id,
                movement.primitive,
                movement.direction,
                movement.from_landmark,
                movement.to_landmark,
                movement.role,
                movement.attach,
                movement.width_profile,
            )
            for movement in program.movements.values()
        ),
        tuple(
            (
                contact.id,
                contact.first_movement,
                contact.second_movement,
                contact.mode,
            )
            for contact in program.contacts
        ),
        tuple(constraint.model_dump_json() for constraint in program.constraints),
    )


def _stage_indices(program: DirectedGlyphProgram, stage: FitStage) -> list[int]:
    if stage == "joint":
        selected = list(range(len(program.parameters)))
    else:
        selected = []
        for index, parameter in enumerate(program.parameters):
            if stage == "alignment" and parameter.category in {
                "global-alignment",
                "style-shared",
            }:
                selected.append(index)
            elif stage == "geometry" and parameter.category == "movement-geometry":
                if ".curvature." not in parameter.target:
                    selected.append(index)
            elif (
                stage == "curvature"
                and (
                    (
                        parameter.category == "movement-geometry"
                        and ".curvature." in parameter.target
                    )
                    or parameter.category == "local-occurrence"
                )
            ) or (stage == "width" and parameter.category == "width"):
                selected.append(index)
    representatives: list[int] = []
    seen_groups: set[str] = set()
    for index in selected:
        group = program.parameters[index].shared_group
        if group is not None and group in seen_groups:
            continue
        representatives.append(index)
        if group is not None:
            seen_groups.add(group)
    return representatives


def fit_known_glyph(
    observation: Image.Image | np.ndarray | GlyphObservation,
    program: DirectedGlyphProgram,
    primitive_templates: Mapping[str, object],
    *,
    initial_values: Mapping[str, float] | None = None,
    config: DirectedFitConfig | None = None,
) -> DirectedGlyphFitResult:
    """Fit bounded numeric parameters while retaining the declared program topology."""
    from scipy.optimize import minimize

    settings = config or DirectedFitConfig()
    prepared = prepare_glyph_observation(
        observation,
        analyze_skeleton=settings.analyze_skeleton,
    )
    objective = ProgramObjective(
        program=program,
        primitive_templates=primitive_templates,
        observation=prepared,
        weights=settings.objective_weights,
        samples_per_movement=settings.samples_per_movement,
        supersampling=settings.supersampling,
    )
    definitions = tuple(program.parameters)
    base_values = _parameter_values(program, initial_values)
    base = np.asarray([base_values[parameter.id] for parameter in definitions], dtype=float)
    lower = np.asarray([parameter.lower for parameter in definitions], dtype=float)
    upper = np.asarray([parameter.upper for parameter in definitions], dtype=float)
    step_sizes = np.maximum((upper - lower) * 0.02, 1e-4)
    for index, parameter in enumerate(definitions):
        if parameter.category == "width":
            step_sizes[index] = max(step_sizes[index], 0.02)
    initial_energy, _ = objective.evaluate(base_values)
    initial_total = initial_energy.total
    signature = _topology_signature(program)
    best_vector = base.copy()
    best_energy = initial_energy
    best_stages: list[ProgramFitStageResult] = []

    for seed_index, seed in enumerate(settings.seeds):
        vector = base.copy()
        if seed_index:
            random = np.random.default_rng(seed)
            vector += (
                random.uniform(-1.0, 1.0, size=len(vector))
                * (upper - lower)
                * settings.seed_perturbation
            )
            vector = np.clip(vector, lower, upper)
        vector = _synchronize_shared(program, vector)
        stage_results: list[ProgramFitStageResult] = []
        for stage in ("alignment", "geometry", "curvature", "width", "joint"):
            indices = _stage_indices(program, stage)
            if not indices:
                continue
            fixed = vector.copy()
            fixed_values = {
                parameter.id: float(value)
                for parameter, value in zip(definitions, fixed, strict=True)
            }
            fixed_energy, _ = objective.evaluate(fixed_values)

            def score(
                active_values: np.ndarray,
                *,
                fixed_values: np.ndarray = fixed,
                active_indices: list[int] = indices,
            ) -> float:
                candidate = fixed_values.copy()
                candidate[active_indices] = active_values
                candidate = _synchronize_shared(program, candidate)
                values = {
                    parameter.id: float(value)
                    for parameter, value in zip(definitions, candidate, strict=True)
                }
                try:
                    energy, _ = objective.evaluate(values)
                except ValueError:
                    return 1e6
                return energy.total

            optimized = minimize(
                score,
                vector[indices],
                method="L-BFGS-B",
                bounds=[(lower[index], upper[index]) for index in indices],
                options={
                    "maxiter": settings.max_iterations_per_stage,
                    "ftol": settings.tolerance,
                    "eps": step_sizes[indices],
                },
            )
            candidate = fixed.copy()
            candidate[indices] = np.clip(optimized.x, lower[indices], upper[indices])
            candidate = _synchronize_shared(program, candidate)
            if score(candidate[indices]) >= 1e6:
                candidate = fixed
            candidate_values = {
                parameter.id: float(value)
                for parameter, value in zip(definitions, candidate, strict=True)
            }
            stage_energy, _ = objective.evaluate(candidate_values)
            if stage_energy.total > fixed_energy.total:
                candidate = fixed
                candidate_values = fixed_values
                stage_energy = fixed_energy
            vector = candidate
            stage_results.append(
                ProgramFitStageResult(
                    stage=stage,
                    objective=stage_energy.total,
                    iterations=int(optimized.nit),
                    success=bool(optimized.success),
                )
            )
        final_values = {
            parameter.id: float(value) for parameter, value in zip(definitions, vector, strict=True)
        }
        final_energy, _ = objective.evaluate(final_values)
        if final_energy.total < best_energy.total or not best_stages:
            best_vector = vector.copy()
            best_energy = final_energy
            best_stages = stage_results
    if _topology_signature(program) != signature:
        raise RuntimeError("continuous fitting changed the declared program topology")
    parameters = {
        parameter.id: float(value)
        for parameter, value in zip(definitions, best_vector, strict=True)
    }
    diagnostics = prepared.topology_diagnostics
    route_ambiguous = diagnostics is None or any(
        (
            diagnostics.component_count > 1,
            diagnostics.junction_count > 0,
            diagnostics.cycle_count > 0,
        )
    )
    return DirectedGlyphFitResult(
        program_id=program.id,
        character=program.character,
        success=best_energy.total <= initial_total + settings.tolerance,
        parameters=parameters,
        energy=best_energy,
        initial_energy=initial_total,
        stages=best_stages,
        topology_preserved=True,
        route_ambiguous=route_ambiguous,
        direction_ambiguous=True,
        direction_uncertainty_note=(
            "A static raster does not independently establish original pen direction."
        ),
        provenance=program.provenance,
    )
