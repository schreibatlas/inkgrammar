from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import hypot

import numpy as np
from planarcurve import Affine2D, evaluate_heading, evaluate_trajectory, evaluate_trajectory_at
from pydantic import Field
from rasterregions import components
from strokeoutline import RasterFrame, StrokeInput, render_strokes

from ..dataset.transforms import glyph_units_to_source_px
from ..fitting.alignment import bilinear_sample
from ..fitting.objective import soft_mask_loss, symmetric_distance_loss
from ..models import StrictModel
from .compiler import CompiledGlyphProgram, compile_glyph_program
from .models import (
    AngleConstraint,
    ContactConstraint,
    CurvatureContinuityConstraint,
    DirectedGlyphProgram,
    DistanceConstraint,
    LoopClosureConstraint,
    OrderingConstraint,
    PositionConstraint,
    ProgramContact,
    TangentContinuityConstraint,
)
from .observation import GlyphObservation
from .parameters import resolve_program_parameter_values


class ProgramObjectiveWeights(StrictModel):
    ink: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    distance: float = Field(default=0.75, ge=0.0, allow_inf_nan=False)
    centerline: float = Field(default=0.35, ge=0.0, allow_inf_nan=False)
    topology: float = Field(default=0.2, ge=0.0, allow_inf_nan=False)
    constraints: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    prior: float = Field(default=0.03, ge=0.0, allow_inf_nan=False)
    width: float = Field(default=0.2, ge=0.0, allow_inf_nan=False)

    complexity: float = Field(default=0.001, ge=0.0, allow_inf_nan=False)


class ProgramComplexityBreakdown(StrictModel):
    primitive_instances: int = Field(ge=0)
    gesture_instances: int | None = Field(default=None, ge=0)
    free_parameters: int = Field(ge=0)
    pen_lifts: int = Field(ge=0)
    exception_overrides: int = Field(ge=0)


class ProgramFitEnergy(StrictModel):
    total: float = Field(ge=0.0, allow_inf_nan=False)
    ink: float = Field(ge=0.0, allow_inf_nan=False)
    distance: float = Field(ge=0.0, allow_inf_nan=False)
    centerline: float = Field(ge=0.0, allow_inf_nan=False)
    topology: float = Field(ge=0.0, allow_inf_nan=False)
    constraints: float = Field(ge=0.0, allow_inf_nan=False)
    prior: float = Field(ge=0.0, allow_inf_nan=False)
    width: float = Field(ge=0.0, allow_inf_nan=False)
    complexity: float = Field(ge=0.0, allow_inf_nan=False)
    complexity_breakdown: ProgramComplexityBreakdown


def _width_in_model_units(width_profile: object) -> float:
    if width_profile is None:
        return 0.12
    if isinstance(width_profile, (int, float, np.number)) and not isinstance(width_profile, bool):
        return float(width_profile)
    if callable(width_profile):
        return float(width_profile(0.5))
    try:
        knots = tuple(width_profile)
    except TypeError as exc:
        raise ValueError("unsupported width profile") from exc
    if not knots:
        raise ValueError("width profile must contain at least one knot")
    positions, widths = zip(*knots, strict=True)
    return float(np.interp(0.5, positions, widths))


def _pixels_per_model_unit(observation: GlyphObservation) -> float:
    frame = observation.frame
    x_scale = np.linalg.norm(frame.x_axis_source_px)
    y_scale = np.linalg.norm(frame.y_axis_source_px)
    return frame.x_height_px * float(np.sqrt(x_scale * y_scale))


def _movement_points(
    compiled: CompiledGlyphProgram,
    observation: GlyphObservation,
    samples_per_movement: int,
) -> tuple[list[np.ndarray], list[float]]:
    paths: list[np.ndarray] = []
    widths_px: list[float] = []
    pixels_per_unit = _pixels_per_model_unit(observation)
    for movement in compiled.movements:
        local = evaluate_trajectory(movement.trajectory, count=samples_per_movement).xy
        world = movement.transform.apply(local)
        model_points = compiled.frame_transform.apply(world)
        paths.append(glyph_units_to_source_px(model_points, observation.frame))
        widths_px.append(_width_in_model_units(movement.width_profile) * pixels_per_unit)
    return paths, widths_px


def render_program_coverage(
    compiled: CompiledGlyphProgram,
    observation: GlyphObservation,
    *,
    samples_per_movement: int = 64,
    supersampling: int = 3,
) -> np.ndarray:
    """Render compiled cubic centerlines with StrokeOutline in the observation frame."""
    if samples_per_movement < 2:
        raise ValueError("samples_per_movement must be at least 2")
    if supersampling < 1:
        raise ValueError("supersampling must be at least 1")

    width, height = observation.image.size
    frame = observation.frame
    glyph_to_pixel = Affine2D.from_basis(
        frame.origin_source_px,
        np.asarray(frame.x_axis_source_px) * frame.x_height_px,
        np.asarray(frame.y_axis_source_px) * frame.x_height_px,
    )
    raster_frame = RasterFrame(
        width,
        height,
        glyph_to_pixel.compose(compiled.frame_transform),
    )
    strokes = tuple(
        StrokeInput(
            movement.movement_id,
            movement.cubic_path,
            movement.width_profile if movement.width_profile is not None else 0.12,
        )
        for movement in compiled.movements
    )
    rendered = render_strokes(strokes, raster_frame=raster_frame, supersampling=supersampling)
    if rendered.combined_coverage is None:
        raise RuntimeError("StrokeOutline did not return raster coverage for the requested frame")
    return rendered.combined_coverage


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _sample_movement(
    movement,
    progress: float,
) -> tuple[np.ndarray, float]:
    points, headings = evaluate_trajectory_at(movement.trajectory, [progress])
    return movement.transform.apply(points)[0], float(headings[0] + movement.transform.rotation_deg)


def _landmark_positions(
    program: DirectedGlyphProgram,
    compiled: CompiledGlyphProgram,
) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    movement_by_id = {movement.movement_id: movement for movement in compiled.movements}
    for movement_id, movement in program.movements.items():
        compiled_movement = movement_by_id[movement_id]
        positions.setdefault(movement.from_landmark, compiled_movement.start_pose.position)
        positions[movement.to_landmark] = compiled_movement.end_pose.position
    return positions


def _contact_residual(
    contact: ProgramContact,
    compiled: CompiledGlyphProgram,
    x_height: float,
) -> float:
    movement_by_id = {movement.movement_id: movement for movement in compiled.movements}
    first_position, first_heading = _sample_movement(
        movement_by_id[contact.first_movement], contact.first_u
    )
    second_position, second_heading = _sample_movement(
        movement_by_id[contact.second_movement], contact.second_u
    )
    distance = float(np.linalg.norm(first_position - second_position))
    if contact.mode == "may-touch":
        return 0.0
    if contact.mode == "must-cross-without-turn":
        acute_angle = _angle_difference(first_heading, second_heading) % 180.0
        acute_angle = min(acute_angle, 180.0 - acute_angle)
        return (
            max(0.0, distance - contact.tolerance) / max(x_height, 1e-9)
            + max(0.0, 1.0 - acute_angle) / 180.0
        )
    if contact.mode == "retrace":
        first = movement_by_id[contact.first_movement]
        second = movement_by_id[contact.second_movement]
        first_points = first.transform.apply(evaluate_trajectory(first.trajectory, count=48).xy)
        second_points = second.transform.apply(evaluate_trajectory(second.trajectory, count=48).xy)
        direct = np.linalg.norm(first_points - second_points, axis=1)
        reverse = np.linalg.norm(first_points - second_points[::-1], axis=1)
        return float(np.mean(np.minimum(direct, reverse))) / max(x_height, 1e-9)
    return max(0.0, distance - contact.tolerance) / max(x_height, 1e-9)


def _constraint_loss(
    program: DirectedGlyphProgram,
    compiled: CompiledGlyphProgram,
) -> float:
    contacts = {contact.id: contact for contact in program.contacts}
    movements = {movement.movement_id: movement for movement in compiled.movements}
    landmarks = _landmark_positions(program, compiled)
    x_height = 1.0
    weighted_errors: list[float] = []
    total_weight = 0.0
    for constraint in program.constraints:
        if constraint.hard or isinstance(constraint, OrderingConstraint):
            continue
        residual = 0.0
        if isinstance(constraint, PositionConstraint):
            target = np.array([constraint.x, constraint.y])
            residual = float(np.linalg.norm(landmarks[constraint.landmark] - target)) / x_height
        elif isinstance(constraint, DistanceConstraint):
            actual = np.linalg.norm(
                landmarks[constraint.first_landmark] - landmarks[constraint.second_landmark]
            )
            residual = abs(float(actual) - constraint.distance) / x_height
        elif isinstance(constraint, LoopClosureConstraint):
            distance = np.linalg.norm(
                landmarks[constraint.start_landmark] - landmarks[constraint.end_landmark]
            )
            residual = max(0.0, float(distance) - constraint.tolerance) / x_height
        elif isinstance(constraint, ContactConstraint):
            contact = contacts[constraint.contact_id]
            residual = _contact_residual(contact, compiled, x_height)
        elif isinstance(constraint, TangentContinuityConstraint):
            first = movements[constraint.first_movement]
            second = movements[constraint.second_movement]
            first_heading = evaluate_heading(first.trajectory, [1.0])[0]
            second_heading = evaluate_heading(second.trajectory, [0.0])[0]
            first_heading += first.transform.rotation_deg
            second_heading += second.transform.rotation_deg
            residual = (
                max(
                    0.0,
                    _angle_difference(float(first_heading), float(second_heading))
                    - constraint.tolerance_deg,
                )
                / 180.0
            )
        elif isinstance(constraint, CurvatureContinuityConstraint):
            first = movements[constraint.first_movement].trajectory.heading_rate
            second = movements[constraint.second_movement].trajectory.heading_rate
            residual = abs(first.end_rate_deg_per_u - second.start_rate_deg_per_u) / 180.0
        elif isinstance(constraint, AngleConstraint):
            first = movements[constraint.first_movement]
            second = movements[constraint.second_movement]
            first_heading = evaluate_heading(first.trajectory, [1.0])[0]
            second_heading = evaluate_heading(second.trajectory, [0.0])[0]
            actual = float(second_heading + second.transform.rotation_deg) - float(
                first_heading + first.transform.rotation_deg
            )
            residual = _angle_difference(actual, constraint.angle_deg) / 180.0
        weighted_errors.append(constraint.weight * residual**2)
        total_weight += constraint.weight
    return float(sum(weighted_errors) / total_weight) if total_weight else 0.0


def _topology_loss(program: DirectedGlyphProgram, observation: GlyphObservation) -> float:
    observed_components = len(components(observation.mask))
    extra_components = max(0, observed_components - len(program.strokes))
    penalty = extra_components / max(1, len(program.strokes))
    diagnostics = observation.topology_diagnostics
    expected_cycles = sum(
        isinstance(constraint, LoopClosureConstraint) for constraint in program.constraints
    )
    if diagnostics is not None:
        penalty += abs(diagnostics.cycle_count - expected_cycles) / max(1, expected_cycles)
    return float(penalty)


def _prior_loss(program: DirectedGlyphProgram, values: Mapping[str, float]) -> float:
    residuals = [
        ((values[parameter.id] - parameter.prior_mean) / parameter.prior_sigma) ** 2
        for parameter in program.parameters
        if parameter.prior_mean is not None and parameter.prior_sigma is not None
    ]
    return float(np.mean(residuals)) if residuals else 0.0


def _complexity_breakdown(program: DirectedGlyphProgram) -> ProgramComplexityBreakdown:
    return ProgramComplexityBreakdown(
        primitive_instances=len(program.movements),
        gesture_instances=program.gesture_instance_count,
        free_parameters=sum(parameter.lower < parameter.upper for parameter in program.parameters),
        pen_lifts=max(0, len(program.strokes) - 1),
        exception_overrides=program.exception_override_count,
    )


def _complexity_score(breakdown: ProgramComplexityBreakdown) -> float:
    return float(
        breakdown.primitive_instances
        + (breakdown.gesture_instances or 0)
        + breakdown.free_parameters
        + breakdown.pen_lifts
        + breakdown.exception_overrides
    )


def _centerline_loss(
    observation: GlyphObservation,
    paths: list[np.ndarray],
    skeleton_distance: np.ndarray | None,
) -> float:
    if skeleton_distance is None:
        return 0.0
    points = np.concatenate(paths, axis=0)
    distances = bilinear_sample(
        skeleton_distance,
        points,
        outside_value=hypot(*skeleton_distance.shape),
    )
    return float(np.mean(distances) / max(observation.frame.x_height_px, 1e-9))


def _width_loss(
    observation: GlyphObservation,
    widths_px: list[float],
) -> float:
    analysis = observation.skeleton_analysis
    if analysis is None or not np.any(analysis.skeleton):
        return 0.0
    observed_diameter = 2.0 * float(np.median(analysis.radius[analysis.skeleton]))
    predicted_width = float(np.median(widths_px))
    return abs(predicted_width - observed_diameter) / max(observation.frame.x_height_px, 1e-9)


@dataclass(frozen=True)
class ProgramObjective:
    program: DirectedGlyphProgram
    primitive_templates: Mapping[str, object]
    observation: GlyphObservation
    weights: ProgramObjectiveWeights
    samples_per_movement: int = 64
    supersampling: int = 3
    _skeleton_distance: np.ndarray | None = field(init=False, default=None, repr=False)
    _topology: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.samples_per_movement < 2:
            raise ValueError("samples_per_movement must be at least 2")
        if self.supersampling < 1:
            raise ValueError("supersampling must be at least 1")
        skeleton_distance = None
        if self.observation.skeleton_analysis is not None:
            from scipy.ndimage import distance_transform_edt

            skeleton_distance = distance_transform_edt(~self.observation.skeleton_analysis.skeleton)
            skeleton_distance.setflags(write=False)
        object.__setattr__(self, "_skeleton_distance", skeleton_distance)
        object.__setattr__(
            self,
            "_topology",
            _topology_loss(self.program, self.observation),
        )

    def evaluate(
        self,
        parameter_values: Mapping[str, float] | None = None,
    ) -> tuple[ProgramFitEnergy, CompiledGlyphProgram]:
        values = resolve_program_parameter_values(self.program, parameter_values)
        compiled = compile_glyph_program(
            self.program,
            self.primitive_templates,
            parameter_values=values,
        )
        coverage = render_program_coverage(
            compiled,
            self.observation,
            samples_per_movement=self.samples_per_movement,
            supersampling=self.supersampling,
        )
        reference = self.observation.mask.astype(np.float32)
        ink = soft_mask_loss(reference, coverage)
        distance = symmetric_distance_loss(
            reference,
            coverage,
            target_distance_field=self.observation.distance_field,
        )
        paths, widths_px = _movement_points(
            compiled,
            self.observation,
            self.samples_per_movement,
        )
        centerline = _centerline_loss(self.observation, paths, self._skeleton_distance)
        topology = self._topology
        constraints = _constraint_loss(self.program, compiled)
        prior = _prior_loss(self.program, values)
        complexity_breakdown = _complexity_breakdown(self.program)
        complexity = _complexity_score(complexity_breakdown)
        width = _width_loss(self.observation, widths_px)
        total = (
            self.weights.ink * ink
            + self.weights.distance * distance
            + self.weights.centerline * centerline
            + self.weights.topology * topology
            + self.weights.constraints * constraints
            + self.weights.prior * prior
            + self.weights.width * width
            + self.weights.complexity * complexity
        )
        energy = ProgramFitEnergy(
            total=total,
            ink=ink,
            distance=distance,
            centerline=centerline,
            topology=topology,
            constraints=constraints,
            prior=prior,
            width=width,
            complexity=complexity,
            complexity_breakdown=complexity_breakdown,
        )
        return energy, compiled


def evaluate_program_objective(
    observation: GlyphObservation,
    program: DirectedGlyphProgram,
    primitive_templates: Mapping[str, object],
    *,
    parameter_values: Mapping[str, float] | None = None,
    weights: ProgramObjectiveWeights | None = None,
) -> ProgramFitEnergy:
    objective = ProgramObjective(
        program=program,
        primitive_templates=primitive_templates,
        observation=observation,
        weights=weights or ProgramObjectiveWeights(),
    )
    energy, _ = objective.evaluate(parameter_values)
    return energy
