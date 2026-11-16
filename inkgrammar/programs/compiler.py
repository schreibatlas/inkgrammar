from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, radians, tan
from typing import TYPE_CHECKING, Literal

import numpy as np
from planarcurve import (
    Affine2D,
    CubicPath,
    CubicSegment,
    DirectedTrajectory,
    HeadingRateChange,
    HeadingRateProfile,
    Pose2D,
    Similarity2D,
    Trajectory2D,
    TrajectoryChain,
    evaluate_heading,
    evaluate_trajectory,
    evaluate_trajectory_at,
    evaluate_trajectory_chain,
    fit_similarity_transform,
    reverse_trajectory,
    trajectory_chain_to_cubic_path,
    trajectory_to_cubic_path,
)
from strokeoutline import WidthProfile

from .models import (
    AngleConstraint,
    ContactConstraint,
    CurvatureContinuityConstraint,
    DirectedGlyphProgram,
    DistanceConstraint,
    LoopClosureConstraint,
    PositionConstraint,
    ProgramContact,
    ProgramMovement,
    TangentContinuityConstraint,
)
from .parameters import (
    ProgramParameterBinding,
    bind_program_parameters,
    resolve_program_parameter_values,
)

if TYPE_CHECKING:
    from ..fitting.models import PrimitiveTemplate


@dataclass(frozen=True)
class CompiledProgramMovement:
    movement_id: str
    stroke_id: str
    direction: Literal["forward", "reverse"]
    trajectory: Trajectory2D
    transform: Similarity2D
    cubic_path: CubicPath
    start_pose: Pose2D
    end_pose: Pose2D
    width_profile: WidthProfile | None


@dataclass(frozen=True)
class CompiledGlyphProgram:
    program_id: str
    movements: tuple[CompiledProgramMovement, ...]
    stroke_paths: tuple[CubicPath, ...]
    landmarks: dict[str, np.ndarray]
    frame_transform: Affine2D


_CHANGE_TARGET = re.compile(r"curvature\.change\[(\d+)\]\.(at|rate)")


def _parameter_values(
    program: DirectedGlyphProgram,
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    return resolve_program_parameter_values(program, overrides)


def _parameter_value(
    program: DirectedGlyphProgram,
    values: Mapping[str, float],
    target: str,
    default: float,
) -> float:
    parameter = next((item for item in program.parameters if item.target == target), None)
    return default if parameter is None else values[parameter.id]


def _optional_parameter_value(
    program: DirectedGlyphProgram,
    values: Mapping[str, float],
    target: str,
) -> float | None:
    parameter = next((item for item in program.parameters if item.target == target), None)
    return None if parameter is None else values[parameter.id]


def _trajectory_for_movement(
    movement: ProgramMovement,
    template: PrimitiveTemplate,
    bindings: tuple[ProgramParameterBinding, ...],
    values: Mapping[str, float],
) -> tuple[Trajectory2D, float | None]:
    if getattr(template, "kind", None) != "curvature-profile":
        raise ValueError(
            f"directed movement requires a curvature-profile primitive: {movement.primitive}"
        )
    geometry = getattr(template, "curvature_stroke", None)
    if geometry is None:
        raise ValueError(f"curvature-profile primitive has no geometry: {movement.primitive}")

    data = geometry.model_dump(mode="python")
    occurrence_scale = 1.0
    movement_rotation = 0.0
    width: float | None = None
    for binding in bindings:
        value = values[binding.parameter_id]
        if binding.root == "style" and binding.path == "stroke_width":
            width = value
            continue
        if binding.owner_id != movement.id:
            continue
        if binding.root == "movement":
            if binding.path == "length":
                data["length"] = value
            elif binding.path == "start_angle":
                data["start_angle_deg"] = value
            elif binding.path == "rotation":
                movement_rotation += value
            elif binding.path == "width":
                width = value
            elif binding.path == "curvature.start_rate":
                data["curvature"]["start_rate_deg"] = value
            elif binding.path == "curvature.end_rate":
                data["curvature"]["end_rate_deg"] = value
            else:
                match = _CHANGE_TARGET.fullmatch(binding.path)
                if match is None:
                    raise ValueError(f"unsupported movement parameter target: {binding.target}")
                index, attribute = int(match.group(1)), match.group(2)
                changes = data["curvature"]["changes"]
                if index >= len(changes):
                    raise ValueError(f"parameter target is absent from primitive: {binding.target}")
                field = "at" if attribute == "at" else "rate_deg"
                changes[index][field] = value
        elif binding.root == "occurrence":
            if binding.path == "scale":
                occurrence_scale *= value
            elif binding.path == "rotation":
                movement_rotation += value
            elif binding.path in {"translation.x", "translation.y"} and value != 0.0:
                raise ValueError(
                    "nonzero occurrence translation cannot preserve a connected stroke"
                )

    geometry = type(geometry).model_validate(data)
    profile = geometry.curvature
    trajectory = Trajectory2D(
        length=geometry.length * occurrence_scale,
        start_heading_deg=geometry.start_angle_deg + movement_rotation,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=profile.start_rate_deg,
            end_rate_deg_per_u=profile.end_rate_deg,
            changes=tuple(
                HeadingRateChange(change.at, change.rate_deg) for change in profile.changes
            ),
            interpolation=profile.interpolation,
        ),
    )
    return trajectory, width


def _transformed_path(path: CubicPath, transform: Similarity2D) -> CubicPath:
    def apply(point: tuple[float, float]) -> tuple[float, float]:
        result = transform.apply(point)
        return float(result[0]), float(result[1])

    return CubicPath(
        start=apply(path.start),
        segments=tuple(
            CubicSegment(c1=apply(segment.c1), c2=apply(segment.c2), end=apply(segment.end))
            for segment in path.segments
        ),
    )


def _contact_points(
    contact: ProgramContact,
    movements: Mapping[str, CompiledProgramMovement],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    first = movements[contact.first_movement]
    second = movements[contact.second_movement]
    first_xy, first_heading = evaluate_trajectory_at(first.trajectory, [contact.first_u])
    second_xy, second_heading = evaluate_trajectory_at(second.trajectory, [contact.second_u])
    first_world = first.transform.apply(first_xy)[0]
    second_world = second.transform.apply(second_xy)[0]
    return (
        first_world,
        second_world,
        float(first_heading[0] + first.transform.rotation_deg),
        float(second_heading[0] + second.transform.rotation_deg),
    )


def _angle_error(actual: float, expected: float = 0.0) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def _validate_hard_constraints(
    program: DirectedGlyphProgram,
    movements: Mapping[str, CompiledProgramMovement],
    landmarks: Mapping[str, np.ndarray],
    *,
    tolerance: float,
) -> None:
    contacts = {contact.id: contact for contact in program.contacts}
    for constraint in program.constraints:
        if not constraint.hard:
            continue
        if isinstance(constraint, PositionConstraint):
            target = np.array([constraint.x, constraint.y], dtype=float)
            if np.linalg.norm(landmarks[constraint.landmark] - target) > tolerance:
                raise ValueError(f"hard position constraint failed: {constraint.id}")
        elif isinstance(constraint, DistanceConstraint):
            actual = np.linalg.norm(
                landmarks[constraint.first_landmark] - landmarks[constraint.second_landmark]
            )
            if abs(float(actual) - constraint.distance) > tolerance:
                raise ValueError(f"hard distance constraint failed: {constraint.id}")
        elif isinstance(constraint, LoopClosureConstraint):
            distance = np.linalg.norm(
                landmarks[constraint.start_landmark] - landmarks[constraint.end_landmark]
            )
            if distance > constraint.tolerance:
                raise ValueError(f"hard loop-closure constraint failed: {constraint.id}")
        elif isinstance(constraint, ContactConstraint):
            contact = contacts[constraint.contact_id]
            first, second, first_heading, second_heading = _contact_points(contact, movements)
            if contact.mode == "must-touch" and np.linalg.norm(first - second) > contact.tolerance:
                raise ValueError(f"hard contact constraint failed: {constraint.id}")
            if contact.mode == "must-cross-without-turn":
                if np.linalg.norm(first - second) > contact.tolerance:
                    raise ValueError(f"hard crossing constraint failed: {constraint.id}")
                acute_angle = _angle_error(first_heading, second_heading) % 180.0
                acute_angle = min(acute_angle, 180.0 - acute_angle)
                if acute_angle <= 1.0:
                    raise ValueError(
                        f"hard crossing is tangent rather than crossing: {constraint.id}"
                    )
            if contact.mode == "retrace":
                first_move = movements[contact.first_movement]
                second_move = movements[contact.second_movement]
                first_samples = first_move.transform.apply(
                    evaluate_trajectory(first_move.trajectory, count=64).xy
                )
                second_samples = second_move.transform.apply(
                    evaluate_trajectory(second_move.trajectory, count=64).xy
                )
                direct_error = np.max(np.linalg.norm(first_samples - second_samples, axis=1))
                reverse_error = np.max(np.linalg.norm(first_samples - second_samples[::-1], axis=1))
                if min(direct_error, reverse_error) > contact.tolerance:
                    raise ValueError(f"hard retrace constraint failed: {constraint.id}")
        elif isinstance(constraint, TangentContinuityConstraint):
            first = movements[constraint.first_movement]
            second = movements[constraint.second_movement]
            first_heading = float(evaluate_heading(first.trajectory, [1.0])[0])
            second_heading = float(evaluate_heading(second.trajectory, [0.0])[0])
            first_heading += first.transform.rotation_deg
            second_heading += second.transform.rotation_deg
            if _angle_error(second_heading, first_heading) > constraint.tolerance_deg:
                raise ValueError(f"hard tangent-continuity constraint failed: {constraint.id}")
        elif isinstance(constraint, CurvatureContinuityConstraint):
            first = movements[constraint.first_movement].trajectory.heading_rate
            second = movements[constraint.second_movement].trajectory.heading_rate
            error = abs(first.end_rate_deg_per_u - second.start_rate_deg_per_u)
            if error > constraint.tolerance:
                raise ValueError(f"hard curvature-continuity constraint failed: {constraint.id}")
        elif isinstance(constraint, AngleConstraint):
            first = movements[constraint.first_movement]
            second = movements[constraint.second_movement]
            first_heading = float(evaluate_heading(first.trajectory, [1.0])[0])
            second_heading = float(evaluate_heading(second.trajectory, [0.0])[0])
            actual = (
                second_heading
                + second.transform.rotation_deg
                - first_heading
                - first.transform.rotation_deg
            )
            if _angle_error(actual, constraint.angle_deg) > tolerance:
                raise ValueError(f"hard angle constraint failed: {constraint.id}")


def compile_glyph_program(
    program: DirectedGlyphProgram,
    primitive_templates: Mapping[str, PrimitiveTemplate],
    *,
    parameter_values: Mapping[str, float] | None = None,
    width_profiles: Mapping[str, WidthProfile] | None = None,
    samples_per_segment: int = 64,
    hard_constraint_tolerance: float = 1e-6,
) -> CompiledGlyphProgram:
    """Compile ordered curvature primitives through PlanarCurve trajectory chains."""
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2")
    if not isfinite(hard_constraint_tolerance) or hard_constraint_tolerance < 0.0:
        raise ValueError("hard_constraint_tolerance must be finite and nonnegative")

    values = _parameter_values(program, parameter_values)
    bindings = bind_program_parameters(program)
    alignment_scale = _parameter_value(program, values, "alignment.scale", 1.0)
    alignment_rotation = _parameter_value(program, values, "alignment.rotation", 0.0)
    alignment_x = _parameter_value(program, values, "alignment.translation.x", 0.0)
    alignment_y = _parameter_value(program, values, "alignment.translation.y", 0.0)
    stroke_width = _optional_parameter_value(program, values, "style.stroke_width")
    slant = _parameter_value(program, values, "style.slant", 0.0)
    frame_transform = Affine2D.from_scale_shear_translation(shear_x=tan(radians(slant)))

    compiled_movements: list[CompiledProgramMovement] = []
    stroke_paths: list[CubicPath] = []
    landmark_positions: dict[str, np.ndarray] = {}
    movement_by_id: dict[str, CompiledProgramMovement] = {}
    width_profiles = width_profiles or {}
    for landmark_id, landmark in program.landmarks.items():
        if landmark.x is not None and landmark.y is not None:
            landmark_positions[landmark_id] = np.array([landmark.x, landmark.y], dtype=float)

    for stroke in program.strokes:
        directed_trajectories: list[DirectedTrajectory] = []
        traversals: list[Trajectory2D] = []
        stroke_movements: list[ProgramMovement] = []
        stroke_widths: list[WidthProfile | None] = []
        for movement_id in stroke.movements:
            movement = program.movements[movement_id]
            template = primitive_templates.get(movement.primitive)
            if template is None:
                raise ValueError(f"program references unknown primitive: {movement.primitive}")
            trajectory, local_width = _trajectory_for_movement(movement, template, bindings, values)
            trajectory = Trajectory2D(
                length=trajectory.length * alignment_scale,
                start_heading_deg=trajectory.start_heading_deg,
                heading_rate=trajectory.heading_rate,
            )
            reversed_direction = movement.direction == "reverse"
            directed_trajectories.append(
                DirectedTrajectory(trajectory=trajectory, reversed=reversed_direction)
            )
            traversals.append(reverse_trajectory(trajectory) if reversed_direction else trajectory)
            stroke_movements.append(movement)
            if movement.width_profile is not None:
                if movement.width_profile not in width_profiles:
                    raise ValueError(f"unknown width profile: {movement.width_profile}")
                stroke_widths.append(width_profiles[movement.width_profile])
            else:
                stroke_widths.append(local_width if local_width is not None else stroke_width)

        if not stroke_movements:
            raise ValueError(f"program stroke has no movements: {stroke.id}")
        first_movement = stroke_movements[0]
        start_landmark = program.landmarks[first_movement.from_landmark]
        start_position = np.array(
            [alignment_x, alignment_y]
            if start_landmark.x is None
            else [start_landmark.x, start_landmark.y],
            dtype=float,
        )
        start_pose = Pose2D(
            position=start_position,
            heading_deg=alignment_rotation + traversals[0].start_heading_deg,
            heading_rate_deg_per_u=0.0,
        )
        chain = TrajectoryChain(tuple(directed_trajectories))
        chain_sample = evaluate_trajectory_chain(
            chain,
            samples_per_segment=samples_per_segment,
            start_pose=start_pose,
        )
        stroke_paths.append(trajectory_chain_to_cubic_path(chain, start_pose=start_pose))

        for index, movement in enumerate(stroke_movements):
            selection = chain_sample.segment_index == index
            world_points = chain_sample.xy[selection]
            if len(world_points) != samples_per_segment:
                raise ValueError("PlanarCurve trajectory-chain sample count mismatch")
            traversal = traversals[index]
            local_sample = evaluate_trajectory(traversal, count=samples_per_segment)
            transform = fit_similarity_transform(world_points, local_sample.xy)
            local_path = trajectory_to_cubic_path(traversal)
            cubic_path = _transformed_path(local_path, transform)
            local_heading = evaluate_heading(traversal, np.array([0.0, 1.0]))
            start = Pose2D(
                position=world_points[0],
                heading_deg=float(local_heading[0] + transform.rotation_deg),
                heading_rate_deg_per_u=traversal.heading_rate.start_rate_deg_per_u,
            )
            end = Pose2D(
                position=world_points[-1],
                heading_deg=float(local_heading[1] + transform.rotation_deg),
                heading_rate_deg_per_u=traversal.heading_rate.end_rate_deg_per_u,
            )
            compiled = CompiledProgramMovement(
                movement_id=movement.id,
                stroke_id=stroke.id,
                direction=movement.direction,
                trajectory=traversal,
                transform=transform,
                cubic_path=cubic_path,
                start_pose=start,
                end_pose=end,
                width_profile=stroke_widths[index],
            )
            compiled_movements.append(compiled)
            movement_by_id[movement.id] = compiled
            for landmark_id, position in (
                (movement.from_landmark, world_points[0]),
                (movement.to_landmark, world_points[-1]),
            ):
                previous = landmark_positions.get(landmark_id)
                if (
                    previous is not None
                    and np.linalg.norm(previous - position) > hard_constraint_tolerance
                ):
                    raise ValueError(f"movement endpoint mismatch at landmark: {landmark_id}")
                landmark_positions[landmark_id] = np.asarray(position, dtype=float).copy()

    missing_landmarks = set(program.landmarks) - set(landmark_positions)
    if missing_landmarks:
        raise ValueError(
            f"program landmarks lack coordinates or movement endpoints: {sorted(missing_landmarks)}"
        )
    _validate_hard_constraints(
        program,
        movement_by_id,
        landmark_positions,
        tolerance=hard_constraint_tolerance,
    )

    return CompiledGlyphProgram(
        program_id=program.id,
        movements=tuple(compiled_movements),
        stroke_paths=tuple(stroke_paths),
        landmarks={
            landmark_id: frame_transform.apply(position)
            for landmark_id, position in landmark_positions.items()
        },
        frame_transform=frame_transform,
    )
