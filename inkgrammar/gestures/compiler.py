from __future__ import annotations

from collections.abc import Mapping

from ..fitting.models import PrimitiveTemplate
from ..programs.compiler import CompiledGlyphProgram, compile_glyph_program
from ..programs.models import (
    ContactConstraint,
    DirectedGlyphProgram,
    LoopClosureConstraint,
    OrderingConstraint,
    ProgramConstraint,
    ProgramContact,
    ProgramLandmark,
    ProgramMovement,
    ProgramStroke,
    TangentContinuityConstraint,
)
from .models import (
    GestureGlyphProgram,
    GestureTemplate,
    ProgramGesture,
)

_LANDMARK_KIND_MAP = {
    "start": "start",
    "end": "end",
    "turn": "turn",
    "apex": "reference",
    "baseline-contact": "contact",
    "closure": "closure",
    "crossing": "crossing",
    "reference": "reference",
}


def _movement_id(gesture: ProgramGesture, slot_id: str) -> str:
    return f"{gesture.id}--{slot_id}"


def _landmark_id(gesture: ProgramGesture, landmark_id: str) -> str:
    return f"{gesture.id}--{landmark_id}"


def _instance_landmarks(
    program: GestureGlyphProgram,
    instance: ProgramGesture,
    template: GestureTemplate,
    landmarks: dict[str, ProgramLandmark],
) -> dict[str, str]:
    unknown = set(instance.landmark_map) - set(template.landmarks)
    if unknown:
        raise ValueError(
            f"gesture landmark map references unknown template landmarks: {sorted(unknown)}"
        )

    result: dict[str, str] = {}
    for landmark_id, landmark in template.landmarks.items():
        explicit = instance.landmark_map.get(landmark_id)
        if landmark.kind == "start":
            if explicit is not None and explicit != instance.from_landmark:
                raise ValueError(f"gesture start landmark mapping conflicts: {instance.id}")
            target_id = instance.from_landmark
        elif landmark.kind == "end":
            if explicit is not None and explicit != instance.to_landmark:
                raise ValueError(f"gesture end landmark mapping conflicts: {instance.id}")
            target_id = instance.to_landmark
        else:
            target_id = explicit or _landmark_id(instance, landmark_id)

        if target_id not in landmarks:
            landmarks[target_id] = ProgramLandmark(
                id=target_id,
                kind=_LANDMARK_KIND_MAP[landmark.kind],
            )
        result[landmark_id] = target_id

    for target_id in (instance.from_landmark, instance.to_landmark):
        if target_id not in program.landmarks:
            raise ValueError(f"gesture references unknown glyph landmark: {target_id}")
    return result


def _movement_role(template: GestureTemplate, index: int) -> str:
    if template.role == "entry" and index == 0:
        return "entry"
    if template.role == "exit" and index == len(template.primitives) - 1:
        return "exit"
    if template.role == "stem":
        return "stem"
    if template.role == "other":
        return "other"
    return "body"


def _expand_template_constraints(
    instance: ProgramGesture,
    template: GestureTemplate,
    landmark_ids: Mapping[str, str],
    movement_ids: Mapping[str, str],
) -> tuple[list[ProgramContact], list[ProgramConstraint]]:
    contacts: list[ProgramContact] = []
    constraints: list[ProgramConstraint] = []
    for constraint in template.constraints:
        constraint_id = f"{instance.id}--{constraint.id}"
        if constraint.kind == "loop-closure":
            constraints.append(
                LoopClosureConstraint(
                    id=constraint_id,
                    kind="loop-closure",
                    start_landmark=landmark_ids[constraint.first],
                    end_landmark=landmark_ids[constraint.second],
                    tolerance=constraint.tolerance,
                    hard=constraint.hard,
                    weight=constraint.weight,
                )
            )
        elif constraint.kind == "ordering":
            constraints.append(
                OrderingConstraint(
                    id=constraint_id,
                    kind="ordering",
                    before_movement=movement_ids[constraint.first],
                    after_movement=movement_ids[constraint.second],
                    hard=constraint.hard,
                    weight=constraint.weight,
                )
            )
        elif constraint.kind == "tangent-continuity":
            constraints.append(
                TangentContinuityConstraint(
                    id=constraint_id,
                    kind="tangent-continuity",
                    first_movement=movement_ids[constraint.first],
                    second_movement=movement_ids[constraint.second],
                    tolerance_deg=constraint.tolerance,
                    hard=constraint.hard,
                    weight=constraint.weight,
                )
            )
        else:
            contact_id = f"{constraint_id}--contact"
            contacts.append(
                ProgramContact(
                    id=contact_id,
                    first_movement=movement_ids[constraint.first],
                    first_u=1.0,
                    second_movement=movement_ids[constraint.second],
                    second_u=0.0,
                    mode="must-touch",
                    tolerance=constraint.tolerance,
                )
            )
            constraints.append(
                ContactConstraint(
                    id=constraint_id,
                    kind="contact",
                    contact_id=contact_id,
                    hard=constraint.hard,
                    weight=constraint.weight,
                )
            )
    return contacts, constraints


def expand_gesture_program(
    program: GestureGlyphProgram,
    gesture_templates: Mapping[str, GestureTemplate],
) -> DirectedGlyphProgram:
    """Expand semantic gesture instances into the existing directed program schema."""
    landmarks = dict(program.landmarks)
    movements: dict[str, ProgramMovement] = {}
    expanded_strokes: list[ProgramStroke] = []
    contacts = list(program.contacts)
    constraints = list(program.constraints)

    for stroke in program.strokes:
        stroke_movement_ids: list[str] = []
        for instance_id in stroke.gestures:
            instance = program.gestures[instance_id]
            template = gesture_templates.get(instance.gesture)
            if template is None:
                raise ValueError(f"program references unknown gesture template: {instance.gesture}")
            landmark_ids = _instance_landmarks(program, instance, template, landmarks)
            movement_ids = {
                slot.id: _movement_id(instance, slot.id) for slot in template.primitives
            }
            previous_end = instance.from_landmark
            for index, slot in enumerate(template.primitives):
                from_landmark = (
                    landmark_ids[slot.from_landmark]
                    if slot.from_landmark is not None
                    else previous_end
                )
                if index == 0 and from_landmark != instance.from_landmark:
                    raise ValueError(
                        "first gesture movement does not start at its instance landmark: "
                        f"{instance.id}"
                    )
                if index > 0 and from_landmark != previous_end:
                    raise ValueError(f"gesture primitives are discontinuous at slot: {slot.id}")

                if slot.to_landmark is not None:
                    to_landmark = landmark_ids[slot.to_landmark]
                elif index == len(template.primitives) - 1:
                    to_landmark = instance.to_landmark
                else:
                    to_landmark = _landmark_id(instance, f"{slot.id}--end")
                    landmarks.setdefault(
                        to_landmark,
                        ProgramLandmark(id=to_landmark, kind="continuation"),
                    )
                if index == len(template.primitives) - 1 and to_landmark != instance.to_landmark:
                    raise ValueError(
                        "last gesture movement does not end at its instance landmark: "
                        f"{instance.id}"
                    )

                movement_id = movement_ids[slot.id]
                movements[movement_id] = ProgramMovement(
                    id=movement_id,
                    primitive=slot.primitive_family,
                    direction=slot.direction,
                    from_landmark=from_landmark,
                    to_landmark=to_landmark,
                    role=_movement_role(template, index),
                    attach=slot.attach_to_previous,
                )
                stroke_movement_ids.append(movement_id)
                previous_end = to_landmark

            generated_contacts, generated_constraints = _expand_template_constraints(
                instance,
                template,
                landmark_ids,
                movement_ids,
            )
            contacts.extend(generated_contacts)
            constraints.extend(generated_constraints)
        expanded_strokes.append(ProgramStroke(id=stroke.id, movements=stroke_movement_ids))

    return DirectedGlyphProgram(
        id=program.id,
        character=program.character,
        variant=program.variant,
        landmarks=landmarks,
        movements=movements,
        strokes=expanded_strokes,
        contacts=contacts,
        constraints=constraints,
        parameters=program.parameters,
        entry_landmark=program.entry_landmark,
        gesture_instance_count=len(program.gestures),
        exception_override_count=sum(
            len(gesture.parameter_overrides) for gesture in program.gestures.values()
        ),
        exit_landmark=program.exit_landmark,
        provenance=program.provenance,
    )




def compile_gesture_program(
    program: GestureGlyphProgram,
    gesture_templates: Mapping[str, GestureTemplate],
    primitive_templates: Mapping[str, PrimitiveTemplate],
    *,
    parameter_values: Mapping[str, float] | None = None,
) -> CompiledGlyphProgram:
    """Expand and compile a gesture program through the existing program compiler."""
    directed = expand_gesture_program(program, gesture_templates)
    values = {
        parameter_id: value
        for gesture in program.gestures.values()
        for parameter_id, value in gesture.parameter_overrides.items()
    }
    if parameter_values is not None:
        values.update(parameter_values)
    return compile_glyph_program(directed, primitive_templates, parameter_values=values)
