from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import ascii_lowercase
from types import MappingProxyType

from ..fitting.models import (
    CurvatureProfile,
    CurvatureStrokeGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
)
from ..programs.models import ProgramContact, ProgramLandmark, ProgramProvenance
from .evidence import GestureProvenance
from .models import (
    GestureConstraint,
    GestureGlyphProgram,
    GestureLandmark,
    GesturePrimitiveSlot,
    GestureRole,
    GestureStroke,
    GestureTemplate,
    ProgramGesture,
)


@dataclass(frozen=True)
class SyntheticLowercaseGrammar:
    templates: Mapping[str, GestureTemplate]
    programs: tuple[GestureGlyphProgram, ...]
    primitive_templates: Mapping[str, PrimitiveTemplate]
    structural_cases: Mapping[str, tuple[str, ...]]


def _primitive(
    primitive_id: str,
    *,
    angle: float,
    rate: float,
    length: float = 1.0,
) -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id=primitive_id,
        kind="curvature-profile",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(0.0, 1.0)),
        curvature_stroke=CurvatureStrokeGeometry(
            length=length,
            start_angle_deg=angle,
            curvature=CurvatureProfile(start_rate_deg=rate, end_rate_deg=rate),
        ),
        provenance={"construction": "synthetic", "trajectory": "synthetic"},
    )


def _template(
    template_id: str,
    role: GestureRole,
    families: Sequence[str],
    *,
    directions: Sequence[str] | None = None,
    detach_slots: frozenset[int] = frozenset(),
    loop: bool = False,
) -> GestureTemplate:
    slot_directions = tuple(directions or ("forward",) * len(families))
    slots = [
        GesturePrimitiveSlot(
            id=f"part-{index + 1}",
            primitive_family=family,
            direction=slot_directions[index],
            attach_to_previous=index not in detach_slots,
        )
        for index, family in enumerate(families)
    ]
    landmarks = (
        {
            "start": GestureLandmark(id="start", kind="start"),
            "end": GestureLandmark(id="end", kind="end"),
        }
        if loop
        else {}
    )
    constraints = (
        [
            GestureConstraint(
                id="loop-closure",
                kind="loop-closure",
                first="start",
                second="end",
                tolerance=0.15,
            )
        ]
        if loop
        else []
    )
    return GestureTemplate(
        id=template_id,
        role=role,
        primitives=slots,
        landmarks=landmarks,
        constraints=constraints,
        provenance=GestureProvenance(
            level="synthetic",
            source_ids=["synthetic:lowercase-grammar"],
            notes="Structural synthetic template, not a reviewed handwriting label.",
        ),
    )


def _program(
    letter: str,
    variant: str,
    stroke_shapes: Sequence[Sequence[str]],
    templates: Mapping[str, GestureTemplate],
    *,
    crossing: bool = False,
    alternative_note: str | None = None,
) -> GestureGlyphProgram:
    gestures: dict[str, ProgramGesture] = {}
    strokes: list[GestureStroke] = []
    landmarks: dict[str, ProgramLandmark] = {}
    ordered_instances: list[tuple[str, str]] = []

    for stroke_index, shapes in enumerate(stroke_shapes):
        instance_ids = []
        previous_end: str | None = None
        for shape_index, template_id in enumerate(shapes):
            instance_id = f"g{stroke_index}-{shape_index}-{template_id}"
            start_id = previous_end or f"s{stroke_index}-start"
            end_id = (
                f"s{stroke_index}-end"
                if shape_index == len(shapes) - 1
                else f"s{stroke_index}-join-{shape_index}"
            )
            landmarks.setdefault(
                start_id,
                ProgramLandmark(
                    id=start_id,
                    kind="start" if previous_end is None else "continuation",
                ),
            )
            landmarks.setdefault(
                end_id,
                ProgramLandmark(
                    id=end_id,
                    kind="end" if shape_index == len(shapes) - 1 else "continuation",
                ),
            )
            gestures[instance_id] = ProgramGesture(
                id=instance_id,
                gesture=template_id,
                from_landmark=start_id,
                to_landmark=end_id,
            )
            instance_ids.append(instance_id)
            ordered_instances.append((instance_id, template_id))
            previous_end = end_id
        strokes.append(GestureStroke(id=f"stroke-{stroke_index + 1}", gestures=instance_ids))

    contacts = []
    if crossing:
        first_instance, second_instance = (item[0] for item in ordered_instances[:2])
        contacts.append(
            ProgramContact(
                id="crossing-contact",
                first_movement=f"{first_instance}--part-1",
                first_u=0.5,
                second_movement=f"{second_instance}--part-1",
                second_u=0.5,
                mode="must-cross-without-turn",
                tolerance=0.1,
            )
        )

    first_id, _ = ordered_instances[0]
    last_id, _ = ordered_instances[-1]
    first_landmark = gestures[first_id].from_landmark
    last_landmark = gestures[last_id].to_landmark
    provenance = ProgramProvenance(
        construction="synthetic",
        trajectory="synthetic",
        sources=["synthetic:lowercase-grammar"],
        notes=alternative_note,
    )
    return GestureGlyphProgram(
        id=f"lowercase-{letter}-{variant}",
        character=letter,
        variant=variant,
        gestures=gestures,
        strokes=strokes,
        landmarks=landmarks,
        contacts=contacts,
        entry_landmark=first_landmark,
        exit_landmark=last_landmark,
        provenance=provenance,
    )


def build_synthetic_lowercase_grammar() -> SyntheticLowercaseGrammar:
    """Build source-free lowercase programs with explicit structural alternatives."""
    family_ids = tuple(f"curve-family-{index:03d}" for index in range(1, 13))
    primitive_templates = {
        family_ids[0]: _primitive(family_ids[0], angle=0.0, rate=280.0),
        family_ids[1]: _primitive(family_ids[1], angle=90.0, rate=0.0),
        family_ids[2]: _primitive(family_ids[2], angle=0.0, rate=-80.0),
        family_ids[3]: _primitive(family_ids[3], angle=0.0, rate=110.0),
        family_ids[4]: _primitive(family_ids[4], angle=90.0, rate=320.0),
        family_ids[5]: _primitive(family_ids[5], angle=-90.0, rate=-320.0),
        family_ids[6]: _primitive(family_ids[6], angle=15.0, rate=0.0, length=0.7),
        family_ids[7]: _primitive(family_ids[7], angle=0.0, rate=0.0),
        family_ids[8]: _primitive(family_ids[8], angle=45.0, rate=0.0),
        family_ids[9]: _primitive(family_ids[9], angle=135.0, rate=0.0),
        family_ids[10]: _primitive(family_ids[10], angle=0.0, rate=0.0),
        family_ids[11]: _primitive(family_ids[11], angle=90.0, rate=0.0, length=0.08),
    }
    templates = {
        "oval": _template("oval", "oval", [family_ids[0]], loop=True),
        "stem": _template("stem", "stem", [family_ids[1]]),
        "underturn": _template("underturn", "underturn", [family_ids[2]]),
        "hump": _template("hump", "hump", [family_ids[3]]),
        "double-hump": _template("double-hump", "hump", [family_ids[3], family_ids[3]]),
        "ascender-loop": _template("ascender-loop", "ascender-loop", [family_ids[4]], loop=True),
        "descender-loop": _template("descender-loop", "descender-loop", [family_ids[5]], loop=True),
        "exit": _template("exit", "exit", [family_ids[6]]),
        "cross-stroke": _template("cross-stroke", "cross-stroke", [family_ids[7]]),
        "diagonal": _template("diagonal", "other", [family_ids[8]]),
        "diagonal-down": _template("diagonal-down", "other", [family_ids[9]]),
        "dot": _template("dot", "dot", [family_ids[11]], detach_slots=frozenset({0})),
        "retrace": _template(
            "retrace", "other", [family_ids[10], family_ids[10]], directions=("forward", "reverse")
        ),
        "retrace-reverse": _template(
            "retrace-reverse",
            "other",
            [family_ids[10], family_ids[10]],
            directions=("reverse", "forward"),
        ),
    }
    default_shapes: dict[str, list[list[str]]] = {
        "a": [["oval", "exit"]],
        "b": [["ascender-loop", "stem"]],
        "c": [["oval"]],
        "d": [["oval", "ascender-loop"]],
        "e": [["oval", "exit"]],
        "f": [["ascender-loop", "cross-stroke"]],
        "g": [["oval", "descender-loop"]],
        "h": [["stem", "hump"]],
        "i": [["stem"], ["dot"]],
        "j": [["stem", "descender-loop"], ["dot"]],
        "k": [["stem", "diagonal", "diagonal-down"]],
        "l": [["stem"]],
        "m": [["double-hump"]],
        "n": [["stem", "underturn"]],
        "o": [["oval"]],
        "p": [["stem", "descender-loop"]],
        "q": [["oval", "descender-loop"]],
        "r": [["stem", "underturn"]],
        "s": [["retrace"]],
        "t": [["stem"], ["cross-stroke"]],
        "u": [["underturn", "exit"]],
        "v": [["underturn"]],
        "w": [["double-hump"]],
        "x": [["diagonal"], ["diagonal-down"]],
        "y": [["double-hump", "descender-loop"]],
        "z": [["cross-stroke", "diagonal", "cross-stroke"]],
    }
    programs = [
        _program(letter, "default", default_shapes[letter], templates, crossing=letter == "x")
        for letter in ascii_lowercase
    ]
    programs.extend(
        (
            _program(
                "i",
                "dot-first",
                [["dot"], ["stem"]],
                templates,
                alternative_note=(
                    "Alternate detached-mark order retained; neither order is reviewed evidence."
                ),
            ),
            _program(
                "x",
                "reverse-order",
                [["diagonal-down"], ["diagonal"]],
                templates,
                crossing=True,
                alternative_note=(
                    "Alternate crossing traversal retained as an explicit synthetic alternative."
                ),
            ),
            _program(
                "s",
                "reverse-first",
                [["retrace-reverse"]],
                templates,
                alternative_note=(
                    "Alternate retracing direction retained as an explicit synthetic alternative."
                ),
            ),
        )
    )
    structural_cases = {
        "oval-plus-exit": ("a",),
        "stem-plus-underturn": ("n", "r"),
        "double-hump": ("m", "w"),
        "ascender-and-descender-loops": ("b", "g", "p", "q", "y"),
        "crossing": ("x",),
        "detached-dot-and-main-stroke": ("i", "j"),
        "retracing": ("s",),
        "ambiguous-traversal-or-order": ("i", "x", "s"),
    }
    return SyntheticLowercaseGrammar(
        templates=MappingProxyType(templates),
        programs=tuple(sorted(programs, key=lambda item: (item.character, item.variant, item.id))),
        primitive_templates=MappingProxyType(primitive_templates),
        structural_cases=MappingProxyType(structural_cases),
    )
