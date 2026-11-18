from __future__ import annotations

import pytest
from pydantic import ValidationError

from inkgrammar.fitting.models import (
    CurvatureProfile,
    CurvatureStrokeGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
)
from inkgrammar.gestures import (
    GestureConstraint,
    GestureGlyphProgram,
    GestureLandmark,
    GesturePrimitiveSlot,
    GestureStroke,
    GestureTemplate,
    ProgramGesture,
    compile_gesture_program,
    expand_gesture_program,
)
from inkgrammar.programs.models import (
    ProgramLandmark,
    ProgramParameter,
    TangentContinuityConstraint,
)


def template() -> GestureTemplate:
    return GestureTemplate(
        id="underturn-basic",
        role="underturn",
        primitives=[
            GesturePrimitiveSlot(
                id="rise",
                primitive_family="curve-family-017",
                direction="reverse",
                from_landmark="start",
                to_landmark="apex",
            ),
            GesturePrimitiveSlot(
                id="fall",
                primitive_family="curve-family-018",
                from_landmark="apex",
                to_landmark="end",
            ),
        ],
        landmarks={
            "start": GestureLandmark(id="start", kind="start"),
            "apex": GestureLandmark(id="apex", kind="apex"),
            "end": GestureLandmark(id="end", kind="end"),
        },
        constraints=[
            GestureConstraint(
                id="smooth-join",
                kind="tangent-continuity",
                first="rise",
                second="fall",
                tolerance=5.0,
            )
        ],
    )


def program() -> GestureGlyphProgram:
    return GestureGlyphProgram(
        id="synthetic-a",
        character="a",
        variant="default",
        gestures={
            "body": ProgramGesture(
                id="body",
                gesture="underturn-basic",
                from_landmark="entry",
                to_landmark="exit",
                landmark_map={"apex": "apex"},
                parameter_overrides={"rise-length": 1.4},
            )
        },
        strokes=[GestureStroke(id="main", gestures=["body"])],
        landmarks={
            "entry": ProgramLandmark(id="entry", kind="start"),
            "apex": ProgramLandmark(id="apex", kind="turn"),
            "exit": ProgramLandmark(id="exit", kind="end"),
        },
        parameters=[
            ProgramParameter(
                id="rise-length",
                target="movement.body--rise.length",
                category="movement-geometry",
                initial=1.0,
                lower=0.5,
                upper=2.0,
            )
        ],
        entry_landmark="entry",
        exit_landmark="exit",
    )


def primitive(primitive_id: str, start_rate: float, end_rate: float) -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id=primitive_id,
        kind="curvature-profile",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(1.0, 0.0)),
        curvature_stroke=CurvatureStrokeGeometry(
            length=1.0,
            start_angle_deg=0.0,
            curvature=CurvatureProfile(
                start_rate_deg=start_rate,
                end_rate_deg=end_rate,
            ),
        ),
    )


def test_gesture_program_expands_and_compiles_through_directed_program_pipeline() -> None:
    gesture_program = program()
    gesture_template = template()
    templates = {gesture_template.id: gesture_template}
    directed = expand_gesture_program(gesture_program, templates)

    assert list(directed.movements) == ["body--rise", "body--fall"]
    assert directed.gesture_instance_count == 1
    assert directed.exception_override_count == 1
    assert [movement.direction for movement in directed.movements.values()] == [
        "reverse",
        "forward",
    ]
    assert directed.strokes[0].movements == ["body--rise", "body--fall"]
    assert directed.movements["body--rise"].to_landmark == "apex"
    assert directed.movements["body--fall"].from_landmark == "apex"
    assert any(isinstance(item, TangentContinuityConstraint) for item in directed.constraints)

    primitives = {
        "curve-family-017": primitive("curve-family-017", 0.0, 15.0),
        "curve-family-018": primitive("curve-family-018", 15.0, 0.0),
    }
    compiled = compile_gesture_program(gesture_program, templates, primitives)

    assert [movement.movement_id for movement in compiled.movements] == [
        "body--rise",
        "body--fall",
    ]
    assert compiled.movements[0].direction == "reverse"
    assert compiled.movements[0].trajectory.length == pytest.approx(1.4)


def test_gesture_program_round_trip_and_reference_validation() -> None:
    valid = program()
    assert GestureGlyphProgram.model_validate_json(valid.model_dump_json()) == valid

    discontinuous = valid.model_dump(mode="python")
    discontinuous["gestures"]["body"]["to_landmark"] = "apex"
    discontinuous["gestures"]["tail"] = {
        "id": "tail",
        "gesture": "underturn-basic",
        "from_landmark": "entry",
        "to_landmark": "exit",
    }
    discontinuous["strokes"][0]["gestures"].append("tail")
    with pytest.raises(ValidationError, match="discontinuous"):
        GestureGlyphProgram.model_validate(discontinuous)

    outside_bounds = valid.model_dump(mode="python")
    outside_bounds["gestures"]["body"]["parameter_overrides"]["rise-length"] = 3.0
    with pytest.raises(ValidationError, match="outside bounds"):
        GestureGlyphProgram.model_validate(outside_bounds)


def test_gesture_expansion_rejects_missing_templates_and_unmapped_landmarks() -> None:
    with pytest.raises(ValueError, match="unknown gesture template"):
        expand_gesture_program(program(), {})

    gesture_program = program()
    gesture_program.gestures["body"].landmark_map["missing"] = "apex"
    with pytest.raises(ValueError, match="unknown template landmarks"):
        expand_gesture_program(gesture_program, {"underturn-basic": template()})
