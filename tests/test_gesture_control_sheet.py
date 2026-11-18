from __future__ import annotations

import json
from pathlib import Path

from inkgrammar.fitting.models import (
    CurvatureProfile,
    CurvatureStrokeGeometry,
    PrimitiveFrame,
    PrimitiveTemplate,
)
from inkgrammar.gestures import (
    GestureGlyphProgram,
    GesturePrimitiveSlot,
    GestureStroke,
    GestureTemplate,
    ProgramGesture,
    write_gesture_alphabet_control_sheet,
)
from inkgrammar.programs.models import ProgramLandmark


def _template() -> GestureTemplate:
    return GestureTemplate(
        id="stem-template",
        role="stem",
        primitives=[GesturePrimitiveSlot(id="main", primitive_family="curve-family-001")],
    )


def _program() -> GestureGlyphProgram:
    gesture = ProgramGesture(
        id="main-instance",
        gesture="stem-template",
        from_landmark="start",
        to_landmark="end",
    )
    return GestureGlyphProgram(
        id="glyph-a",
        character="a",
        variant="default",
        gestures={gesture.id: gesture},
        strokes=[GestureStroke(id="body", gestures=[gesture.id])],
        landmarks={
            "start": ProgramLandmark(id="start", kind="start", x=0.0, y=0.0),
            "end": ProgramLandmark(id="end", kind="end", x=0.0, y=1.0),
        },
        entry_landmark="start",
        exit_landmark="end",
    )


def _primitive() -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id="curve-family-001",
        kind="curvature-profile",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(0.0, 1.0)),
        curvature_stroke=CurvatureStrokeGeometry(
            length=1.0,
            start_angle_deg=90.0,
            curvature=CurvatureProfile(start_rate_deg=0.0, end_rate_deg=0.0),
        ),
    )


def test_gesture_control_sheet_is_deterministic_and_reports_semantic_geometry(
    tmp_path: Path,
) -> None:
    template = _template()
    program = _program()
    templates = {template.id: template}
    primitives = {"curve-family-001": _primitive()}
    first_path = tmp_path / "alphabet.png"
    second_path = tmp_path / "alphabet-second.png"

    first = write_gesture_alphabet_control_sheet(
        [program], templates, primitives, first_path, columns=1
    )
    second = write_gesture_alphabet_control_sheet(
        [program], templates, primitives, second_path, columns=1
    )

    assert first == second
    assert first["renderer"] == "StrokeOutline"
    assert first["target_count"] == first["rendered_count"] == 1
    assert first["glyphs"][0]["gesture_template_ids"] == [template.id]
    assert first["glyphs"][0]["gesture_roles"] == ["stem"]
    assert first["glyphs"][0]["geometry_family_ids"] == ["curve-family-001"]
    assert first["glyphs"][0]["ambiguity_status"] is None
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.with_suffix(".json").read_text()) == first
