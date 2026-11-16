from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.programs.compiler import compile_glyph_program
from inkgrammar.programs.diagnostics import MOVEMENT_COLORS, render_program_diagnostic
from inkgrammar.programs.models import DirectedGlyphProgram

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = ROOT / "data" / "programs" / "synthetic" / "lowercase-a-v1.yaml"
PRIMITIVES_PATH = ROOT / "data" / "programs" / "synthetic" / "primitives.yaml"


def load_inputs() -> tuple[DirectedGlyphProgram, dict[str, PrimitiveTemplate]]:
    program = DirectedGlyphProgram.model_validate(yaml.safe_load(PROGRAM_PATH.read_text()))
    primitive_data = yaml.safe_load(PRIMITIVES_PATH.read_text())["primitives"]
    primitives = {
        primitive_id: PrimitiveTemplate.model_validate(value)
        for primitive_id, value in primitive_data.items()
    }
    return program, primitives


def test_synthetic_program_compiles_deterministically_through_planarcurve() -> None:
    program, primitives = load_inputs()
    compiled = compile_glyph_program(program, primitives)
    repeated = compile_glyph_program(program, primitives)

    assert compiled.program_id == program.id
    assert [movement.movement_id for movement in compiled.movements] == [
        "entry",
        "bowl",
        "stem",
        "exit",
    ]
    assert len(compiled.stroke_paths) == 1
    assert all(path.segments for path in compiled.stroke_paths)
    assert set(compiled.landmarks) == set(program.landmarks)
    for left, right in zip(compiled.movements[:-1], compiled.movements[1:], strict=True):
        np.testing.assert_allclose(left.end_pose.position, right.start_pose.position, atol=1e-9)
    for first, second in zip(compiled.movements, repeated.movements, strict=True):
        assert first.transform == second.transform
        assert first.cubic_path == second.cubic_path


def test_compiler_preserves_reverse_direction_and_applies_parameter_bounds() -> None:
    program_data = yaml.safe_load(PROGRAM_PATH.read_text())
    program_data["movements"]["bowl"]["direction"] = "reverse"
    program = DirectedGlyphProgram.model_validate(program_data)
    _, primitives = load_inputs()

    compiled = compile_glyph_program(program, primitives)

    bowl = next(movement for movement in compiled.movements if movement.movement_id == "bowl")
    assert bowl.direction == "reverse"
    with pytest.raises(ValueError, match="outside bounds"):
        compile_glyph_program(program, primitives, parameter_values={"alignment-scale": 3.0})


def test_compiler_rejects_missing_primitives_and_hard_constraint_failures() -> None:
    program, primitives = load_inputs()
    missing = dict(primitives)
    missing.pop("synthetic-entry")
    with pytest.raises(ValueError, match="unknown primitive"):
        compile_glyph_program(program, missing)

    data = yaml.safe_load(PROGRAM_PATH.read_text())
    data["constraints"].append(
        {
            "id": "impossible-position",
            "kind": "position",
            "landmark": "end",
            "x": 1000.0,
            "y": 1000.0,
            "hard": True,
        }
    )
    impossible = DirectedGlyphProgram.model_validate(data)
    with pytest.raises(ValueError, match="hard position constraint failed"):
        compile_glyph_program(impossible, primitives)


def test_compiler_validates_hard_endpoint_coordinates() -> None:
    program_data = yaml.safe_load(PROGRAM_PATH.read_text())
    program_data["landmarks"]["entry-turn"].update(x=1000.0, y=1000.0)
    program = DirectedGlyphProgram.model_validate(program_data)
    _, primitives = load_inputs()

    with pytest.raises(ValueError, match="endpoint mismatch"):
        compile_glyph_program(program, primitives)


def test_compiler_rejects_changes_to_nonexistent_primitive_parameters() -> None:
    program_data = yaml.safe_load(PROGRAM_PATH.read_text())
    program_data["parameters"].append(
        {
            "id": "missing-change",
            "target": "movement.bowl.curvature.change[9].at",
            "category": "movement-geometry",
            "initial": 0.5,
            "lower": 0.1,
            "upper": 0.9,
        }
    )
    program = DirectedGlyphProgram.model_validate(program_data)
    _, primitives = load_inputs()

    with pytest.raises(ValueError, match="absent from primitive"):
        compile_glyph_program(program, primitives)


def test_diagnostic_labels_each_movement_and_landmark_in_color() -> None:
    program, primitives = load_inputs()
    compiled = compile_glyph_program(program, primitives)

    image = render_program_diagnostic(compiled, program, size=(480, 360))
    color_counts = image.getcolors(maxcolors=image.width * image.height)
    assert color_counts is not None
    colors = {color for _, color in color_counts} - {(255, 255, 255)}

    assert image.size == (480, 360)
    assert len(colors.intersection(set(MOVEMENT_COLORS))) >= 4


def test_program_model_rejects_unknown_constraint_when_compiling() -> None:
    data = yaml.safe_load(PROGRAM_PATH.read_text())
    data["constraints"].append({"id": "bad", "kind": "unknown"})

    with pytest.raises(ValidationError):
        DirectedGlyphProgram.model_validate(data)
