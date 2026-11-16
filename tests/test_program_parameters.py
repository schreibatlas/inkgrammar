from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inkgrammar.programs.models import DirectedGlyphProgram
from inkgrammar.programs.parameters import (
    bind_program_parameters,
    resolve_program_parameter_values,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_PROGRAM = ROOT / "data" / "programs" / "synthetic" / "lowercase-a-v1.yaml"


def load_program() -> DirectedGlyphProgram:
    return DirectedGlyphProgram.model_validate(yaml.safe_load(SYNTHETIC_PROGRAM.read_text()))


def test_synthetic_program_is_explicitly_labeled_and_human_readable() -> None:
    program = load_program()

    assert program.character == "a"
    assert program.provenance.construction == "synthetic"
    assert program.provenance.trajectory == "synthetic"
    assert program.strokes[0].movements == ["entry", "bowl", "stem", "exit"]
    assert program.parameters


def test_parameter_binding_preserves_order_and_stage_categories() -> None:
    program = load_program()
    bindings = bind_program_parameters(program)
    geometry = bind_program_parameters(program, category="movement-geometry")

    assert [binding.parameter_id for binding in bindings[:4]] == [
        "alignment-scale",
        "alignment-rotation",
        "alignment-translation-x",
        "alignment-translation-y",
    ]

    assert [(binding.owner_id, binding.path) for binding in geometry] == [
        ("entry", "length"),
        ("exit", "length"),
        ("bowl", "length"),
        ("stem", "start_angle"),
    ]
    assert bindings[4].shared_group == bindings[5].shared_group == "terminal-length"
    assert bindings[-1].category == "width"


def test_shared_parameter_overrides_resolve_to_one_value() -> None:
    program = load_program()

    values = resolve_program_parameter_values(program, {"entry-length": 0.31})
    assert values["entry-length"] == values["exit-length"] == 0.31
    with pytest.raises(ValueError, match="conflicting overrides"):
        resolve_program_parameter_values(
            program,
            {"entry-length": 0.31, "exit-length": 0.32},
        )


def test_parameter_model_rejects_invalid_bounds_and_prior_pairs() -> None:
    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"][0]["lower"] = 1.1
    with pytest.raises(ValidationError, match="initial value must be within bounds"):
        DirectedGlyphProgram.model_validate(data)

    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"][0].pop("prior_sigma")
    with pytest.raises(ValidationError, match="must be provided together"):
        DirectedGlyphProgram.model_validate(data)


def test_parameter_model_rejects_target_category_and_movement_errors() -> None:
    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"][0]["category"] = "width"
    with pytest.raises(ValidationError, match="does not match parameter category"):
        DirectedGlyphProgram.model_validate(data)

    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"][4]["target"] = "movement.unknown.length"
    with pytest.raises(ValidationError, match="unknown movement"):
        DirectedGlyphProgram.model_validate(data)


def test_parameter_model_rejects_incompatible_shared_group_definitions() -> None:
    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"][5]["initial"] = 0.3

    with pytest.raises(ValidationError, match="must share bounds and priors"):
        DirectedGlyphProgram.model_validate(data)


def test_parameter_model_rejects_duplicate_targets() -> None:
    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    duplicate = copy.deepcopy(data["parameters"][0])
    duplicate["id"] = "alignment-scale-copy"
    data["parameters"].append(duplicate)

    with pytest.raises(ValidationError, match="targets must be unique"):
        DirectedGlyphProgram.model_validate(data)


def test_parameter_rejects_unknown_movement_for_width_target() -> None:
    data = yaml.safe_load(SYNTHETIC_PROGRAM.read_text())
    data["parameters"].append(
        {
            "id": "missing-width",
            "target": "movement.unknown.width",
            "category": "width",
            "initial": 0.1,
            "lower": 0.05,
            "upper": 0.2,
        }
    )

    with pytest.raises(ValidationError, match="unknown movement"):
        DirectedGlyphProgram.model_validate(data)
