from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from inkgrammar.programs.models import (
    ContactConstraint,
    DirectedGlyphProgram,
    OrderingConstraint,
)


def program_data() -> dict:
    return {
        "id": "test-a",
        "character": "a",
        "variant": "synthetic",
        "landmarks": {
            "start": {"id": "start", "kind": "start"},
            "join": {"id": "join", "kind": "contact"},
            "end": {"id": "end", "kind": "end"},
        },
        "movements": {
            "entry": {
                "id": "entry",
                "primitive": "entry-primitive",
                "from_landmark": "start",
                "to_landmark": "join",
                "role": "entry",
            },
            "body": {
                "id": "body",
                "primitive": "body-primitive",
                "from_landmark": "join",
                "to_landmark": "end",
            },
        },
        "strokes": [{"id": "main", "movements": ["entry", "body"]}],
        "contacts": [
            {
                "id": "join-contact",
                "first_movement": "entry",
                "first_u": 1.0,
                "second_movement": "body",
                "second_u": 0.0,
                "mode": "must-touch",
                "tolerance": 0.02,
            }
        ],
        "constraints": [
            {"id": "contact-rule", "kind": "contact", "contact_id": "join-contact"},
            {
                "id": "order-rule",
                "kind": "ordering",
                "before_movement": "entry",
                "after_movement": "body",
            },
        ],
        "provenance": {"construction": "synthetic", "trajectory": "synthetic"},
    }


def test_program_models_validate_order_contacts_and_discriminated_constraints() -> None:
    program = DirectedGlyphProgram.model_validate(program_data())

    assert program.strokes[0].movements == ["entry", "body"]
    assert isinstance(program.constraints[0], ContactConstraint)
    assert isinstance(program.constraints[1], OrderingConstraint)
    assert program.contacts[0].mode == "must-touch"


def test_every_constraint_variant_is_discriminated_and_reference_checked() -> None:
    data = program_data()
    data["constraints"].extend(
        [
            {"id": "position", "kind": "position", "landmark": "join", "x": 0.0, "y": 0.0},
            {
                "id": "distance",
                "kind": "distance",
                "first_landmark": "start",
                "second_landmark": "join",
                "distance": 1.0,
            },
            {
                "id": "angle",
                "kind": "angle",
                "first_movement": "entry",
                "second_movement": "body",
                "angle_deg": 90.0,
            },
            {
                "id": "tangent",
                "kind": "tangent-continuity",
                "first_movement": "entry",
                "second_movement": "body",
            },
            {
                "id": "curvature",
                "kind": "curvature-continuity",
                "first_movement": "entry",
                "second_movement": "body",
            },
            {
                "id": "closure",
                "kind": "loop-closure",
                "start_landmark": "start",
                "end_landmark": "end",
                "tolerance": 0.0,
            },
        ]
    )

    program = DirectedGlyphProgram.model_validate(data)

    assert [constraint.kind for constraint in program.constraints[2:]] == [
        "position",
        "distance",
        "angle",
        "tangent-continuity",
        "curvature-continuity",
        "loop-closure",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["movements"]["entry"].update(to_landmark="missing"),
            "unknown to_landmark",
        ),
        (
            lambda data: data["strokes"][0].update(movements=["entry", "missing"]),
            "unknown movement",
        ),
        (lambda data: data["strokes"][0].update(movements=["body", "entry"]), "discontinuous"),
        (lambda data: data["strokes"][0].update(movements=["entry"]), "not assigned to a stroke"),
        (lambda data: data["contacts"][0].update(first_u=1.1), "less than or equal to 1"),
        (lambda data: data["constraints"][0].update(contact_id="missing"), "unknown contact"),
        (
            lambda data: data["constraints"][1].update(after_movement="entry"),
            "declared movement order",
        ),
    ],
)
def test_program_rejects_invalid_references_and_order(mutate, message: str) -> None:
    data = copy.deepcopy(program_data())
    mutate(data)

    with pytest.raises(ValidationError, match=message):
        DirectedGlyphProgram.model_validate(data)


def test_program_requires_landmark_coordinates_as_a_pair() -> None:
    data = program_data()
    data["landmarks"]["join"]["x"] = 1.0

    with pytest.raises(ValidationError, match="both x and y"):
        DirectedGlyphProgram.model_validate(data)


def test_program_rejects_duplicate_or_unused_movements() -> None:
    data = program_data()
    data["movements"]["body"]["to_landmark"] = "join"
    data["strokes"][0]["movements"] = ["entry", "body", "body"]

    with pytest.raises(ValidationError, match="used exactly once"):
        DirectedGlyphProgram.model_validate(data)
