from __future__ import annotations

import pytest
from pydantic import ValidationError

from inkgrammar.gestures import (
    GestureConstraint,
    GesturePrimitiveSlot,
    GestureTemplate,
    GestureTrainingConfig,
    MovementAnnotation,
    MovementAnnotations,
    WritingStyleParameters,
)


def test_gesture_template_round_trips_semantics_and_validates_references() -> None:
    template = GestureTemplate(
        id="underturn-basic",
        role="underturn",
        primitives=[
            GesturePrimitiveSlot(id="entry", primitive_family="curve-family-017"),
            GesturePrimitiveSlot(
                id="turn",
                primitive_family="curve-family-021",
                attach_to_previous=True,
            ),
        ],
        landmarks={
            "start": {"id": "start", "kind": "start"},
            "end": {"id": "end", "kind": "end"},
        },
        constraints=[
            GestureConstraint(
                id="join",
                kind="attachment",
                first="entry",
                second="turn",
                hard=True,
            )
        ],
        parameters=[
            {
                "id": "turn-length",
                "target": "turn.length",
                "category": "relative-length",
                "initial": 1.0,
                "lower": 0.8,
                "upper": 1.2,
                "shared_group": "underturn-length",
            }
        ],
    )

    restored = GestureTemplate.model_validate_json(template.model_dump_json())

    assert restored == template
    assert restored.role == "underturn"
    assert restored.primitives[0].primitive_family == "curve-family-017"

    with pytest.raises(ValidationError, match="unknown item"):
        GestureTemplate(
            id="bad",
            role="oval",
            primitives=[GesturePrimitiveSlot(id="one", primitive_family="curve-family-001")],
            constraints=[
                GestureConstraint(
                    id="bad-join",
                    kind="attachment",
                    first="one",
                    second="missing",
                )
            ],
        )


def test_movement_annotations_keep_clean_observation_separate_from_annotation_source() -> None:
    movement = MovementAnnotation(
        id="sample-1-main",
        source_observation_id="clean-glyph-1",
        annotation_source_id="annotated-page-1",
        stroke_id="main",
        stroke_order=0,
        stroke_order_status="reviewed",
        direction="forward",
        direction_status="reviewed",
        ordered_landmarks=[(0.0, 0.0), (0.5, 0.3), (1.0, 0.0)],
        gesture_spans=[{"id": "body", "start_index": 0, "end_index": 2, "role": "underturn"}],
    )
    annotations = MovementAnnotations(movements=[movement])
    restored = MovementAnnotations.model_validate_json(annotations.model_dump_json())

    assert restored.movements[0] == movement
    assert restored.movements[0].source_observation_id != restored.movements[0].annotation_source_id

    with pytest.raises(ValidationError, match="requires direction"):
        MovementAnnotation(
            id="missing-direction",
            source_observation_id="clean-glyph-1",
            stroke_id="main",
            direction_status="reviewed",
            ordered_landmarks=[(0.0, 0.0), (1.0, 1.0)],
        )


def test_versioned_training_config_defaults_and_validation() -> None:
    config = GestureTrainingConfig.model_validate(
        {
            "dataset": "data/training/sas-1968/corpus.yaml",
            "style": "sas-1968",
            "geometry_learning": {"allowed_change_counts": [0, 1, 2], "held_out_fraction": 0.25},
            "stroke_hypotheses": {"max_candidates": 4, "max_repeated_edges": 1},
            "validation": {"letters": "acn"},
        }
    )

    assert config.format == "inkgrammar-gesture-training"
    assert config.format_version == 1
    assert config.gesture_dictionary.seeded_roles
    assert config.validation.letters == "acn"
    assert GestureTrainingConfig.model_validate_json(config.model_dump_json()) == config

    with pytest.raises(ValidationError, match="must be unique"):
        GestureTrainingConfig(
            dataset="dataset",
            style="style",
            geometry_learning={"allowed_change_counts": [0, 1, 1]},
        )
    with pytest.raises(ValidationError):
        WritingStyleParameters(slant_deg=60.0, body_height=1.0, stroke_width=0.1)
