from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from inkgrammar.cli import app
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
    OccurrenceEvidence,
    ProgramGesture,
    load_gesture_training_project,
)
from inkgrammar.programs.models import ProgramLandmark

_CURVE_FAMILY = "curve-family-001"


def _primitive() -> PrimitiveTemplate:
    return PrimitiveTemplate(
        id=_CURVE_FAMILY,
        kind="curvature-profile",
        frame=PrimitiveFrame(start=(0.0, 0.0), nominal_end=(0.0, 1.0)),
        curvature_stroke=CurvatureStrokeGeometry(
            length=1.0,
            start_angle_deg=90.0,
            curvature=CurvatureProfile(start_rate_deg=0.0, end_rate_deg=0.0),
        ),
    )


def _program() -> GestureGlyphProgram:
    instance = ProgramGesture(
        id="main-instance",
        gesture="stem-template",
        from_landmark="start",
        to_landmark="end",
    )
    return GestureGlyphProgram(
        id="synthetic-a",
        character="a",
        variant="default",
        gestures={instance.id: instance},
        strokes=[GestureStroke(id="main", gestures=[instance.id])],
        landmarks={
            "start": ProgramLandmark(id="start", kind="start", x=0.0, y=0.0),
            "end": ProgramLandmark(id="end", kind="end", x=0.0, y=1.0),
        },
        entry_landmark="start",
        exit_landmark="end",
    )


def _template() -> GestureTemplate:
    return GestureTemplate(
        id="stem-template",
        role="stem",
        primitives=[GesturePrimitiveSlot(id="main", primitive_family=_CURVE_FAMILY)],
    )


def _points(index: int = 0) -> list[list[float]]:
    shift = index * 0.004
    return [[0.0, 0.0], [0.15 + shift, 0.3], [0.25 + shift, 0.7], [0.3, 1.0]]


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _write_project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    geometry_occurrences = [
        {
            "id": f"geometry-{index}",
            "source_observation_id": f"source-{index}",
            "points": _points(index),
            "split": "held-out" if index == 3 else "training",
            "evidence_ids": [f"evidence-{index}"],
        }
        for index in range(4)
    ]
    gesture_occurrences = [
        {
            "id": f"gesture-{index}",
            "primitive_curves": {"main": _points(index)},
            "evidence": OccurrenceEvidence(
                source_observation_id=f"source-{index}",
                evidence_level="inferred-ambiguous",
            ).model_dump(mode="json"),
            "split": "held-out" if index == 3 else "training",
        }
        for index in range(4)
    ]
    _write_yaml(
        root / "geometry.yaml",
        {
            "format": "inkgrammar-geometry-occurrences",
            "format_version": 1,
            "families": {_CURVE_FAMILY: geometry_occurrences},
        },
    )
    _write_yaml(
        root / "templates.yaml",
        {
            "format": "inkgrammar-gesture-templates",
            "format_version": 1,
            "templates": [_template().model_dump(mode="json")],
        },
    )
    _write_yaml(
        root / "occurrences.yaml",
        {
            "format": "inkgrammar-gesture-occurrences",
            "format_version": 1,
            "occurrences": gesture_occurrences,
        },
    )
    _write_yaml(
        root / "programs.yaml",
        {
            "format": "inkgrammar-gesture-programs",
            "format_version": 1,
            "programs": [_program().model_dump(mode="json")],
        },
    )
    _write_yaml(
        root / "primitives.yaml",
        {
            "format": "inkgrammar-primitive-templates",
            "format_version": 1,
            "primitives": {_CURVE_FAMILY: _primitive().model_dump(mode="json")},
        },
    )
    project_path = root / "project.yaml"
    _write_yaml(
        project_path,
        {
            "format": "inkgrammar-gesture-project",
            "format_version": 1,
            "dataset": "synthetic-test",
            "style": "test-style",
            "geometry_learning": {
                "allowed_change_counts": [0],
                "held_out_fraction": 0.25,
            },
            "validation": {"letters": "a"},
            "files": {
                "geometry_occurrences": "geometry.yaml",
                "gesture_templates": "templates.yaml",
                "gesture_occurrences": "occurrences.yaml",
                "gesture_programs": "programs.yaml",
                "primitive_templates": "primitives.yaml",
            },
            "output_dir": "runs/grammar",
        },
    )
    return project_path


def _invoke(runner: CliRunner, arguments: list[str]) -> str:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return result.output


def test_grammar_commands_create_deterministic_run_artifacts(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path / "project")
    bundle = load_gesture_training_project(project_path)
    runner = CliRunner()
    project_arg = str(project_path)

    first_validation = _invoke(runner, ["grammar", "validate", project_arg])
    second_validation = _invoke(runner, ["grammar", "validate", project_arg])
    assert first_validation == second_validation
    validation = json.loads(first_validation)
    assert validation["status"] == "valid"
    assert validation["compiled_program_count"] == 1

    materialized = json.loads(_invoke(runner, ["grammar", "materialize", project_arg]))
    snapshot_path = bundle.output_dir / "materialize" / "project.yaml"
    snapshot_bundle = load_gesture_training_project(snapshot_path)
    assert snapshot_bundle.project.dataset == "synthetic-test"
    manifest_path = bundle.output_dir / "materialize" / "manifest.json"
    before_manifest = manifest_path.read_bytes()
    _invoke(runner, ["grammar", "materialize", project_arg])
    assert manifest_path.read_bytes() == before_manifest
    assert materialized["format"] == "inkgrammar-gesture-materialization"

    learned_output = _invoke(runner, ["grammar", "learn-primitives", project_arg])
    learned = json.loads(learned_output)
    assert learned["family_count"] == 1
    assert _invoke(runner, ["grammar", "learn-primitives", project_arg]) == learned_output
    fitted_output = _invoke(runner, ["grammar", "fit-gestures", project_arg])
    fitted = json.loads(fitted_output)
    assert fitted["template_fits"][0]["template_id"] == "stem-template"
    assert all("family_fit_v5" not in line for line in (json.dumps(fitted),))
    assert _invoke(runner, ["grammar", "fit-gestures", project_arg]) == fitted_output

    glyphs_output = _invoke(runner, ["grammar", "fit-glyphs", project_arg])
    glyphs = json.loads(glyphs_output)
    assert glyphs["program_count"] == 1
    assert glyphs["programs"][0]["compiled"]["stroke_count"] == 1
    assert _invoke(runner, ["grammar", "fit-glyphs", project_arg]) == glyphs_output

    control_output = _invoke(runner, ["grammar", "control-sheet", project_arg])
    control = json.loads(control_output)
    assert control["renderer"] == "StrokeOutline"
    assert (bundle.output_dir / "control-sheet" / "alphabet.png").is_file()
    control_path = bundle.output_dir / "control-sheet" / "alphabet.png"
    first_control_sheet = control_path.read_bytes()
    assert _invoke(runner, ["grammar", "control-sheet", project_arg]) == control_output
    assert control_path.read_bytes() == first_control_sheet

    first_inspection = _invoke(
        runner, ["grammar", "inspect-letter", project_arg, "--letter", "a"]
    )
    second_inspection = _invoke(
        runner, ["grammar", "inspect-letter", project_arg, "--letter", "a"]
    )
    assert first_inspection == second_inspection
    inspection = json.loads(first_inspection)
    assert inspection["programs"][0]["gesture_instances"][0]["geometry_family_ids"] == [
        _CURVE_FAMILY
    ]
    assert (bundle.output_dir / "inspect-letter" / "a.json").is_file()


def test_project_inputs_are_confined_to_project_directory(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path / "project")
    outside = tmp_path / "outside.yaml"
    outside.write_text("{}\n", encoding="utf-8")
    project = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    project["files"]["geometry_occurrences"] = "../outside.yaml"
    _write_yaml(project_path, project)

    result = CliRunner().invoke(app, ["grammar", "validate", str(project_path)])

    assert result.exit_code != 0
    assert "grammar project inputs must be files beneath the project" in result.output



def test_fitting_generation_deprecation_gates_are_documented() -> None:
    documentation = (Path(__file__).resolve().parents[1] / "docs" / "GESTURE-GRAMMAR.md").read_text(
        encoding="utf-8"
    )

    assert "no `family_fit_v5` stack" in documentation
    assert "Do not deprecate or delete an existing generation" in documentation
    assert "training and held-out family and gesture metrics" in documentation
    assert "StrokeOutline coverage" in documentation
