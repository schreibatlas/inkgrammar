from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .gestures.project import load_gesture_training_project
from .grammar_workflows import (
    fit_grammar_gestures,
    fit_grammar_glyphs,
    inspect_grammar_letter,
    learn_grammar_primitives,
    materialize_grammar_project,
    validate_grammar_project,
    write_grammar_control_sheet,
)

app = typer.Typer(no_args_is_help=True, help="Deterministic gesture grammar workflows.")


def _project(path: Path):
    try:
        return load_gesture_training_project(path)
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _report(payload: dict) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@app.command()
def materialize(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Copy validated project inputs into a deterministic run snapshot."""
    bundle = _project(project)
    try:
        report = materialize_grammar_project(bundle)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command("learn-primitives")
def learn_primitives(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Fit reusable geometry families from directed training occurrences."""
    bundle = _project(project)
    try:
        report = learn_grammar_primitives(bundle)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command("fit-gestures")
def fit_gestures(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Fit seeded gesture templates against learned geometry families."""
    bundle = _project(project)
    try:
        report = fit_grammar_gestures(bundle)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command("fit-glyphs")
def fit_glyphs(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Compile gesture glyphs through the existing directed program pipeline."""
    bundle = _project(project)
    try:
        report = fit_grammar_glyphs(bundle)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command()
def validate(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Validate project configuration, input references, and program composition."""
    bundle = _project(project)
    try:
        report = validate_grammar_project(bundle)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command("control-sheet")
def control_sheet(project: Path = typer.Argument(..., exists=True)) -> None:  # noqa: B008
    """Render the configured gesture alphabet to StrokeOutline previews."""
    bundle = _project(project)
    try:
        report = write_grammar_control_sheet(bundle)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)


@app.command("inspect-letter")
def inspect_letter(
    project: Path = typer.Argument(..., exists=True),  # noqa: B008
    letter: str = typer.Option(..., "--letter"),  # noqa: B008
) -> None:
    """Report configured templates, geometry families, and evidence for one letter."""
    bundle = _project(project)
    try:
        report = inspect_grammar_letter(bundle, letter)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _report(report)
