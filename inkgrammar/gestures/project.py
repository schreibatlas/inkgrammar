from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import yaml
from pydantic import Field

from ..fitting.models import PrimitiveTemplate
from ..models import StrictModel
from .dictionary import GestureOccurrence
from .evidence import OccurrenceEvidence
from .geometry_fit import DirectedCurveOccurrence
from .models import GestureGlyphProgram, GestureTemplate
from .training import (
    GeometryLearningConfig,
    GestureDictionaryConfig,
    MovementAnnotations,
    StrokeHypothesesConfig,
    TrainingValidationConfig,
)


class GestureProjectFiles(StrictModel):
    geometry_occurrences: str = Field(min_length=1)
    gesture_templates: str = Field(min_length=1)
    gesture_occurrences: str = Field(min_length=1)
    gesture_programs: str = Field(min_length=1)
    primitive_templates: str = Field(min_length=1)
    movement_annotations: str | None = Field(default=None, min_length=1)


class GestureTrainingProject(StrictModel):
    format: Literal["inkgrammar-gesture-project"] = "inkgrammar-gesture-project"
    format_version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    style: str = Field(min_length=1)
    geometry_learning: GeometryLearningConfig = Field(default_factory=GeometryLearningConfig)
    stroke_hypotheses: StrokeHypothesesConfig = Field(default_factory=StrokeHypothesesConfig)
    gesture_dictionary: GestureDictionaryConfig = Field(default_factory=GestureDictionaryConfig)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
    files: GestureProjectFiles
    output_dir: str = Field(min_length=1)


@dataclass(frozen=True)
class GestureProjectBundle:
    project_path: Path
    project_dir: Path
    output_dir: Path
    project: GestureTrainingProject
    input_paths: Mapping[str, Path]
    geometry_occurrences_by_family: Mapping[str, tuple[DirectedCurveOccurrence, ...]]
    gesture_templates: Mapping[str, GestureTemplate]
    gesture_occurrences: tuple[GestureOccurrence, ...]
    gesture_programs: tuple[GestureGlyphProgram, ...]
    primitive_templates: Mapping[str, PrimitiveTemplate]
    reviewed_annotations: MovementAnnotations | None


def _document(path: Path, expected_format: str) -> Mapping[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    version = value.get("format_version")
    if value.get("format") != expected_format or type(version) is not int or version != 1:
        raise ValueError(f"{path.name} has an unsupported {expected_format} format")
    return value


def _input_path(project_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("grammar project inputs must use project-relative paths")
    resolved = (project_dir / path).resolve(strict=True)
    if not resolved.is_relative_to(project_dir) or not resolved.is_file():
        raise ValueError("grammar project inputs must be files beneath the project directory")
    return resolved


def _geometry_occurrences(path: Path) -> dict[str, tuple[DirectedCurveOccurrence, ...]]:
    document = _document(path, "inkgrammar-geometry-occurrences")
    raw_families = document.get("families")
    if not isinstance(raw_families, Mapping) or not raw_families:
        raise ValueError("geometry occurrence file requires a nonempty families mapping")
    result: dict[str, tuple[DirectedCurveOccurrence, ...]] = {}
    occurrence_ids: set[str] = set()
    for family_id, raw_occurrences in sorted(raw_families.items()):
        if not isinstance(family_id, str) or not isinstance(raw_occurrences, list):
            raise ValueError("geometry families must map IDs to occurrence lists")
        occurrences = []
        for item in raw_occurrences:
            if not isinstance(item, Mapping):
                raise ValueError(f"geometry occurrence in {family_id} must be a mapping")
            occurrence = DirectedCurveOccurrence(
                id=item["id"],
                source_observation_id=item["source_observation_id"],
                points=np.asarray(item["points"], dtype=float),
                split=item.get("split", "training"),
                evidence_ids=tuple(item.get("evidence_ids", ())),
            )
            if occurrence.id in occurrence_ids:
                raise ValueError(f"duplicate geometry occurrence ID: {occurrence.id}")
            occurrence_ids.add(occurrence.id)
            occurrences.append(occurrence)
        if not occurrences:
            raise ValueError(f"geometry family has no occurrences: {family_id}")
        result[family_id] = tuple(occurrences)
    return result


def _gesture_templates(path: Path) -> dict[str, GestureTemplate]:
    document = _document(path, "inkgrammar-gesture-templates")
    raw_templates = document.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("gesture template file requires a nonempty templates list")
    templates = [GestureTemplate.model_validate(item) for item in raw_templates]
    result = {template.id: template for template in templates}
    if len(result) != len(templates):
        raise ValueError("gesture template IDs must be unique")
    return result


def _gesture_occurrences(path: Path) -> tuple[GestureOccurrence, ...]:
    document = _document(path, "inkgrammar-gesture-occurrences")
    raw_occurrences = document.get("occurrences")
    if not isinstance(raw_occurrences, list) or not raw_occurrences:
        raise ValueError("gesture occurrence file requires a nonempty occurrences list")
    occurrences = []
    for raw in raw_occurrences:
        if not isinstance(raw, Mapping):
            raise ValueError("gesture occurrences must be mappings")
        values = dict(raw)
        values["evidence"] = OccurrenceEvidence.model_validate(values["evidence"])
        occurrences.append(GestureOccurrence(**values))
    ids = [occurrence.id for occurrence in occurrences]
    if len(ids) != len(set(ids)):
        raise ValueError("gesture occurrence IDs must be unique")
    return tuple(sorted(occurrences, key=lambda item: item.id))


def _gesture_programs(path: Path) -> tuple[GestureGlyphProgram, ...]:
    document = _document(path, "inkgrammar-gesture-programs")
    raw_programs = document.get("programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        raise ValueError("gesture program file requires a nonempty programs list")
    programs = tuple(GestureGlyphProgram.model_validate(item) for item in raw_programs)
    ids = [program.id for program in programs]
    character_variants = [(program.character, program.variant) for program in programs]
    if len(ids) != len(set(ids)) or len(character_variants) != len(set(character_variants)):
        raise ValueError("gesture programs require unique IDs and character/variant pairs")
    return tuple(sorted(programs, key=lambda item: (item.character, item.variant, item.id)))


def _primitive_templates(path: Path) -> dict[str, PrimitiveTemplate]:
    document = _document(path, "inkgrammar-primitive-templates")
    raw_primitives = document.get("primitives")
    if not isinstance(raw_primitives, Mapping) or not raw_primitives:
        raise ValueError("primitive template file requires a nonempty primitives mapping")
    primitives = {
        primitive_id: PrimitiveTemplate.model_validate(raw)
        for primitive_id, raw in raw_primitives.items()
    }
    if any(primitive_id != primitive.id for primitive_id, primitive in primitives.items()):
        raise ValueError("primitive template keys must match primitive IDs")
    return primitives


def load_gesture_training_project(path: str | Path) -> GestureProjectBundle:
    """Load and validate a versioned grammar project and its project-owned inputs."""
    project_path = Path(path).resolve(strict=True)
    if not project_path.is_file():
        raise ValueError("grammar project path must be a file")
    project_dir = project_path.parent.resolve(strict=True)
    project = GestureTrainingProject.model_validate(
        yaml.safe_load(project_path.read_text(encoding="utf-8"))
    )
    input_paths = {
        key: _input_path(project_dir, value)
        for key, value in project.files.model_dump(exclude_none=True).items()
    }
    output_path = Path(project.output_dir)
    output_dir = output_path if output_path.is_absolute() else project_dir / output_path
    output_dir = output_dir.resolve()

    geometry_occurrences = _geometry_occurrences(input_paths["geometry_occurrences"])
    templates = _gesture_templates(input_paths["gesture_templates"])
    occurrences = _gesture_occurrences(input_paths["gesture_occurrences"])
    programs = _gesture_programs(input_paths["gesture_programs"])
    invalid_letters = {program.character for program in programs} - set(project.validation.letters)
    if invalid_letters:
        raise ValueError(
            f"gesture programs are outside configured validation letters: {sorted(invalid_letters)}"
        )
    primitives = _primitive_templates(input_paths["primitive_templates"])
    annotations = (
        MovementAnnotations.model_validate(
            yaml.safe_load(input_paths["movement_annotations"].read_text(encoding="utf-8"))
        )
        if "movement_annotations" in input_paths
        else None
    )

    family_ids = set(geometry_occurrences)
    if not family_ids.issubset(primitives):
        raise ValueError("primitive templates must include every geometry family ID")
    for template in templates.values():
        missing = {slot.primitive_family for slot in template.primitives} - family_ids
        if missing:
            raise ValueError(
                f"gesture template {template.id} references unknown geometry families: "
                f"{sorted(missing)}"
            )
    template_ids = set(templates)
    for program in programs:
        missing = {gesture.gesture for gesture in program.gestures.values()} - template_ids
        if missing:
            raise ValueError(
                f"gesture program {program.id} references unknown templates: {sorted(missing)}"
            )

    return GestureProjectBundle(
        project_path=project_path,
        project_dir=project_dir,
        output_dir=output_dir,
        project=project,
        input_paths=MappingProxyType(dict(sorted(input_paths.items()))),
        geometry_occurrences_by_family=MappingProxyType(geometry_occurrences),
        gesture_templates=MappingProxyType(templates),
        gesture_occurrences=occurrences,
        gesture_programs=programs,
        primitive_templates=MappingProxyType(primitives),
        reviewed_annotations=annotations,
    )
