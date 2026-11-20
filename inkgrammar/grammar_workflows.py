from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .gestures.compiler import expand_gesture_program
from .gestures.diagnostics import write_gesture_alphabet_control_sheet
from .gestures.dictionary import learn_gesture_dictionary
from .gestures.geometry import GeometryPrimitiveFamily
from .gestures.geometry_fit import fit_geometry_family
from .gestures.project import GestureProjectBundle
from .programs.compiler import compile_glyph_program


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_grammar_project(bundle: GestureProjectBundle) -> dict[str, Any]:
    output = bundle.output_dir / "materialize"
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for key, source in bundle.input_paths.items():
        target = input_dir / f"{key}{source.suffix}"
        shutil.copyfile(source, target)
        snapshots[key] = {
            "source": source.relative_to(bundle.project_dir).as_posix(),
            "snapshot": target.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    project_snapshot = output / "project.yaml"
    snapshot_config = bundle.project.model_dump(mode="json")
    for key, snapshot in snapshots.items():
        snapshot_config["files"][key] = snapshot["snapshot"]
    snapshot_config["output_dir"] = "run"
    project_snapshot.write_text(yaml.safe_dump(snapshot_config, sort_keys=True), encoding="utf-8")
    report = {
        "format": "inkgrammar-gesture-materialization",
        "format_version": 1,
        "project": bundle.project.model_dump(mode="json"),
        "inputs": snapshots,
        "project_snapshot": project_snapshot.relative_to(output).as_posix(),
    }
    _write_json(output / "manifest.json", report)
    return report


def learn_grammar_primitives(bundle: GestureProjectBundle) -> dict[str, Any]:
    from planarcurve.fitting import TrajectoryFitOptions, TrajectoryModelSelectionOptions

    settings = bundle.project.geometry_learning
    fit_options = TrajectoryFitOptions(
        change_count=0,
        samples=24,
        restarts=0,
        max_nfev=60,
    )
    selection_options = TrajectoryModelSelectionOptions(
        change_counts=tuple(settings.allowed_change_counts),
        validation_fraction=settings.held_out_fraction,
    )
    families = {}
    for family_id, occurrences in sorted(bundle.geometry_occurrences_by_family.items()):
        fit = fit_geometry_family(
            family_id,
            tuple(sorted(occurrences, key=lambda item: item.id)),
            fit_options=fit_options,
            selection_options=selection_options,
        )
        families[family_id] = {
            "family": fit.family.model_dump(mode="json"),
            "training_metrics": dict(fit.family.training_metrics),
            "held_out_metrics": dict(fit.family.held_out_metrics),
            "occurrence_fits": [
                {
                    "occurrence_id": item.occurrence_id,
                    "split": item.split,
                    "metrics": dict(item.metrics),
                }
                for item in fit.occurrence_fits
            ],
        }
    report = {
        "format": "inkgrammar-learned-geometry-families",
        "format_version": 1,
        "family_count": len(families),
        "families": families,
    }
    _write_json(bundle.output_dir / "learn-primitives" / "geometry-families.json", report)
    return report


def fit_grammar_gestures(bundle: GestureProjectBundle) -> dict[str, Any]:
    family_path = bundle.output_dir / "learn-primitives" / "geometry-families.json"
    if not family_path.is_file():
        raise ValueError("run grammar learn-primitives before fit-gestures")
    family_report = json.loads(family_path.read_text(encoding="utf-8"))
    geometry_families = {
        family_id: GeometryPrimitiveFamily.model_validate(value["family"])
        for family_id, value in family_report["families"].items()
    }
    result = learn_gesture_dictionary(
        tuple(bundle.gesture_templates[key] for key in sorted(bundle.gesture_templates)),
        geometry_families,
        bundle.gesture_occurrences,
        max_candidates=bundle.project.stroke_hypotheses.max_candidates,
    )
    report = {
        "format": "inkgrammar-fitted-gesture-dictionary",
        "format_version": 1,
        "template_fits": [
            {
                "template_id": fit.template.id,
                "fitted_parameter_values": dict(fit.fitted_parameter_values),
                "primitive_reuse_counts": dict(fit.primitive_reuse_counts),
                "training_metrics": dict(fit.training_metrics),
                "held_out_metrics": dict(fit.held_out_metrics),
            }
            for fit in result.template_fits
        ],
        "ranked_matches": [
            {
                "template_id": match.template_id,
                "occurrence_id": match.occurrence_id,
                "rank": match.rank,
                "ambiguous": match.ambiguous,
                "breakdown": {
                    "geometry_fit": match.breakdown.geometry_fit,
                    "constraint_fit": match.breakdown.constraint_fit,
                    "topology_compatibility": match.breakdown.topology_compatibility,
                    "parameter_deviation": match.breakdown.parameter_deviation,
                    "complexity": match.breakdown.complexity,
                    "evidence_level": match.breakdown.evidence_level,
                    "evidence_source_ids": list(match.breakdown.evidence_source_ids),
                },
            }
            for match in result.ranked_matches
        ],
    }
    _write_json(bundle.output_dir / "fit-gestures" / "gesture-dictionary.json", report)
    return report


def fit_grammar_glyphs(bundle: GestureProjectBundle) -> dict[str, Any]:
    programs = []
    for program in bundle.gesture_programs:
        directed = expand_gesture_program(program, bundle.gesture_templates)
        compiled = compile_glyph_program(directed, bundle.primitive_templates)
        programs.append(
            {
                "program_id": program.id,
                "character": program.character,
                "variant": program.variant,
                "directed_program": directed.model_dump(mode="json"),
                "compiled": {
                    "movement_ids": [movement.movement_id for movement in compiled.movements],
                    "stroke_count": len(compiled.stroke_paths),
                    "landmarks": sorted(compiled.landmarks),
                    "gesture_instance_count": directed.gesture_instance_count,
                    "exception_override_count": directed.exception_override_count,
                },
            }
        )
    report = {
        "format": "inkgrammar-compiled-gesture-glyphs",
        "format_version": 1,
        "program_count": len(programs),
        "programs": programs,
    }
    _write_json(bundle.output_dir / "fit-glyphs" / "glyph-programs.json", report)
    return report


def validate_grammar_project(bundle: GestureProjectBundle) -> dict[str, Any]:
    for program in bundle.gesture_programs:
        directed = expand_gesture_program(program, bundle.gesture_templates)
        compile_glyph_program(directed, bundle.primitive_templates)
    report = {
        "format": "inkgrammar-gesture-project-validation",
        "format_version": 1,
        "status": "valid",
        "project_format_version": bundle.project.format_version,
        "dataset": bundle.project.dataset,
        "style": bundle.project.style,
        "geometry_family_count": len(bundle.geometry_occurrences_by_family),
        "geometry_occurrence_count": sum(
            len(items) for items in bundle.geometry_occurrences_by_family.values()
        ),
        "gesture_template_count": len(bundle.gesture_templates),
        "gesture_occurrence_count": len(bundle.gesture_occurrences),
        "gesture_program_count": len(bundle.gesture_programs),
        "compiled_program_count": len(bundle.gesture_programs),
        "reviewed_annotation_count": (
            len(bundle.reviewed_annotations.movements) if bundle.reviewed_annotations else 0
        ),
        "validation_letters": bundle.project.validation.letters,
        "input_files": sorted(bundle.input_paths),
    }
    _write_json(bundle.output_dir / "validate" / "project.json", report)
    return report


def write_grammar_control_sheet(bundle: GestureProjectBundle) -> dict[str, Any]:
    path = bundle.output_dir / "control-sheet" / "alphabet.png"
    report = write_gesture_alphabet_control_sheet(
        bundle.gesture_programs,
        bundle.gesture_templates,
        bundle.primitive_templates,
        path,
        columns=8,
        supersampling=3,
    )
    return report


def inspect_grammar_letter(bundle: GestureProjectBundle, letter: str) -> dict[str, Any]:
    if len(letter) != 1 or letter not in bundle.project.validation.letters:
        raise ValueError("letter must be one configured lowercase ASCII character")
    programs = [program for program in bundle.gesture_programs if program.character == letter]
    if not programs:
        raise ValueError(f"no gesture program is configured for letter {letter}")
    report = {
        "format": "inkgrammar-letter-inspection",
        "format_version": 1,
        "letter": letter,
        "programs": [
            {
                "program_id": program.id,
                "variant": program.variant,
                "gesture_instances": [
                    {
                        "instance_id": instance.id,
                        "template_id": instance.gesture,
                        "role": bundle.gesture_templates[instance.gesture].role,
                        "geometry_family_ids": [
                            slot.primitive_family
                            for slot in bundle.gesture_templates[instance.gesture].primitives
                        ],
                    }
                    for stroke in program.strokes
                    for instance in (program.gestures[gesture_id] for gesture_id in stroke.gestures)
                ],
                "provenance": program.provenance.model_dump(mode="json"),
            }
            for program in programs
        ],
    }
    _write_json(bundle.output_dir / "inspect-letter" / f"{letter}.json", report)
    return report
