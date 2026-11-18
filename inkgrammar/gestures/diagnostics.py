from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..fitting.models import PrimitiveTemplate
from ..programs.compiler import CompiledGlyphProgram
from ..programs.objective import render_program_coverage
from ..programs.observation import (
    GlyphObservation,
    prepare_glyph_observation,
    standard_frame,
)
from .compiler import compile_gesture_program
from .models import GestureGlyphProgram, GestureTemplate


def _program_metadata(
    program: GestureGlyphProgram,
    templates: Mapping[str, GestureTemplate],
) -> dict[str, object]:
    instances = [
        program.gestures[instance_id]
        for stroke in program.strokes
        for instance_id in stroke.gestures
    ]
    used_templates = [templates[instance.gesture] for instance in instances]
    family_ids = sorted(
        {slot.primitive_family for template in used_templates for slot in template.primitives}
    )
    return {
        "program_id": program.id,
        "character": program.character,
        "variant": program.variant,
        "gesture_instance_ids": [instance.id for instance in instances],
        "gesture_template_ids": [instance.gesture for instance in instances],
        "gesture_roles": [template.role for template in used_templates],
        "geometry_family_ids": family_ids,
        "complexity": {
            "primitive_instances": sum(len(template.primitives) for template in used_templates),
            "gesture_instances": len(instances),
            "free_parameters": sum(
                parameter.lower < parameter.upper for parameter in program.parameters
            ),
            "pen_lifts": max(0, len(program.strokes) - 1),
            "exception_overrides": sum(len(instance.parameter_overrides) for instance in instances),
        },
        "provenance": program.provenance.model_dump(mode="json"),
        "ambiguity_status": None,
    }


def _control_observation(
    cell_size: tuple[int, int],
    x_height_px: float,
) -> tuple[Image.Image, GlyphObservation]:
    width, height = cell_size
    image = Image.new("L", cell_size, 255)
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    frame = standard_frame((width / 6.0, height - 18.0), x_height_px)
    return image, prepare_glyph_observation(image, mask=mask, frame=frame)


def _render_cell(
    compiled: CompiledGlyphProgram,
    observation: GlyphObservation,
    supersampling: int,
) -> Image.Image:
    coverage = render_program_coverage(
        compiled,
        observation,
        samples_per_movement=64,
        supersampling=supersampling,
    )
    grayscale = np.asarray(np.clip((1.0 - coverage) * 255.0, 0, 255), dtype=np.uint8)
    return Image.fromarray(grayscale).convert("RGB")


def write_gesture_alphabet_control_sheet(
    programs: Sequence[GestureGlyphProgram],
    gesture_templates: Mapping[str, GestureTemplate],
    primitive_templates: Mapping[str, PrimitiveTemplate],
    output_path: str | Path,
    *,
    columns: int = 8,
    cell_size: tuple[int, int] = (112, 140),
    x_height_px: float = 78.0,
    supersampling: int = 3,
) -> dict[str, object]:
    """Render deterministic StrokeOutline previews and evidence metadata for gesture glyphs."""
    if not programs:
        raise ValueError("at least one gesture program is required")
    if columns < 1:
        raise ValueError("columns must be positive")
    if min(cell_size) < 64:
        raise ValueError("control-sheet cells must be at least 64 pixels in each dimension")
    if x_height_px <= 0.0:
        raise ValueError("x_height_px must be positive")
    if supersampling < 1:
        raise ValueError("supersampling must be positive")

    ordered = tuple(sorted(programs, key=lambda item: (item.character, item.variant, item.id)))
    keys = [(program.character, program.variant) for program in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("control-sheet programs must have unique character and variant pairs")

    cell_width, cell_height = cell_size
    rows = (len(ordered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height + 24), "white")
    font = ImageFont.load_default()
    image, observation = _control_observation(cell_size, x_height_px)
    glyphs: list[dict[str, object]] = []

    for index, program in enumerate(ordered):
        compiled = compile_gesture_program(program, gesture_templates, primitive_templates)
        panel = _render_cell(compiled, observation, supersampling)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, cell_width - 1, 17), fill=(239, 242, 246))
        draw.text(
            (3, 3),
            f"{program.character} | {program.variant}"[:24],
            font=font,
            fill=(20, 20, 20),
        )
        metadata = _program_metadata(program, gesture_templates)
        family_label = ",".join(metadata["geometry_family_ids"])
        gesture_label = ",".join(metadata["gesture_template_ids"])
        draw.rectangle((0, cell_height - 30, cell_width - 1, cell_height - 1), fill=(248, 248, 248))
        draw.text((3, cell_height - 27), f"G:{gesture_label}"[:24], font=font, fill=(35, 35, 35))
        draw.text((3, cell_height - 15), f"F:{family_label}"[:24], font=font, fill=(35, 35, 35))
        left = index % columns * cell_width
        top = index // columns * cell_height + 24
        sheet.paste(panel, (left, top))
        glyphs.append(metadata)

    ImageDraw.Draw(sheet).text(
        (6, 6),
        f"Gesture grammar control sheet | {len(ordered)} program(s) | StrokeOutline",
        font=font,
        fill=(20, 20, 20),
    )
    image_path = Path(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path, format="PNG", optimize=False, compress_level=9)
    report = {
        "format": "inkgrammar-gesture-control-sheet",
        "format_version": 1,
        "renderer": "StrokeOutline",
        "target_count": len(ordered),
        "rendered_count": len(ordered),
        "columns": columns,
        "cell_size": list(cell_size),
        "x_height_px": x_height_px,
        "supersampling": supersampling,
        "glyphs": glyphs,
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
