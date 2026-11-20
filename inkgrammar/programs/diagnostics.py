from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from planarcurve import evaluate_trajectory

from .compiler import CompiledGlyphProgram
from .models import DirectedGlyphProgram
from .objective import ProgramFitEnergy, render_program_coverage
from .observation import GlyphObservation

MOVEMENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (35, 105, 190),
    (220, 85, 55),
    (40, 150, 100),
    (155, 85, 180),
    (205, 145, 30),
    (35, 155, 165),
)


PANEL_TITLES = (
    "Original crop",
    "Foreground mask",
    "Segmentation alternatives",
    "Skeleton graph",
    "Stroke hypotheses",
    "Selected directed path",
    "Gesture boundaries",
    "Gesture names",
    "Geometry families",
    "Heading-rate profiles",
    "Re-rendered outline",
    "Raster overlay",
    "Loss breakdown",
    "Complexity",
    "Provenance and ambiguity",
)


def render_program_diagnostic(
    compiled: CompiledGlyphProgram,
    program: DirectedGlyphProgram,
    *,
    size: tuple[int, int] = (640, 640),
    margin: int = 32,
    samples_per_movement: int = 96,
) -> Image.Image:
    """Render movement-colored centerlines and labeled logical landmarks."""
    width, height = size
    if width < 64 or height < 64:
        raise ValueError("diagnostic image dimensions must be at least 64 pixels")
    if margin < 0 or margin * 2 >= min(width, height):
        raise ValueError("diagnostic margin must be nonnegative and smaller than half the image")
    if samples_per_movement < 2:
        raise ValueError("samples_per_movement must be at least 2")
    program_ids = {movement.id for movement in program.movements.values()}
    compiled_ids = {movement.movement_id for movement in compiled.movements}
    if program_ids != compiled_ids:
        raise ValueError("compiled movement IDs do not match the program")

    paths: list[np.ndarray] = []
    for movement in compiled.movements:
        local = evaluate_trajectory(movement.trajectory, count=samples_per_movement).xy
        world = movement.transform.apply(local)
        paths.append(compiled.frame_transform.apply(world))
    landmark_points = np.asarray(list(compiled.landmarks.values()), dtype=float).reshape(-1, 2)
    all_points = np.concatenate([*paths, landmark_points], axis=0)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def to_pixel(points: np.ndarray) -> list[tuple[int, int]]:
        values = (points - minimum) * scale
        values[:, 1] = height - margin - values[:, 1]
        values[:, 0] += margin
        return [(int(round(x)), int(round(y))) for x, y in values]

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for index, (movement, points) in enumerate(zip(compiled.movements, paths, strict=True)):
        color = MOVEMENT_COLORS[index % len(MOVEMENT_COLORS)]
        pixels = to_pixel(points)
        draw.line(pixels, fill=color, width=4, joint="curve")
        arrow_index = max(1, (len(pixels) * 2) // 3)
        tip = np.asarray(pixels[arrow_index], dtype=float)
        delta = tip - np.asarray(pixels[arrow_index - 1], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm > 0.0:
            unit = delta / norm
            perpendicular = np.array([-unit[1], unit[0]])
            base = tip - unit * 10.0
            arrow = [
                tuple(tip),
                tuple(base + perpendicular * 4.0),
                tuple(base - perpendicular * 4.0),
            ]
            draw.polygon(arrow, fill=color)
        for point in (pixels[0], pixels[-1]):
            radius = 4
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=color,
            )
        draw.text((pixels[0][0] + 5, pixels[0][1] - 12), movement.movement_id, fill=color)

    for landmark_id, position in compiled.landmarks.items():
        point = to_pixel(np.asarray(position, dtype=float).reshape(1, 2))[0]
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(20, 20, 20))
        draw.text((point[0] + 5, point[1] + 3), landmark_id, fill=(20, 20, 20))
    return image


_PANEL_WIDTH = 224
_PANEL_HEIGHT = 174
_PANEL_CONTENT_SIZE = (_PANEL_WIDTH - 16, _PANEL_HEIGHT - 36)


def _preview_image(value: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("diagnostic previews must be two-dimensional masks")
    if array.dtype == bool or (array.size and float(np.max(array)) <= 1.0):
        array = np.where(array, 0, 255)
    return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")


def _fit_preview(image: Image.Image) -> Image.Image:
    result = image.convert("RGB")
    result.thumbnail(_PANEL_CONTENT_SIZE, Image.Resampling.LANCZOS)
    return result


def _draw_panel(
    sheet: Image.Image,
    index: int,
    title: str,
    *,
    image: Image.Image | None = None,
    lines: Sequence[str] = (),
) -> None:
    left = index % 3 * _PANEL_WIDTH
    top = index // 3 * _PANEL_HEIGHT
    draw = ImageDraw.Draw(sheet)
    draw.rectangle(
        (left, top, left + _PANEL_WIDTH - 1, top + _PANEL_HEIGHT - 1),
        outline=(170, 170, 170),
    )
    draw.rectangle((left + 1, top + 1, left + _PANEL_WIDTH - 2, top + 19), fill=(235, 239, 244))
    draw.text((left + 5, top + 4), title, font=ImageFont.load_default(), fill=(20, 20, 20))
    if image is not None:
        preview = _fit_preview(image)
        x = left + (_PANEL_WIDTH - preview.width) // 2
        y = top + 24 + (_PANEL_HEIGHT - 30 - preview.height) // 2
        sheet.paste(preview, (x, y))
        return
    for line_index, line in enumerate(lines[:10]):
        draw.text(
            (left + 6, top + 26 + line_index * 13),
            str(line)[:36],
            font=ImageFont.load_default(),
            fill=(35, 35, 35),
        )


def _paths_preview(paths: Sequence[np.ndarray]) -> Image.Image | None:
    nonempty = [
        np.asarray(path, dtype=float).reshape(-1, 2) for path in paths if np.asarray(path).size
    ]
    if not nonempty:
        return None
    all_points = np.concatenate(nonempty, axis=0)
    minimum = all_points.min(axis=0)
    span = np.maximum(all_points.max(axis=0) - minimum, 1e-9)
    width, height = _PANEL_CONTENT_SIZE
    scale = min((width - 12) / span[0], (height - 12) / span[1])
    image = Image.new("RGB", _PANEL_CONTENT_SIZE, "white")
    draw = ImageDraw.Draw(image)
    for index, path in enumerate(nonempty):
        points = (path - minimum) * scale
        pixels = [(int(x + 6), int(height - y - 6)) for x, y in points]
        draw.line(pixels, fill=MOVEMENT_COLORS[index % len(MOVEMENT_COLORS)], width=2)
    return image


def _alternatives_preview(alternatives: Sequence[Image.Image | np.ndarray]) -> Image.Image | None:
    if not alternatives:
        return None
    width, height = _PANEL_CONTENT_SIZE
    tile_size = (width // 2, height // 2)
    sheet = Image.new("RGB", _PANEL_CONTENT_SIZE, "white")
    for index, alternative in enumerate(alternatives[:4]):
        tile = _preview_image(alternative)
        tile.thumbnail((tile_size[0] - 4, tile_size[1] - 4), Image.Resampling.LANCZOS)
        left = (index % 2) * tile_size[0] + (tile_size[0] - tile.width) // 2
        top = (index // 2) * tile_size[1] + (tile_size[1] - tile.height) // 2
        sheet.paste(tile, (left, top))
    return sheet


def _coverage_overlay(observation: GlyphObservation, coverage: np.ndarray) -> Image.Image:
    source = np.asarray(observation.image.convert("RGB"), dtype=float)
    alpha = np.clip(coverage, 0.0, 1.0)[..., None] * 0.65
    red = np.zeros_like(source)
    red[..., 0] = 255.0
    blended = source * (1.0 - alpha) + red * alpha
    return Image.fromarray(np.asarray(np.clip(blended, 0, 255), dtype=np.uint8))


def _loss_lines(energy: ProgramFitEnergy) -> list[str]:
    return [
        f"{name}: {value:.5f}"
        for name, value in energy.model_dump(mode="json").items()
        if name not in {"complexity_breakdown"}
    ]


def _complexity_lines(energy: ProgramFitEnergy) -> list[str]:
    return [
        f"{name}: {value if value is not None else 'unknown'}"
        for name, value in energy.complexity_breakdown.model_dump(mode="json").items()
    ]


def _heading_rate_preview(compiled: CompiledGlyphProgram) -> Image.Image | None:
    if not compiled.movements:
        return None
    width, height = _PANEL_CONTENT_SIZE
    image = Image.new("RGB", _PANEL_CONTENT_SIZE, "white")
    draw = ImageDraw.Draw(image)
    draw.line((8, height - 12, width - 4, height - 12), fill=(120, 120, 120), width=1)
    profiles = [movement.trajectory.heading_rate for movement in compiled.movements]
    rates = [
        rate
        for profile in profiles
        for rate in (
            profile.start_rate_deg_per_u,
            *(change.heading_rate_deg_per_u for change in profile.changes),
            profile.end_rate_deg_per_u,
        )
    ]
    low, high = min(rates), max(rates)
    span = max(high - low, 1e-9)
    for index, (movement, profile) in enumerate(zip(compiled.movements, profiles, strict=True)):
        positions = (0.0, *(change.at for change in profile.changes), 1.0)
        values = (
            profile.start_rate_deg_per_u,
            *(change.heading_rate_deg_per_u for change in profile.changes),
            profile.end_rate_deg_per_u,
        )
        points = [
            (
                int(8 + position * (width - 16)),
                int(height - 18 - (value - low) / span * (height - 34)),
            )
            for position, value in zip(positions, values, strict=True)
        ]
        color = MOVEMENT_COLORS[index % len(MOVEMENT_COLORS)]
        draw.line(points, fill=color, width=2)
        draw.text((points[0][0] + 2, points[0][1] - 10), movement.movement_id[:12], fill=color)
    return image


def _fit_diagnostic_payload(
    program: DirectedGlyphProgram,
    energy: ProgramFitEnergy,
    *,
    segmentation_alternatives: Sequence[Image.Image | np.ndarray],
    stroke_hypotheses: Sequence[np.ndarray],
    gesture_boundaries: Mapping[str, tuple[float, float]],
    gesture_names: Mapping[str, str],
    geometry_family_ids: Mapping[str, str],
    route_ambiguous: bool | None,
    direction_ambiguous: bool | None,
) -> dict[str, object]:
    return {
        "format": "inkgrammar-program-fit-diagnostic",
        "format_version": 1,
        "program_id": program.id,
        "character": program.character,
        "variant": program.variant,
        "panels": list(PANEL_TITLES),
        "segmentation_alternative_count": len(segmentation_alternatives),
        "stroke_hypothesis_count": len(stroke_hypotheses),
        "gesture_boundaries": {
            key: list(value) for key, value in sorted(gesture_boundaries.items())
        },
        "gesture_names": dict(sorted(gesture_names.items())),
        "geometry_family_ids": dict(sorted(geometry_family_ids.items())),
        "loss_breakdown": energy.model_dump(mode="json"),
        "complexity": energy.complexity_breakdown.model_dump(mode="json"),
        "provenance": program.provenance.model_dump(mode="json"),
        "ambiguity": {
            "route": route_ambiguous,
            "direction": direction_ambiguous,
        },
    }


def render_program_fit_diagnostic(
    compiled: CompiledGlyphProgram,
    program: DirectedGlyphProgram,
    observation: GlyphObservation,
    energy: ProgramFitEnergy,
    *,
    segmentation_alternatives: Sequence[Image.Image | np.ndarray] = (),
    stroke_hypotheses: Sequence[np.ndarray] = (),
    gesture_boundaries: Mapping[str, tuple[float, float]] | None = None,
    gesture_names: Mapping[str, str] | None = None,
    geometry_family_ids: Mapping[str, str] | None = None,
    route_ambiguous: bool | None = None,
    direction_ambiguous: bool | None = None,
) -> Image.Image:
    """Render the fifteen-panel evidence, rendering, loss, and ambiguity sheet."""
    boundaries = gesture_boundaries or {}
    names = gesture_names or {}
    families = geometry_family_ids or {}
    coverage = render_program_coverage(compiled, observation)
    mask_image = Image.fromarray(np.where(observation.mask, 0, 255).astype(np.uint8))
    skeleton = (
        Image.fromarray(np.where(observation.skeleton_analysis.skeleton, 0, 255).astype(np.uint8))
        if observation.skeleton_analysis is not None
        else None
    )
    panels: list[tuple[Image.Image | None, Sequence[str]]] = [
        (observation.image, ()),
        (mask_image, ()),
        (
            _alternatives_preview(segmentation_alternatives),
            () if segmentation_alternatives else ("No alternatives supplied",),
        ),
        (skeleton, () if skeleton is not None else ("Skeleton analysis not supplied",)),
        (
            _paths_preview(stroke_hypotheses),
            () if stroke_hypotheses else ("No path hypotheses supplied",),
        ),
        (
            render_program_diagnostic(compiled, program, size=_PANEL_CONTENT_SIZE, margin=8),
            (),
        ),
        (
            None,
            [
                f"{key}: {value[0]:.3f} to {value[1]:.3f}"
                for key, value in sorted(boundaries.items())
            ]
            or ["No boundaries supplied"],
        ),
        (
            None,
            [f"{key}: {value}" for key, value in sorted(names.items())]
            or ["No gesture labels supplied"],
        ),
        (
            None,
            [f"{key}: {value}" for key, value in sorted(families.items())]
            or ["No family IDs supplied"],
        ),
        (
            _heading_rate_preview(compiled),
            () if compiled.movements else ("No movements",),
        ),
        (Image.fromarray(np.asarray((1.0 - coverage) * 255.0, dtype=np.uint8)), ()),
        (_coverage_overlay(observation, coverage), ()),
        (None, _loss_lines(energy)),
        (None, _complexity_lines(energy)),
        (
            None,
            [
                f"construction: {program.provenance.construction}",
                f"trajectory: {program.provenance.trajectory}",
                f"sources: {', '.join(program.provenance.sources) or 'none'}",
                f"route ambiguous: {route_ambiguous if route_ambiguous is not None else 'unknown'}",
                f"direction ambiguous: "
                f"{direction_ambiguous if direction_ambiguous is not None else 'unknown'}",
            ],
        ),
    ]
    sheet = Image.new("RGB", (_PANEL_WIDTH * 3, _PANEL_HEIGHT * 5), "white")
    for index, (image, lines) in enumerate(panels):
        _draw_panel(sheet, index, PANEL_TITLES[index], image=image, lines=lines)
    return sheet


def write_program_fit_diagnostic(
    compiled: CompiledGlyphProgram,
    program: DirectedGlyphProgram,
    observation: GlyphObservation,
    energy: ProgramFitEnergy,
    output_path: str | Path,
    **diagnostic_inputs,
) -> dict[str, object]:
    image_path = Path(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_program_fit_diagnostic(
        compiled, program, observation, energy, **diagnostic_inputs
    )
    image.save(image_path, format="PNG", optimize=False, compress_level=9)
    payload = _fit_diagnostic_payload(
        program,
        energy,
        segmentation_alternatives=diagnostic_inputs.get("segmentation_alternatives", ()),
        stroke_hypotheses=diagnostic_inputs.get("stroke_hypotheses", ()),
        gesture_boundaries=diagnostic_inputs.get("gesture_boundaries") or {},
        gesture_names=diagnostic_inputs.get("gesture_names") or {},
        geometry_family_ids=diagnostic_inputs.get("geometry_family_ids") or {},
        route_ambiguous=diagnostic_inputs.get("route_ambiguous"),
        direction_ambiguous=diagnostic_inputs.get("direction_ambiguous"),
    )
    image_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
