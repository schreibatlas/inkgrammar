from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from planarcurve import evaluate_trajectory

from .compiler import CompiledGlyphProgram
from .models import DirectedGlyphProgram

MOVEMENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (35, 105, 190),
    (220, 85, 55),
    (40, 150, 100),
    (155, 85, 180),
    (205, 145, 30),
    (35, 155, 165),
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
