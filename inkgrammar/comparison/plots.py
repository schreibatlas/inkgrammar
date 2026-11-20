from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PlotSeries = tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]


def write_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[PlotSeries],
    event_progress: Sequence[float] = (),
    event_annotations: Sequence[tuple[float, str]] = (),
) -> None:
    width, height = 900, 440
    scale = 4
    left, top, right, bottom = 100, 70, 32, 76
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12 * scale)
    draw.text((left * scale, 15 * scale), title, fill=(20, 20, 20), font=font)

    all_x = np.concatenate([np.asarray(item[1], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item[2], dtype=float) for item in series])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    y_padding = (y_max - y_min) * 0.05
    y_min, y_max = y_min - y_padding, y_max + y_padding

    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(left + fraction * (plot_right - left)) * scale
        y = round(plot_bottom - fraction * (plot_bottom - top)) * scale
        draw.line((x, top * scale, x, plot_bottom * scale), fill=(232, 232, 232), width=scale)
        draw.line((left * scale, y, plot_right * scale, y), fill=(232, 232, 232), width=scale)
        draw.text(
            (x - 8 * scale, (plot_bottom + 8) * scale),
            f"{x_min + fraction * (x_max - x_min):.2g}",
            fill=(55, 55, 55),
            font=font,
        )
        draw.text(
            ((left - 52) * scale, y - 4 * scale),
            f"{y_min + fraction * (y_max - y_min):.2g}",
            fill=(55, 55, 55),
            font=font,
        )
    draw.line(
        (left * scale, top * scale, left * scale, plot_bottom * scale),
        fill=(40, 40, 40),
        width=2 * scale,
    )
    draw.line(
        (left * scale, plot_bottom * scale, plot_right * scale, plot_bottom * scale),
        fill=(40, 40, 40),
        width=2 * scale,
    )
    draw.text((left * scale, (height - 24) * scale), x_label, fill=(30, 30, 30), font=font)
    draw.text((left * scale, 42 * scale), y_label, fill=(30, 30, 30), font=font)

    for event in event_progress:
        x = left + (float(event) - x_min) / (x_max - x_min) * (plot_right - left)
        x_pixel = round(x * scale)
        draw.line(
            (x_pixel, top * scale, x_pixel, plot_bottom * scale),
            fill=(185, 185, 185),
            width=scale,
        )

    for index, (event, label) in enumerate(event_annotations):
        x = left + (float(event) - x_min) / (x_max - x_min) * (plot_right - left)
        x_pixel = round(x * scale)
        draw.line(
            (x_pixel, top * scale, x_pixel, plot_bottom * scale),
            fill=(145, 145, 145),
            width=scale,
        )
        draw.text(
            (
                min(x_pixel + scale, plot_right * scale - 36 * scale),
                (top + 2 + index % 3 * 14) * scale,
            ),
            label,
            fill=(85, 85, 85),
            font=font,
        )

    for index, (label, x_values, y_values, color) in enumerate(series):
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        x_pixels = left + (x_values - x_min) / (x_max - x_min) * (plot_right - left)
        y_pixels = plot_bottom - (y_values - y_min) / (y_max - y_min) * (plot_bottom - top)
        points = list(
            zip(
                np.rint(x_pixels * scale).astype(int),
                np.rint(y_pixels * scale).astype(int),
                strict=True,
            )
        )
        if len(points) >= 2:
            draw.line(points, fill=color, width=2 * scale)
        legend_y = (top + 10 + index * 16) * scale
        draw.line(
            (
                (plot_right - 160) * scale,
                legend_y + 5 * scale,
                (plot_right - 145) * scale,
                legend_y + 5 * scale,
            ),
            fill=color,
            width=3 * scale,
        )
        draw.text(
            ((plot_right - 140) * scale, legend_y),
            label,
            fill=(30, 30, 30),
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.Resampling.LANCZOS).save(path, format="PNG")
