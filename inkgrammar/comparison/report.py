from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from planarcurve.trajectory import evaluate_heading_rate
from skeletonpaths import SubpixelCenterlineOptions

from .metrics import ComparisonAnalysis, analyze_comparison, reconstruction_model_payload
from .models import ComparisonResult, ExtractedLetter, TopologySummary, WritingRouteCandidate
from .plots import PlotSeries, write_plot
from .routes import ROUTE_REPORT_LIMIT, ROUTE_SEARCH_LIMIT, align_writing_routes

_REFERENCE = (35, 105, 215)
_SAMPLE = (225, 75, 55)

_MIN_DIAGNOSTIC_SIDE = 640
_ROUTE_PAIR_REPORT_LIMIT = 8

_PURPLE = (145, 55, 165)

_ARTIFACTS = {
    "input_reference": "input-reference.png",
    "input_sample": "input-sample.png",
    "extraction_reference": "extraction-reference.png",
    "extraction_sample": "extraction-sample.png",
    "mask_reference": "mask-reference.png",
    "mask_sample": "mask-sample.png",
    "skeleton_reference": "skeleton-reference.png",
    "skeleton_sample": "skeleton-sample.png",
    "overlay_raster": "overlay-raster.png",
    "overlay_centerline": "overlay-centerline.png",
    "overlay_error_vectors": "overlay-error-vectors.png",
    "plot_position_error": "plot-position-error.png",
    "plot_xy": "plot-x-y.png",
    "plot_heading": "plot-heading.png",
    "plot_tangent_error": "plot-tangent-error.png",
    "plot_curvature": "plot-curvature.png",
    "plot_curvature_error": "plot-curvature-error.png",
    "plot_cumulative_turn": "plot-cumulative-turn.png",
    "plot_width": "plot-width.png",
    "writing_path_overlay": "writing-path-overlay.png",
    "writing_path_overlay_svg": "writing-path-overlay.svg",
    "steering_profile": "steering-profile.png",
    "route_alternatives": "route-alternatives.png",
    "route_cost_breakdown": "route-cost-breakdown.png",
    "route_pairs": "route-pairs.json",
    "ink_reconstruction_reference": "ink-reconstruction-reference.png",
    "ink_reconstruction_sample": "ink-reconstruction-sample.png",
    "ink_residual_reference": "ink-residual-reference.png",
    "ink_residual_sample": "ink-residual-sample.png",
    "report_json": "report.json",
    "report_markdown": "report.md",
    "report_html": "report.html",
}


def _alignment_transform(result: ComparisonResult) -> Any:
    if result.alignment is not None:
        return result.alignment.transform
    if result.route_alignment is not None:
        return result.route_alignment.transform
    raise ValueError("comparison has no fitted alignment")


def _alignment_direction(result: ComparisonResult) -> str:
    if result.alignment is not None:
        return result.alignment.direction
    if result.route_alignment is not None:
        return result.route_alignment.direction
    raise ValueError("comparison has no fitted alignment")


def _alignment_error(result: ComparisonResult) -> float:
    if result.alignment is not None:
        return result.alignment.mean_squared_error
    if result.route_alignment is not None:
        return result.route_alignment.mean_squared_error
    raise ValueError("comparison has no fitted alignment")


def _direction_label(candidate: WritingRouteCandidate) -> str:
    return "inferred" if candidate.orientation_status == "resolved" else "unresolved"


def _route_direction_interpretation(result: ComparisonResult) -> str:
    selection = result.route_selection
    if selection is None:
        return "No jointly selected route is available, so direction is unavailable."
    reference = _direction_label(selection.reference)
    sample = _direction_label(selection.sample)
    if reference == sample == "inferred":
        return (
            "Reference and sample directions are inferred from static image geometry, "
            "not observed historical pen traces."
        )
    return (
        f"Reference direction: {reference}; sample direction: {sample}. "
        "Arrows show a plausible geometric alignment and do not establish historical pen order."
    )


@dataclass(frozen=True)
class WrittenComparisonReport:
    output_dir: Path
    payload: dict[str, Any]

    @property
    def json_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_json"]

    @property
    def markdown_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_markdown"]

    @property
    def html_path(self) -> Path:
        return self.output_dir / _ARTIFACTS["report_html"]


def _save_rgb_crop(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="PNG")


def _draw_point(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    x, y = xy
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
    if label is not None:
        draw.text((x + 5, y - 8), label, fill=color, font=ImageFont.load_default())


def _write_extraction_images(letter: ExtractedLetter, output_dir: Path, side: str) -> None:
    image = letter.image.convert("RGB")
    mask = letter.mask_result.mask
    mask_path = output_dir / _ARTIFACTS[f"mask_{side}"]
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    extraction = image.copy()
    draw = ImageDraw.Draw(extraction)
    coordinates = np.argwhere(mask)
    if len(coordinates):
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        draw.rectangle((left, top, right, bottom), outline=(255, 180, 0), width=2)
    else:
        draw.text((4, 4), "No ink detected", fill=(220, 30, 30))
    endpoints = letter.diagnostics.topology.endpoints
    for point in endpoints[:2]:
        _draw_point(draw, point, (20, 110, 230))
    extraction.save(output_dir / _ARTIFACTS[f"extraction_{side}"], format="PNG")

    skeleton_image = image.copy()
    skeleton_draw = ImageDraw.Draw(skeleton_image)
    ys, xs = np.nonzero(letter.diagnostics.skeleton)
    skeleton_draw.point(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(20, 195, 95))
    for point in letter.diagnostics.topology.endpoints:
        _draw_point(skeleton_draw, point, (20, 110, 230), "E")
    for point in letter.diagnostics.topology.junctions:
        _draw_point(skeleton_draw, point, (220, 40, 180), "J")
    skeleton_image.save(output_dir / _ARTIFACTS[f"skeleton_{side}"], format="PNG")


def _canvas_bounds(
    first: np.ndarray,
    second: np.ndarray,
    padding: int = 12,
) -> tuple[int, int, int, int]:
    points = np.concatenate((first, second), axis=0)
    minimum = np.floor(points.min(axis=0)).astype(int) - padding
    maximum = np.ceil(points.max(axis=0)).astype(int) + padding
    return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])


def _canvas_size(
    bounds: tuple[int, int, int, int],
    scale: float = 1.0,
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    return (
        max(1, int(np.ceil((right - left + 1) * scale))),
        max(1, int(np.ceil((bottom - top + 1) * scale))),
    )


def _viewport_scale(bounds: tuple[int, int, int, int]) -> float:
    width, height = _canvas_size(bounds)
    return max(1.0, _MIN_DIAGNOSTIC_SIDE / min(width, height))


def _canvas_points(
    points: np.ndarray,
    bounds: tuple[int, int, int, int],
    scale: float = 1.0,
) -> list[tuple[int, int]]:
    left, top, _, _ = bounds
    coordinates = np.rint((points - np.array((left, top))) * scale).astype(int)
    return list(zip(coordinates[:, 0].tolist(), coordinates[:, 1].tolist(), strict=True))


def _draw_centerline_overlay(
    output: Path,
    reference_xy: np.ndarray,
    sample_xy: np.ndarray,
    *,
    error_vectors: bool = False,
    reference_raw_skeleton_xy: np.ndarray | None = None,
    sample_raw_skeleton_xy: np.ndarray | None = None,
    reference_topology: TopologySummary | None = None,
    sample_topology: TopologySummary | None = None,
    sample_transform: Any | None = None,
    reference_event_xy: np.ndarray | None = None,
    sample_event_xy: np.ndarray | None = None,
    reference_stroke_offsets: np.ndarray | None = None,
    sample_stroke_offsets: np.ndarray | None = None,
) -> None:
    reference_points_for_bounds = [reference_xy]
    sample_points_for_bounds = [sample_xy]
    if reference_raw_skeleton_xy is not None:
        reference_points_for_bounds.append(reference_raw_skeleton_xy)
    if sample_raw_skeleton_xy is not None:
        sample_points_for_bounds.append(sample_raw_skeleton_xy)
    if reference_event_xy is not None:
        reference_points_for_bounds.append(reference_event_xy)
    if sample_event_xy is not None:
        sample_points_for_bounds.append(sample_event_xy)
    bounds = _canvas_bounds(
        np.concatenate(reference_points_for_bounds, axis=0),
        np.concatenate(sample_points_for_bounds, axis=0),
    )
    scale = _viewport_scale(bounds)
    image = Image.new("RGB", _canvas_size(bounds, scale), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(2 * scale)))

    if reference_raw_skeleton_xy is not None:
        draw.point(_canvas_points(reference_raw_skeleton_xy, bounds, scale), fill=(130, 145, 140))
    if sample_raw_skeleton_xy is not None:
        draw.point(_canvas_points(sample_raw_skeleton_xy, bounds, scale), fill=(75, 165, 105))

    def split_strokes(points: np.ndarray, offsets: np.ndarray | None) -> tuple[np.ndarray, ...]:
        if offsets is None:
            return (points,)
        boundaries = np.asarray(offsets, dtype=int)
        return tuple(
            points[start:end]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            if end > start
        )

    reference_segments = split_strokes(
        _canvas_points(reference_xy, bounds, scale), reference_stroke_offsets
    )
    sample_segments = split_strokes(_canvas_points(sample_xy, bounds, scale), sample_stroke_offsets)
    for side, segments, color in (
        ("R", reference_segments, _REFERENCE),
        ("S", sample_segments, _SAMPLE),
    ):
        for stroke_index, points in enumerate(segments, start=1):
            draw.line(points, fill=color, width=max(2, round(2 * scale)))
            for endpoint, label in (
                (points[0], f"{side}{stroke_index} start"),
                (points[-1], f"{side}{stroke_index} end"),
            ):
                x, y = endpoint
                radius = max(4, round(4 * scale))
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline="white",
                    width=max(1, round(scale)),
                )
                draw.text(
                    (x + max(5, round(5 * scale)), y - max(8, round(8 * scale))),
                    label,
                    fill=color,
                    font=font,
                )
    for topology, transform in (
        (reference_topology, None),
        (sample_topology, sample_transform),
    ):
        if topology is None:
            continue
        for coordinates, color, label in (
            (topology.endpoints, (20, 110, 230), "E"),
            (topology.junctions, (220, 40, 180), "J"),
        ):
            points = np.asarray(coordinates, dtype=float).reshape((-1, 2))
            if transform is not None and len(points):
                points = transform.apply(points)
            for x, y in _canvas_points(points, bounds, scale):
                radius = max(3, round(3 * scale))
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline="white",
                    width=max(1, round(scale)),
                )
                draw.text((x + radius, y - radius), label, fill=color, font=font)

    for event_xy in (reference_event_xy, sample_event_xy):
        if event_xy is None:
            continue
        for x, y in _canvas_points(event_xy, bounds, scale):
            radius = max(3, round(3 * scale))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(235, 165, 20),
            )
    if error_vectors:
        for reference_segment, sample_segment in zip(
            reference_segments,
            sample_segments,
            strict=False,
        ):
            indexes = np.linspace(0, len(reference_segment) - 1, 24).round().astype(int)
            for index in indexes:
                start = reference_segment[index]
                sample_index = round(
                    index * (len(sample_segment) - 1) / max(len(reference_segment) - 1, 1)
                )
                end = sample_segment[sample_index]
                draw.line((start, end), fill=(55, 55, 55), width=max(1, round(scale)))
                radius = max(2, round(2 * scale))
                draw.ellipse(
                    (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
                    fill=_REFERENCE,
                )
                radius = max(2, round(2 * scale))
                draw.ellipse(
                    (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
                    fill=_SAMPLE,
                )
    image.save(output, format="PNG")


def _point_on_polyline(points: np.ndarray, progress: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(np.clip(progress, 0.0, 1.0)) * cumulative[-1]
    segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
    fraction = (target - cumulative[segment]) / max(lengths[segment], 1e-12)
    return points[segment] + fraction * (points[segment + 1] - points[segment])


def _display_strokes(candidate: WritingRouteCandidate, reverse: bool) -> tuple[np.ndarray, ...]:
    strokes = candidate.continuous_strokes[::-1] if reverse else candidate.continuous_strokes
    return tuple(stroke.xy[::-1] if reverse else stroke.xy for stroke in strokes)


def _route_event_point(
    candidate: WritingRouteCandidate,
    progress: float,
    reverse: bool,
) -> np.ndarray:
    target_progress = 1.0 - progress if reverse else progress
    total_length = sum(stroke.arc_length_px for stroke in candidate.continuous_strokes)
    target_length = target_progress * total_length
    strokes = _display_strokes(candidate, reverse)
    length_before = 0.0
    for stroke, points in zip(
        candidate.continuous_strokes[::-1] if reverse else candidate.continuous_strokes,
        strokes,
        strict=True,
    ):
        if target_length <= length_before + stroke.arc_length_px:
            local_progress = (target_length - length_before) / stroke.arc_length_px
            return _point_on_polyline(points, local_progress)
        length_before += stroke.arc_length_px
    return strokes[-1][-1]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    bounds: tuple[int, int, int, int],
    progress: float,
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(progress) * cumulative[-1]
    segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
    first, second = _canvas_points(points[[segment, segment + 1]], bounds, scale)
    draw.line((first, second), fill=color, width=max(2, round(2 * scale)))
    dx, dy = second[0] - first[0], second[1] - first[1]
    magnitude = max(float(np.hypot(dx, dy)), 1e-12)
    ux, uy = dx / magnitude, dy / magnitude
    arrow_length = 5 * scale
    arrow_width = 3 * scale
    left = (
        round(second[0] - arrow_length * ux - arrow_width * uy),
        round(second[1] - arrow_length * uy + arrow_width * ux),
    )
    right = (
        round(second[0] - arrow_length * ux + arrow_width * uy),
        round(second[1] - arrow_length * uy - arrow_width * ux),
    )
    draw.polygon((second, left, right), fill=color)


def _write_writing_path_overlay(result: ComparisonResult, output: Path) -> None:
    selection = result.route_selection
    if selection is None:
        return
    reverse_sample = _alignment_direction(result) == "reversed-for-best-match"
    reference_strokes = _display_strokes(selection.reference, False)
    sample_strokes = (
        tuple(path.xy for path in result.route_alignment.aligned_sample_strokes)
        if result.route_alignment is not None
        else tuple(
            _alignment_transform(result).apply(points)
            for points in _display_strokes(selection.sample, reverse_sample)
        )
    )
    reference_junctions = np.asarray(
        result.reference.diagnostics.topology.junctions, dtype=float
    ).reshape((-1, 2))
    sample_junctions = _alignment_transform(result).apply(
        np.asarray(result.sample.diagnostics.topology.junctions, dtype=float).reshape((-1, 2))
    )
    all_points = [*reference_strokes, *sample_strokes]
    if len(reference_junctions):
        all_points.append(reference_junctions)
    if len(sample_junctions):
        all_points.append(sample_junctions)
    bounds = _canvas_bounds(np.concatenate(all_points, axis=0), np.empty((0, 2), dtype=float))
    left, top, right, bottom = bounds
    bounds = (left, top - 24, right, bottom)
    scale = _viewport_scale(bounds)
    image = Image.new("RGB", _canvas_size(bounds, scale), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(2.5 * scale)))
    status_font = ImageFont.load_default(size=max(12, round(2.5 * scale)))
    draw.text(
        (round(4 * scale), round(2 * scale)),
        f"R direction: {_direction_label(selection.reference)}",
        fill=_REFERENCE,
        font=status_font,
    )
    draw.text(
        (round(4 * scale), round(7 * scale)),
        f"S direction: {_direction_label(selection.sample)}",
        fill=_SAMPLE,
        font=status_font,
    )
    draw.text(
        (round(4 * scale), round(12 * scale)),
        "Alignment arrows are not observed historical pen traces.",
        fill=(65, 65, 65),
        font=status_font,
    )
    for side, strokes, color in (
        ("R", reference_strokes, _REFERENCE),
        ("S", sample_strokes, _SAMPLE),
    ):
        for stroke_index, points in enumerate(strokes, start=1):
            pixel_points = _canvas_points(points, bounds, scale)
            draw.line(pixel_points, fill=color, width=max(2, round(3 * scale)))
            for fraction in (0.25, 0.5, 0.75):
                _draw_arrow(draw, points, bounds, fraction, color, scale)
            start_x, start_y = pixel_points[0]
            start_radius = max(4, round(4 * scale))
            draw.ellipse(
                (
                    start_x - start_radius,
                    start_y - start_radius,
                    start_x + start_radius,
                    start_y + start_radius,
                ),
                fill=color,
                outline="white",
                width=max(1, round(scale)),
            )
            label_offset = max(6, round(6 * scale))
            draw.text(
                (start_x + label_offset, start_y - label_offset),
                f"{side}{stroke_index} start",
                fill=color,
                font=font,
            )
            end_x, end_y = pixel_points[-1]
            end_radius = max(4, round(4 * scale))
            draw.rectangle(
                (end_x - end_radius, end_y - end_radius, end_x + end_radius, end_y + end_radius),
                fill=color,
                outline="white",
                width=max(1, round(scale)),
            )
            draw.text(
                (end_x + label_offset, end_y - label_offset),
                f"{side}{stroke_index} end",
                fill=color,
                font=font,
            )
    junction_radius = max(7, round(7 * scale))
    for points, color in ((reference_junctions, _REFERENCE), (sample_junctions, _SAMPLE)):
        for x, y in _canvas_points(points, bounds, scale):
            draw.ellipse(
                (
                    x - junction_radius,
                    y - junction_radius,
                    x + junction_radius,
                    y + junction_radius,
                ),
                outline=color,
                width=max(2, round(2 * scale)),
            )
            draw.text((x + junction_radius, y - junction_radius), "J", fill=color, font=font)
    for _side, candidate, reverse, transform in (
        ("R", selection.reference, False, None),
        ("S", selection.sample, reverse_sample, _alignment_transform(result)),
    ):
        for encounter_index, event in enumerate(
            (item for item in candidate.events if item.kind == "junction-encounter"),
            start=1,
        ):
            point = _route_event_point(candidate, event.progress, reverse)
            if transform is not None:
                point = transform.apply(point)
            x, y = _canvas_points(point.reshape((1, 2)), bounds, scale)[0]
            event_radius = max(3, round(3 * scale))
            draw.ellipse(
                (x - event_radius, y - event_radius, x + event_radius, y + event_radius),
                fill=(245, 165, 20),
            )
            draw.text(
                (x + event_radius, y + event_radius),
                str(event.occurrence or encounter_index),
                fill=(95, 70, 5),
                font=font,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def _write_writing_path_overlay_svg(result: ComparisonResult, output: Path) -> None:
    selection = result.route_selection
    if selection is None:
        return
    reverse_sample = _alignment_direction(result) == "reversed-for-best-match"
    reference_strokes = _display_strokes(selection.reference, False)
    sample_strokes = (
        tuple(path.xy for path in result.route_alignment.aligned_sample_strokes)
        if result.route_alignment is not None
        else tuple(
            _alignment_transform(result).apply(points)
            for points in _display_strokes(selection.sample, reverse_sample)
        )
    )
    reference_junctions = np.asarray(
        result.reference.diagnostics.topology.junctions, dtype=float
    ).reshape((-1, 2))
    sample_junctions = _alignment_transform(result).apply(
        np.asarray(result.sample.diagnostics.topology.junctions, dtype=float).reshape((-1, 2))
    )
    all_points = [*reference_strokes, *sample_strokes]
    if len(reference_junctions):
        all_points.append(reference_junctions)
    if len(sample_junctions):
        all_points.append(sample_junctions)
    bounds = _canvas_bounds(np.concatenate(all_points, axis=0), np.empty((0, 2), dtype=float))
    left, top, right, bottom = bounds
    bounds = (left, top - 24, right, bottom)
    scale = _viewport_scale(bounds)
    width, height = _canvas_size(bounds, scale)
    stroke_width = max(2.0, 3.0 * scale)
    font_size = max(12.0, 2.5 * scale)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<g font-family="sans-serif" font-size="{font_size:g}">',
        f'<text x="{4 * scale:g}" y="{14 * scale:g}" fill="#235fd7">'
        f"Reference direction: {_direction_label(selection.reference)}</text>",
        f'<text x="{4 * scale:g}" y="{19 * scale:g}" fill="#e14b37">'
        f"Sample direction: {_direction_label(selection.sample)}</text>",
        f'<text x="{4 * scale:g}" y="{24 * scale:g}" fill="#414141">'
        "Alignment arrows are not observed historical pen traces.</text>",
    ]
    colors = (("R", reference_strokes, "#2369d7"), ("S", sample_strokes, "#e14b37"))
    for side, strokes, color in colors:
        for stroke_index, points in enumerate(strokes, start=1):
            pixels = _canvas_points(points, bounds, scale)
            point_list = " ".join(f"{x},{y}" for x, y in pixels)
            lines.append(
                f'<polyline points="{point_list}" fill="none" stroke="{color}" '
                f'stroke-width="{stroke_width:g}" stroke-linecap="round" '
                'stroke-linejoin="round"/>'
            )
            start_x, start_y = pixels[0]
            end_x, end_y = pixels[-1]
            marker_radius = max(4.0, 4.0 * scale)
            lines.append(
                f'<circle cx="{start_x}" cy="{start_y}" r="{marker_radius:g}" '
                f'fill="{color}" stroke="white"/> '
                f'<text x="{start_x + 6 * scale:g}" y="{start_y - 6 * scale:g}" '
                f'fill="{color}">{side}{stroke_index} start</text>'
            )
            end_size = max(8.0, 8.0 * scale)
            lines.append(
                f'<rect x="{end_x - end_size / 2:g}" y="{end_y - end_size / 2:g}" '
                f'width="{end_size:g}" height="{end_size:g}" fill="{color}" '
                f'stroke="white"/> <text x="{end_x + 6 * scale:g}" '
                f'y="{end_y - 6 * scale:g}" fill="{color}">{side}{stroke_index} end</text>'
            )
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
            target = 0.5 * cumulative[-1]
            segment = min(
                int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1
            )
            first, second = np.asarray(
                _canvas_points(points[[segment, segment + 1]], bounds, scale)
            )
            dx, dy = second - first
            magnitude = max(float(np.hypot(dx, dy)), 1e-12)
            ux, uy = dx / magnitude, dy / magnitude
            arrow_length, arrow_width = 5 * scale, 3 * scale
            arrow_points = (
                second,
                second - arrow_length * np.array((ux, uy)) + arrow_width * np.array((-uy, ux)),
                second - arrow_length * np.array((ux, uy)) - arrow_width * np.array((-uy, ux)),
            )
            arrow = " ".join(f"{x:.2f},{y:.2f}" for x, y in arrow_points)
            lines.append(f'<polygon points="{arrow}" fill="{color}"/>')
    for points, color in ((reference_junctions, "#2369d7"), (sample_junctions, "#e14b37")):
        for x, y in _canvas_points(points, bounds, scale):
            radius = max(7.0, 7.0 * scale)
            lines.append(
                f'<circle cx="{x}" cy="{y}" r="{radius:g}" fill="none" '
                f'stroke="{color}" stroke-width="{max(2.0, 2.0 * scale):g}"/>'
                f'<text x="{x + radius:g}" y="{y - radius:g}" fill="{color}">J</text>'
            )
    for _side, candidate, reverse, transform in (
        ("R", selection.reference, False, None),
        ("S", selection.sample, reverse_sample, _alignment_transform(result)),
    ):
        for encounter_index, event in enumerate(
            (item for item in candidate.events if item.kind == "junction-encounter"),
            start=1,
        ):
            point = _route_event_point(candidate, event.progress, reverse)
            if transform is not None:
                point = transform.apply(point)
            x, y = _canvas_points(point.reshape((1, 2)), bounds, scale)[0]
            radius = max(3.0, 3.0 * scale)
            label = event.occurrence or encounter_index
            lines.append(
                f'<circle cx="{x}" cy="{y}" r="{radius:g}" fill="#f5a514"/>'
                f'<text x="{x + radius:g}" y="{y + radius:g}" fill="#5f4605">{label}</text>'
            )
    lines.extend(("</g>", "</svg>"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _route_event_annotations(
    candidate: WritingRouteCandidate,
    side: str,
    reverse: bool,
) -> list[tuple[int, float, str]]:
    total_length = sum(stroke.arc_length_px for stroke in candidate.continuous_strokes)
    annotations = []
    for event in candidate.events:
        if event.kind != "junction-encounter":
            continue
        progress = 1.0 - event.progress if reverse else event.progress
        target_length = progress * total_length
        length_before = 0.0
        stroke_order = (
            candidate.continuous_strokes[::-1] if reverse else candidate.continuous_strokes
        )
        for stroke_index, stroke in enumerate(stroke_order):
            if target_length <= length_before + stroke.arc_length_px:
                local_progress = (target_length - length_before) / stroke.arc_length_px
                annotations.append(
                    (stroke_index, float(local_progress), f"{side}-J{event.occurrence or 1}")
                )
                break
            length_before += stroke.arc_length_px
    return annotations


def _write_steering_profile(result: ComparisonResult, output: Path) -> None:
    selection = result.route_selection
    if selection is None:
        return
    progress = np.linspace(0.0, 1.0, 101)
    series = []
    annotations: list[tuple[float, str]] = []
    for side, candidate, reverse, color in (
        ("Reference", selection.reference, False, _REFERENCE),
        (
            "Sample",
            selection.sample,
            _alignment_direction(result) == "reversed-for-best-match",
            _SAMPLE,
        ),
    ):
        event_annotations = _route_event_annotations(candidate, side[0], reverse)
        for stroke_index, model in enumerate(candidate.steering_models):
            rates = evaluate_heading_rate(model.selected.trajectory.heading_rate, progress)
            if reverse:
                rates = -rates[::-1]
            series.append(_series(f"{side} stroke {stroke_index + 1}", progress, rates, color))
            for change_index, change in enumerate(
                model.selected.trajectory.heading_rate.changes, start=1
            ):
                knot = 1.0 - change.at if reverse else change.at
                annotations.append((float(knot), f"{side[0]}-K{change_index}"))
            annotations.extend(
                (position, label)
                for index, position, label in event_annotations
                if index == stroke_index
            )
    write_plot(
        output,
        title="Fitted heading-rate profiles",
        x_label="Inferred stroke progress",
        y_label="Heading rate (degrees / progress)",
        series=series,
        event_annotations=annotations,
    )


def _write_ink_reconstruction_panels(analysis: ComparisonAnalysis, output_dir: Path) -> None:
    for side in ("reference", "sample"):
        observed = np.asarray(analysis.series[f"{side}_observed_ink"], dtype=bool)
        coverage = np.asarray(analysis.series[f"{side}_rendered_coverage"], dtype=float)
        under = np.asarray(analysis.series[f"{side}_under_coverage"], dtype=bool)
        over = np.asarray(analysis.series[f"{side}_over_coverage"], dtype=bool)
        height, width = observed.shape
        scale = _viewport_scale((0, 0, width - 1, height - 1))
        panel_size = _canvas_size((0, 0, width - 1, height - 1), scale)
        label_height = 22
        gap = 8
        observed_image = np.full((height, width, 3), 255, dtype=np.uint8)
        observed_image[observed] = (25, 25, 25)
        rendered_gray = np.rint(255 * (1.0 - coverage)).astype(np.uint8)
        rendered_image = np.repeat(rendered_gray[:, :, None], 3, axis=2)
        under_image = np.full((height, width, 3), 255, dtype=np.uint8)
        under_image[under] = (225, 70, 45)
        over_image = np.full((height, width, 3), 255, dtype=np.uint8)
        over_image[over] = (135, 55, 165)
        nearest = Image.Resampling.NEAREST
        panels = (
            ("Observed ink", Image.fromarray(observed_image).resize(panel_size, nearest)),
            (
                "Rendered coverage",
                Image.fromarray(rendered_image).resize(panel_size, Image.Resampling.BILINEAR),
            ),
            ("Under-coverage", Image.fromarray(under_image).resize(panel_size, nearest)),
            ("Over-coverage", Image.fromarray(over_image).resize(panel_size, nearest)),
        )
        panel_width, panel_height = panel_size
        sheet = Image.new(
            "RGB",
            (2 * panel_width + gap, 2 * (panel_height + label_height) + gap),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=14)
        for index, (label, panel) in enumerate(panels):
            x = (index % 2) * (panel_width + gap)
            y = (index // 2) * (panel_height + label_height + gap)
            draw.text((x + 2, y + 2), label, fill=(30, 30, 30), font=font)
            sheet.paste(panel, (x, y + label_height))
        sheet.save(output_dir / _ARTIFACTS[f"ink_reconstruction_{side}"], format="PNG")

        residual = np.full((height, width, 3), 255, dtype=np.uint8)
        residual[under] = (225, 70, 45)
        residual[over] = (135, 55, 165)
        Image.fromarray(residual).resize(panel_size, nearest).save(
            output_dir / _ARTIFACTS[f"ink_residual_{side}"], format="PNG"
        )


def _write_raster_overlay(result: ComparisonResult, output: Path) -> None:
    reference_yx = np.argwhere(result.reference.mask_result.mask)
    sample_yx = np.argwhere(result.sample.mask_result.mask)
    reference_xy = reference_yx[:, ::-1].astype(float)
    sample_xy = _alignment_transform(result).apply(sample_yx[:, ::-1].astype(float))
    bounds = _canvas_bounds(reference_xy, sample_xy, padding=4)
    width, height = _canvas_size(bounds)
    left, top, _, _ = bounds
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    reference_pixels = np.zeros((height, width), dtype=bool)
    sample_pixels = np.zeros((height, width), dtype=bool)
    for points, target in ((reference_xy, reference_pixels), (sample_xy, sample_pixels)):
        coordinates = np.rint(points - np.array((left, top))).astype(int)
        x = np.clip(coordinates[:, 0], 0, width - 1)
        y = np.clip(coordinates[:, 1], 0, height - 1)
        target[y, x] = True
    image[sample_pixels] = _SAMPLE
    image[reference_pixels] = _REFERENCE
    image[reference_pixels & sample_pixels] = _PURPLE
    Image.fromarray(image).save(output, format="PNG")


def _series(label: str, x: np.ndarray, y: np.ndarray, color: tuple[int, int, int]) -> PlotSeries:
    return label, x, y, color


def _write_plots(analysis: ComparisonAnalysis, output_dir: Path) -> None:
    q = analysis.progress
    data = analysis.series
    plots = [
        (
            "plot_position_error",
            "Normalized position error",
            "Normalized stroke progress q",
            "Error / reference arc length",
            [_series("distance", q, data["position_error"], _PURPLE)],
        ),
        (
            "plot_xy",
            "X/Y coordinates by progress",
            "Normalized stroke progress q",
            "Coordinate / reference arc length",
            [
                _series("reference x", q, data["reference_xy_normalized"][:, 0], _REFERENCE),
                _series("sample x", q, data["sample_xy_normalized"][:, 0], _SAMPLE),
                _series("reference y", q, data["reference_xy_normalized"][:, 1], (25, 155, 145)),
                _series("sample y", q, data["sample_xy_normalized"][:, 1], (225, 145, 30)),
            ],
        ),
        (
            "plot_heading",
            "Heading by progress",
            "Normalized stroke progress q",
            "Heading (degrees)",
            [
                _series("reference", q, data["reference_heading_deg"], _REFERENCE),
                _series("sample", q, data["sample_heading_deg"], _SAMPLE),
            ],
        ),
        (
            "plot_tangent_error",
            "Signed tangent error",
            "Normalized stroke progress q",
            "Sample minus reference (degrees)",
            [_series("tangent error", q, data["tangent_error_deg"], _PURPLE)],
        ),
        (
            "plot_curvature",
            "Normalized turn-rate",
            "Normalized stroke progress q",
            "Turn-rate (radians / progress)",
            [
                _series("reference", q, data["reference_turn_rate"], _REFERENCE),
                _series("sample", q, data["sample_turn_rate"], _SAMPLE),
            ],
        ),
        (
            "plot_curvature_error",
            "Turn-rate error",
            "Normalized stroke progress q",
            "Sample minus reference (radians / progress)",
            [_series("turn-rate error", q, data["turn_rate_error"], _PURPLE)],
        ),
        (
            "plot_cumulative_turn",
            "Cumulative turn",
            "Normalized stroke progress q",
            "Turn from start (radians)",
            [
                _series("reference", q, data["reference_cumulative_turn"], _REFERENCE),
                _series("sample", q, data["sample_cumulative_turn"], _SAMPLE),
            ],
        ),
        (
            "plot_width",
            "Stroke-width diagnostic",
            "Normalized stroke progress q",
            "Estimated width (pixels in each crop)",
            [
                _series("reference", q, data["reference_width_px"], _REFERENCE),
                _series("sample", q, data["sample_width_px"], _SAMPLE),
            ],
        ),
    ]
    event_plot_keys = {
        "plot_heading",
        "plot_tangent_error",
        "plot_curvature",
        "plot_curvature_error",
        "plot_cumulative_turn",
    }
    route_events = np.concatenate(
        (
            data["reference_route_event_progress"],
            data["sample_route_event_progress"],
        )
    )
    for key, title, x_label, y_label, series in plots:
        write_plot(
            output_dir / _ARTIFACTS[key],
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=series,
            event_progress=route_events if key in event_plot_keys else (),
        )


def _json_series(analysis: ComparisonAnalysis) -> dict[str, list]:
    return {name: np.asarray(values).tolist() for name, values in analysis.series.items()}


def _crop_description(crop: dict[str, int]) -> str:
    return f"x={crop['x']}, y={crop['y']}, width={crop['width']}, height={crop['height']}"


def _route_candidate_diagnostics(
    extracted: ExtractedLetter,
    selected_rank: int | None,
) -> list[dict[str, Any]]:
    ranked_candidates = list(enumerate(extracted.route_candidates, start=1))
    reported_candidates = ranked_candidates[:ROUTE_REPORT_LIMIT]
    if selected_rank is not None and selected_rank > ROUTE_REPORT_LIMIT and reported_candidates:
        selected_candidate = (
            extracted.route_candidates[selected_rank - 1]
            if selected_rank <= len(extracted.route_candidates)
            else extracted.selected_route
        )
        if selected_candidate is not None:
            reported_candidates[-1] = (selected_rank, selected_candidate)
    return [
        {
            "rank": rank,
            "selected": rank == selected_rank,
            "ambiguous": candidate.ambiguous,
            "stroke_count": len(candidate.continuous_strokes),
            "pen_lift_count": sum(event.kind == "pen-lift" for event in candidate.events),
            "physical_junction_count": candidate.physical_junction_count,
            "junction_encounter_count": sum(
                event.kind == "junction-encounter" for event in candidate.events
            ),
            "cycle_count": candidate.cycle_count,
            "repeated_length_ratio": candidate.score.repeat_cost,
            "full_coverage": not candidate.hypothesis.uncovered_edge_ids,
            "events": [
                {
                    "kind": event.kind,
                    "progress": event.progress,
                    "junction_node_id": event.junction_node_id,
                    "occurrence": event.occurrence,
                }
                for event in candidate.events
            ],
            "score": {
                "coverage_cost": candidate.score.coverage_cost,
                "repeat_cost": candidate.score.repeat_cost,
                "lift_cost": candidate.score.lift_cost,
                "continuity_cost": candidate.score.continuity_cost,
                "steering_complexity_cost": candidate.score.steering_complexity_cost,
                "intrinsic_cost": candidate.score.intrinsic_cost,
            },
            "minimum_complexity_reconstruction": reconstruction_model_payload(
                candidate,
                route_rank=rank,
            ),
        }
        for rank, candidate in reported_candidates
    ]


def _route_candidate_for_rank(
    result: ComparisonResult,
    side: str,
    rank: int,
) -> WritingRouteCandidate | None:
    selection = result.route_selection
    if selection is not None:
        selected_rank = (
            selection.score.reference_rank if side == "reference" else selection.score.sample_rank
        )
        if rank == selected_rank:
            return selection.reference if side == "reference" else selection.sample
    extracted = result.reference if side == "reference" else result.sample
    if 1 <= rank <= len(extracted.route_candidates):
        return extracted.route_candidates[rank - 1]
    return None


def _route_pairs_payload(result: ComparisonResult) -> dict[str, Any]:
    selection = result.route_selection
    if selection is None:
        return {"selected_pair": None, "evaluated_pair_count": 0, "pairs": []}

    selected_key = (selection.score.reference_rank, selection.score.sample_rank)
    pairs = []
    for pair_rank, score in enumerate(selection.alternatives, start=1):
        reference = _route_candidate_for_rank(result, "reference", score.reference_rank)
        sample = _route_candidate_for_rank(result, "sample", score.sample_rank)
        pairs.append(
            {
                "pair_rank": pair_rank,
                "reference_route_rank": score.reference_rank,
                "sample_route_rank": score.sample_rank,
                "selected": (score.reference_rank, score.sample_rank) == selected_key,
                "eligible": score.eligible,
                "reference": {
                    "available": reference is not None,
                    "stroke_count": (
                        len(reference.continuous_strokes) if reference is not None else None
                    ),
                    "pen_lift_count": (
                        reference.hypothesis.pen_lift_count if reference is not None else None
                    ),
                    "ambiguous": reference.ambiguous if reference is not None else None,
                    "minimum_complexity_reconstruction": (
                        reconstruction_model_payload(reference, route_rank=score.reference_rank)
                        if reference is not None
                        else None
                    ),
                },
                "sample": {
                    "available": sample is not None,
                    "stroke_count": len(sample.continuous_strokes) if sample is not None else None,
                    "pen_lift_count": (
                        sample.hypothesis.pen_lift_count if sample is not None else None
                    ),
                    "ambiguous": sample.ambiguous if sample is not None else None,
                    "minimum_complexity_reconstruction": (
                        reconstruction_model_payload(sample, route_rank=score.sample_rank)
                        if sample is not None
                        else None
                    ),
                },
                "pen_lift_count": score.pen_lift_count,
                "geometry_cost": score.geometry_cost,
                "cost_breakdown": score.cost_breakdown,
                "total_cost": score.total_cost,
            }
        )
    selected_pair = next(pair for pair in pairs if pair["selected"])
    return {
        "selected_pair": selected_pair,
        "evaluated_pair_count": len(pairs),
        "pairs": pairs,
    }


def _write_route_alternatives(
    result: ComparisonResult,
    route_pairs: dict[str, Any],
    output: Path,
) -> None:
    pairs = route_pairs["pairs"][:_ROUTE_PAIR_REPORT_LIMIT]
    width, panel_height, columns = 1200, 230, 2
    rows = max(1, (len(pairs) + columns - 1) // columns)
    image = Image.new("RGB", (width, 42 + rows * panel_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=16)
    body_font = ImageFont.load_default(size=12)
    draw.text((12, 10), "Evaluated route-pair alternatives", fill=(20, 20, 20), font=title_font)

    if not pairs:
        draw.text((12, 55), "No joint route pairs are available.", fill=(75, 75, 75))
        image.save(output, format="PNG")
        return

    selection = result.route_selection
    assert selection is not None
    for index, pair in enumerate(pairs):
        column, row = index % columns, index // columns
        left, top = column * 600 + 8, 42 + row * panel_height + 4
        right, bottom = left + 584, top + panel_height - 8
        draw.rectangle((left, top, right, bottom), outline=(190, 190, 190), width=1)
        reference = _route_candidate_for_rank(
            result,
            "reference",
            pair["reference_route_rank"],
        )
        sample = _route_candidate_for_rank(result, "sample", pair["sample_route_rank"])
        if reference is None or sample is None:
            state = "selected" if pair["selected"] else "alternative"
            eligibility = "eligible" if pair["eligible"] else "not eligible"
            draw.text(
                (left + 12, top + 8),
                f"Pair #{pair['pair_rank']}: R{pair['reference_route_rank']} / "
                f"S{pair['sample_route_rank']} ({state}, {eligibility})",
                fill=(20, 20, 20),
                font=body_font,
            )
            draw.text(
                (left + 12, top + 26),
                f"Candidate geometry unavailable; total cost {pair['total_cost']:.4f}",
                fill=(55, 55, 55),
                font=body_font,
            )
            costs = pair["cost_breakdown"]
            draw.text(
                (left + 12, top + 44),
                f"Ink fit {costs['reconstruction_fit_cost']:.3f}; "
                f"model complexity {costs['model_complexity_cost']:.3f}",
                fill=(75, 75, 75),
                font=body_font,
            )
            continue
        alignment = align_writing_routes(reference, sample)
        reference_strokes = reference.continuous_strokes
        sample_strokes = alignment.aligned_sample_strokes
        all_points = np.concatenate(
            [stroke.xy for stroke in (*reference_strokes, *sample_strokes)], axis=0
        )
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        plot_left, plot_top = left + 14, top + 66
        plot_width, plot_height = right - left - 28, bottom - top - 96
        scale = min(plot_width / span[0], plot_height / span[1])

        def project(
            points: np.ndarray,
            *,
            origin: np.ndarray,
            plot_origin: tuple[int, int],
            scale: float,
        ) -> list[tuple[int, int]]:
            return [
                (
                    round(plot_origin[0] + (point[0] - origin[0]) * scale),
                    round(plot_origin[1] + (point[1] - origin[1]) * scale),
                )
                for point in points
            ]

        for stroke in reference_strokes:
            points = project(
                stroke.xy,
                origin=minimum,
                plot_origin=(plot_left, plot_top),
                scale=scale,
            )
            draw.line(points, fill=_REFERENCE, width=3)
        for stroke in sample_strokes:
            points = project(
                stroke.xy,
                origin=minimum,
                plot_origin=(plot_left, plot_top),
                scale=scale,
            )
            draw.line(points, fill=_SAMPLE, width=3)

        state = "selected" if pair["selected"] else "alternative"
        eligibility = "eligible" if pair["eligible"] else "not eligible"
        draw.text(
            (left + 12, top + 8),
            f"Pair #{pair['pair_rank']}: R{pair['reference_route_rank']} / "
            f"S{pair['sample_route_rank']} ({state}, {eligibility})",
            fill=(20, 20, 20),
            font=body_font,
        )
        draw.text(
            (left + 12, top + 26),
            f"R: {len(reference_strokes)} strokes, {reference.hypothesis.pen_lift_count} lifts; "
            f"S: {len(sample_strokes)} strokes, {sample.hypothesis.pen_lift_count} lifts; "
            f"total cost {pair['total_cost']:.4f}",
            fill=(55, 55, 55),
            font=body_font,
        )
        costs = pair["cost_breakdown"]
        draw.text(
            (left + 12, top + 44),
            f"Ink fit {costs['reconstruction_fit_cost']:.3f}; "
            f"model complexity {costs['model_complexity_cost']:.3f}",
            fill=(75, 75, 75),
            font=body_font,
        )
        draw.text((left + 14, bottom - 22), "Reference", fill=_REFERENCE, font=body_font)
        draw.text((left + 92, bottom - 22), "Aligned sample", fill=_SAMPLE, font=body_font)
    image.save(output, format="PNG")


def _write_route_cost_breakdown(route_pairs: dict[str, Any], output: Path) -> None:
    pairs = route_pairs["pairs"][:_ROUTE_PAIR_REPORT_LIMIT]
    width, row_height = 1200, 34
    height = 62 + max(len(pairs), 1) * row_height + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=16)
    body_font = ImageFont.load_default(size=12)
    draw.text((12, 10), "Additive route-pair cost breakdown", fill=(20, 20, 20), font=title_font)
    if not pairs:
        draw.text((12, 46), "No route-pair costs are available.", fill=(75, 75, 75))
        image.save(output, format="PNG")
        return

    terms = (
        ("intrinsic_route_cost", "route plausibility", (95, 95, 95)),
        ("descriptor_matching_cost", "trajectory descriptor", (145, 55, 165)),
        ("aligned_centerline_cost", "centerline", (35, 105, 215)),
        ("steering_profile_cost", "steering", (225, 75, 55)),
        ("topology_event_compatibility_cost", "topology", (20, 145, 95)),
        ("writing_prior_cost", "writing prior", (230, 160, 35)),
        ("reconstruction_fit_cost", "ink fit", (60, 150, 190)),
        ("model_complexity_cost", "complexity", (125, 85, 155)),
    )
    maximum = max(max(pair["total_cost"] for pair in pairs), 1e-12)
    bar_left, bar_width = 240, 780
    for index, pair in enumerate(pairs):
        y = 48 + index * row_height
        state = "selected" if pair["selected"] else "alternative"
        eligibility = "eligible" if pair["eligible"] else "not eligible"
        draw.text(
            (12, y + 6),
            f"#{pair['pair_rank']} R{pair['reference_route_rank']}/S{pair['sample_route_rank']} "
            f"{state}, {eligibility}",
            fill=(35, 35, 35),
            font=body_font,
        )
        x = bar_left
        for key, _label, color in terms:
            segment_width = round(pair["cost_breakdown"][key] / maximum * bar_width)
            if segment_width:
                draw.rectangle((x, y + 4, x + segment_width, y + 24), fill=color)
            x += segment_width
        draw.text(
            (bar_left + bar_width + 10, y + 6),
            f"{pair['total_cost']:.4f}",
            fill=(20, 20, 20),
        )

    legend_y = 52 + len(pairs) * row_height
    x = 12
    for _key, label, color in terms:
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=color)
        draw.text((x + 17, legend_y - 1), label, fill=(40, 40, 40), font=body_font)
        x += 145
    image.save(output, format="PNG")


def _write_route_pair_artifacts(result: ComparisonResult, output_dir: Path) -> dict[str, Any]:
    route_pairs = _route_pairs_payload(result)
    (output_dir / _ARTIFACTS["route_pairs"]).write_text(
        json.dumps(route_pairs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_route_alternatives(
        result,
        route_pairs,
        output_dir / _ARTIFACTS["route_alternatives"],
    )
    _write_route_cost_breakdown(
        route_pairs,
        output_dir / _ARTIFACTS["route_cost_breakdown"],
    )
    return route_pairs


def _route_pair_rationale(route_pairs: dict[str, Any]) -> str:
    selected = route_pairs["selected_pair"]
    if selected is None:
        return "No jointly evaluated route pair is available."
    rationale = (
        f"Route pair R{selected['reference_route_rank']}/S{selected['sample_route_rank']} "
        f"was selected as the lowest-cost eligible pair with additive total cost "
        f"{selected['total_cost']:.4f}."
    )
    closest = next((pair for pair in route_pairs["pairs"] if not pair["selected"]), None)
    if closest is not None:
        rationale += (
            f" The closest listed alternative is pair #{closest['pair_rank']} "
            f"(R{closest['reference_route_rank']}/S{closest['sample_route_rank']}) at "
            f"{closest['total_cost']:.4f}."
        )
    return rationale


def _writing_payload(result: ComparisonResult, writing_metrics: dict[str, Any]) -> dict[str, Any]:
    selection = result.route_selection
    payload: dict[str, Any] = {
        "available": bool(writing_metrics.get("available")),
        "direction_source": "static_image_geometry_not_observed",
        "feedback_status": (
            "final" if result.identity_status == "accepted" else "candidate_diagnostic"
        ),
        "direction_status": "unavailable",
        "direction_interpretation": _route_direction_interpretation(result),
        "route_ambiguous": selection.ambiguous if selection is not None else None,
        "reference_candidate_count": len(result.reference.route_candidates),
        "sample_candidate_count": len(result.sample.route_candidates),
    }
    if selection is None or not writing_metrics.get("available"):
        payload["unavailable_reason"] = writing_metrics.get(
            "unavailable_reason", "joint writing-route selection is unavailable"
        )
        payload["reference"] = {
            "orientation": None,
            "writing_prior": None,
            "physical_junction_count": result.reference.diagnostics.topology.junction_count,
        }
        payload["sample"] = {
            "physical_junction_count": result.sample.diagnostics.topology.junction_count
        }
        payload["differences"] = {}
        return payload
    payload["direction_status"] = (
        "inferred"
        if _direction_label(selection.reference) == _direction_label(selection.sample) == "inferred"
        else "unresolved"
    )

    for side, candidate in (("reference", selection.reference), ("sample", selection.sample)):
        payload[side] = {
            "selected_route_rank": (
                selection.score.reference_rank
                if side == "reference"
                else selection.score.sample_rank
            ),
            "stroke_count": len(candidate.continuous_strokes),
            "pen_lift_count": sum(event.kind == "pen-lift" for event in candidate.events),
            "physical_junction_count": candidate.physical_junction_count,
            "junction_encounter_count": sum(
                event.kind == "junction-encounter" for event in candidate.events
            ),
            "cycle_count": candidate.cycle_count,
            "retrace_ratio": candidate.score.repeat_cost,
            "full_coverage": not candidate.hypothesis.uncovered_edge_ids,
            "orientation": {
                "status": candidate.orientation_status,
                "interpretation": _direction_label(candidate),
                "selected": ("forward" if candidate.orientation_variant == "native" else "reverse"),
                "variant": candidate.orientation_variant,
                "reason": candidate.orientation_evidence,
                "margin": candidate.orientation_margin,
            },
            "writing_prior": (
                {
                    "pen_lift_cost": candidate.writing_prior.pen_lift_cost,
                    "start_endpoint_cost": candidate.writing_prior.start_endpoint_cost,
                    "end_endpoint_cost": candidate.writing_prior.end_endpoint_cost,
                    "direction_cost": candidate.writing_prior.direction_cost,
                    "junction_continuity_cost": (candidate.writing_prior.junction_continuity_cost),
                    "total": candidate.writing_prior.total,
                }
                if candidate.writing_prior is not None
                else None
            ),
            "steering_models": writing_metrics["steering"][f"{side}_strokes"],
            "width": writing_metrics["width"][side],
            "path_allocation": writing_metrics["path_allocation"][side],
        }
    reference = payload["reference"]
    sample = payload["sample"]
    payload["differences"] = {
        "stroke_count": sample["stroke_count"] - reference["stroke_count"],
        "physical_junction_count": (
            sample["physical_junction_count"] - reference["physical_junction_count"]
        ),
        "cycle_count": sample["cycle_count"] - reference["cycle_count"],
        "total_signed_turn_deg": writing_metrics["steering"]["total_signed_turn_difference_deg"],
        "steering_profile_distance": selection.score.steering_profile_cost,
        "entry_exit": writing_metrics["entry_exit"],
        "sample_to_reference_median_width_ratio": (
            writing_metrics["width"]["sample_to_reference_median_ratio"]
        ),
        "route_pair_components": {
            "intrinsic_route_cost": selection.score.intrinsic_route_cost,
            "aligned_centerline_cost": selection.score.aligned_centerline_cost,
            "steering_profile_cost": selection.score.steering_profile_cost,
            "topology_event_compatibility_cost": (
                selection.score.topology_event_compatibility_cost
            ),
            "geometry_cost": selection.score.geometry_cost,
            "writing_prior_cost": selection.score.writing_prior_cost,
            "pen_lift_count": selection.score.pen_lift_count,
        },
    }
    return payload


def _report_payload(result: ComparisonResult, analysis: ComparisonAnalysis) -> dict[str, Any]:
    transform = _alignment_transform(result)
    reference_path = result.reference.continuous_path
    sample_path = result.sample.continuous_path
    continuous_geometry = (
        reference_path is not None
        and sample_path is not None
        and (result.alignment is not None and result.alignment.aligned_sample_path is not None)
    )
    selection = result.route_selection

    route_pairs = _route_pairs_payload(result)

    def input_payload(letter: ExtractedLetter, side: str) -> dict[str, Any]:
        path = letter.continuous_path
        return {
            "source_name": letter.source_path.name,
            "crop": letter.crop.model_dump(mode="json"),
            "mask_method": letter.mask_result.method,
            "mask_confidence": letter.mask_result.confidence,
            "warnings": letter.mask_result.warnings,
            "topology": letter.diagnostics.topology.model_dump(mode="json"),
            "path_coverage": (
                letter.ordered_path.coverage if letter.ordered_path is not None else None
            ),
            "continuous_centerline": {
                "available": path is not None,
                "algorithm": path.source if path is not None else None,
                "boundary_evidence": path.boundary_evidence if path is not None else None,
                "arc_length_px": path.arc_length_px if path is not None else None,
                "route_edge_ids": list(path.edge_ids) if path is not None else [],
                "route_edge_directions": list(path.edge_directions) if path is not None else [],
            },
            "original_artifact": _ARTIFACTS[f"input_{side}"],
            "mask_artifact": _ARTIFACTS[f"mask_{side}"],
            "skeleton_artifact": _ARTIFACTS[f"skeleton_{side}"],
            "extraction_artifact": _ARTIFACTS[f"extraction_{side}"],
        }

    warnings = [
        *result.warnings,
        *result.reference.mask_result.warnings,
        *result.sample.mask_result.warnings,
    ]
    reconstruction_metadata = (
        {
            "algorithm": "skeletonpaths-subpixel-v1",
            "boundary_evidence": {
                "reference": reference_path.boundary_evidence,
                "sample": sample_path.boundary_evidence,
            },
            "curve_model": "planarcurve-cubic-b-spline",
            "refinement_passes": SubpixelCenterlineOptions().refinement_passes,
            "fallback": False,
            "fallback_reason": None,
        }
        if continuous_geometry
        else {
            "algorithm": "legacy-polyline",
            "boundary_evidence": {
                "reference": (
                    reference_path.boundary_evidence if reference_path is not None else None
                ),
                "sample": sample_path.boundary_evidence if sample_path is not None else None,
            },
            "curve_model": "polyline",
            "refinement_passes": 0,
            "fallback": True,
            "fallback_reason": [
                warning
                for warning in warnings
                if "refin" in warning.lower() or "continuous" in warning.lower()
            ]
            or [
                "Continuous centerline geometry was unavailable; "
                "legacy polyline metrics were selected."
            ],
        }
    )
    diagnostic_metrics = {
        name: values
        for name, values in analysis.metrics.items()
        if name not in {"writing", "reconstruction"}
    }
    reference_rank = selection.score.reference_rank if selection is not None else None
    sample_rank = selection.score.sample_rank if selection is not None else None
    candidate_routes = {
        "reference": _route_candidate_diagnostics(result.reference, reference_rank),
        "sample": _route_candidate_diagnostics(result.sample, sample_rank),
    }
    route_search = {
        "hypothesis_limit": ROUTE_SEARCH_LIMIT,
        "report_limit": ROUTE_REPORT_LIMIT,
        "reference_candidates_available": len(result.reference.route_candidates),
        "sample_candidates_available": len(result.sample.route_candidates),
        "reference_candidates_reported": len(candidate_routes["reference"]),
        "sample_candidates_reported": len(candidate_routes["sample"]),
    }
    selected_route_pair = (
        None
        if selection is None
        else {
            "reference_rank": selection.score.reference_rank,
            "sample_rank": selection.score.sample_rank,
            "ambiguous": selection.ambiguous,
            "eligible": selection.score.eligible,
            "score": {
                "intrinsic_route_cost": selection.score.intrinsic_route_cost,
                "aligned_centerline_cost": selection.score.aligned_centerline_cost,
                "steering_profile_cost": selection.score.steering_profile_cost,
                "topology_event_compatibility_cost": (
                    selection.score.topology_event_compatibility_cost
                ),
                "geometry_cost": selection.score.geometry_cost,
                "writing_prior_cost": selection.score.writing_prior_cost,
                "pen_lift_count": selection.score.pen_lift_count,
                "descriptor_matching_cost": selection.score.descriptor_matching_cost,
                "cost_breakdown": selection.score.cost_breakdown,
                "total_cost": selection.score.total_cost,
            },
            "alternatives": [
                {
                    "reference_rank": score.reference_rank,
                    "sample_rank": score.sample_rank,
                    "total_cost": score.total_cost,
                    "eligible": score.eligible,
                    "descriptor_matching_cost": score.descriptor_matching_cost,
                    "cost_breakdown": score.cost_breakdown,
                }
                for score in selection.alternatives
            ],
        }
    )
    diagnostic_artifacts = [
        _ARTIFACTS[name]
        for name in (
            "extraction_reference",
            "extraction_sample",
            "mask_reference",
            "mask_sample",
            "skeleton_reference",
            "skeleton_sample",
            "overlay_raster",
            "overlay_centerline",
            "overlay_error_vectors",
            "plot_position_error",
            "plot_xy",
            "plot_heading",
            "plot_tangent_error",
            "plot_curvature",
            "plot_curvature_error",
            "plot_cumulative_turn",
            "plot_width",
            "route_alternatives",
            "route_cost_breakdown",
            "route_pairs",
        )
    ]
    primary_artifacts = [_ARTIFACTS["input_reference"], _ARTIFACTS["input_sample"]]
    if selection is not None:
        primary_artifacts.extend(
            (
                _ARTIFACTS["writing_path_overlay"],
                _ARTIFACTS["writing_path_overlay_svg"],
                _ARTIFACTS["steering_profile"],
                _ARTIFACTS["route_alternatives"],
                _ARTIFACTS["route_cost_breakdown"],
                _ARTIFACTS["route_pairs"],
            )
        )
    if analysis.metrics["reconstruction"]["available"]:
        primary_artifacts.extend(
            _ARTIFACTS[name]
            for name in (
                "ink_reconstruction_reference",
                "ink_reconstruction_sample",
                "ink_residual_reference",
                "ink_residual_sample",
            )
        )
    writing_payload = _writing_payload(result, analysis.metrics["writing"])
    return {
        "format": "inkgrammar-letter-comparison",
        "format_version": 3,
        "letter": result.letter,
        "identity": {
            "sample_id": result.identity.sample_id,
            "reference_id": result.identity.reference_id,
            "label": result.identity.label,
            "status": result.identity_status,
            "accepted_reference_id": (
                result.identity.reference_id if result.identity_status == "accepted" else None
            ),
            "final_feedback_available": result.identity_status == "accepted",
        },
        "interpretation": {
            "direction_source": "static_image_geometry_not_observed",
            "direction_status": writing_payload["direction_status"],
            "direction_summary": writing_payload["direction_interpretation"],
            "route_ambiguous": selection.ambiguous if selection is not None else None,
            "reference_candidate_count": len(result.reference.route_candidates),
            "sample_candidate_count": len(result.sample.route_candidates),
        },
        "capabilities": {
            "raster": True,
            "topology": True,
            "ordered_path": (
                result.reference.ordered_path is not None and result.sample.ordered_path is not None
            ),
            "continuous_centerline": continuous_geometry or result.route_alignment is not None,
            "writing_routes": selection is not None,
            "ink_reconstruction": analysis.metrics["reconstruction"]["available"],
        },
        "provenance": {
            "observed_raster_mask": "RasterRegions foreground evidence",
            "discrete_skeleton_topology": (
                "SkeletonPaths graph derived from the observed raster mask"
            ),
            "route_orientation": writing_payload["direction_interpretation"],
            "continuous_centerline": (
                "reconstructed continuous centerline consistent with raster evidence and route "
                "hypothesis; not a proven historical pen trace"
            ),
            "unmeasured_properties": (
                "A static image does not establish writing speed, elapsed time, pressure, or force."
            ),
        },
        "inputs": {
            "reference": input_payload(result.reference, "reference"),
            "sample": input_payload(result.sample, "sample"),
        },
        "alignment": {
            "orientation": _alignment_direction(result),
            "strategy": (
                result.route_alignment.strategy
                if result.route_alignment is not None
                else "single-path"
            ),
            "per_stroke_mean_squared_error_px2": (
                list(result.route_alignment.stroke_mean_squared_errors)
                if result.route_alignment is not None
                else []
            ),
            "scale": transform.scale,
            "rotation_deg": transform.rotation_deg,
            "translation_xy": [transform.dx, transform.dy],
            "reflection": False,
            "mean_squared_error_px2": _alignment_error(result),
            "interpretation": "Nuisance normalization, not a handwriting-quality score.",
        },
        "writing": writing_payload,
        "writing_feedback": (writing_payload if result.identity_status == "accepted" else None),
        "reconstruction": analysis.metrics["reconstruction"],
        "diagnostics": {
            "geometry": {
                "sample_count": len(analysis.progress),
                "progress": "normalized arc-length progress from 0 to 1",
                "position_error_normalization": "reference arc length equals 1.0",
                "centerline_reconstruction": reconstruction_metadata,
                "differential_geometry": {
                    "heading": (
                        "analytic tangent" if continuous_geometry else "legacy finite difference"
                    ),
                    "turn_rate": (
                        "analytic curvature times aligned arc length"
                        if continuous_geometry
                        else "legacy finite difference"
                    ),
                    "route_events": "semantic writing events on normalized arc-length progress",
                    "curvature_summary_guard_progress": diagnostic_metrics["turn_rate"][
                        "route_event_guard_progress"
                    ],
                },
                "legacy_fallback": (
                    {
                        "selected": True,
                        "algorithm": "Savitzky-Golay",
                        "window_length": result.smoothing_window,
                        "polynomial_order": result.smoothing_polynomial_order,
                    }
                    if not continuous_geometry
                    else None
                ),
            },
            "metrics": diagnostic_metrics,
            "writing_metrics": analysis.metrics["writing"],
            "series": _json_series(analysis),
            "candidate_routes": candidate_routes,
            "route_search": route_search,
            "selected_route_pair": selected_route_pair,
            "route_pairs": route_pairs,
        },
        "artifacts": {
            **_ARTIFACTS,
            "primary": primary_artifacts,
            "diagnostics": diagnostic_artifacts,
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def _writing_summary_rows(writing: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    reference = writing["reference"]
    sample = writing["sample"]
    differences = writing["differences"]
    if not writing["available"]:
        return [
            (
                "Route-based writing measurements",
                "Unavailable",
                "Unavailable",
                writing.get("unavailable_reason", "No jointly selected route is available."),
            )
        ]

    def physical_junctions(value: dict[str, Any]) -> str:
        return (
            f"{value['physical_junction_count']} physical, "
            f"{value['junction_encounter_count']} ordered encounters"
        )

    reference_changes = sum(
        model["selected_change_count"] for model in reference["steering_models"]
    )
    sample_changes = sum(model["selected_change_count"] for model in sample["steering_models"])
    width_ratio = differences["sample_to_reference_median_width_ratio"]
    return [
        (
            "Plausible strokes",
            str(reference["stroke_count"]),
            str(sample["stroke_count"]),
            f"difference {differences['stroke_count']:+d}; inferred route structure",
        ),
        (
            "Direction interpretation",
            reference["orientation"]["interpretation"],
            sample["orientation"]["interpretation"],
            writing["direction_interpretation"],
        ),
        (
            "Physical junctions / encounters",
            physical_junctions(reference),
            physical_junctions(sample),
            "Physical regions are counted once; encounters preserve ordered traversal.",
        ),
        (
            "Cycles",
            str(reference["cycle_count"]),
            str(sample["cycle_count"]),
            f"difference {differences['cycle_count']:+d}",
        ),
        (
            "Retracing ratio",
            f"{reference['retrace_ratio']:.4g}",
            f"{sample['retrace_ratio']:.4g}",
            "Inferred repeated route length, not a correctness score.",
        ),
        (
            "Fitted steering changes",
            str(reference_changes),
            str(sample_changes),
            f"profile distance {differences['steering_profile_distance']:.4g}",
        ),
        (
            "Inferred entry / exit heading",
            "Reference orientation",
            "Sample orientation",
            f"start {differences['entry_exit']['start_heading_difference_deg']:.4g}°, "
            f"end {differences['entry_exit']['end_heading_difference_deg']:.4g}° difference",
        ),
        (
            "Median width / glyph height",
            f"{reference['width']['median_width_over_glyph_height']:.4g}",
            f"{sample['width']['median_width_over_glyph_height']:.4g}",
            (
                f"sample/reference ratio {width_ratio:.4g}"
                if width_ratio is not None
                else "unavailable"
            ),
        ),
        (
            "Route alternatives",
            str(writing["reference_candidate_count"]),
            str(writing["sample_candidate_count"]),
            (
                "near-tied alternatives remain"
                if writing["route_ambiguous"]
                else "selected pair is not near-tied"
            ),
        ),
    ]


def _markdown(result: ComparisonResult, payload: dict[str, Any]) -> str:
    writing = payload["writing"]
    route_pairs = payload["diagnostics"]["route_pairs"]
    writing_heading = (
        "Candidate diagnostic, not final handwriting feedback"
        if result.identity_status == "candidate_diagnostic"
        else "Writing summary"
    )
    rows = [
        f"# {_comparison_title(result)}",
        "",
        "![Reference crop](input-reference.png)",
        "![My writing crop](input-sample.png)",
        "",
        f"## 1. {writing_heading}",
        "",
        writing["direction_interpretation"],
        "",
        "| Feature | Reference | My writing | Difference / interpretation |",
        "| --- | --- | --- | --- |",
    ]
    rows.extend(
        f"| {feature} | {reference} | {sample} | {difference} |"
        for feature, reference, sample, difference in _writing_summary_rows(writing)
    )
    rows.extend(
        (
            "",
            "## 2. How the selected route differs",
            "",
            "The route is a plausible reconstruction from raster and topology evidence, "
            "not a proven historical trace.",
        )
    )
    if writing["available"]:
        rows.extend(
            (
                "![Aligned plausible routes](writing-path-overlay.png)",
                "[SVG route overlay](writing-path-overlay.svg)",
            )
        )
    else:
        rows.append(f"Route overlay unavailable: {writing['unavailable_reason']}")
    rows.extend(
        (
            "",
            "## 3. Routing alternatives",
            "",
            f"Why this pair won: {_route_pair_rationale(route_pairs)}",
            "",
            "![Ranked route-pair alternatives](route-alternatives.png)",
            "![Additive route-pair costs](route-cost-breakdown.png)",
            "All evaluated pairs and cost terms are in [route-pairs.json](route-pairs.json).",
        )
    )
    rows.extend(("", "## 4. Fitted steering profile", ""))
    if writing["available"]:
        rows.append("![Fitted heading-rate profiles](steering-profile.png)")
    else:
        rows.append("A fitted steering profile is unavailable without a jointly selected route.")
    rows.extend(("", "## 5. Ink reconstruction", ""))
    reconstruction = payload["reconstruction"]
    if reconstruction["available"]:
        rows.extend(
            (
                "Rendered width is derived from subpixel radius evidence. "
                "Residuals compare a retained observed mask with rendered coverage.",
                "",
                "![Reference ink reconstruction and residuals](ink-reconstruction-reference.png)",
                "![Sample ink reconstruction and residuals](ink-reconstruction-sample.png)",
            )
        )
    else:
        rows.append(reconstruction.get("unavailable_reason", "Ink reconstruction is unavailable."))
    rows.extend(
        (
            "",
            "<details><summary>6. Engineering diagnostics</summary>",
            "",
            "Raw geometry, skeleton pixels, candidate routes, and numeric series are retained "
            "in [report.json](report.json).",
            "Diagnostic overlays are annotations, not source ink; "
            "clean masks are shown separately.",
            "",
            "![Clean reference mask](mask-reference.png)",
            "![Clean sample mask](mask-sample.png)",
            "",
            "![Position error diagnostic](plot-position-error.png)",
            "![Raw X/Y diagnostic](plot-x-y.png)",
            "![Heading diagnostic](plot-heading.png)",
            "![Raw turn-rate diagnostic](plot-curvature.png)",
            "![Cumulative turn diagnostic](plot-cumulative-turn.png)",
            "![Width diagnostic](plot-width.png)",
            "",
            "</details>",
            "",
        )
    )
    return "\n".join(rows)


def _comparison_title(result: ComparisonResult) -> str:
    if result.letter is not None:
        title = f"Letter comparison: {result.letter}"
    else:
        identity = result.identity
        assert identity is not None
        title = f"Glyph comparison: {identity.sample_id} to {identity.reference_id}"
    if result.identity_status == "candidate_diagnostic":
        return f"Candidate diagnostic (identity not accepted): {title}"
    return title


def _html(result: ComparisonResult, payload: dict[str, Any]) -> str:
    writing = payload["writing"]
    route_pairs = payload["diagnostics"]["route_pairs"]
    writing_heading = (
        "Candidate diagnostic, not final handwriting feedback"
        if result.identity_status == "candidate_diagnostic"
        else "Writing summary"
    )
    reconstruction = payload["reconstruction"]

    def image(name: str, alt: str) -> str:
        return (
            f'<figure><img src="{html.escape(name)}" alt="{html.escape(alt)}">'
            f"<figcaption>{html.escape(alt)}</figcaption></figure>"
        )

    summary_rows = "".join(
        "<tr>"
        f"<th>{html.escape(feature)}</th>"
        f"<td>{html.escape(reference)}</td>"
        f"<td>{html.escape(sample)}</td>"
        f"<td>{html.escape(difference)}</td>"
        "</tr>"
        for feature, reference, sample, difference in _writing_summary_rows(writing)
    )
    rows = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(_comparison_title(result))}</title>",
        "<style>body{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%;height:auto;border:1px solid #bbb}figure{margin:.5rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}"
        "td,th{padding:.35rem .7rem;border:1px solid #ccc;text-align:left}"
        "table{border-collapse:collapse;width:100%}details{margin-top:2rem}</style></head><body>",
        f"<h1>{html.escape(_comparison_title(result))}</h1>",
        '<div class="grid">',
        image(_ARTIFACTS["input_reference"], "Reference crop"),
        image(_ARTIFACTS["input_sample"], "My writing crop"),
        "</div>",
        f"<h2>1. {html.escape(writing_heading)}</h2>",
        f"<p>{html.escape(writing['direction_interpretation'])} "
        "Differences are descriptive, not correctness scores.</p>",
        "<table><thead><tr><th>Feature</th><th>Reference</th><th>My writing</th>"
        "<th>Difference / interpretation</th></tr></thead><tbody>",
        summary_rows,
        "</tbody></table>",
        "<h2>2. How the selected route differs</h2>",
    ]
    if writing["available"]:
        rows.extend(
            (
                f"<p>{html.escape(writing['direction_interpretation'])} "
                "Centerline arrows show the displayed geometric alignment, which may reverse "
                "a sample route for comparison. Junction circles mark physical semantic regions; "
                "nearby numbers identify ordered encounters.</p>",
                '<p><a href="writing-path-overlay.svg">Open SVG route overlay</a></p>',
                '<div class="grid">',
                image(
                    _ARTIFACTS["writing_path_overlay"],
                    "Aligned inferred route with direction and junction markers",
                ),
                "</div>",
            )
        )
    else:
        unavailable_reason = writing.get(
            "unavailable_reason", "A jointly selected route is unavailable."
        )
        rows.append(f"<p>{html.escape(unavailable_reason)}</p>")
    rows.extend(
        (
            "<h2>3. Routing alternatives</h2>",
            f"<p>Why this pair won: {html.escape(_route_pair_rationale(route_pairs))}</p>",
            '<div class="grid">',
            image(_ARTIFACTS["route_alternatives"], "Ranked aligned route-pair alternatives"),
            image(_ARTIFACTS["route_cost_breakdown"], "Additive route-pair cost breakdown"),
            "</div>",
            '<p><a href="route-pairs.json">Complete evaluated route pairs and cost terms</a></p>',
        )
    )
    rows.extend(("<h2>4. Fitted steering profile</h2>",))
    if writing["available"]:
        rows.extend(
            (
                "<p>Profiles are fitted heading-rate models. K labels mark selected model changes, "
                "J labels mark semantic junction encounters.</p>",
                '<div class="grid">',
                image(_ARTIFACTS["steering_profile"], "Fitted heading-rate profiles"),
                "</div>",
            )
        )
    else:
        rows.append(
            "<p>A fitted steering profile is unavailable without a jointly selected route.</p>"
        )
    rows.append("<h2>5. Ink reconstruction</h2>")
    if reconstruction["available"]:
        rows.extend(
            (
                "<p>Rendered width is derived from subpixel radius evidence. "
                "Under-coverage and over-coverage are kept separate from observed ink.</p>",
                '<div class="grid">',
                image(
                    _ARTIFACTS["ink_reconstruction_reference"],
                    "Reference observed ink, rendered coverage, and residuals",
                ),
                image(
                    _ARTIFACTS["ink_reconstruction_sample"],
                    "Sample observed ink, rendered coverage, and residuals",
                ),
                image(
                    _ARTIFACTS["ink_residual_reference"],
                    "Reference under-coverage in red and over-coverage in purple",
                ),
                image(
                    _ARTIFACTS["ink_residual_sample"],
                    "Sample under-coverage in red and over-coverage in purple",
                ),
                "</div>",
            )
        )
    else:
        unavailable_reason = reconstruction.get(
            "unavailable_reason", "Ink reconstruction is unavailable."
        )
        rows.append(f"<p>{html.escape(unavailable_reason)}</p>")
    rows.extend(
        (
            "<details><summary>6. Engineering diagnostics</summary>",
            "<p>Raw geometry, skeleton pixels, candidate routes, and "
            "numeric series are secondary diagnostics.</p>",
            "<p>Diagnostic overlays are annotations, not source ink; "
            "clean masks are shown separately.</p>",
            '<div class="grid">',
            image(_ARTIFACTS["extraction_reference"], "Reference extraction diagnostics"),
            image(_ARTIFACTS["extraction_sample"], "Sample extraction diagnostics"),
            image(_ARTIFACTS["mask_reference"], "Clean reference mask"),
            image(_ARTIFACTS["mask_sample"], "Clean sample mask"),
            image(_ARTIFACTS["skeleton_reference"], "Reference skeleton diagnostics"),
            image(_ARTIFACTS["skeleton_sample"], "Sample skeleton diagnostics"),
            image(_ARTIFACTS["plot_position_error"], "Position error diagnostic"),
            image(_ARTIFACTS["plot_xy"], "Raw X/Y diagnostic"),
            image(_ARTIFACTS["plot_heading"], "Heading diagnostic"),
            image(_ARTIFACTS["plot_tangent_error"], "Tangent error diagnostic"),
            image(_ARTIFACTS["plot_curvature"], "Raw turn-rate diagnostic"),
            image(_ARTIFACTS["plot_curvature_error"], "Raw turn-rate error diagnostic"),
            image(_ARTIFACTS["plot_cumulative_turn"], "Cumulative turn diagnostic"),
            image(_ARTIFACTS["plot_width"], "Width diagnostic"),
            "</div>",
            "<h3>Diagnostic metrics</h3><pre>",
            html.escape(json.dumps(payload["diagnostics"]["metrics"], indent=2)),
            '</pre><p><a href="report.json">Full report JSON</a> includes route candidates '
            "and numeric series.</p>",
            "</details>",
            "</body></html>",
        )
    )
    return "\n".join(rows)


def _write_raster_only_report(
    result: ComparisonResult,
    output_dir: Path,
) -> WrittenComparisonReport:
    reference = result.reference
    sample = result.sample
    identity = result.identity
    assert identity is not None
    _save_rgb_crop(reference.image, output_dir / _ARTIFACTS["input_reference"])
    _save_rgb_crop(sample.image, output_dir / _ARTIFACTS["input_sample"])

    panels = []
    inputs: dict[str, Any] = {}
    for side, extracted in (("reference", reference), ("sample", sample)):
        Image.fromarray(extracted.mask_result.mask.astype(np.uint8) * 255).save(
            output_dir / _ARTIFACTS[f"mask_{side}"]
        )
        skeleton = extracted.image.convert("RGB")
        ys, xs = np.nonzero(extracted.diagnostics.skeleton)
        ImageDraw.Draw(skeleton).point(
            list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(20, 195, 95)
        )
        skeleton.save(output_dir / _ARTIFACTS[f"skeleton_{side}"], format="PNG")
        panels.append((side.title(), extracted.mask_result.mask))
        path = extracted.continuous_path
        inputs[side] = {
            "source_name": extracted.source_path.name,
            "crop": extracted.crop.model_dump(mode="json"),
            "mask_method": extracted.mask_result.method,
            "mask_confidence": extracted.mask_result.confidence,
            "warnings": extracted.mask_result.warnings,
            "topology": extracted.diagnostics.topology.model_dump(mode="json"),
            "path_coverage": (
                extracted.ordered_path.coverage if extracted.ordered_path is not None else None
            ),
            "continuous_centerline": {
                "available": path is not None,
                "algorithm": path.source if path is not None else None,
                "boundary_evidence": path.boundary_evidence if path is not None else None,
                "arc_length_px": path.arc_length_px if path is not None else None,
                "route_edge_ids": list(path.edge_ids) if path is not None else [],
                "route_edge_directions": list(path.edge_directions) if path is not None else [],
            },
            "original_artifact": _ARTIFACTS[f"input_{side}"],
            "mask_artifact": _ARTIFACTS[f"mask_{side}"],
            "skeleton_artifact": _ARTIFACTS[f"skeleton_{side}"],
        }

    gap = 16
    label_height = 20
    panel_height = max(mask.shape[0] for _, mask in panels)
    panel_width = sum(mask.shape[1] for _, mask in panels) + gap
    overlay = Image.new("RGB", (panel_width, panel_height + label_height), "white")
    draw = ImageDraw.Draw(overlay)
    x_offset = 0
    for label, mask in panels:
        rgb = np.full((*mask.shape, 3), 255, dtype=np.uint8)
        rgb[mask] = _REFERENCE if label == "Reference" else _SAMPLE
        overlay.paste(Image.fromarray(rgb), (x_offset, label_height))
        draw.text(
            (x_offset + 2, 3), f"{label} (unaligned)", fill="black", font=ImageFont.load_default()
        )
        x_offset += mask.shape[1] + gap
    overlay.save(output_dir / _ARTIFACTS["overlay_raster"], format="PNG")

    unavailable = "; ".join(result.warnings) or (
        "A jointly comparable route was unavailable, so route-based writing "
        "measurements were not calculated."
    )
    writing = {
        "available": False,
        "unavailable_reason": unavailable,
        "direction_source": "unavailable_without_jointly_selected_route",
        "direction_status": "unavailable",
        "direction_interpretation": (
            "No jointly selected route is available, so direction is unavailable."
        ),
        "route_ambiguous": None,
        "reference_candidate_count": len(reference.route_candidates),
        "sample_candidate_count": len(sample.route_candidates),
        "reference": {"physical_junction_count": reference.diagnostics.topology.junction_count},
        "sample": {"physical_junction_count": sample.diagnostics.topology.junction_count},
        "differences": {},
    }
    reconstruction = {"available": False, "unavailable_reason": unavailable}
    artifacts = {
        name: _ARTIFACTS[name]
        for name in (
            "input_reference",
            "input_sample",
            "mask_reference",
            "mask_sample",
            "skeleton_reference",
            "skeleton_sample",
            "overlay_raster",
            "route_alternatives",
            "route_cost_breakdown",
            "route_pairs",
            "report_json",
            "report_markdown",
            "report_html",
        )
    }
    payload = {
        "format": "inkgrammar-letter-comparison",
        "format_version": 3,
        "letter": result.letter,
        "identity": {
            "sample_id": identity.sample_id,
            "reference_id": identity.reference_id,
            "label": identity.label,
            "status": result.identity_status,
            "accepted_reference_id": (
                identity.reference_id if result.identity_status == "accepted" else None
            ),
            "final_feedback_available": result.identity_status == "accepted",
        },
        "interpretation": {
            "direction_source": "unavailable_without_jointly_selected_route",
            "direction_status": "unavailable",
            "direction_summary": writing["direction_interpretation"],
            "route_ambiguous": None,
            "reference_candidate_count": len(reference.route_candidates),
            "sample_candidate_count": len(sample.route_candidates),
        },
        "capabilities": {
            "raster": True,
            "topology": True,
            "ordered_path": reference.ordered_path is not None and sample.ordered_path is not None,
            "continuous_centerline": (
                reference.continuous_path is not None and sample.continuous_path is not None
            ),
            "writing_routes": False,
            "ink_reconstruction": False,
        },
        "provenance": {
            "observed_raster_mask": "RasterRegions foreground evidence",
            "discrete_skeleton_topology": (
                "SkeletonPaths graph derived from the observed raster mask"
            ),
            "route_orientation": "unavailable because no comparable jointly selected route exists",
            "continuous_centerline": (
                "reconstructed from static raster and topology evidence; "
                "not a proven historical pen trace"
            ),
            "unmeasured_properties": (
                "A static image does not establish writing speed, elapsed time, pressure, or force."
            ),
        },
        "inputs": inputs,
        "alignment": None,
        "writing": writing,
        "writing_feedback": (writing if result.identity_status == "accepted" else None),
        "reconstruction": reconstruction,
        "diagnostics": {
            "geometry": {
                "sample_count": 0,
                "progress": None,
                "centerline_reconstruction": {
                    "algorithm": "unavailable_without_joint_route",
                    "fallback": True,
                    "fallback_reason": unavailable,
                },
                "differential_geometry": {
                    "heading": "unavailable",
                    "turn_rate": "unavailable",
                    "route_events": "unavailable without a selected route",
                },
                "legacy_fallback": None,
            },
            "metrics": {
                "raster": {
                    "reference_ink_pixels": int(reference.mask_result.mask.sum()),
                    "sample_ink_pixels": int(sample.mask_result.mask.sum()),
                },
                "topology": {
                    "reference": inputs["reference"]["topology"],
                    "sample": inputs["sample"]["topology"],
                },
            },
            "writing_metrics": {"available": False, "unavailable_reason": unavailable},
            "series": {
                "reference_raw_skeleton_xy": np.column_stack(
                    np.nonzero(reference.diagnostics.skeleton)[::-1]
                )
                .astype(float)
                .tolist(),
                "sample_raw_skeleton_xy": np.column_stack(
                    np.nonzero(sample.diagnostics.skeleton)[::-1]
                )
                .astype(float)
                .tolist(),
                "reference_subpixel_xy": (
                    reference.continuous_path.xy.tolist()
                    if reference.continuous_path is not None
                    else []
                ),
                "sample_subpixel_xy": (
                    sample.continuous_path.xy.tolist() if sample.continuous_path is not None else []
                ),
            },
            "candidate_routes": {
                "reference": _route_candidate_diagnostics(reference, None),
                "sample": _route_candidate_diagnostics(sample, None),
            },
            "selected_route_pair": None,
            "route_pairs": _route_pairs_payload(result),
        },
        "artifacts": {
            **artifacts,
            "primary": [artifacts["input_reference"], artifacts["input_sample"]],
            "diagnostics": [
                artifacts["mask_reference"],
                artifacts["mask_sample"],
                artifacts["skeleton_reference"],
                artifacts["skeleton_sample"],
                artifacts["overlay_raster"],
            ],
        },
        "warnings": list(
            dict.fromkeys(
                [*result.warnings, *reference.mask_result.warnings, *sample.mask_result.warnings]
            )
        ),
    }
    report_title = _comparison_title(result)
    writing_heading = (
        "Candidate diagnostic, not final handwriting feedback"
        if result.identity_status == "candidate_diagnostic"
        else "Writing summary"
    )
    (output_dir / _ARTIFACTS["report_json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        f"# {report_title}",
        "",
        "![Reference crop](input-reference.png)",
        "![My writing crop](input-sample.png)",
        "",
        f"## 1. {writing_heading}",
        "",
        writing["direction_interpretation"],
        "",
        f"Route-based writing measurements are unavailable: {unavailable}",
        "",
        "## 2. Route geometry comparison",
        "",
        "No comparable jointly selected route is available. No historical route, "
        "direction, speed, elapsed time, pressure, or force is inferred.",
        "",
        "## 3. Fitted steering profile",
        "",
        "Unavailable without a jointly selected route.",
        "",
        "## 4. Ink reconstruction",
        "",
        "Unavailable without a comparable route and width model.",
        "",
        "<details><summary>5. Engineering diagnostics</summary>",
        "",
        "![Unaligned observed masks](overlay-raster.png)",
        "![Reference mask](mask-reference.png)",
        "![Sample mask](mask-sample.png)",
        "",
        "Raw skeleton coordinates, candidate routes, and topology metrics are retained "
        "in [report.json](report.json).",
        "",
        "</details>",
        "",
    ]
    (output_dir / _ARTIFACTS["report_markdown"]).write_text("\n".join(markdown), encoding="utf-8")
    image_html = "".join(
        f'<figure><img src="{name}" alt="{html.escape(description)}">'
        f"<figcaption>{html.escape(description)}</figcaption></figure>"
        for name, description in (
            (artifacts["input_reference"], "Reference crop"),
            (artifacts["input_sample"], "My writing crop"),
        )
    )
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(report_title)}</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%;height:auto;border:1px solid #bbb}"
        "figure{display:inline-block;vertical-align:top}"
        "details{margin-top:2rem}</style></head><body>"
        f"<h1>{html.escape(report_title)}</h1><div>{image_html}</div>"
        f"<h2>1. {html.escape(writing_heading)}</h2>"
        f"<p>{html.escape(writing['direction_interpretation'])}</p>"
        f"<p>Route-based writing measurements are unavailable: {html.escape(unavailable)}</p>"
        "<h2>2. Route geometry comparison</h2>"
        "<p>No comparable jointly selected route is available. No historical route, "
        "direction, speed, elapsed time, pressure, or force is inferred.</p>"
        "<h2>3. Fitted steering profile</h2><p>Unavailable without a jointly selected route.</p>"
        "<h2>4. Ink reconstruction</h2>"
        "<p>Unavailable without a comparable route and width model.</p>"
        "<details><summary>5. Engineering diagnostics</summary>"
        f'<p><img src="{artifacts["overlay_raster"]}" alt="Unaligned observed masks"></p>'
        f'<p><img src="{artifacts["mask_reference"]}" alt="Clean reference mask"></p>'
        f'<p><img src="{artifacts["mask_sample"]}" alt="Clean sample mask"></p>'
        "<p>Diagnostic overlays are annotations, not source ink; "
        "clean masks are shown separately.</p>"
        f"<pre>{html.escape(json.dumps(payload['diagnostics']['metrics'], indent=2))}</pre>"
        '<p><a href="report.json">Full report JSON</a></p></details></body></html>'
    )
    (output_dir / _ARTIFACTS["report_html"]).write_text(page, encoding="utf-8")
    return WrittenComparisonReport(output_dir, payload)


def write_comparison_report(
    result: ComparisonResult,
    output_dir: str | Path,
) -> WrittenComparisonReport:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_route_pair_artifacts(result, destination)
    if result.alignment is None and result.route_alignment is None:
        return _write_raster_only_report(result, destination)

    analysis = analyze_comparison(result)
    _save_rgb_crop(result.reference.image, destination / _ARTIFACTS["input_reference"])
    _save_rgb_crop(result.sample.image, destination / _ARTIFACTS["input_sample"])
    _write_extraction_images(result.reference, destination, "reference")
    _write_extraction_images(result.sample, destination, "sample")
    _write_raster_overlay(result, destination / _ARTIFACTS["overlay_raster"])
    reference_continuous = result.reference.continuous_path
    sample_continuous = (
        result.alignment.aligned_sample_path if result.alignment is not None else None
    )
    reference_centerline = (
        reference_continuous.xy
        if reference_continuous is not None
        else analysis.series["reference_resampled_xy"]
    )
    sample_centerline = (
        sample_continuous.xy
        if sample_continuous is not None
        else analysis.series["sample_resampled_xy"]
    )
    reference_raw_skeleton_xy = np.argwhere(result.reference.diagnostics.skeleton)[:, ::-1].astype(
        float
    )
    sample_raw_skeleton_xy = _alignment_transform(result).apply(
        np.argwhere(result.sample.diagnostics.skeleton)[:, ::-1].astype(float)
    )
    reference_event_xy = (
        reference_continuous.xy[list(reference_continuous.route_event_indices)]
        if reference_continuous is not None
        else analysis.series.get("reference_route_event_xy", np.empty((0, 2), dtype=float))
    )
    sample_event_xy = (
        sample_continuous.xy[list(sample_continuous.route_event_indices)]
        if sample_continuous is not None
        else analysis.series.get("sample_route_event_xy", np.empty((0, 2), dtype=float))
    )
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_centerline"],
        reference_centerline,
        sample_centerline,
        reference_raw_skeleton_xy=reference_raw_skeleton_xy,
        sample_raw_skeleton_xy=sample_raw_skeleton_xy,
        reference_topology=result.reference.diagnostics.topology,
        sample_topology=result.sample.diagnostics.topology,
        sample_transform=_alignment_transform(result),
        reference_event_xy=reference_event_xy,
        sample_event_xy=sample_event_xy,
        reference_stroke_offsets=analysis.series.get("reference_resampled_stroke_offsets"),
        sample_stroke_offsets=analysis.series.get("sample_resampled_stroke_offsets"),
    )
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_error_vectors"],
        reference_centerline,
        sample_centerline,
        error_vectors=True,
        reference_stroke_offsets=analysis.series.get("reference_resampled_stroke_offsets"),
        sample_stroke_offsets=analysis.series.get("sample_resampled_stroke_offsets"),
    )
    _write_plots(analysis, destination)
    _write_writing_path_overlay(result, destination / _ARTIFACTS["writing_path_overlay"])
    _write_writing_path_overlay_svg(result, destination / _ARTIFACTS["writing_path_overlay_svg"])
    _write_steering_profile(result, destination / _ARTIFACTS["steering_profile"])
    if analysis.metrics["reconstruction"]["available"]:
        _write_ink_reconstruction_panels(analysis, destination)

    payload = _report_payload(result, analysis)
    (destination / _ARTIFACTS["report_json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / _ARTIFACTS["report_markdown"]).write_text(
        _markdown(result, payload),
        encoding="utf-8",
    )
    (destination / _ARTIFACTS["report_html"]).write_text(
        _html(result, payload),
        encoding="utf-8",
    )
    return WrittenComparisonReport(destination, payload)
