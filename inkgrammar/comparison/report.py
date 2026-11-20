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

from .metrics import ComparisonAnalysis, analyze_comparison
from .models import ComparisonResult, ExtractedLetter, TopologySummary, WritingRouteCandidate
from .plots import PlotSeries, write_plot

_REFERENCE = (35, 105, 215)
_SAMPLE = (225, 75, 55)
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
    "steering_profile": "steering-profile.png",
    "ink_reconstruction_reference": "ink-reconstruction-reference.png",
    "ink_reconstruction_sample": "ink-reconstruction-sample.png",
    "ink_residual_reference": "ink-residual-reference.png",
    "ink_residual_sample": "ink-residual-sample.png",
    "report_json": "report.json",
    "report_markdown": "report.md",
    "report_html": "report.html",
}


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
    label: str,
) -> None:
    x, y = xy
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
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
    for index, point in enumerate(endpoints[:2]):
        _draw_point(draw, point, _REFERENCE if index == 0 else _SAMPLE, "AB"[index])
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


def _canvas_size(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = bounds
    return max(1, right - left + 1), max(1, bottom - top + 1)


def _canvas_points(points: np.ndarray, bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, _, _ = bounds
    coordinates = np.rint(points - np.array((left, top))).astype(int)
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
    image = Image.new("RGB", _canvas_size(bounds), "white")
    draw = ImageDraw.Draw(image)

    if reference_raw_skeleton_xy is not None:
        draw.point(_canvas_points(reference_raw_skeleton_xy, bounds), fill=(130, 145, 140))
    if sample_raw_skeleton_xy is not None:
        draw.point(_canvas_points(sample_raw_skeleton_xy, bounds), fill=(75, 165, 105))
    reference_points = _canvas_points(reference_xy, bounds)
    sample_points = _canvas_points(sample_xy, bounds)
    draw.line(reference_points, fill=_REFERENCE, width=2)
    draw.line(sample_points, fill=_SAMPLE, width=2)

    for points, color, start_label, end_label in (
        (reference_points, _REFERENCE, "R start", "R end"),
        (sample_points, _SAMPLE, "S start", "S end"),
    ):
        for point, label in ((points[0], start_label), (points[-1], end_label)):
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
            draw.text((x + 5, y - 8), label, fill=color, font=ImageFont.load_default())

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
            for x, y in _canvas_points(points, bounds):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="white", width=1)
                draw.text((x + 3, y - 7), label, fill=color, font=ImageFont.load_default())

    for event_xy in (reference_event_xy, sample_event_xy):
        if event_xy is None:
            continue
        for x, y in _canvas_points(event_xy, bounds):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(235, 165, 20))
    if error_vectors:
        indexes = np.linspace(0, len(reference_points) - 1, 24).round().astype(int)
        for index in indexes:
            start = reference_points[index]
            end = sample_points[index]
            draw.line((start, end), fill=(55, 55, 55), width=1)
            draw.ellipse((start[0] - 2, start[1] - 2, start[0] + 2, start[1] + 2), fill=_REFERENCE)
            draw.ellipse((end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2), fill=_SAMPLE)
    output.parent.mkdir(parents=True, exist_ok=True)
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
) -> None:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(progress) * cumulative[-1]
    segment = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(lengths) - 1)
    first, second = _canvas_points(points[[segment, segment + 1]], bounds)
    draw.line((first, second), fill=color, width=2)
    dx, dy = second[0] - first[0], second[1] - first[1]
    magnitude = max(float(np.hypot(dx, dy)), 1e-12)
    ux, uy = dx / magnitude, dy / magnitude
    left = (round(second[0] - 5 * ux - 3 * uy), round(second[1] - 5 * uy + 3 * ux))
    right = (round(second[0] - 5 * ux + 3 * uy), round(second[1] - 5 * uy - 3 * ux))
    draw.polygon((second, left, right), fill=color)


def _write_writing_path_overlay(result: ComparisonResult, output: Path) -> None:
    selection = result.route_selection
    if selection is None:
        return
    reverse_sample = result.alignment.direction == "reversed-for-best-match"
    reference_strokes = _display_strokes(selection.reference, False)
    sample_strokes = _display_strokes(selection.sample, reverse_sample)
    reference_junctions = np.asarray(
        result.reference.diagnostics.topology.junctions, dtype=float
    ).reshape((-1, 2))
    sample_junctions = result.alignment.transform.apply(
        np.asarray(result.sample.diagnostics.topology.junctions, dtype=float).reshape((-1, 2))
    )
    all_points = [*reference_strokes, *sample_strokes]
    if len(reference_junctions):
        all_points.append(reference_junctions)
    if len(sample_junctions):
        all_points.append(sample_junctions)
    bounds = _canvas_bounds(np.concatenate(all_points, axis=0), np.empty((0, 2), dtype=float))
    image = Image.new("RGB", _canvas_size(bounds), "white")
    draw = ImageDraw.Draw(image)
    for side, strokes, color in (
        ("R", reference_strokes, _REFERENCE),
        ("S", sample_strokes, _SAMPLE),
    ):
        for stroke_index, points in enumerate(strokes):
            pixel_points = _canvas_points(points, bounds)
            draw.line(pixel_points, fill=color, width=3)
            for fraction in (0.25, 0.5, 0.75):
                _draw_arrow(draw, points, bounds, fraction, color)
            if stroke_index == 0:
                x, y = pixel_points[0]
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=1)
                draw.text(
                    (x + 6, y - 8), f"{side} start", fill=color, font=ImageFont.load_default()
                )
            if stroke_index == len(strokes) - 1:
                x, y = pixel_points[-1]
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
                draw.text((x + 6, y - 8), f"{side} end", fill=color, font=ImageFont.load_default())
    for points, color in ((reference_junctions, _REFERENCE), (sample_junctions, _SAMPLE)):
        for x, y in _canvas_points(points, bounds):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
            draw.text((x + 7, y - 12), "J", fill=color, font=ImageFont.load_default())
    for _side, candidate, reverse, transform in (
        ("R", selection.reference, False, None),
        ("S", selection.sample, reverse_sample, result.alignment.transform),
    ):
        for encounter_index, event in enumerate(
            (item for item in candidate.events if item.kind == "junction-encounter"),
            start=1,
        ):
            point = _route_event_point(candidate, event.progress, reverse)
            if transform is not None:
                point = transform.apply(point)
            x, y = _canvas_points(point.reshape((1, 2)), bounds)[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(245, 165, 20))
            draw.text((x + 3, y + 2), str(event.occurrence or encounter_index), fill=(95, 70, 5))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


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
            result.alignment.direction == "reversed-for-best-match",
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
        label_height = 20
        gap = 8
        panels = []
        observed_image = np.full((height, width, 3), 255, dtype=np.uint8)
        observed_image[observed] = (25, 25, 25)
        rendered_gray = np.rint(255 * (1.0 - coverage)).astype(np.uint8)
        rendered_image = np.repeat(rendered_gray[:, :, None], 3, axis=2)
        under_image = np.full((height, width, 3), 255, dtype=np.uint8)
        under_image[under] = (225, 70, 45)
        over_image = np.full((height, width, 3), 255, dtype=np.uint8)
        over_image[over] = (135, 55, 165)
        panels.extend(
            (
                ("Observed ink", Image.fromarray(observed_image)),
                ("Rendered coverage", Image.fromarray(rendered_image)),
                ("Under-coverage", Image.fromarray(under_image)),
                ("Over-coverage", Image.fromarray(over_image)),
            )
        )
        sheet = Image.new("RGB", (2 * width + gap, 2 * (height + label_height) + gap), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (label, panel) in enumerate(panels):
            x = (index % 2) * (width + gap)
            y = (index // 2) * (height + label_height + gap)
            draw.text((x + 2, y + 2), label, fill=(30, 30, 30), font=ImageFont.load_default())
            sheet.paste(panel, (x, y + label_height))
        sheet.save(output_dir / _ARTIFACTS[f"ink_reconstruction_{side}"], format="PNG")

        residual = np.full((height, width, 3), 255, dtype=np.uint8)
        residual[under] = (225, 70, 45)
        residual[over] = (135, 55, 165)
        Image.fromarray(residual).save(
            output_dir / _ARTIFACTS[f"ink_residual_{side}"], format="PNG"
        )


def _write_raster_overlay(result: ComparisonResult, output: Path) -> None:
    reference_yx = np.argwhere(result.reference.mask_result.mask)
    sample_yx = np.argwhere(result.sample.mask_result.mask)
    reference_xy = reference_yx[:, ::-1].astype(float)
    sample_xy = result.alignment.transform.apply(sample_yx[:, ::-1].astype(float))
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
        }
        for rank, candidate in enumerate(extracted.route_candidates, start=1)
    ]


def _writing_payload(result: ComparisonResult, writing_metrics: dict[str, Any]) -> dict[str, Any]:
    selection = result.route_selection
    payload: dict[str, Any] = {
        "available": bool(writing_metrics.get("available")),
        "direction_source": "inferred_from_static_image_geometry",
        "route_ambiguous": selection.ambiguous if selection is not None else None,
        "reference_candidate_count": len(result.reference.route_candidates),
        "sample_candidate_count": len(result.sample.route_candidates),
    }
    if selection is None or not writing_metrics.get("available"):
        payload["unavailable_reason"] = writing_metrics.get(
            "unavailable_reason", "joint writing-route selection is unavailable"
        )
        payload["reference"] = {
            "physical_junction_count": result.reference.diagnostics.topology.junction_count
        }
        payload["sample"] = {
            "physical_junction_count": result.sample.diagnostics.topology.junction_count
        }
        payload["differences"] = {}
        return payload

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
        },
    }
    return payload


def _report_payload(result: ComparisonResult, analysis: ComparisonAnalysis) -> dict[str, Any]:
    transform = result.alignment.transform
    reference_path = result.reference.continuous_path
    sample_path = result.sample.continuous_path
    continuous_geometry = (
        reference_path is not None
        and sample_path is not None
        and result.alignment.aligned_sample_path is not None
    )
    selection = result.route_selection

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
    selected_route_pair = (
        None
        if selection is None
        else {
            "reference_rank": selection.score.reference_rank,
            "sample_rank": selection.score.sample_rank,
            "ambiguous": selection.ambiguous,
            "score": {
                "intrinsic_route_cost": selection.score.intrinsic_route_cost,
                "aligned_centerline_cost": selection.score.aligned_centerline_cost,
                "steering_profile_cost": selection.score.steering_profile_cost,
                "topology_event_compatibility_cost": (
                    selection.score.topology_event_compatibility_cost
                ),
                "total_cost": selection.score.total_cost,
            },
            "alternatives": [
                {
                    "reference_rank": score.reference_rank,
                    "sample_rank": score.sample_rank,
                    "total_cost": score.total_cost,
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
        )
    ]
    primary_artifacts = [_ARTIFACTS["input_reference"], _ARTIFACTS["input_sample"]]
    if selection is not None:
        primary_artifacts.extend(
            (_ARTIFACTS["writing_path_overlay"], _ARTIFACTS["steering_profile"])
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
    return {
        "format": "inkgrammar-letter-comparison",
        "format_version": 3,
        "letter": result.letter,
        "identity": {
            "sample_id": result.identity.sample_id,
            "reference_id": result.identity.reference_id,
            "label": result.identity.label,
        },
        "interpretation": {
            "direction_source": "inferred_from_static_image_geometry",
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
            "continuous_centerline": continuous_geometry,
            "writing_routes": selection is not None,
            "ink_reconstruction": analysis.metrics["reconstruction"]["available"],
        },
        "provenance": {
            "observed_raster_mask": "RasterRegions foreground evidence",
            "discrete_skeleton_topology": (
                "SkeletonPaths graph derived from the observed raster mask"
            ),
            "route_orientation": (
                "inferred from static image geometry; not observed historical pen order"
            ),
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
            "orientation": result.alignment.direction,
            "scale": transform.scale,
            "rotation_deg": transform.rotation_deg,
            "translation_xy": [transform.dx, transform.dy],
            "reflection": False,
            "mean_squared_error_px2": result.alignment.mean_squared_error,
            "interpretation": "Nuisance normalization, not a handwriting-quality score.",
        },
        "writing": _writing_payload(result, analysis.metrics["writing"]),
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
            "selected_route_pair": selected_route_pair,
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
    rows = [
        f"# {_comparison_title(result)}",
        "",
        "![Reference crop](input-reference.png)",
        "![My writing crop](input-sample.png)",
        "",
        "## 1. Writing summary",
        "",
        "Route orientation and direction are inferred from static image geometry. "
        "They are not observed historical pen traces.",
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
            "## 2. How the inferred pen path differs",
            "",
            "The route is a plausible reconstruction from raster and topology evidence, "
            "not a proven historical trace.",
            "",
        )
    )
    if writing["available"]:
        rows.append("![Aligned inferred routes](writing-path-overlay.png)")
    else:
        rows.append(f"Route overlay unavailable: {writing['unavailable_reason']}")
    rows.extend(("", "## 3. Fitted steering profile", ""))
    if writing["available"]:
        rows.append("![Fitted heading-rate profiles](steering-profile.png)")
    else:
        rows.append("A fitted steering profile is unavailable without a jointly selected route.")
    rows.extend(("", "## 4. Ink reconstruction", ""))
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
            "<details><summary>5. Engineering diagnostics</summary>",
            "",
            "Raw geometry, skeleton pixels, candidate routes, and numeric series are retained "
            "in [report.json](report.json).",
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
        return f"Letter comparison: {result.letter}"
    identity = result.identity
    assert identity is not None
    return f"Glyph comparison: {identity.sample_id} to {identity.reference_id}"


def _html(result: ComparisonResult, payload: dict[str, Any]) -> str:
    writing = payload["writing"]
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
        "<h2>1. Writing summary</h2>",
        "<p>Route orientation and direction are inferred from static image geometry. "
        "They are not observed historical pen traces. "
        "Differences are descriptive, not correctness scores.</p>",
        "<table><thead><tr><th>Feature</th><th>Reference</th><th>My writing</th>"
        "<th>Difference / interpretation</th></tr></thead><tbody>",
        summary_rows,
        "</tbody></table>",
        "<h2>2. How the inferred pen path differs</h2>",
    ]
    if writing["available"]:
        rows.extend(
            (
                "<p>Centerline arrows show inferred direction. "
                "Junction circles mark physical semantic regions; nearby numbers "
                "identify ordered encounters.</p>",
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
    rows.extend(("<h2>3. Fitted steering profile</h2>",))
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
    rows.append("<h2>4. Ink reconstruction</h2>")
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
            "<details><summary>5. Engineering diagnostics</summary>",
            "<p>Raw geometry, skeleton pixels, candidate routes, and numeric series "
            "are secondary diagnostics.</p>",
            '<div class="grid">',
            image(_ARTIFACTS["extraction_reference"], "Reference extraction diagnostics"),
            image(_ARTIFACTS["extraction_sample"], "Sample extraction diagnostics"),
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
        },
        "interpretation": {
            "direction_source": "unavailable_without_jointly_selected_route",
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
        "## 1. Writing summary",
        "",
        "Route orientation and direction are inferred from static image geometry, "
        "not observed historical pen traces.",
        "",
        f"Route-based writing measurements are unavailable: {unavailable}",
        "",
        "## 2. How the inferred pen path differs",
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
        "<h2>1. Writing summary</h2>"
        "<p>Route orientation and direction are inferred from static image geometry, "
        "not observed historical pen traces.</p>"
        f"<p>Route-based writing measurements are unavailable: {html.escape(unavailable)}</p>"
        "<h2>2. How the inferred pen path differs</h2>"
        "<p>No comparable jointly selected route is available. No historical route, "
        "direction, speed, elapsed time, pressure, or force is inferred.</p>"
        "<h2>3. Fitted steering profile</h2><p>Unavailable without a jointly selected route.</p>"
        "<h2>4. Ink reconstruction</h2>"
        "<p>Unavailable without a comparable route and width model.</p>"
        "<details><summary>5. Engineering diagnostics</summary>"
        f'<p><img src="{artifacts["overlay_raster"]}" alt="Unaligned observed masks"></p>'
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
    if result.alignment is None:
        return _write_raster_only_report(result, destination)

    analysis = analyze_comparison(result)
    _save_rgb_crop(result.reference.image, destination / _ARTIFACTS["input_reference"])
    _save_rgb_crop(result.sample.image, destination / _ARTIFACTS["input_sample"])
    _write_extraction_images(result.reference, destination, "reference")
    _write_extraction_images(result.sample, destination, "sample")
    _write_raster_overlay(result, destination / _ARTIFACTS["overlay_raster"])
    reference_continuous = result.reference.continuous_path
    sample_continuous = result.alignment.aligned_sample_path
    reference_centerline = (
        reference_continuous.xy
        if reference_continuous is not None
        else result.reference.resampled_xy
    )
    sample_centerline = (
        sample_continuous.xy
        if sample_continuous is not None
        else result.alignment.aligned_sample_xy
    )
    reference_raw_skeleton_xy = np.argwhere(result.reference.diagnostics.skeleton)[:, ::-1].astype(
        float
    )
    sample_raw_skeleton_xy = result.alignment.transform.apply(
        np.argwhere(result.sample.diagnostics.skeleton)[:, ::-1].astype(float)
    )
    reference_event_xy = (
        reference_continuous.xy[list(reference_continuous.route_event_indices)]
        if reference_continuous is not None
        else np.empty((0, 2), dtype=float)
    )
    sample_event_xy = (
        sample_continuous.xy[list(sample_continuous.route_event_indices)]
        if sample_continuous is not None
        else np.empty((0, 2), dtype=float)
    )
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_centerline"],
        reference_centerline,
        sample_centerline,
        reference_raw_skeleton_xy=reference_raw_skeleton_xy,
        sample_raw_skeleton_xy=sample_raw_skeleton_xy,
        reference_topology=result.reference.diagnostics.topology,
        sample_topology=result.sample.diagnostics.topology,
        sample_transform=result.alignment.transform,
        reference_event_xy=reference_event_xy,
        sample_event_xy=sample_event_xy,
    )
    _draw_centerline_overlay(
        destination / _ARTIFACTS["overlay_error_vectors"],
        result.reference.resampled_xy,
        result.alignment.aligned_sample_xy,
        error_vectors=True,
    )
    _write_plots(analysis, destination)
    _write_writing_path_overlay(result, destination / _ARTIFACTS["writing_path_overlay"])
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
