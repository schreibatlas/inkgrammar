from __future__ import annotations

import json
from dataclasses import replace

from PIL import Image, ImageDraw
from planarcurve.transforms import Similarity2D

from inkgrammar.comparison import (
    AlignmentResult,
    ComparisonResult,
    CropRect,
    ExtractedLetter,
    analyze_skeleton,
    extract_ink_mask,
    recover_ordered_path,
    resample_path,
)
from inkgrammar.comparison.path import refine_continuous_path, resample_continuous_path


def _extracted_letter(path, color: str) -> ExtractedLetter:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((10, 50, 25, 15, 50, 48), fill=color, width=5)
    mask = extract_ink_mask(image)
    diagnostics = analyze_skeleton(mask.mask)
    ordered_path = recover_ordered_path(diagnostics)
    continuous_path = refine_continuous_path(diagnostics, ordered_path)
    resampled = resample_continuous_path(continuous_path)
    return ExtractedLetter(
        source_path=path,
        crop=CropRect(x=0, y=0, width=64, height=64),
        image=image,
        mask_result=mask,
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        continuous_path=continuous_path,
        resampled_xy=resampled.xy,
    )


def test_report_writes_machine_and_human_outputs(tmp_path) -> None:
    reference = _extracted_letter("reference.png", "black")
    sample = _extracted_letter("sample.png", "black")
    alignment = AlignmentResult(
        transform=Similarity2D(),
        direction="forward",
        aligned_sample_xy=sample.resampled_xy,
        mean_squared_error=0.0,
        aligned_sample_path=resample_continuous_path(sample.continuous_path),
    )
    result = ComparisonResult("e", reference, sample, alignment)

    from inkgrammar.comparison.report import write_comparison_report

    report = write_comparison_report(result, tmp_path / "comparison")

    required = {
        "report.html",
        "report.md",
        "report.json",
        "input-reference.png",
        "input-sample.png",
        "mask-reference.png",
        "mask-sample.png",
        "skeleton-reference.png",
        "skeleton-sample.png",
        "overlay-raster.png",
        "overlay-centerline.png",
        "overlay-error-vectors.png",
        "plot-position-error.png",
        "plot-x-y.png",
        "plot-heading.png",
        "plot-tangent-error.png",
        "plot-curvature.png",
        "plot-curvature-error.png",
        "plot-cumulative-turn.png",
        "plot-width.png",
    }
    assert required <= {item.name for item in report.output_dir.iterdir()}

    payload = json.loads((report.output_dir / "report.json").read_text())
    assert payload["letter"] == "e"
    assert payload["alignment"]["scale"] > 0.0
    assert payload["metrics"]["path"]["mean_error"] < 1e-12
    assert payload["metrics"]["width"]["scored"] is False
    assert len(payload["series"]["progress"]) == 256
    assert "score" not in payload
    assert payload["format_version"] == 2
    reconstruction = payload["geometry"]["centerline_reconstruction"]
    assert reconstruction["algorithm"] == "skeletonpaths-subpixel-v1"
    assert reconstruction["curve_model"] == "planarcurve-cubic-b-spline"
    assert reconstruction["fallback"] is False
    assert payload["geometry"]["legacy_fallback"] is None
    assert payload["geometry"]["differential_geometry"]["curvature_summary_guard_progress"] == 0.005
    assert payload["metrics"]["turn_rate"]["summary_sample_count"] < 256
    assert payload["metrics"]["width"]["source"] == "subpixel-cross-section"
    assert "not a proven historical pen trace" in payload["provenance"]["continuous_centerline"]
    for name in (
        "reference_raw_skeleton_xy",
        "sample_raw_skeleton_xy",
        "reference_subpixel_xy",
        "sample_subpixel_xy",
        "reference_centerline_confidence",
        "sample_centerline_confidence",
        "reference_route_event_progress",
        "sample_route_event_progress",
    ):
        assert name in payload["series"]
    assert payload["series"]["reference_raw_skeleton_xy"]
    assert payload["series"]["reference_subpixel_xy"]
    assert payload["series"]["reference_route_event_progress"]
    with Image.open(report.output_dir / "plot-heading.png") as plot:
        assert plot.size == (900, 440)
    html = (report.output_dir / "report.html").read_text()
    assert "overlay-centerline.png" in html
    assert "plot-curvature-error.png" in html
    assert all(name in html for name in required if name.endswith(".png"))
    assert (report.output_dir / "input-reference.png").read_bytes()


def test_report_records_selected_legacy_fallback(tmp_path) -> None:
    reference = _extracted_letter("reference.png", "black")
    sample = _extracted_letter("sample.png", "black")
    reference = replace(
        reference,
        continuous_path=None,
        resampled_xy=resample_path(reference.ordered_path.xy),
    )
    sample = replace(
        sample,
        continuous_path=None,
        resampled_xy=resample_path(sample.ordered_path.xy),
    )
    alignment = AlignmentResult(
        transform=Similarity2D(),
        direction="forward",
        aligned_sample_xy=reference.resampled_xy,
        mean_squared_error=0.0,
    )
    result = ComparisonResult("e", reference, sample, alignment)

    from inkgrammar.comparison.report import write_comparison_report

    report = write_comparison_report(result, tmp_path / "legacy")
    reconstruction = report.payload["geometry"]["centerline_reconstruction"]
    assert report.payload["format_version"] == 2
    assert reconstruction["fallback"] is True
    assert reconstruction["algorithm"] == "legacy-polyline"
    assert report.payload["geometry"]["legacy_fallback"]["algorithm"] == "Savitzky-Golay"
    assert report.payload["metrics"]["width"]["source"] == "edt-bilinear-fallback"
