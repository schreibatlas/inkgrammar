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
    metrics = payload["diagnostics"]["metrics"]
    geometry = payload["diagnostics"]["geometry"]
    series = payload["diagnostics"]["series"]
    assert metrics["path"]["mean_error"] < 1e-12
    assert metrics["width"]["scored"] is False
    assert payload["format_version"] == 3
    assert payload["capabilities"]["writing_routes"] is False
    assert payload["writing"]["available"] is False
    assert geometry["centerline_reconstruction"]["algorithm"] == "skeletonpaths-subpixel-v1"
    assert geometry["centerline_reconstruction"]["fallback"] is False
    assert geometry["legacy_fallback"] is None
    assert geometry["differential_geometry"]["curvature_summary_guard_progress"] == 0.005
    assert metrics["turn_rate"]["summary_sample_count"] < 256
    assert metrics["width"]["source"] == "subpixel-cross-section"
    assert "not a proven historical pen trace" in payload["provenance"]["continuous_centerline"]
    assert "not observed historical pen order" in payload["provenance"]["route_orientation"]
    assert "speed" in payload["provenance"]["unmeasured_properties"]
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
        assert name in series
    assert series["reference_raw_skeleton_xy"]
    assert series["reference_subpixel_xy"]
    assert series["reference_route_event_progress"]
    with Image.open(report.output_dir / "plot-heading.png") as plot:
        assert plot.size == (900, 440)
    html = (report.output_dir / "report.html").read_text()
    markdown = (report.output_dir / "report.md").read_text()
    assert "Engineering diagnostics" in html
    assert "plot-curvature-error.png" in html
    assert "1. Writing summary" in markdown
    assert "not observed historical pen traces" in markdown
    assert "score" not in payload
    assert (report.output_dir / "input-reference.png").read_bytes()


def test_report_v3_renders_lowercase_e_routes_and_ink_residuals(tmp_path) -> None:
    from inkgrammar.comparison import (
        ComparisonIdentity,
        compare_extracted_letters,
        extract_letter,
    )
    from inkgrammar.comparison.report import write_comparison_report

    image_path = tmp_path / "lowercase-e.png"
    image = Image.new("RGB", (64, 64), "white")
    points = [
        (15, 27),
        (34, 27),
        (36, 22),
        (33, 16),
        (25, 14),
        (18, 18),
        (15, 26),
        (17, 34),
        (24, 38),
        (33, 37),
    ]
    ImageDraw.Draw(image).line(points, fill="black", width=4)
    image.save(image_path)
    crop = CropRect(x=0, y=0, width=64, height=64)
    extracted = extract_letter(image_path, crop)
    assert len(extracted.diagnostics.topology.junction_regions) == 3
    result = compare_extracted_letters(
        extracted,
        extracted,
        identity=ComparisonIdentity("sample", "reference", "e"),
    )
    assert result.route_selection is not None
    selected_events = [
        event
        for event in result.route_selection.reference.events
        if event.kind == "junction-encounter"
    ]
    assert len(selected_events) > result.route_selection.reference.physical_junction_count
    assert {event.junction_node_id for event in selected_events} == {
        region.node_id for region in extracted.diagnostics.topology.junction_regions
    }
    assert any(event.occurrence == 2 for event in selected_events)
    report = write_comparison_report(result, tmp_path / "writing-aware")

    assert report.payload["format_version"] == 3
    assert report.payload["writing"]["available"] is True
    assert report.payload["writing"]["reference"]["physical_junction_count"] == 3
    assert report.payload["writing"]["reference"]["junction_encounter_count"] == len(
        selected_events
    )
    assert report.payload["writing"]["route_ambiguous"] is True
    assert (
        report.payload["interpretation"]["direction_source"]
        == "inferred_from_static_image_geometry"
    )
    assert report.payload["reconstruction"]["available"] is True
    assert report.payload["artifacts"]["primary"][:4] == [
        "input-reference.png",
        "input-sample.png",
        "writing-path-overlay.png",
        "steering-profile.png",
    ]
    for artifact in (
        "writing-path-overlay.png",
        "steering-profile.png",
        "ink-reconstruction-reference.png",
        "ink-reconstruction-sample.png",
        "ink-residual-reference.png",
        "ink-residual-sample.png",
    ):
        assert (report.output_dir / artifact).is_file()
    markdown = report.markdown_path.read_text()
    assert "2. How the inferred pen path differs" in markdown
    assert "4. Ink reconstruction" in markdown
    html = report.html_path.read_text()
    assert "junction circles mark physical semantic regions" in html.lower()
    assert "under-coverage" in html


def test_raster_only_report_uses_v3_unavailable_sections(tmp_path) -> None:
    from inkgrammar.comparison.report import write_comparison_report

    reference = _extracted_letter("reference.png", "black")
    sample = _extracted_letter("sample.png", "black")
    result = ComparisonResult("e", reference, sample, None)
    report = write_comparison_report(result, tmp_path / "raster-only")

    assert report.payload["format_version"] == 3
    assert report.payload["alignment"] is None
    assert report.payload["writing"]["available"] is False
    assert report.payload["reconstruction"]["available"] is False
    assert report.payload["diagnostics"]["selected_route_pair"] is None
    assert report.payload["provenance"]["unmeasured_properties"]
    assert (report.output_dir / "overlay-raster.png").is_file()
    assert "Engineering diagnostics" in report.markdown_path.read_text()
    assert "Full report JSON" in report.html_path.read_text()


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
    reconstruction = report.payload["diagnostics"]["geometry"]["centerline_reconstruction"]
    assert report.payload["format_version"] == 3
    assert reconstruction["fallback"] is True
    assert reconstruction["algorithm"] == "legacy-polyline"
    assert (
        report.payload["diagnostics"]["geometry"]["legacy_fallback"]["algorithm"]
        == "Savitzky-Golay"
    )
    assert report.payload["diagnostics"]["metrics"]["width"]["source"] == "edt-bilinear-fallback"
