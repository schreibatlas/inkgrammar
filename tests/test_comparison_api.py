from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw
from typer.testing import CliRunner

from inkgrammar import compare_letter_images as root_compare_letter_images
from inkgrammar.cli import app
from inkgrammar.comparison import compare_letter_images


def _write_letter_images(directory):
    reference_path = directory / "reference.png"
    sample_path = directory / "sample.png"
    for path, ink in ((reference_path, "black"), (sample_path, (20, 70, 220))):
        image = Image.new("RGB", (80, 72), "white")
        draw = ImageDraw.Draw(image)
        if path == sample_path:
            draw.line((2, 40, 77, 40), fill=(180, 180, 180), width=1)
        draw.line(
            [
                (16, 36),
                (30, 36),
                (43, 36),
                (41, 31),
                (37, 27),
                (31, 25),
                (25, 26),
                (20, 30),
                (18, 36),
                (20, 43),
                (25, 48),
                (32, 50),
                (39, 49),
                (45, 45),
            ],
            fill=ink,
            width=5,
        )
        image.save(path)
    return reference_path, sample_path


def _compare_options(output_dir):
    return {
        "letter": "e",
        "reference_crop": (5, 5, 70, 62),
        "sample_crop": (5, 5, 70, 62),
        "output_dir": output_dir,
    }


def test_compare_letter_images_api_writes_complete_report_and_keeps_crop_pixels(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)
    output_dir = tmp_path / "api-report"

    report = compare_letter_images(
        reference_path,
        sample_path,
        **_compare_options(output_dir),
    )

    assert report.html_path.is_file()
    assert report.json_path.is_file()
    assert report.markdown_path.is_file()
    assert report.payload["letter"] == "e"
    assert report.payload["inputs"]["reference"]["mask_method"] == "grayscale"
    assert report.payload["inputs"]["sample"]["mask_method"] == "color"
    assert report.payload["inputs"]["sample"]["topology"]["component_count"] == 1
    diagnostics = report.payload["diagnostics"]
    assert diagnostics["geometry"]["sample_count"] == 256
    assert diagnostics["metrics"]["path"]["mean_error"] < 0.01
    writing = diagnostics["writing_metrics"]
    assert writing["direction_source"] == "inferred_from_static_image_geometry"
    assert writing["entry_exit"]["direction_source"] == "inferred_from_static_image_geometry"
    assert writing["steering"]["reference_strokes"]
    reconstruction = report.payload["reconstruction"]
    assert reconstruction["available"]
    assert reconstruction["reference"]["renderer"] == "strokeoutline.render_strokes"
    assert reconstruction["sample"]["stroke_width_profiles"]
    with (
        Image.open(reference_path) as source,
        Image.open(output_dir / "input-reference.png") as retained,
    ):
        expected = source.crop((5, 5, 75, 67))
        assert ImageChops.difference(expected, retained).getbbox() is None


def test_compare_letter_images_is_exported_from_package_root(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)

    report = root_compare_letter_images(
        reference_path,
        sample_path,
        **_compare_options(tmp_path / "root-report"),
    )

    assert report.html_path.is_file()


def test_compare_letter_cli_writes_report_for_manual_crops(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)
    output_dir = tmp_path / "cli-report"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference_path),
            str(sample_path),
            "--letter",
            "e",
            "--reference-crop",
            "5,5,70,62",
            "--sample-crop",
            "5,5,70,62",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "report.html").is_file()
    assert "HTML report:" in result.output
    assert "JSON report:" in result.output


def test_compare_letter_cli_rejects_malformed_crop_coordinates(tmp_path):
    reference_path, sample_path = _write_letter_images(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference_path),
            str(sample_path),
            "--letter",
            "e",
            "--reference-crop",
            "5,5,70",
            "--sample-crop",
            "5,5,70,62",
            "--output",
            str(tmp_path / "invalid-report"),
        ],
    )

    assert result.exit_code == 2
    assert "--reference-crop must be x,y,width,height" in result.output


def test_label_free_comparison_writes_raster_report_when_route_search_fails(tmp_path, monkeypatch):
    import inkgrammar.comparison.api as comparison_api
    from inkgrammar.comparison import (
        ComparisonIdentity,
        compare_extracted_letters,
        extract_letter,
    )
    from inkgrammar.comparison.path import PathExtractionError
    from inkgrammar.comparison.report import write_comparison_report

    reference_path, sample_path = _write_letter_images(tmp_path)

    def fail_route_search(_diagnostics, **_kwargs):
        raise PathExtractionError("synthetic multi-stroke path")

    monkeypatch.setattr(comparison_api, "generate_writing_route_candidates", fail_route_search)
    reference = extract_letter(reference_path, (0, 0, 80, 72), allow_path_failure=True)
    sample = extract_letter(sample_path, (0, 0, 80, 72), allow_path_failure=True)
    result = compare_extracted_letters(
        reference,
        sample,
        identity=ComparisonIdentity("sample-000", "ref-004"),
    )
    report = write_comparison_report(result, tmp_path / "raster-only")

    assert result.alignment is None
    assert report.payload["letter"] is None
    assert report.payload["identity"] == {
        "sample_id": "sample-000",
        "reference_id": "ref-004",
        "label": None,
    }
    capabilities = report.payload["capabilities"]
    assert capabilities["raster"] is True
    assert capabilities["topology"] is True
    assert capabilities["ordered_path"] is True
    assert capabilities["continuous_centerline"] is False
    assert capabilities["writing_routes"] is False
    assert capabilities["ink_reconstruction"] is False
    assert "synthetic multi-stroke path" in report.payload["writing"]["unavailable_reason"]
    assert report.payload["format_version"] == 3
    diagnostics = report.payload["diagnostics"]
    reconstruction = diagnostics["geometry"]["centerline_reconstruction"]
    assert reconstruction["fallback"] is True
    assert reconstruction["algorithm"] == "unavailable_without_joint_route"
    assert diagnostics["series"]["reference_raw_skeleton_xy"]
    centerline_provenance = report.payload["provenance"]["continuous_centerline"]
    assert "not a proven historical pen trace" in centerline_provenance
    assert report.html_path.is_file()
    assert report.markdown_path.is_file()
    assert (report.output_dir / "overlay-raster.png").is_file()
    assert "Route-based writing measurements are unavailable" in report.markdown_path.read_text()


def test_changed_smoothing_options_warn_and_do_not_change_continuous_geometry(tmp_path):
    import numpy as np

    from inkgrammar.comparison import (
        ComparisonIdentity,
        compare_extracted_letters,
        extract_letter,
    )
    from inkgrammar.comparison.metrics import analyze_comparison

    reference_path, sample_path = _write_letter_images(tmp_path)
    reference = extract_letter(reference_path, (5, 5, 70, 62))
    sample = extract_letter(sample_path, (5, 5, 70, 62))
    identity = ComparisonIdentity("sample", "reference", "e")
    default = compare_extracted_letters(reference, sample, identity=identity)
    changed = compare_extracted_letters(
        reference,
        sample,
        identity=identity,
        smoothing_window=7,
        smoothing_polynomial_order=2,
    )

    warning = next(message for message in changed.warnings if "Legacy smoothing options" in message)
    assert "window=7" in warning
    assert "only by the ordered-polyline fallback" in warning
    assert not any("Legacy smoothing options" in message for message in default.warnings)
    default_analysis = analyze_comparison(default)
    changed_analysis = analyze_comparison(changed)
    np.testing.assert_array_equal(
        default_analysis.reference_geometry.heading_rad,
        changed_analysis.reference_geometry.heading_rad,
    )
    np.testing.assert_array_equal(
        default_analysis.reference_geometry.turn_rate_rad_per_progress,
        changed_analysis.reference_geometry.turn_rate_rad_per_progress,
    )
