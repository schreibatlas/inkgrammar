from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw
from typer.testing import CliRunner

from inkgrammar import compare_letter_images as root_compare_letter_images
from inkgrammar.cli import app
from inkgrammar.comparison import (
    ComparisonIdentity,
    CropRect,
    analyze_comparison,
    compare_extracted_letters,
    compare_letter_images,
    extract_letter,
)


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


def test_multistroke_comparison_preserves_and_scores_every_selected_stroke(tmp_path):
    image_path = tmp_path / "crossing-strokes.png"
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 50, 50), fill="black", width=3)
    draw.line((10, 50, 50, 10), fill="black", width=3)
    image.save(image_path)
    extracted = extract_letter(image_path, CropRect(x=0, y=0, width=64, height=64))

    result = compare_extracted_letters(
        extracted,
        extracted,
        identity=ComparisonIdentity("sample", "reference", "x"),
    )

    selection = result.route_selection
    assert selection is not None
    stroke_count = len(selection.reference.continuous_strokes)
    assert stroke_count > 1
    assert result.reference.selected_route is selection.reference
    assert result.reference.ordered_path is None
    assert result.reference.continuous_path is None
    assert result.reference.resampled_xy is None
    assert result.alignment is None
    assert result.route_alignment is not None
    assert len(result.route_alignment.aligned_sample_strokes) == stroke_count

    analysis = analyze_comparison(result)
    assert analysis.series["reference_stroke_offsets"].tolist() == [
        0,
        *np.cumsum([len(path.xy) for path in selection.reference.continuous_strokes]),
    ]
    assert analysis.series["reference_resampled_stroke_offsets"][-1] == 256
    assert len(analysis.series["reference_resampled_stroke_offsets"]) == stroke_count + 1
    assert len(analysis.metrics["writing"]["entry_exit"]["per_stroke"]) == stroke_count
    assert (
        len(
            analysis.metrics["writing"]["route_pair_score"][
                "per_stroke_alignment_mean_squared_error_px2"
            ]
        )
        == stroke_count
    )


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
        "status": "accepted",
        "accepted_reference_id": "ref-004",
        "final_feedback_available": True,
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


def test_segmented_extraction_passes_assigned_mark_stroke_prior(monkeypatch):
    import inkgrammar.comparison.api as comparison_api
    from inkgrammar.comparison.models import WritingStructurePolicy
    from inkgrammar.comparison.segmentation import SegmentedGlyph

    mask = np.zeros((16, 16), dtype=bool)
    mask[4, 2:9] = True
    mask[1, 11] = True
    mask[12, 12] = True
    candidate = SegmentedGlyph(
        glyph_id="sample",
        bbox=CropRect(x=0, y=0, width=16, height=16),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=Image.new("RGB", (16, 16), "white"),
        mask=mask,
        component_indices=(0, 1, 2),
        primary_component_indices=(0,),
        secondary_component_indices=(1,),
        ambiguous_component_indices=(2,),
        secondary_mark_centroids=((0.8, 0.1),),
    )
    policy = WritingStructurePolicy(preferred_body_strokes=2)
    captured = {}

    def capture_search(_diagnostics, **kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(comparison_api, "generate_writing_route_candidates", capture_search)
    extracted = comparison_api.extract_segmented_glyph(
        candidate,
        source_path="sample.png",
        writing_structure_policy=policy,
    )
    assert extracted.component_role_sequence == ("primary", "secondary", "ambiguous")
    assert extracted.secondary_mark_centroids == ((0.8, 0.1),)

    assert captured["expected_strokes"] == 3
    assert captured["structure_policy"] is policy
    assert captured["ambiguous_component_count"] == 1


def test_comparison_reuses_the_route_pair_selection_from_retrieval(monkeypatch) -> None:
    import inkgrammar.comparison.api as comparison_api
    from inkgrammar.comparison.models import ComparisonIdentity
    from inkgrammar.comparison.routes import evaluate_writing_route_pairs
    from inkgrammar.comparison.segmentation import SegmentedGlyph
    from inkgrammar.comparison.steering import fit_writing_route_candidates

    mask = np.zeros((32, 32), dtype=bool)
    mask[15, 5:28] = True
    candidate = SegmentedGlyph(
        glyph_id="glyph",
        bbox=CropRect(x=0, y=0, width=32, height=32),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=Image.new("RGB", (32, 32), "white"),
        mask=mask,
        component_indices=(0,),
        primary_component_indices=(0,),
    )
    extracted = comparison_api.extract_segmented_glyph(candidate, source_path="glyph.png")
    fitted = fit_writing_route_candidates(extracted.route_candidates)
    selection = evaluate_writing_route_pairs(
        fitted,
        fitted,
        reference_descriptors=extracted.route_descriptors,
        sample_descriptors=extracted.route_descriptors,
    )

    def unexpected_pair_evaluation(*_args, **_kwargs):
        pytest.fail("retrieval's selected route pair must be reused")

    monkeypatch.setattr(
        comparison_api,
        "evaluate_writing_route_pairs",
        unexpected_pair_evaluation,
    )
    result = comparison_api.compare_extracted_letters(
        extracted,
        extracted,
        identity=ComparisonIdentity("sample", "reference", "x"),
        route_pair_selection=selection,
    )

    assert result.route_selection is selection
    assert result.reference.selected_route is selection.reference
    assert result.sample.selected_route is selection.sample
