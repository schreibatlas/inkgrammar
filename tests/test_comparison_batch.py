from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

import inkgrammar.comparison.api as comparison_api
from inkgrammar.comparison.batch import compare_letter_page


def _draw_glyph(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "e":
        draw.line(
            [
                (x, y + 15),
                (x + 19, y + 15),
                (x + 21, y + 10),
                (x + 18, y + 4),
                (x + 10, y + 2),
                (x + 3, y + 6),
                (x, y + 14),
                (x + 2, y + 22),
                (x + 9, y + 26),
                (x + 18, y + 25),
            ],
            fill="black",
            width=4,
        )
    elif kind == "n":
        draw.line(
            [
                (x, y + 26),
                (x, y + 5),
                (x + 6, y + 2),
                (x + 15, y + 4),
                (x + 20, y + 11),
                (x + 20, y + 26),
            ],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 8, y, x + 8, y + 26), fill="black", width=4)


def _page(path, glyphs: tuple[str, ...], *, y: int = 24):
    image = Image.new("RGB", (250, 88), "white")
    draw = ImageDraw.Draw(image)
    for x, kind in zip((15, 95, 175), glyphs, strict=True):
        _draw_glyph(draw, x, y, kind)
    image.save(path)
    return path


def test_batch_writes_pair_reports_summaries_and_reuses_hash_keyed_reference_cache(
    tmp_path, monkeypatch
):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "automatic-run"

    first = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
    )

    assert first.payload["sample"]["detected_count"] == 3
    assert first.payload["reference_cache"]["hit"] is False
    assert first.json_path.is_file()
    assert first.markdown_path.is_file()
    assert first.html_path.is_file()
    for filename in (
        "reference-mask.png",
        "reference-rows.png",
        "reference-components.png",
        "reference-gallery.png",
        "ref-000-mask.png",
        "reference-candidates.png",
        "reference-candidates.json",
        "reference-segmentation-hypotheses.json",
        "sample-hypotheses.json",
        "sample-mask.png",
        "sample-boundaries.png",
        "sample-candidates.png",
        "sample-candidates.json",
    ):
        assert (output / "segmentation" / filename).is_file()
    assert (output / "reference-index" / "gallery.json").is_file()
    assert all(
        (output / "reference-index" / f"ref-{index:03d}.png").is_file() for index in range(3)
    )
    reference_manifest = json.loads(
        (output / "segmentation" / "reference-candidates.json").read_text()
    )
    assert reference_manifest["reference_profile"] == "alphabet"
    assert reference_manifest["row_count"] == 1
    assert reference_manifest["per_row_candidate_counts"] == {"0": 3}
    candidate = reference_manifest["candidates"][0]
    assert candidate["dimensions"] == {
        "width": candidate["bbox"]["width"],
        "height": candidate["bbox"]["height"],
    }

    summary = json.loads(first.json_path.read_text())
    assert len(summary["matches"]) == 3
    assert "not a probability" in summary["matching"]["cost_interpretation"]
    assert "<img" in first.html_path.read_text()
    for match in summary["matches"]:
        sample_dir = output / match["sample_id"]
        assert (sample_dir / "sample.png").is_file()
        assert (sample_dir / "matched-reference.png").is_file()
        assert (sample_dir / "alternatives.png").is_file()
        assert (sample_dir / "match.json").is_file()
        assert (sample_dir / "report.json").is_file()
        assert (sample_dir / "report.html").is_file()
        assert len(match["alternatives"]) == 2

    def no_rebuild(*args, **kwargs):
        raise AssertionError("reference segmentation should be loaded from the content-hash cache")

    monkeypatch.setattr("inkgrammar.comparison.batch.segment_reference_gallery", no_rebuild)
    analysis_calls = 0
    original_analyze = comparison_api.analyze_skeleton

    def count_sample_analysis(mask):
        nonlocal analysis_calls
        analysis_calls += 1
        return original_analyze(mask)

    monkeypatch.setattr(comparison_api, "analyze_skeleton", count_sample_analysis)


    def fail_repeat_analysis(mask):
        raise AssertionError("prepared sample paths must be reused for every top-K candidate")

    monkeypatch.setattr("inkgrammar.comparison.matching.analyze_skeleton", fail_repeat_analysis)
    second = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
    )

    assert second.payload["reference_cache"]["hit"] is True
    public_gallery = json.loads((output / "reference-index" / "gallery.json").read_text())
    assert public_gallery["cache_hit"] is True
    assert analysis_calls == 3



def test_reference_cache_rebuilds_for_old_versions_profiles_and_mask_changes(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "cache-invalidation"

    first = compare_letter_page(reference, sample, expected_letters=3, output_dir=output)
    cache_dir = next((output / ".reference-cache").iterdir())
    metadata_path = cache_dir / "gallery.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["cache_version"] = 1
    metadata_path.write_text(json.dumps(metadata))

    old_cache = compare_letter_page(reference, sample, expected_letters=3, output_dir=output)
    assert old_cache.payload["reference_cache"]["hit"] is False

    generic = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        reference_profile="generic",
    )
    assert generic.payload["reference_cache"]["hit"] is False
    public_gallery = json.loads((output / "reference-index" / "gallery.json").read_text())
    assert public_gallery["reference_profile"] == "generic"

    cache_dir = next((output / ".reference-cache").iterdir())
    with Image.open(cache_dir / "crops" / "ref-000-mask.png") as mask:
        Image.new("L", mask.size, 255).save(cache_dir / "crops" / "ref-000-mask.png")
    corrupted_mask = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        reference_profile="generic",
    )
    assert corrupted_mask.payload["reference_cache"]["hit"] is False
    assert first.payload["reference_cache"]["hit"] is False



def test_progress_and_run_files_cover_success_failure_and_quiet_mode(tmp_path, capsys):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "progress-run"

    compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        verbose=True,
    )

    stderr = capsys.readouterr().err
    assert "[compare-letter] 01/13 Validate inputs" in stderr
    assert "[compare-letter] 13/13 Done" in stderr
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    stages = {event["stage"] for event in events}
    extraction_events = [event for event in events if event["stage"] == "reference_extraction"]
    assert extraction_events
    assert all("elapsed_seconds" in event["details"] for event in extraction_events)
    assert {
        "input_validation",
        "reference_hash",
        "reference_cache",
        "reference_mask_rows",
        "reference_segmentation",
        "reference_descriptors",
        "sample_segmentation",
        "coarse_ranking",
        "reference_extraction",
        "path_refinement",
        "pair_report",
        "batch_summary",
        "completion",
    } <= stages
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["completed_sample_count"] == run["expected_sample_count"] == 3
    assert run["artifact_paths"]["summary_json"] == str(output / "summary.json")

    quiet_output = tmp_path / "quiet-run"
    compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=quiet_output,
        no_progress=True,
    )
    assert capsys.readouterr().err == ""
    assert (quiet_output / "progress.jsonl").is_file()

    from inkgrammar.comparison.segmentation import SegmentationError

    failed_output = tmp_path / "failed-run"
    with pytest.raises(SegmentationError):
        compare_letter_page(
            reference,
            sample,
            expected_letters=4,
            output_dir=failed_output,
            no_progress=True,
        )
    failed_run = json.loads((failed_output / "run.json").read_text())
    assert failed_run["status"] == "failed"
    assert failed_run["failure_details"]["type"] == "SegmentationError"
    assert failed_run["artifact_paths"]["diagnostic"].endswith("sample-boundaries.png")
    failure_event = json.loads((failed_output / "progress.jsonl").read_text().splitlines()[-1])
    assert failure_event["stage"] == "failure"


def test_segment_only_writes_diagnostics_without_matching_or_skeleton_analysis(
    tmp_path, monkeypatch
):
    import inkgrammar.comparison.batch as comparison_batch

    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "segment-only"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("segment-only must stop before descriptors, extraction, and matching")

    monkeypatch.setattr(comparison_batch, "build_reference_index", fail_if_called)
    monkeypatch.setattr(comparison_batch, "extract_segmented_glyph", fail_if_called)
    monkeypatch.setattr(comparison_batch, "rank_reference_candidates", fail_if_called)

    report = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        segment_only=True,
        no_progress=True,
    )

    assert report is None
    summary = json.loads((output / "summary.json").read_text())
    assert summary["operation"] == "segment-only"
    assert summary["sample"]["detected_count"] == 3
    assert (output / "segmentation" / "reference-gallery.png").is_file()
    assert (output / "segmentation" / "sample-boundaries.png").is_file()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["completed_sample_count"] == 3
    stages = {
        json.loads(line)["stage"] for line in (output / "progress.jsonl").read_text().splitlines()
    }
    assert "reference_descriptors" not in stages
    assert "reference_extraction" not in stages
    assert "path_refinement" not in stages



def test_reference_segmentation_ambiguity_is_separate_from_match_ambiguity(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import inkgrammar.comparison.batch as comparison_batch

    original_segment = comparison_batch.segment_reference_gallery

    def ambiguous_reference(*args, **kwargs):
        gallery = original_segment(*args, **kwargs)
        segmentation = replace(
            gallery.segmentation,
            status="ambiguous",
            warnings=("synthetic reference-row uncertainty",),
        )
        return replace(gallery, segmentation=segmentation)

    monkeypatch.setattr(comparison_batch, "segment_reference_gallery", ambiguous_reference)
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "ambiguous-reference"

    report = compare_letter_page(reference, sample, expected_letters=3, output_dir=output)

    assert report is not None
    assert report.payload["status"] == "ambiguous"
    assert report.payload["segmentation_ambiguous_count"] == 3
    assert report.payload["ambiguous_count"] == report.payload["match_ambiguous_count"]
    for item in report.payload["matches"]:
        assert item["segmentation_status"] == "ambiguous"
        assert item["status"] == item["match_status"]
        assert any(
            "synthetic reference-row uncertainty" in warning
            for warning in item["segmentation_warnings"]
        )
        match = json.loads((output / item["match_json"]).read_text())
        assert match["segmentation_status"] == "ambiguous"
        assert match["status"] == match["match_status"]
    assert "Segmentation-ambiguous samples: 3" in report.markdown_path.read_text()
    assert "Segmentation status" in report.html_path.read_text()
