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
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
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
        "reference-anchor-regions.png",
        "reference-selected-regions.png",
        "reference-cleanup-overlay.png",
        "reference-segmentation-decision.json",
        "reference-segmentation.md",
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
    assert (output / "segmentation" / "reference-hypotheses" / "hypotheses.json").is_file()
    assert (output / "reference-index" / "gallery.json").is_file()
    assert all(
        (output / "reference-index" / f"ref-{index:03d}.png").is_file() for index in range(3)
    )
    reference_manifest = json.loads(
        (output / "segmentation" / "reference-candidates.json").read_text()
    )
    assert reference_manifest["reference_profile"] == "generic"
    assert reference_manifest["row_count"] == 1
    assert reference_manifest["per_row_candidate_counts"] == {"0": 3}
    assert isinstance(reference_manifest["search_truncated"], bool)
    assert reference_manifest["cleanup_diagnostics"]["component_decisions"]
    assert reference_manifest["segmentation_decision"]["accepted"] is True
    assert reference_manifest["search_limit"] is None
    assert isinstance(reference_manifest["anchor_region_count"], int)
    candidate = reference_manifest["candidates"][0]
    assert candidate["primary_component_ids"]
    assert isinstance(candidate["secondary_component_ids"], list)
    assert isinstance(candidate["ambiguous_component_ids"], list)
    assert isinstance(candidate["secondary_mark_centroids"], list)
    assert candidate["dimensions"] == {
        "width": candidate["bbox"]["width"],
        "height": candidate["bbox"]["height"],
    }

    summary = json.loads(first.json_path.read_text())
    assert len(summary["matches"]) == 3
    reference_segmentation = summary["reference"]["segmentation"]
    assert summary["writing_direction_ambiguous_count"] == sum(
        any(
            source["source"] == "writing_direction_unresolved"
            for source in item["ambiguity_sources"]
        )
        for item in summary["matches"]
    )
    assert reference_segmentation["status"] == "ok"
    assert reference_segmentation["reference_profile"] == "generic"
    assert reference_segmentation["row_count"] == 1
    assert reference_segmentation["candidate_count"] == 3
    assert reference_segmentation["per_row_candidate_counts"] == {"0": 3}
    assert isinstance(reference_segmentation["search_truncated"], bool)
    assert reference_segmentation["search_limit"] is None
    assert (
        reference_segmentation["unassigned_ink_fraction"]
        == reference_manifest["unassigned_ink_fraction"]
    )
    assert "not a probability" in summary["matching"]["cost_interpretation"]
    assert "<img" in first.html_path.read_text()
    batch_html = first.html_path.read_text()
    assert 'width="100"' in batch_html
    assert 'width="180"' in batch_html
    readable_pair_diagnostic = False
    for match in summary["matches"]:
        sample_dir = output / match["sample_id"]
        assert (sample_dir / "sample.png").is_file()
        assert (sample_dir / "matched-reference.png").is_file()
        assert (sample_dir / "alternatives.png").is_file()
        assert (sample_dir / "match.json").is_file()
        assert (sample_dir / "report.json").is_file()
        assert (sample_dir / "report.html").is_file()
        assert len(match["alternatives"]) + 1 <= match["search"]["refined_candidate_count"]
        assert match["search"]["coarse_candidate_count"] == 3
        assert match["search"]["refinement_limit"] == 32
        assert match["search"]["stop_reason"] in {
            "all_candidates_refined",
            "coarse_lower_bound",
            "candidate_refinement_limit",
        }
        assert isinstance(match["search"]["search_truncated"], bool)
        leading = match["leading_match"]
        assert leading["reference_id"] == match["leading_reference_id"]
        assert "missing_evidence_penalty" in leading
        assert leading["feature_coverage"]["available_features"]
        assert all("feature_coverage" in alternative for alternative in match["alternatives"])
        assert all("route_descriptor_cost" in alternative for alternative in match["alternatives"])
        assert match["search"]["refined_candidate_count"] >= 1
        if match["status"] == "clear":
            assert match["accepted_reference_id"] == match["leading_reference_id"]
        else:
            assert match["accepted_reference_id"] is None

        pair_overlay = output / match["report_dir"] / "writing-path-overlay.png"
        if pair_overlay.is_file():
            with Image.open(pair_overlay) as diagnostic:
                assert min(diagnostic.size) >= 640
            readable_pair_diagnostic = True
    assert readable_pair_diagnostic
    assert all(f"{match['report_dir']}/report.html" in batch_html for match in summary["matches"])

    def no_rebuild(*args, **kwargs):
        raise AssertionError("reference segmentation should be loaded from the content-hash cache")

    monkeypatch.setattr("inkgrammar.comparison.batch.segment_reference_gallery", no_rebuild)
    import inkgrammar.comparison.batch as batch_module

    original_extract = batch_module.extract_segmented_glyph

    def fail_reference_route_extraction(candidate, **kwargs):
        if candidate.glyph_id.startswith("ref-"):
            raise AssertionError("cache hits must not run reference route extraction")
        return original_extract(candidate, **kwargs)

    monkeypatch.setattr(batch_module, "extract_segmented_glyph", fail_reference_route_extraction)
    analysis_calls = 0
    original_analyze = comparison_api.analyze_skeleton

    def count_sample_analysis(mask):
        nonlocal analysis_calls
        analysis_calls += 1
        return original_analyze(mask)

    monkeypatch.setattr(comparison_api, "analyze_skeleton", count_sample_analysis)

    def fail_repeat_analysis(mask):
        raise AssertionError("prepared sample paths must be reused during progressive refinement")

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
    cached_match = second.payload["matches"][0]
    cached_pair = json.loads((output / cached_match["report_json"]).read_text())
    assert cached_pair["diagnostics"]["selected_route_pair"] is not None


def test_reference_cache_rebuilds_for_old_versions_profiles_and_mask_changes(tmp_path, monkeypatch):
    import inkgrammar.comparison.api as comparison_api

    original_route_generator = comparison_api.generate_writing_route_candidates

    def bounded_route_generator(diagnostics, **kwargs):
        kwargs["limit"] = 4
        return original_route_generator(diagnostics, **kwargs)

    monkeypatch.setattr(
        comparison_api,
        "generate_writing_route_candidates",
        bounded_route_generator,
    )
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "cache-invalidation"

    first = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
    )
    cache_dir = next((output / ".reference-cache").iterdir())
    metadata_path = cache_dir / "gallery.json"
    metadata = json.loads(metadata_path.read_text())
    candidate_ids = [item["id"] for item in metadata["candidates"]]
    assert metadata["cleanup_diagnostics"]["component_decisions"]
    assert metadata["semantic_config_fingerprint"]
    assert metadata["cleanup_config_fingerprint"]
    assert metadata["reference_profile_fingerprint"]
    assert metadata["cache_version"] == 7
    assert metadata["reference_profile_version"] == "2"
    semantic = metadata["semantic_configuration"]
    assert semantic["rasterregions_version"] == metadata["rasterregions_version"]
    assert "cleanup" in semantic["rasterregions_config"]
    assert "spacing" in semantic["rasterregions_config"]
    assert semantic["reference_profile_settings"]["name"] == "auto"
    assert semantic["reference_segmentation_version"] == "7"
    assert semantic["segmentation_policy_version"] == "3"
    assert semantic["reference_profile_version"] == "2"
    metadata["cache_version"] = 6
    metadata["segmentation_algorithm_version"] = "6"
    metadata["segmentation_policy_version"] = "2"
    metadata["cleanup_policy_version"] = "2"
    metadata["reference_profile_version"] = "1"
    metadata["semantic_configuration"]["reference_segmentation_version"] = "6"
    metadata["semantic_configuration"]["segmentation_policy_version"] = "2"
    metadata["semantic_configuration"]["reference_profile_version"] = "1"
    metadata_path.write_text(json.dumps(metadata))

    old_cache = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
    )
    assert old_cache.payload["reference_cache"]["hit"] is False
    rebuilt_metadata = json.loads(metadata_path.read_text())
    assert rebuilt_metadata["segmentation_algorithm_version"] == "7"
    assert rebuilt_metadata["segmentation_policy_version"] == "3"
    assert [item["id"] for item in rebuilt_metadata["candidates"]] == candidate_ids

    reused_cache = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
    )
    assert reused_cache.payload["reference_cache"]["hit"] is True

    from inkgrammar.comparison.segmentation import ReferenceSheetProfile

    profile = ReferenceSheetProfile(
        name="generic",
        allowed_row_count_patterns=((3,),),
    )
    generic = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
        reference_profile=profile,
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
        refinement_limit=1,
        reference_profile=profile,
    )
    assert corrupted_mask.payload["reference_cache"]["hit"] is False
    assert first.payload["reference_cache"]["hit"] is False


def test_progress_and_run_files_cover_success_failure_and_quiet_mode(tmp_path, capsys):
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "progress-run"

    compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
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
    assert all("route_status" in event["details"] for event in extraction_events)
    assert all("route_candidate_count" in event["details"] for event in extraction_events)
    assert all("route_search_truncated" in event["details"] for event in extraction_events)
    assert all("junction_count" in event["details"] for event in extraction_events)
    assert all("max_junction_degree" in event["details"] for event in extraction_events)
    assert all("stroke_hypothesis_count" in event["details"] for event in extraction_events)
    segmentation_events = [event for event in events if event["stage"] == "reference_segmentation"]
    assert segmentation_events
    assert segmentation_events[0]["details"]["reference_profile"] == "generic"
    assert segmentation_events[0]["details"]["candidate_count"] == 3
    assert isinstance(segmentation_events[0]["details"]["search_truncated"], bool)
    details = segmentation_events[0]["details"]
    assert details["message"].startswith("Reference segmentation accepted")
    assert "profile: generic" in details["message"]
    assert "foreground ownership: valid" in details["message"]
    assert "search truncated: no" in details["message"]
    assert details["foreground_ownership_valid"] is True
    assert isinstance(details["removed_page_artifact_count"], int)
    assert isinstance(details["uncertain_secondary_count"], int)
    assert isinstance(details["ambiguous_component_ids"], list)
    assert isinstance(details["uncertain_secondary_component_ids"], list)
    assert isinstance(details["unassigned_secondary_component_ids"], list)
    assert isinstance(details["uncertain_secondary_components"], list)
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
        refinement_limit=1,
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


def test_reference_segmentation_ambiguity_is_separate_from_match_ambiguity(tmp_path, monkeypatch):
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

    report = compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
        refinement_limit=1,
    )

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
        assert item["accepted_reference_id"] is None
        assert item["identity_status"] == "candidate_diagnostic"
        pair_payload = json.loads((output / item["report_json"]).read_text())
        assert pair_payload["identity"]["final_feedback_available"] is False
        assert pair_payload["writing_feedback"] is None
        assert match["status"] == match["match_status"]
        assert item["match_reason"]
        assert any(
            source["source"] == "segmentation_uncertainty" for source in item["ambiguity_sources"]
        )
        assert match["match_reason"] == item["match_reason"]
        assert match["ambiguity_sources"] == item["ambiguity_sources"]
    assert "Segmentation-ambiguous samples: 3" in report.markdown_path.read_text()
    assert "Segmentation status" in report.html_path.read_text()
    assert "Match reason" in report.markdown_path.read_text()
    assert "Ambiguity sources" in report.html_path.read_text()
    assert "segmentation_uncertainty" in report.markdown_path.read_text()


def test_batch_reason_and_ambiguity_sources_explain_each_uncertainty(tmp_path) -> None:
    from inkgrammar.comparison.batch_report import ambiguity_sources, match_reason
    from inkgrammar.comparison.matching import MatchScore

    best = MatchScore(
        reference_id="ref-best",
        rank=1,
        coarse_cost=0.12,
        raster_cost=0.1,
        aspect_cost=0.2,
        fill_cost=0.3,
        topology_cost=0.2,
        refined_cost=0.12,
    )
    next_best = MatchScore(
        reference_id="ref-next",
        rank=2,
        coarse_cost=0.13,
        raster_cost=0.2,
        aspect_cost=0.2,
        fill_cost=0.3,
        topology_cost=0.4,
        refined_cost=0.13,
    )
    reason = match_reason(best, (next_best,))
    assert "ref-next" in reason
    assert "raster and topology" in reason

    pair_report = tmp_path / "report.json"
    pair_report.write_text(
        json.dumps(
            {
                "writing": {"route_ambiguous": True},
                "diagnostics": {
                    "selected_route_pair": {"reference_rank": 1, "sample_rank": 2},
                    "candidate_routes": {
                        "reference": [{"rank": 1, "ambiguous": True}],
                        "sample": [{"rank": 2, "ambiguous": False}],
                    },
                },
            }
        )
    )
    sources = ambiguity_sources(
        best,
        (next_best,),
        "ambiguous",
        "ambiguous",
        ("uncertain crop boundary",),
        pair_report,
    )
    assert {item["source"] for item in sources} == {
        "match_cost_tie",
        "segmentation_uncertainty",
        "writing_route_hypothesis",
    }
    assert "within the configured ambiguity margin" in sources[0]["reason"]


def test_reference_failure_copies_diagnostics_and_records_phase(tmp_path, monkeypatch):
    from inkgrammar.comparison.segmentation import SegmentationError

    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "failed-reference"
    dependency_error = ValueError("synthetic rasterregions failure")

    def fail_segmentation(*args, **kwargs):
        raise dependency_error

    monkeypatch.setattr(
        "inkgrammar.comparison.segmentation.rr.detect_spaced_region_hypotheses",
        fail_segmentation,
    )
    with pytest.raises(SegmentationError, match="Rasterregions failed") as error:
        compare_letter_page(reference, sample, expected_letters=3, output_dir=output)

    diagnostic = output / "segmentation" / "reference-segmentation-hypotheses.json"
    assert error.value.diagnostic_path == diagnostic
    assert f"Diagnostics: {diagnostic}" in str(error.value)
    assert diagnostic.is_file()
    assert (output / "segmentation" / "reference-hypotheses" / "failure.json").is_file()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["failure_details"]["failure_phase"] == "rasterregions_detection"
    assert run["artifact_paths"]["diagnostic"] == str(diagnostic)
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    reference_events = [event for event in events if event["stage"] == "reference_mask_rows"]
    assert reference_events[-1]["details"]["failure_phase"] == "rasterregions_detection"
    assert events[-1]["stage"] == "failure"
    assert events[-1]["details"]["diagnostic_path"] == str(diagnostic)


def test_batch_runs_one_glyph_end_to_end_and_writes_match_reports(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    sample = tmp_path / "single-sample.png"
    image = Image.new("RGB", (250, 88), "white")
    _draw_glyph(ImageDraw.Draw(image), 95, 30, "e")
    image.save(sample)
    output = tmp_path / "one-glyph-run"

    report = compare_letter_page(reference, sample, expected_letters=1, output_dir=output)

    assert report.payload["sample"]["detected_count"] == 1
    assert report.payload["reference"]["segmentation"]["candidate_count"] == 3
    assert len(report.payload["matches"]) == 1
    match = report.payload["matches"][0]
    assert match["leading_reference_id"] in {"ref-000", "ref-001", "ref-002"}
    assert match["accepted_reference_id"] is None or match["accepted_reference_id"] in {
        "ref-000",
        "ref-001",
    }
    if match["leading_reference_id"] == "ref-002":
        assert match["status"] == "ambiguous"
        assert match["accepted_reference_id"] is None
    assert (output / match["match_json"]).is_file()
    assert (output / match["report_json"]).is_file()
    assert (output / match["report_dir"] / "report.html").is_file()


def test_identity_acceptance_requires_complete_route_evidence_and_a_certain_pair():
    from dataclasses import replace
    from types import SimpleNamespace

    from inkgrammar.comparison.batch import _identity_acceptance_reasons
    from inkgrammar.comparison.matching import MatchScore

    score = MatchScore(
        reference_id="ref-e",
        rank=1,
        coarse_cost=0.1,
        raster_cost=0.1,
        aspect_cost=0.0,
        fill_cost=0.0,
        topology_cost=0.0,
    )
    search = {"search_truncated": False}
    missing = _identity_acceptance_reasons(
        score,
        progressive_ambiguous=False,
        segmentation_status="ok",
        search_metadata=search,
    )
    assert "missing_essential_route_evidence" in missing
    assert "route_pair_unavailable" in missing

    route_ambiguous_score = replace(
        score,
        centerline_cost=0.0,
        fitted_steering_cost=0.0,
        route_structure_cost=0.0,
        route_descriptor_cost=0.0,
        route_topology_cost=0.0,
        route_pair_selection=SimpleNamespace(ambiguous=True),
    )
    clear_score = replace(
        route_ambiguous_score,
        route_pair_selection=SimpleNamespace(ambiguous=False),
    )
    assert not _identity_acceptance_reasons(
        clear_score,
        progressive_ambiguous=False,
        segmentation_status="ok",
        search_metadata=search,
    )
    route_ambiguous = _identity_acceptance_reasons(
        route_ambiguous_score,
        progressive_ambiguous=False,
        segmentation_status="ok",
        search_metadata=search,
    )
    assert "route_pair_ambiguous" in route_ambiguous
    assert "missing_essential_route_evidence" not in route_ambiguous
    assert "segmentation_uncertainty" in _identity_acceptance_reasons(
        route_ambiguous_score,
        progressive_ambiguous=False,
        segmentation_status="ambiguous",
        search_metadata=search,
    )


def test_refinement_cap_with_a_viable_candidate_marks_batch_result_ambiguous(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    sample = tmp_path / "sample.png"
    image = Image.new("RGB", (250, 88), "white")
    _draw_glyph(ImageDraw.Draw(image), 95, 30, "e")
    image.save(sample)

    report = compare_letter_page(
        reference,
        sample,
        expected_letters=1,
        output_dir=tmp_path / "capped-run",
        refinement_limit=1,
        no_progress=True,
    )
    match = report.payload["matches"][0]
    assert match["leading_reference_id"] == match["reference_id"]
    assert match["accepted_reference_id"] is None
    assert match["identity_status"] == "candidate_diagnostic"
    assert "reference_search_truncated" in match["acceptance_reasons"]
    pair_payload = json.loads((tmp_path / "capped-run" / match["report_json"]).read_text())
    assert pair_payload["identity"]["status"] == "candidate_diagnostic"
    assert pair_payload["identity"]["accepted_reference_id"] is None
    assert pair_payload["identity"]["final_feedback_available"] is False
    assert pair_payload["writing_feedback"] is None
    pair_html = (tmp_path / "capped-run" / match["report_dir"] / "report.html").read_text()
    assert "Candidate diagnostic" in pair_html
    assert "not final handwriting feedback" in pair_html
    assert "Leading candidate (not accepted)" in report.markdown_path.read_text()

    assert match["status"] == "ambiguous"
    assert match["search"]["stop_reason"] == "candidate_refinement_limit"
    assert match["search"]["search_truncated"] is True
    assert any(
        source["source"] == "reference_search_truncated" for source in match["ambiguity_sources"]
    )


def test_rejected_reference_segmentation_stops_before_downstream_work(tmp_path, monkeypatch):
    import inkgrammar.comparison.batch as comparison_batch
    from inkgrammar.comparison.segmentation import SegmentationError

    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    output = tmp_path / "rejected-reference"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("rejected reference segmentation reached downstream work")

    for name in (
        "build_reference_index",
        "segment_sample_glyphs",
        "extract_segmented_glyph",
        "rank_reference_candidates",
        "refine_matches_progressively",
        "fit_writing_route_candidates",
        "analyze_skeleton",
    ):
        monkeypatch.setattr(comparison_batch, name, fail_if_called)

    with pytest.raises(SegmentationError, match="Reference segmentation rejected"):
        compare_letter_page(
            reference,
            sample,
            expected_letters=3,
            output_dir=output,
            reference_profile="alphabet",
            no_progress=True,
        )

    segmentation_dir = output / "segmentation"
    assert (segmentation_dir / "reference-candidates.json").is_file()
    assert (segmentation_dir / "reference-segmentation-hypotheses.json").is_file()
    assert (segmentation_dir / "reference-segmentation-decision.json").is_file()
    assert not (segmentation_dir / "sample-segmentation.json").exists()
    assert not (output / "reference-index" / "gallery.json").exists()
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    validation = next(event for event in events if event["stage"] == "reference_segmentation")
    assert validation["details"]["status"] == "rejected"
    assert validation["details"]["accepted"] is False
    assert "recognized alphabet layout" in " ".join(validation["details"]["reasons"])
    assert validation["details"]["message"].startswith("Reference segmentation rejected")
    assert isinstance(validation["details"]["ambiguous_component_ids"], list)
    assert isinstance(validation["details"]["uncertain_secondary_components"], list)


def test_reference_cache_round_trips_cleanup_evidence_and_rejects_old_policy(tmp_path):

    import shutil

    from inkgrammar.comparison.batch import _load_reference_cache, _save_reference_cache
    from inkgrammar.comparison.matching import build_reference_index
    from inkgrammar.comparison.segmentation import segment_reference_gallery

    reference = _page(tmp_path / "reference.png", ("e", "e", "l"), y=22)
    diagnostics_dir = tmp_path / "reference-diagnostics"
    gallery = segment_reference_gallery(
        reference,
        ink_mode="dark",
        output_dir=diagnostics_dir,
        reference_profile="generic",
    )
    assert gallery.decision is not None and gallery.decision.accepted
    index = build_reference_index(gallery)
    cache_dir = tmp_path / "reference-cache"
    shutil.copytree(diagnostics_dir, cache_dir / "segmentation")
    metadata = _save_reference_cache(cache_dir, gallery, index, "dark", "generic")

    restored, restored_index, loaded_metadata = _load_reference_cache(
        cache_dir,
        reference,
        gallery.source_sha256,
        "dark",
        "generic",
    )
    assert restored.decision == gallery.decision
    assert restored.cleanup_diagnostics == json.loads(json.dumps(gallery.cleanup_diagnostics))
    assert len(restored_index.descriptors) == len(index.descriptors)
    assert loaded_metadata["semantic_config_fingerprint"] == metadata["semantic_config_fingerprint"]
    assert loaded_metadata["cleanup_config_fingerprint"] == metadata["cleanup_config_fingerprint"]

    metadata["segmentation_policy_version"] = "1"
    (cache_dir / "gallery.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reference cache metadata"):
        _load_reference_cache(
            cache_dir,
            reference,
            gallery.source_sha256,
            "dark",
            "generic",
        )


def test_cached_route_search_fallback_is_reused_without_route_generation(tmp_path, monkeypatch):
    import numpy as np
    from PIL import ImageDraw
    from skeletonpaths import JunctionPairingLimitError

    from inkgrammar.comparison import api
    from inkgrammar.comparison.batch import (
        _load_reference_extraction,
        _save_reference_extraction,
    )
    from inkgrammar.comparison.models import CropRect
    from inkgrammar.comparison.segmentation import SegmentedGlyph
    from inkgrammar.comparison.skeleton import analyze_skeleton

    mask = np.zeros((24, 24), dtype=bool)
    mask[3:21, 11:14] = True
    image = Image.new("RGB", (24, 24), "white")
    ImageDraw.Draw(image).rectangle((11, 3, 13, 20), fill="black")
    candidate = SegmentedGlyph(
        glyph_id="ref-000",
        bbox=CropRect(x=0, y=0, width=24, height=24),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=image,
        mask=mask,
    )
    analysis = analyze_skeleton(mask).analysis
    assert analysis is not None
    error = JunctionPairingLimitError(
        "max_junction_pairings", 1000, 1001, "junction-pairing-generation"
    )

    def fail_with_limit(*args, **kwargs):
        raise error

    monkeypatch.setattr(api, "generate_writing_route_candidates", fail_with_limit)
    extracted = api.extract_segmented_glyph(
        candidate,
        source_path=tmp_path / "reference.png",
        allow_path_failure=True,
        analysis=analysis,
    )
    cache_dir = tmp_path / "cache"
    _save_reference_extraction(
        cache_dir, "source-sha", "auto", candidate.glyph_id, candidate, extracted
    )

    def fail_route_search(*args, **kwargs):
        raise AssertionError("cached fallback must not repeat route generation")

    monkeypatch.setattr(api, "generate_writing_route_candidates", fail_route_search)
    cached = _load_reference_extraction(
        cache_dir,
        tmp_path / "reference.png",
        "source-sha",
        candidate,
        "auto",
        prepared_analysis=analysis,
    )

    assert cached.route_candidates == ()
    assert cached.geometry_warning == extracted.geometry_warning
    assert "JunctionPairingLimitError" in cached.geometry_warning
    assert cached.route_evidence_status == "search-truncated-fallback"
    assert cached.route_search_truncated is True
    assert cached.route_search_failure["type"] == "JunctionPairingLimitError"


def test_pathological_reference_route_does_not_abort_comparison_batch(tmp_path, monkeypatch):
    from skeletonpaths import JunctionPairingLimitError

    import inkgrammar.comparison.batch as batch_module
    from inkgrammar.comparison import api

    reference = _page(tmp_path / "reference.png", ("e", "n", "l"), y=22)
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"), y=30)
    original_extract = batch_module.extract_segmented_glyph
    original_route_search = api.generate_writing_route_candidates

    def fail_route_search(*args, **kwargs):
        raise JunctionPairingLimitError(
            "max_junction_pairings", 1000, 1001, "junction-pairing-generation"
        )

    def degrade_one_reference(candidate, **kwargs):
        if candidate.glyph_id == "ref-000":
            monkeypatch.setattr(api, "generate_writing_route_candidates", fail_route_search)
            try:
                return original_extract(candidate, **kwargs)
            finally:
                monkeypatch.setattr(api, "generate_writing_route_candidates", original_route_search)
        return original_extract(candidate, **kwargs)

    monkeypatch.setattr(batch_module, "extract_segmented_glyph", degrade_one_reference)
    output = tmp_path / "contained-batch"
    report = batch_module.compare_letter_page(
        reference,
        sample,
        expected_letters=3,
        output_dir=output,
    )

    run = json.loads((output / "run.json").read_text())
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    failed_reference = next(
        event
        for event in progress
        if event["stage"] == "reference_extraction"
        and event["details"]["reference_id"] == "ref-000"
    )
    assert report.payload["sample"]["detected_count"] == 3
    assert run["status"] == "completed"
    assert failed_reference["details"]["route_status"] == "search-truncated-fallback"
    assert failed_reference["details"]["route_candidate_count"] == 0
    assert all(item["search"]["coarse_candidate_count"] == 3 for item in report.payload["matches"])
    degraded_match = report.payload["matches"][0]
    assert degraded_match["leading_reference_id"] == "ref-000"
    assert degraded_match["leading_match"]["route_evidence_status"] == "search-truncated-fallback"
    assert degraded_match["leading_match"]["route_search_truncated"] is True
    assert (
        degraded_match["leading_match"]["route_search_failure"]["stage"]
        == "junction-pairing-generation"
    )
    assert degraded_match["leading_match"]["centerline_cost"] is None
    assert degraded_match["leading_match"]["feature_coverage"]["available_features"] == [
        "raster",
        "aspect",
        "fill",
        "topology",
    ]
    pair_report = json.loads((output / degraded_match["report_json"]).read_text())
    assert any(
        "incomplete writing-route evidence" in warning for warning in pair_report["warnings"]
    )
    assert any(
        "raster and topology evidence remained available" in warning
        for warning in pair_report["warnings"]
    )
