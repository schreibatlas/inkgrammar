from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterregions as rr
from PIL import Image, ImageDraw

from inkgrammar.comparison.models import WritingStructurePolicy
from inkgrammar.comparison.segmentation import (
    ReferenceSheetProfile,
    SegmentationError,
    _adapt_candidates,
    detect_ink_rows,
    evaluate_reference_segmentation,
    segment_reference_gallery,
    segment_sample_glyphs,
)


def _draw_body(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    body_width: int,
    *,
    detached_mark: bool = False,
) -> None:
    draw.line(
        ((x, y + 45), (x + round(body_width * 0.45), y + 25), (x + body_width, y + 2)),
        fill="black",
        width=4,
    )
    if detached_mark:
        mark_x = x + body_width - 5
        draw.ellipse((mark_x - 2, y - 5, mark_x + 2, y - 1), fill="black")


def _reference_page(path):
    image = Image.new("RGB", (260, 170), "white")
    draw = ImageDraw.Draw(image)
    for x in (20, 75, 130, 185):
        _draw_body(draw, x, 20, 60, detached_mark=x == 20)
    for x in (20, 80, 145, 200):
        _draw_body(draw, x, 105, 35)
    image.save(path)
    return path


def _sample_page(path):
    image = Image.new("RGB", (260, 135), "white")
    draw = ImageDraw.Draw(image)
    for x in (20, 100, 180):
        _draw_body(draw, x, 35, 34)
    draw.line((4, 100, 255, 100), fill=(180, 180, 180), width=1)
    draw.point((250, 8), fill="black")
    image.save(path)
    return path


def test_reference_gallery_segments_rows_and_reports_detached_mark_ambiguity(tmp_path):
    path = _reference_page(tmp_path / "alphabet.png")
    output = tmp_path / "reference-diagnostics"

    gallery = segment_reference_gallery(path, output_dir=output)

    assert len(gallery.candidates) == 8
    assert [candidate.glyph_id for candidate in gallery.candidates] == [
        f"ref-{index:03d}" for index in range(8)
    ]
    assert [candidate.row_index for candidate in gallery.candidates] == [0] * 4 + [1] * 4
    assert [candidate.column_index for candidate in gallery.candidates] == [0, 1, 2, 3] * 2
    assert gallery.segmentation.status == "ambiguous"
    assert gallery.segmentation.unassigned_ink_fraction > 0.0
    assert len(detect_ink_rows(gallery.mask_result.mask)) == 2
    assert (output / "reference-mask.png").is_file()
    assert (output / "reference-candidates.png").is_file()
    assert (output / "reference-segmentation-hypotheses.json").is_file()
    for filename in (
        "reference-anchor-regions.png",
        "reference-selected-regions.png",
        "reference-cleanup-overlay.png",
        "reference-segmentation-decision.json",
        "reference-segmentation.md",
    ):
        assert (output / filename).is_file()
    manifest = json.loads((output / "reference-candidates.json").read_text())
    assert "component_decisions" in manifest["cleanup_diagnostics"]
    assert "secondary_mark_assignments" in manifest["cleanup_diagnostics"]
    assert manifest["candidate_count"] == 8
    assert manifest["candidates"][2]["id"] == "ref-002"


def test_sample_segmentation_recovers_exact_count_and_writes_review_artifacts(tmp_path):
    path = _sample_page(tmp_path / "sample.png")
    output = tmp_path / "sample-diagnostics"

    samples = segment_sample_glyphs(path, expected_count=3, ink_mode="dark", output_dir=output)

    assert [candidate.glyph_id for candidate in samples] == [
        "sample-000",
        "sample-001",
        "sample-002",
    ]
    assert samples[1].bbox.y == 36
    assert all(candidate.segmentation_confidence >= 0.3 for candidate in samples)
    for filename in (
        "sample-mask.png",
        "sample-components.png",
        "sample-boundaries.png",
        "sample-candidates.png",
        "sample-hypotheses.json",
        "sample-segmentation.json",
    ):
        assert (output / filename).is_file()
    manifest = json.loads((output / "sample-segmentation.json").read_text())
    assert manifest["expected_count"] == manifest["detected_count"] == 3
    assert manifest["status"] == "ok"
    assert any("rule" in warning for warning in manifest["warnings"])
    assert manifest["hypothesis_count"] >= 1
    assert isinstance(manifest["search_truncated"], bool)
    assert (output / "sample-hypotheses" / "hypotheses.json").is_file()
    with Image.open(output / "sample-mask.png") as mask_image:
        assert mask_image.crop((0, 100, 260, 101)).getbbox() is None


def test_wrong_sample_count_fails_with_segmentation_diagnostics(tmp_path):
    path = _sample_page(tmp_path / "sample.png")
    output = tmp_path / "failed-segmentation"

    with pytest.raises(SegmentationError, match="Unable to recover 4 credible") as error:
        segment_sample_glyphs(path, expected_count=4, ink_mode="dark", output_dir=output)

    assert error.value.diagnostic_path == output / "sample-boundaries.png"
    assert (output / "sample-mask.png").is_file()
    assert (output / "sample-components.png").is_file()
    assert (output / "sample-boundaries.png").is_file()
    manifest = json.loads((output / "sample-segmentation.json").read_text())
    assert manifest["expected_count"] == 4
    assert manifest["status"] == "failed"


def test_page_mask_adapter_is_rasterregions_foreground_mask():
    from rasterregions import MaskResult, foreground_mask

    from inkgrammar.comparison.segmentation import extract_page_ink_mask

    image = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(image).rectangle((20, 15, 39, 44), fill="black")

    result = extract_page_ink_mask(image, mode="dark")

    assert extract_page_ink_mask is foreground_mask
    assert isinstance(result, MaskResult)
    assert result.method == "dark"
    assert result.mask[25, 25]
    assert isinstance(result.warnings, tuple)


def _alphabet_like_page(path):
    image = Image.new("RGB", (820, 530), "white")
    draw = ImageDraw.Draw(image)
    draw.line(
        [(20, 30), (55, 15), (90, 32), (125, 15), (160, 32), (195, 15), (230, 30)],
        fill="black",
        width=5,
    )
    for x in (300, 330, 360, 390):
        draw.line((x, 65, x + 12, 65, x + 4, 82), fill="black", width=3)

    row_counts = (12, 8, 6, 11, 10, 5, 10)
    for row_index, count in enumerate(row_counts):
        y = 120 + row_index * 55
        for column in range(count):
            x = 20 + column * 58
            width = 14 + column % 3 * 3
            draw.rectangle((x, y, x + width, y + 25), fill="black")
    image.save(path)
    return path


def _anchor_first_reference_page(path):
    row_counts = (12, 8, 6, 11, 11, 5, 10)
    image = Image.new("RGB", (1081, 1198), "white")
    draw = ImageDraw.Draw(image)
    for row_index, count in enumerate(row_counts):
        y = 35 + row_index * 155
        spacing = (image.width - 150) / max(1, count - 1)
        for index in range(count):
            x = 30 + index * spacing
            body_width = 86 + (index * 7) % 12
            draw.line(
                (
                    (round(x), y + 55),
                    (round(x + body_width * 0.45), y + 30),
                    (round(x + body_width), y + 5),
                ),
                fill="black",
                width=4,
            )
            if index % 8 == 0:
                dot_x = round(x + body_width - 20)
                draw.ellipse((dot_x - 1, y - 3, dot_x + 1, y - 1), fill="black")
    image.save(path)
    return path


def _reference_page_with_two_detached_dots(path):
    image = Image.new("RGB", (260, 140), "white")
    draw = ImageDraw.Draw(image)
    for y in (30, 90):
        for column, x in enumerate((20, 90, 160)):
            draw.rectangle((x, y, x + 6, y + 30), fill="black")
            if column == 1:
                draw.ellipse((x, y - 12, x + 6, y - 6), fill="black")
    image.save(path)
    return path


def test_reference_gallery_assigns_meaningful_detached_dots(tmp_path):
    path = _reference_page_with_two_detached_dots(tmp_path / "detached-dots.png")

    gallery = segment_reference_gallery(path, ink_mode="dark", reference_profile="generic")

    assert gallery.decision is not None and gallery.decision.accepted
    assert gallery.decision.retained_secondary_count == 2
    assert gallery.decision.uncertain_secondary_count == 0
    assert gallery.decision.unassigned_secondary_count == 0
    assert [
        candidate.glyph_id
        for candidate in gallery.candidates
        if candidate.secondary_component_indices
    ] == [
        "ref-001",
        "ref-004",
    ]
    assignments = gallery.cleanup_diagnostics["secondary_mark_assignments"]
    assert len(assignments) == 2
    assert all(item["status"] == "assigned" and item["mark_kind"] == "dot" for item in assignments)


def test_generated_anchor_first_reference_has_63_disjoint_candidates(tmp_path):
    row_counts = [12, 8, 6, 11, 11, 5, 10]
    path = _anchor_first_reference_page(tmp_path / "anchor-reference.png")
    output = tmp_path / "anchor-reference-diagnostics"

    gallery = segment_reference_gallery(
        path,
        ink_mode="dark",
        output_dir=output,
        reference_profile="auto",
    )

    assert len(gallery.candidates) == 63
    assert [row.candidate_count for row in gallery.segmentation.rows] == row_counts
    assert gallery.decision is not None
    assert gallery.segmentation.reference_profile == "alphabet"
    assert gallery.decision.accepted
    assert gallery.decision.accepted_count == 63
    assert gallery.decision.anchor_row_counts == tuple(row_counts)
    assert gallery.decision.selected_row_counts == tuple(row_counts)
    assert not gallery.decision.search_truncated
    assert gallery.decision.retained_secondary_count == 0
    assert all(row.included for row in gallery.segmentation.rows)
    assert gallery.segmentation.search_truncated is False
    assert gallery.segmentation.search_limit is None
    assert gallery.segmentation.unassigned_ink_fraction == 0.0
    assert any(
        left.bbox.x + left.bbox.width > right.bbox.x
        for left, right in zip(gallery.candidates[:12], gallery.candidates[1:12], strict=False)
    )

    page_masks = []
    for candidate in gallery.candidates:
        page_mask = np.zeros(gallery.mask_result.mask.shape, dtype=bool)
        x0, y0 = candidate.bbox.x, candidate.bbox.y
        page_mask[y0 : y0 + candidate.bbox.height, x0 : x0 + candidate.bbox.width] = candidate.mask
        page_masks.append(page_mask)
    assert all(
        not np.any(page_masks[left] & page_masks[right])
        for left in range(len(page_masks))
        for right in range(left + 1, len(page_masks))
    )

    manifest = json.loads((output / "reference-candidates.json").read_text())
    assert manifest["candidate_count"] == 63
    assert manifest["reference_profile"] == "alphabet"
    assert manifest["per_row_candidate_count"] == {
        str(index): count for index, count in enumerate(row_counts)
    }

    decision_payload = json.loads((output / "reference-segmentation-decision.json").read_text())
    assert decision_payload["accepted"] is True
    assert decision_payload["accepted_count"] == 63
    assert isinstance(decision_payload["ambiguous_component_ids"], list)
    assert isinstance(decision_payload["uncertain_secondary_component_ids"], list)
    assert isinstance(decision_payload["unassigned_secondary_component_ids"], list)
    assert isinstance(decision_payload["uncertain_secondary_components"], list)


def test_reference_hypothesis_invariant_falls_back_to_detected_anchor_counts(tmp_path, monkeypatch):
    row_counts = [12, 8, 6, 11, 11, 5, 10]
    path = _anchor_first_reference_page(tmp_path / "anchor-reference.png")
    output = tmp_path / "anchor-fallback-diagnostics"

    def fail_hypothesis_conversion(*args, **kwargs):
        raise ValueError("row hypothesis count and boundaries must match its regions")

    monkeypatch.setattr(
        "inkgrammar.comparison.segmentation.rr.detect_spaced_region_hypotheses",
        fail_hypothesis_conversion,
    )
    gallery = segment_reference_gallery(
        path,
        ink_mode="dark",
        output_dir=output,
        reference_profile="alphabet",
    )

    assert len(gallery.candidates) == 63
    assert [row.candidate_count for row in gallery.segmentation.rows] == row_counts
    assert gallery.segmentation.anchor_region_count == 63
    assert gallery.segmentation.status == "ambiguous"
    assert gallery.segmentation.hypothesis_count == 1
    assert any("row-boundary invariant" in warning for warning in gallery.segmentation.warnings)
    hypotheses = json.loads((output / "reference-segmentation-hypotheses.json").read_text())
    assert hypotheses["page_hypotheses"] == []
    assert len(hypotheses["selected"]["regions"]) == 63


def test_reference_ownership_validation_rejects_overlapping_regions(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from inkgrammar.comparison.segmentation import _reference_ownership_error

    path = tmp_path / "reference.png"
    image = Image.new("RGB", (80, 50), "white")
    ImageDraw.Draw(image).rectangle((20, 12, 50, 38), fill="black")
    image.save(path)
    result = rr.detect_spaced_regions(path, mode="dark")
    region = result.regions[0]
    duplicate = replace(region, reading_order=region.reading_order + 1)
    overlapping = SimpleNamespace(
        status=result.status,
        clean_mask=result.clean_mask,
        rows=result.rows,
        regions=(region, duplicate),
        unassigned_mask=result.unassigned_mask,
        diagnostics=result.diagnostics,
    )

    assert _reference_ownership_error(overlapping) == (
        "Reference regions have overlapping foreground ownership."
    )


def test_alphabet_profile_excludes_title_and_year_and_isolates_all_glyph_rows(tmp_path):
    path = _alphabet_like_page(tmp_path / "synthetic-alphabet.png")
    output = tmp_path / "reference-diagnostics"

    gallery = segment_reference_gallery(
        path,
        ink_mode="dark",
        output_dir=output,
        reference_profile="alphabet",
    )

    assert [row.candidate_count for row in gallery.segmentation.rows] == [
        1,
        4,
        12,
        8,
        6,
        11,
        10,
        5,
        10,
    ]
    assert [row.candidate_count for row in gallery.segmentation.rows if row.included] == [
        12,
        8,
        6,
        11,
        10,
        5,
        10,
    ]
    assert [row.classification for row in gallery.segmentation.rows[:2]] == ["title", "metadata"]
    assert len(gallery.candidates) == 62
    assert [candidate.glyph_id for candidate in gallery.candidates] == [
        f"ref-{index:03d}" for index in range(62)
    ]
    assert all(candidate.label is None for candidate in gallery.candidates)
    assert all(candidate.component_indices for candidate in gallery.candidates)

    source = np.asarray(Image.open(path).convert("RGB"))
    page_masks = []
    for candidate in gallery.candidates:
        x0, y0 = candidate.bbox.x, candidate.bbox.y
        crop = source[y0 : y0 + candidate.bbox.height, x0 : x0 + candidate.bbox.width]
        candidate_image = np.asarray(candidate.image.convert("RGB"))
        assert np.all(candidate_image[~candidate.mask] == 255)
        assert np.array_equal(candidate_image[candidate.mask], crop[candidate.mask])
        full_mask = np.zeros(source.shape[:2], dtype=bool)
        full_mask[y0 : y0 + candidate.bbox.height, x0 : x0 + candidate.bbox.width] = candidate.mask
        page_masks.append(full_mask)
    assert all(
        not np.any(page_masks[left] & page_masks[right])
        for left in range(len(page_masks))
        for right in range(left + 1, len(page_masks))
    )

    manifest = json.loads((output / "reference-candidates.json").read_text())
    assert manifest["reference_profile"] == "alphabet"
    assert manifest["row_count"] == 9
    assert [row["row"] for row in manifest["excluded_rows"]] == [0, 1]
    assert manifest["candidate_count"] == 62

    from inkgrammar.comparison.segmentation import ReferenceSheetProfile

    generic = segment_reference_gallery(path, ink_mode="dark", reference_profile="generic")
    assert len(generic.candidates) == 67
    assert all(row.included for row in generic.segmentation.rows)

    overridden = segment_reference_gallery(
        path,
        ink_mode="dark",
        reference_profile=ReferenceSheetProfile(name="alphabet", include_rows=(2, 4)),
    )
    assert [candidate.row_index for candidate in overridden.candidates] == [2] * 12 + [4] * 6
    for filename in (
        "reference-mask.png",
        "reference-rows.png",
        "reference-components.png",
        "reference-candidates.png",
        "reference-gallery.png",
        "ref-000.png",
        "ref-000-mask.png",
    ):
        assert (output / filename).is_file()


def test_auto_reference_profile_selects_stable_alphabet_layout(tmp_path):
    path = _alphabet_like_page(tmp_path / "synthetic-alphabet.png")

    gallery = segment_reference_gallery(path, ink_mode="dark", reference_profile="auto")

    assert gallery.segmentation.reference_profile == "alphabet"
    assert tuple(row.candidate_count for row in gallery.segmentation.rows if row.included) == (
        12,
        8,
        6,
        11,
        10,
        5,
        10,
    )
    assert gallery.decision is not None
    assert gallery.decision.profile_name == "alphabet"


def test_auto_reference_profile_keeps_unrelated_layout_generic(tmp_path):
    path = _reference_page(tmp_path / "unrelated-reference.png")

    gallery = segment_reference_gallery(path, ink_mode="dark", reference_profile="auto")

    assert gallery.segmentation.reference_profile == "generic"
    assert gallery.decision is not None
    assert gallery.decision.profile_name == "generic"


def test_explicit_alphabet_profile_rejects_unrecognized_layout(tmp_path):
    path = _reference_page(tmp_path / "unrelated-reference.png")

    gallery = segment_reference_gallery(path, ink_mode="dark", reference_profile="alphabet")

    assert gallery.decision is not None
    assert not gallery.decision.accepted
    assert "recognized alphabet layout has no compatible hypothesis" in gallery.decision.reasons


def test_isolated_candidate_image_whitens_other_ink_in_overlapping_boxes():
    from inkgrammar.comparison.segmentation import _isolated_candidate_image

    source = Image.new("RGB", (12, 8), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((4, 3, 5, 4), fill="black")
    draw.rectangle((6, 3, 7, 4), fill="black")
    left_crop = source.crop((0, 0, 8, 8))
    right_crop = source.crop((4, 0, 12, 8))
    left_mask = np.zeros((8, 8), dtype=bool)
    right_mask = np.zeros((8, 8), dtype=bool)
    left_mask[3:5, 4:6] = True
    right_mask[3:5, 2:4] = True

    left = np.asarray(_isolated_candidate_image(left_crop, left_mask))
    right = np.asarray(_isolated_candidate_image(right_crop, right_mask))

    assert left_crop.size == right_crop.size
    assert not np.any(np.min(left, axis=2)[~left_mask] < 128)
    assert not np.any(np.min(right, axis=2)[~right_mask] < 128)


@pytest.mark.parametrize(
    ("failure_type", "message", "search_truncated", "search_limit"),
    [
        (ValueError, "row hypothesis invariant failed", False, None),
        (
            rr.SegmentationSearchLimitError,
            "partition search budget exhausted before a feasible result",
            True,
            "row-partition-budget",
        ),
    ],
)
def test_reference_dependency_failures_write_diagnostics(
    tmp_path, monkeypatch, failure_type, message, search_truncated, search_limit
):
    path = tmp_path / "reference.png"
    image = Image.new("RGB", (80, 50), "white")
    ImageDraw.Draw(image).rectangle((20, 12, 50, 38), fill="black")
    image.save(path)
    output = tmp_path / "reference-failure"
    failure = failure_type(message)

    def fail_segmentation(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        "inkgrammar.comparison.segmentation.rr.detect_spaced_region_hypotheses",
        fail_segmentation,
    )
    with pytest.raises(SegmentationError, match="Reference segmentation") as error:
        segment_reference_gallery(path, ink_mode="dark", output_dir=output)

    diagnostic_path = output / "reference-segmentation-hypotheses.json"
    assert error.value.__cause__ is failure
    assert error.value.diagnostic_path == diagnostic_path
    assert "Diagnostics:" in str(error.value)
    assert diagnostic_path.is_file()
    for filename in (
        "reference-mask.png",
        "reference-rows.png",
        "reference-components.png",
        "reference-candidates.png",
        "reference-candidates.json",
        "reference-gallery.png",
    ):
        assert (output / filename).is_file()
    assert (output / "reference-hypotheses" / "failure.json").is_file()

    manifest = json.loads((output / "reference-candidates.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["row_count"] == 1
    assert manifest["search_truncated"] is search_truncated
    assert manifest["search_limit"] == search_limit
    hypotheses = json.loads(diagnostic_path.read_text())
    assert hypotheses["error"]["type"] == failure_type.__name__


def test_segmented_glyph_preserves_rasterregions_component_roles_and_mark_centroids():
    image = Image.new("RGB", (16, 16), "white")
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:11, 2:7] = True
    mask[2:4, 9:11] = True
    mask[13:15, 13:15] = True
    components = tuple(rr.components(mask))
    roles = {
        "primary": components[1].index,
        "secondary": components[0].index,
        "ambiguous": components[2].index,
    }
    region = rr.Region(
        region_id="region-0",
        bbox=rr.BBox(0, 0, 16, 16),
        row_index=0,
        column_index=0,
        reading_order=0,
        confidence=1.0,
        component_indices=tuple(component.index for component in components),
        image=image,
        mask=mask,
        primary_component_indices=(roles["primary"],),
        secondary_component_indices=(roles["secondary"],),
        ambiguous_component_indices=(roles["ambiguous"],),
    )
    result = rr.DetectionResult(
        source_path=None,
        image=image,
        mask_result=rr.MaskResult(mask, 1.0, "grayscale"),
        clean_mask=mask,
        components=components,
        rows=(),
        regions=(region,),
    )

    candidate = _adapt_candidates(result, "sample")[0]

    assert candidate.primary_component_indices == (roles["primary"],)
    assert candidate.secondary_component_indices == (roles["secondary"],)
    assert candidate.ambiguous_component_indices == (roles["ambiguous"],)
    assert candidate.expected_stroke_count() == 2
    assert candidate.expected_stroke_count(WritingStructurePolicy(preferred_body_strokes=2)) == 3
    assert candidate.component_role_sequence == ("secondary", "primary", "ambiguous")
    assert len(candidate.secondary_mark_centroids) == 1


def _reference_decision_fixture(
    monkeypatch,
    *,
    anchor_counts,
    selected_counts,
    profile=None,
    assignments=(),
    component_decisions=(),
):

    monkeypatch.setattr(
        "inkgrammar.comparison.segmentation._reference_ownership_error",
        lambda result: None,
    )
    row_diagnostics = tuple(
        SimpleNamespace(
            row_index=index,
            anchor_count=anchor_count,
            repair_count=max(0, selected_count - anchor_count),
            split_anchor_indices=(),
            merged_anchor_groups=(),
            truncated_counts=(),
            split_evidence=(),
        )
        for index, (anchor_count, selected_count) in enumerate(
            zip(anchor_counts, selected_counts, strict=True)
        )
    )
    diagnostics = SimpleNamespace(
        raw_component_count=sum(selected_counts),
        row_segmentation=row_diagnostics,
        secondary_mark_assignments=tuple(assignments),
        search_truncated=False,
    )
    result = SimpleNamespace(
        diagnostics=diagnostics,
        regions=(),
        components=(),
        component_decisions=tuple(component_decisions),
    )
    segmentation_rows = tuple(
        SimpleNamespace(
            row_index=index,
            candidate_count=count,
            included=True,
            classification="glyph",
            warnings=(),
        )
        for index, count in enumerate(selected_counts)
    )
    candidates = tuple(
        SimpleNamespace(ambiguous_component_indices=(), warnings=())
        for _ in range(sum(selected_counts))
    )
    segmentation = SimpleNamespace(rows=segmentation_rows, candidates=candidates)
    hypotheses = SimpleNamespace(truncated=False, page_hypotheses=())
    return evaluate_reference_segmentation(
        result,
        segmentation,
        hypotheses,
        profile or ReferenceSheetProfile(name="alphabet"),
    )


def test_reference_decision_rejects_65_candidate_repairs_against_safe_63_anchor_layout(
    monkeypatch,
):
    decision = _reference_decision_fixture(
        monkeypatch,
        anchor_counts=(12, 8, 6, 11, 11, 5, 10),
        selected_counts=(13, 9, 6, 11, 11, 5, 10),
    )

    assert decision.accepted is False
    assert decision.accepted_count is None
    assert decision.recognized_layout == (12, 8, 6, 11, 11, 5, 10)
    assert decision.detected_count == 65
    assert "unresolved count-changing split/merge ambiguity" in decision.reasons
    assert any("conflict with safe recognized layout" in reason for reason in decision.reasons)


def test_reference_decision_accepts_generic_counts_and_nonblocking_removed_noise(
    monkeypatch,
):
    decision = _reference_decision_fixture(
        monkeypatch,
        anchor_counts=(3, 4),
        selected_counts=(3, 4),
        profile=ReferenceSheetProfile(name="generic"),
        assignments=(SimpleNamespace(status="assigned", component_index=99),),
        component_decisions=(SimpleNamespace(action="removed", role="micro-noise"),),
    )

    assert ReferenceSheetProfile(name="generic").allowed_row_count_patterns == ()
    assert ReferenceSheetProfile().name == "generic"
    assert ReferenceSheetProfile().allowed_row_count_patterns == ()
    assert decision.accepted
    assert decision.recognized_layout is None
    assert decision.removed_micro_noise_count == 1
    assert decision.retained_secondary_count == 1

    assert "profile: generic" in decision.summary
    assert "rows: [3, 4]" in decision.summary
    assert "removed micro-noise: 1" in decision.summary
    assert "detached marks retained: 1" in decision.summary
    assert "foreground ownership: valid" in decision.summary
    assert "search truncated: no" in decision.summary


def test_reference_decision_rejects_uncertain_bbox_changing_secondary_mark(tmp_path, monkeypatch):
    decision = _reference_decision_fixture(
        monkeypatch,
        anchor_counts=(12, 8, 6, 11, 11, 5, 10),
        selected_counts=(12, 8, 6, 11, 11, 5, 10),
        assignments=(
            SimpleNamespace(
                status="ambiguous",
                component_index=99,
                bbox_expansion_ratio=1.2,
                candidate_region_indices=(35, 36),
                best_score=0.31,
                second_best_score=0.33,
                margin=0.02,
                reason="best-target score margin is below the configured threshold",
            ),
        ),
        component_decisions=(
            SimpleNamespace(
                component_index=99,
                bbox=SimpleNamespace(x0=10, y0=20, x1=12, y1=22),
                action="kept",
                role="secondary",
            ),
        ),
    )

    assert decision.accepted is False
    assert decision.uncertain_secondary_count == 1
    assert "uncertain secondary ownership can change candidate bounds" in decision.reasons
    assert decision.ambiguous_component_ids == (99,)
    assert decision.uncertain_secondary_component_ids == (99,)
    assert decision.unassigned_secondary_component_ids == ()
    component = decision.uncertain_secondary_components[0]
    assert component["bbox"] == (10, 20, 12, 22)
    assert component["candidate_region_ids"] == ("region-035", "region-036")
    assert component["best_score"] == 0.31
    assert component["second_best_score"] == 0.33
    assert component["margin"] == 0.02
    assert component["bbox_expansion_ratio"] == 1.2
    assert "secondary component 99 is ambiguous between region-035 and region-036" in (
        decision.summary
    )
    assert "uncertain secondary component IDs: [99]" in decision.summary

    from inkgrammar.comparison.segmentation import _write_reference_report

    report_path = tmp_path / "reference-segmentation.md"
    _write_reference_report(
        SimpleNamespace(diagnostics=None),
        SimpleNamespace(warnings=()),
        decision,
        report_path,
    )
    report = report_path.read_text()
    assert "## Uncertain secondary components" in report
    assert "best score=0.31" in report
    assert "bbox expansion=1.2" in report
