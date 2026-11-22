from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from rasterregions import MaskResult

import inkgrammar.comparison.matching as matching_module
from inkgrammar.comparison.descriptors import canonicalize_mask, glyph_descriptor
from inkgrammar.comparison.matching import (
    AMBIGUITY_ABSOLUTE_MARGIN,
    ASPECT_WEIGHT,
    FILL_WEIGHT,
    RASTER_WEIGHT,
    TOPOLOGY_WEIGHT,
    MatchEvidence,
    MatchScore,
    build_reference_index,
    coarse_match_cost,
    match_status,
    rank_reference_candidates,
    refine_match,
    refine_matches_progressively,
    refine_top_k_matches,
)
from inkgrammar.comparison.models import CropRect
from inkgrammar.comparison.segmentation import PageSegmentation, ReferenceGallery, SegmentedGlyph
from inkgrammar.comparison.steering import fitted_steering_distance


def _shape(kind: str, *, size: int = 64, offset: tuple[int, int] = (0, 0)) -> SegmentedGlyph:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    x, y = offset
    if kind == "e":
        draw.line(
            [
                (x + 12, y + 31),
                (x + 33, y + 31),
                (x + 36, y + 24),
                (x + 32, y + 17),
                (x + 24, y + 15),
                (x + 16, y + 19),
                (x + 12, y + 28),
                (x + 14, y + 38),
                (x + 21, y + 43),
                (x + 32, y + 42),
            ],
            fill="black",
            width=4,
        )
    elif kind == "6":
        draw.line(
            [
                (x + 35, y + 13),
                (x + 26, y + 13),
                (x + 18, y + 18),
                (x + 13, y + 26),
                (x + 12, y + 34),
                (x + 15, y + 40),
                (x + 21, y + 44),
                (x + 29, y + 43),
                (x + 35, y + 38),
                (x + 36, y + 31),
                (x + 33, y + 25),
                (x + 27, y + 22),
                (x + 20, y + 24),
                (x + 15, y + 29),
            ],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 25, y + 12, x + 25, y + 45), fill="black", width=4)
    elif kind == "n":
        draw.line(
            [
                (x + 13, y + 45),
                (x + 13, y + 18),
                (x + 20, y + 15),
                (x + 31, y + 18),
                (x + 35, y + 25),
                (x + 35, y + 45),
            ],
            fill="black",
            width=4,
        )
    elif kind == "two-strokes":
        draw.line((x + 16, y + 15, x + 16, y + 45), fill="black", width=4)
        draw.line((x + 36, y + 15, x + 36, y + 45), fill="black", width=4)
    mask = np.asarray(image.convert("L")) < 128
    return SegmentedGlyph(
        glyph_id=kind,
        bbox=CropRect(x=0, y=0, width=size, height=size),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=image,
        mask=mask,
    )


def _gallery(*candidates: SegmentedGlyph) -> ReferenceGallery:
    page_mask = np.zeros((64, 64), dtype=bool)
    segmentation = PageSegmentation(Path("reference.png"), candidates)
    return ReferenceGallery(
        source_path=Path("reference.png"),
        source_sha256="a" * 64,
        candidates=tuple(candidates),
        mask_result=MaskResult(page_mask, 1.0, "grayscale", []),
        segmentation=segmentation,
    )


def _translated_scaled_e() -> SegmentedGlyph:
    source = _shape("e").image
    rotated = source.rotate(4, resample=Image.Resampling.NEAREST, expand=False)
    resized = rotated.resize((48, 48), resample=Image.Resampling.NEAREST)
    image = Image.new("RGB", (80, 80), "white")
    image.paste(resized, (20, 16))
    mask = np.asarray(image.convert("L")) < 128
    return SegmentedGlyph(
        glyph_id="sample",
        bbox=CropRect(x=0, y=0, width=80, height=80),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=1.0,
        image=image,
        mask=mask,
    )


def test_canonical_mask_preserves_aspect_ratio_and_centers_shape():
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 4:36] = True

    normalized = canonicalize_mask(mask, size=96, padding=8)
    ys, xs = np.nonzero(normalized)

    assert normalized.shape == (96, 96)
    assert abs((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) - 3.2) < 0.15
    assert abs(float(xs.mean()) - float(ys.mean())) < 2.0


def test_transformed_query_ranks_its_reference_first_without_unique_assignment():
    reference_e = _shape("e")
    gallery = _gallery(reference_e, _shape("n"), _shape("l"))
    index = build_reference_index(gallery)
    sample = _translated_scaled_e()

    ranked = rank_reference_candidates(sample, index, top_k=3)
    repeated = rank_reference_candidates(sample, index, top_k=3)

    assert ranked[0].reference_id == "e"
    assert [score.rank for score in ranked] == [1, 2, 3]
    assert repeated[0].reference_id == "e"
    assert RASTER_WEIGHT == 1.0
    assert ASPECT_WEIGHT == 0.15
    assert FILL_WEIGHT == 0.10
    assert TOPOLOGY_WEIGHT == 0.15


def test_continuous_refinement_keeps_the_known_candidate_ranked_first():
    gallery = _gallery(_shape("e"), _shape("n"), _shape("l"))
    sample = _shape("e")
    coarse = rank_reference_candidates(sample, gallery, top_k=3)

    refined = refine_top_k_matches(sample, gallery, coarse)

    assert coarse[0].reference_id == "e"
    assert refined[0].reference_id == "e"
    assert refined[0].centerline_cost is not None
    assert refined[0].heading_cost is not None
    assert refined[0].fitted_steering_cost is not None


def test_refinement_is_limited_to_the_coarse_top_k_candidates():
    gallery = _gallery(_shape("e"), _shape("n"), _shape("l"))
    coarse = rank_reference_candidates(_shape("e"), gallery, top_k=2)

    refined = refine_top_k_matches(_shape("e"), gallery, coarse)

    assert len(refined) == 2
    assert [score.rank for score in refined] == [1, 2]
    assert {score.reference_id for score in refined} <= {score.reference_id for score in coarse}


def test_ambiguity_uses_a_deterministic_best_second_margin():
    first = MatchScore("ref-a", 1, 0.1, 0.1, 0, 0, 0)
    second = MatchScore("ref-b", 2, 0.1 + AMBIGUITY_ABSOLUTE_MARGIN / 2, 0.1, 0, 0, 0)

    assert match_status([first, second]) == "ambiguous"
    assert match_status([first]) == "clear"


def test_refinement_uses_paths_when_available_and_falls_back_when_not():
    reference = _shape("l")
    descriptor = glyph_descriptor(reference)
    coarse = rank_reference_candidates(reference, _gallery(reference))[0]

    refined = refine_match(reference, reference, coarse)
    raster_only = refine_match(_shape("two-strokes"), _shape("l"), coarse)

    assert refined.centerline_cost is not None
    assert refined.heading_cost is not None
    assert refined.fitted_steering_cost is not None
    assert refined.refined_cost >= refined.coarse_cost
    assert refined.refined_cost == pytest.approx(
        refined.coarse_cost + refined.evidence.missing_evidence_penalty
    )
    assert descriptor.path_available is True
    assert descriptor.endpoint_count == 2
    assert descriptor.junction_count == 0
    assert descriptor.cycle_count == 0
    assert refined.evidence.missing_evidence_penalty > 0.0
    assert {
        "raster",
        "aspect",
        "fill",
        "topology",
        "route_centerline",
        "route_heading",
        "route_steering",
    }.issubset(refined.evidence.available_features)
    assert raster_only.refined_cost >= (
        raster_only.coarse_cost + raster_only.evidence.missing_evidence_penalty
    )
    assert raster_only.evidence.missing_evidence_penalty > 0.0
    assert {"route_centerline", "route_heading", "route_steering"}.issubset(
        raster_only.evidence.unavailable_features
    )
    assert raster_only.centerline_cost is None


def test_glyph_descriptor_reuses_prepared_skeleton_analysis(monkeypatch) -> None:
    from inkgrammar.comparison import descriptors as descriptor_module
    from inkgrammar.comparison.skeleton import analyze_skeleton

    candidate = _shape("l")
    analysis = analyze_skeleton(candidate.mask).analysis
    assert analysis is not None

    def unexpected_analysis(_mask):
        raise AssertionError("prepared skeleton analysis should be reused")

    monkeypatch.setattr(descriptor_module, "analyze_skeleton", unexpected_analysis)
    descriptor = glyph_descriptor(candidate, analysis=analysis)

    assert descriptor.analysis is analysis
    assert descriptor.endpoint_count == 2
    assert descriptor.junction_count == 0
    assert descriptor.cycle_count == 0
    assert descriptor.path_available is True


def test_semantic_topology_mismatch_changes_coarse_cost() -> None:
    descriptor = glyph_descriptor(_shape("l"))
    same = coarse_match_cost(descriptor, descriptor)
    different = coarse_match_cost(
        descriptor,
        replace(descriptor, cycle_count=descriptor.cycle_count + 1),
    )

    assert same.topology_cost == 0.0
    assert different.topology_cost > same.topology_cost
    assert different.coarse_cost > same.coarse_cost


def test_fitted_steering_distance_ignores_raw_curvature_spikes() -> None:
    from inkgrammar.comparison.api import extract_segmented_glyph

    extracted = extract_segmented_glyph(_shape("l"), source_path=Path("l.png"))
    path = extracted.continuous_path
    assert path is not None
    spiked_curvature = path.curvature.copy()
    spiked_curvature[::2] += 1_000_000.0
    spiked_path = replace(path, curvature=spiked_curvature)

    assert fitted_steering_distance(path, spiked_path) == 0.0


def test_refinement_keeps_topology_steering_and_route_structure_separate() -> None:
    from inkgrammar.comparison.api import extract_segmented_glyph

    sample = _shape("two-strokes")
    reference = _shape("l")
    sample_extracted = extract_segmented_glyph(sample, source_path=Path("sample.png"))
    reference_extracted = extract_segmented_glyph(reference, source_path=Path("reference.png"))
    coarse = rank_reference_candidates(sample, _gallery(reference))[0]

    refined = refine_match(
        sample,
        reference,
        coarse,
        sample_extracted=sample_extracted,
        reference_extracted=reference_extracted,
    )

    assert refined.topology_cost > 0.0
    assert refined.topology_cost > 0.0
    assert refined.fitted_steering_cost is not None
    assert refined.route_descriptor_cost is not None
    assert refined.route_topology_cost is not None
    assert refined.route_pair_selection is not None
    assert "route_steering" in refined.evidence.available_features
    assert "route_descriptor" in refined.evidence.available_features
    assert "route_topology" in refined.evidence.available_features
    assert refined.route_structure_cost is not None
    assert refined.route_structure_cost > 0.0
    assert not hasattr(refined, "turn_rate_cost")


def test_match_evidence_rejects_negative_feature_costs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MatchEvidence(raster_cost=-0.1, aspect_cost=0.0, fill_cost=0.0, topology_cost=0.0)


def test_progressive_refinement_reaches_competitive_candidates_beyond_rank_five(
    monkeypatch,
) -> None:
    references = tuple(replace(_shape("e"), glyph_id=f"ref-{index}") for index in range(7))
    gallery = _gallery(*references)
    coarse = [
        MatchScore(
            reference.glyph_id,
            rank=index + 1,
            coarse_cost=index / 100.0 if index < 6 else 0.5,
            raster_cost=0.0,
            aspect_cost=0.0,
            fill_cost=0.0,
            topology_cost=0.0,
        )
        for index, reference in enumerate(references)
    ]
    final_costs = {
        "ref-0": 0.40,
        "ref-1": 0.35,
        "ref-2": 0.30,
        "ref-3": 0.25,
        "ref-4": 0.20,
        "ref-5": 0.02,
        "ref-6": 0.50,
    }
    refined_ids: list[str] = []
    extracted_ids: list[str] = []

    def extract_reference(candidate):
        extracted_ids.append(candidate.glyph_id)
        return None

    def fake_refine(_sample, reference, coarse_score, **_kwargs):
        refined_ids.append(reference.glyph_id)
        return replace(coarse_score, refined_cost=final_costs[reference.glyph_id])

    monkeypatch.setattr(matching_module, "refine_match", fake_refine)
    result = refine_matches_progressively(
        _shape("e"),
        gallery,
        coarse,
        max_refinements=10,
        reference_extractor=extract_reference,
    )

    assert result.best.reference_id == "ref-5"
    assert result.refined_candidate_count == 6
    assert result.coarse_candidate_count == 7
    assert "ref-5" in refined_ids
    assert "ref-6" not in refined_ids
    assert extracted_ids == refined_ids
    assert result.stop_reason == "coarse_lower_bound"
    assert result.search_truncated is False

    refined_ids.clear()
    extracted_ids.clear()
    truncated = refine_matches_progressively(
        _shape("e"),
        gallery,
        coarse,
        max_refinements=1,
        reference_extractor=extract_reference,
    )
    assert truncated.refined_candidate_count == 1
    assert truncated.search_truncated is True
    assert truncated.ambiguous is True
    assert truncated.stop_reason == "candidate_refinement_limit"


def test_controlled_e_vs_6_never_accepts_the_6_route_as_a_clear_identity(
    tmp_path, monkeypatch
) -> None:
    import json

    import inkgrammar.comparison.api as comparison_api
    from inkgrammar.comparison.api import compare_extracted_letters, extract_segmented_glyph
    from inkgrammar.comparison.batch import _identity_acceptance_reasons
    from inkgrammar.comparison.models import ComparisonIdentity
    from inkgrammar.comparison.report import write_comparison_report

    sample = replace(_shape("e"), glyph_id="controlled-sample-e")
    e_reference = replace(_shape("e"), glyph_id="controlled-e")
    six_reference = replace(_shape("6"), glyph_id="controlled-6")
    gallery = _gallery(e_reference, six_reference)
    gallery_candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    coarse = rank_reference_candidates(sample, gallery, top_k=2)
    assert {score.reference_id for score in coarse} == {"controlled-e", "controlled-6"}

    original_route_generator = comparison_api.generate_writing_route_candidates

    def bounded_route_generator(diagnostics, **kwargs):
        kwargs["limit"] = 8
        return original_route_generator(diagnostics, **kwargs)

    monkeypatch.setattr(
        comparison_api,
        "generate_writing_route_candidates",
        bounded_route_generator,
    )

    sample_extracted = extract_segmented_glyph(sample, source_path=Path("sample-e.png"))
    reference_extractions = {
        candidate.glyph_id: extract_segmented_glyph(
            candidate,
            source_path=Path(f"{candidate.glyph_id}.png"),
        )
        for candidate in gallery.candidates
    }
    refined_by_id = {
        coarse_score.reference_id: refine_match(
            sample,
            gallery_candidates[coarse_score.reference_id],
            coarse_score,
            sample_extracted=sample_extracted,
            reference_extracted=reference_extractions[coarse_score.reference_id],
        )
        for coarse_score in coarse
    }
    e_score = refined_by_id["controlled-e"]
    six_score = refined_by_id["controlled-6"]
    assert e_score.route_descriptor_cost is not None
    assert six_score.route_descriptor_cost is not None
    assert six_score.route_descriptor_cost > e_score.route_descriptor_cost

    progressive = refine_matches_progressively(
        sample,
        gallery,
        coarse,
        sample_extracted=sample_extracted,
        reference_extractions=reference_extractions,
        max_refinements=2,
    )
    leader = progressive.best
    reasons = _identity_acceptance_reasons(
        leader,
        progressive_ambiguous=progressive.ambiguous,
        segmentation_status="ok",
        search_metadata={"search_truncated": progressive.search_truncated},
    )
    accepted_reference_id = leader.reference_id if not reasons else None
    assert accepted_reference_id != "controlled-6"
    assert six_score.cost > e_score.cost or progressive.ambiguous
    assert leader.route_pair_selection is not None

    result = compare_extracted_letters(
        reference_extractions[leader.reference_id],
        sample_extracted,
        identity=ComparisonIdentity("controlled-sample-e", leader.reference_id, "e"),
        route_pair_selection=leader.route_pair_selection,
        identity_status="accepted" if accepted_reference_id is not None else "candidate_diagnostic",
    )
    report = write_comparison_report(result, tmp_path / "controlled-e-vs-6")
    selected_pair = report.payload["diagnostics"]["selected_route_pair"]
    assert selected_pair["reference_rank"] == leader.route_pair_selection.score.reference_rank
    assert selected_pair["sample_rank"] == leader.route_pair_selection.score.sample_rank
    assert report.payload["diagnostics"]["route_pairs"]["selected_pair"]["selected"] is True
    route_pairs = json.loads((report.output_dir / "route-pairs.json").read_text())
    selected_route_pair = route_pairs["selected_pair"]
    assert selected_route_pair["reference_route_rank"] == selected_pair["reference_rank"]
    assert selected_route_pair["sample_route_rank"] == selected_pair["sample_rank"]
    assert not (
        report.payload["identity"]["status"] == "accepted"
        and report.payload["identity"]["accepted_reference_id"] == "controlled-6"
    )
