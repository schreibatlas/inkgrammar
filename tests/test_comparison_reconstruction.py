from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from rasterregions import ForegroundEvidence, foreground_pyramid

from inkgrammar.comparison.reconstruction import (
    compare_rendered_coverage,
    reconstruct_route_ink,
    render_route_coverage,
)
from inkgrammar.comparison.refinement import (
    MinimumComplexityReconstruction,
    ReconstructionLevelResult,
    ReconstructionRefinementConfig,
    SparseWidthChange,
    SparseWidthProfile,
    fit_minimum_complexity_reconstruction,
    seed_reconstructed_strokes,
)
from inkgrammar.comparison.routes import generate_writing_route_candidates
from inkgrammar.comparison.skeleton import analyze_skeleton


def _candidate():
    mask = np.zeros((32, 40), dtype=bool)
    mask[14:19, 5:35] = True
    diagnostics = analyze_skeleton(mask)
    return generate_writing_route_candidates(diagnostics, limit=1)[0]


def _evidence(mask: np.ndarray) -> ForegroundEvidence:
    return ForegroundEvidence(
        mask=mask,
        decision_field=np.where(mask, 1.0, -1.0),
        confidence=1.0,
        method="binary",
    )


def test_matching_render_has_negligible_residual_and_perturbation_has_under_and_over_coverage():
    candidate = _candidate()
    coverage, profiles = render_route_coverage(
        candidate,
        frame_width=40,
        frame_height=32,
    )
    observed = coverage >= 0.5
    evidence = _evidence(observed)
    reconstruction = compare_rendered_coverage(coverage, evidence, width_profiles=profiles)

    assert reconstruction.metrics["under_coverage_pixel_count"] == 0
    assert reconstruction.metrics["over_coverage_pixel_count"] == 0
    assert reconstruction.metrics["intersection_over_union"] == 1.0
    assert reconstruction.width_profiles

    stroke = candidate.continuous_strokes[0]
    shifted_stroke = replace(stroke, xy=stroke.xy + np.asarray((3.0, 0.0)))
    shifted_candidate = replace(candidate, continuous_strokes=(shifted_stroke,))
    perturbed = reconstruct_route_ink(shifted_candidate, evidence)

    assert perturbed.metrics["under_coverage_pixel_count"] > 0
    assert perturbed.metrics["over_coverage_pixel_count"] > 0
    assert (
        perturbed.metrics["intersection_over_union"]
        < reconstruction.metrics["intersection_over_union"]
    )
    assert np.array_equal(evidence.mask, observed)


def test_sparse_width_profile_is_constant_or_interpolates_sparse_changes():
    constant = SparseWidthProfile.constant(3.0)
    assert constant.parameter_count == 1
    assert constant.width_change_count == 0
    assert constant.width_at(0.5) == 3.0

    profile = SparseWidthProfile(
        start_width_px=2.0,
        end_width_px=6.0,
        changes=(SparseWidthChange(at=0.5, width_px=4.0),),
    )
    assert profile.width_change_count == 1
    assert profile.parameter_count == 3
    assert profile.width_at(0.25) == 3.0
    assert profile.width_at(0.75) == 5.0
    with pytest.raises(ValueError):
        SparseWidthChange(at=0.0, width_px=2.0)


def test_refinement_configuration_is_coarse_to_fine_and_strict():
    config = ReconstructionRefinementConfig()
    assert config.scales == (0.25, 0.5, 1.0)
    assert config.max_route_candidates == 5
    with pytest.raises(ValueError):
        ReconstructionRefinementConfig(scales=(0.5, 0.25))
    with pytest.raises(ValueError):
        ReconstructionRefinementConfig(unexpected_option=1)


def test_seed_uses_fitted_trajectory_and_constant_robust_width_independent_of_sampling():
    from inkgrammar.comparison.steering import fit_writing_route_candidate

    candidate = fit_writing_route_candidate(_candidate())
    path = candidate.continuous_strokes[0]
    radii = np.full_like(path.radius_px, 2.0)
    radii[len(radii) // 2] = 50.0
    path_with_radius_outlier = replace(path, radius_px=radii)
    candidate = replace(candidate, continuous_strokes=(path_with_radius_outlier,))

    seeded = seed_reconstructed_strokes(candidate)[0]
    selected = candidate.steering_models[0].selected
    assert seeded.trajectory is selected.trajectory
    assert seeded.transform is selected.transform
    assert seeded.width == SparseWidthProfile.constant(4.0)

    indices = np.linspace(0, len(path.xy) - 1, max(2, len(path.xy) // 2), dtype=int)
    sparse_path = replace(
        path_with_radius_outlier,
        xy=path.xy[indices],
        tangent=path.tangent[indices],
        curvature=path.curvature[indices],
        radius_px=radii[indices],
        confidence=path.confidence[indices],
        progress=path.progress[indices],
        route_event_indices=(),
        route_event_progress=(),
        edge_ids=(),
        edge_directions=(),
    )
    sparse_candidate = replace(candidate, continuous_strokes=(sparse_path,))
    sparse_seed = seed_reconstructed_strokes(sparse_candidate)[0]
    assert sparse_seed.trajectory is seeded.trajectory
    assert sparse_seed.transform is seeded.transform
    assert sparse_seed.width == seeded.width

    coverage = np.ones((2, 2), dtype=np.float32)
    result = MinimumComplexityReconstruction(
        strokes=(seeded,),
        levels=(
            ReconstructionLevelResult(
                level_index=0,
                scale_x=0.25,
                scale_y=0.25,
                objective_before=1.0,
                objective_after=0.5,
                iterations=2,
                accepted_geometry_changes=0,
                accepted_width_changes=0,
            ),
        ),
        rendered_coverage=coverage,
        energy=0.5,
        topology_preserved=True,
        ambiguous=False,
    )
    coverage[0, 0] = 0.0
    assert result.rendered_coverage[0, 0] == 1.0
    assert not result.rendered_coverage.flags.writeable


def test_multiscale_fit_preserves_fractional_detached_ink_and_route_topology():
    from inkgrammar.comparison.steering import fit_writing_route_candidate

    mask = np.zeros((32, 40), dtype=bool)
    mask[14:19, 5:35] = True
    mask[3, 3] = True
    evidence = _evidence(mask)
    candidate = fit_writing_route_candidate(_candidate())
    config = ReconstructionRefinementConfig(
        max_geometry_changes=0,
        max_width_changes=0,
        max_iterations_per_level=4,
    )
    calls = []

    def render_at_level(strokes, level, supersampling):
        calls.append((level.index, supersampling, len(strokes)))
        return level.coverage

    result = fit_minimum_complexity_reconstruction(
        candidate,
        evidence,
        render_at_level,
        config=config,
    )
    pyramid = foreground_pyramid(evidence, scales=config.scales)
    coarse = pyramid.levels[0]

    assert len(result.levels) == 3
    assert [item.level_index for item in result.levels] == [0, 1, 2]
    assert all(item.objective_after <= item.objective_before + 1e-9 for item in result.levels)
    assert np.array_equal(result.rendered_coverage, mask.astype(np.float32))
    assert result.topology_preserved
    assert len(result.strokes) == len(candidate.continuous_strokes)
    assert not any(
        item.accepted_geometry_changes or item.accepted_width_changes for item in result.levels
    )
    assert 0.0 < coarse.coverage[0, 0] < 0.5
    assert not coarse.mask[0, 0]
    assert result.objective_evaluations == len(calls)
    assert calls[0] == (0, config.supersampling_coarse, len(candidate.continuous_strokes))
    assert calls[-1][0] == 2


def test_raster_objective_refines_global_placement_from_the_route_seed():
    from inkgrammar.comparison.steering import fit_writing_route_candidate

    mask = np.zeros((32, 40), dtype=bool)
    mask[14:19, 18:23] = True
    evidence = _evidence(mask)
    candidate = fit_writing_route_candidate(_candidate())
    seed_dx = candidate.steering_models[0].selected.transform.dx

    def render_at_level(strokes, level, supersampling):
        stroke = strokes[0]
        height, width = level.shape
        yy, xx = np.mgrid[:height, :width]
        center_x = stroke.transform.dx * level.scale_x
        center_y = stroke.transform.dy * level.scale_y
        sigma = 8.0
        return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))

    result = fit_minimum_complexity_reconstruction(
        candidate,
        evidence,
        render_at_level,
        config=ReconstructionRefinementConfig(
            scales=(1.0,),
            max_geometry_changes=0,
            max_width_changes=0,
            max_iterations_per_level=30,
        ),
    )

    assert result.levels[0].objective_after < result.levels[0].objective_before
    assert abs(result.strokes[0].transform.dx - 20.0) < abs(seed_dx - 20.0)


def test_raster_complexity_penalty_rejects_width_changes_without_coverage_gain():
    from inkgrammar.comparison.steering import fit_writing_route_candidate

    candidate = fit_writing_route_candidate(_candidate())
    path = candidate.continuous_strokes[0]
    radii = np.full_like(path.radius_px, 2.0)
    localized = (path.progress > 0.35) & (path.progress < 0.55)
    radii[localized] = 6.0
    candidate = replace(
        candidate,
        continuous_strokes=(replace(path, radius_px=radii),),
    )
    evidence = _evidence(np.ones((32, 40), dtype=bool))

    result = fit_minimum_complexity_reconstruction(
        candidate,
        evidence,
        lambda strokes, level, supersampling: level.coverage,
        config=ReconstructionRefinementConfig(
            scales=(1.0,),
            max_geometry_changes=0,
            max_width_changes=1,
            max_iterations_per_level=3,
        ),
    )

    assert result.strokes[0].width.width_change_count == 0
    assert result.levels[0].accepted_width_changes == 0


def test_route_orchestration_ranks_bounded_strokeoutline_reconstructions(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw

    import inkgrammar.comparison.api as comparison_api
    from inkgrammar.comparison.api import compare_extracted_letters, extract_letter
    from inkgrammar.comparison.models import ComparisonIdentity

    path = tmp_path / "horizontal.png"
    image = Image.new("L", (40, 32), 255)
    ImageDraw.Draw(image).line((5, 16, 34, 16), fill=0, width=5)
    image.save(path)
    reference = extract_letter(path, (0, 0, 40, 32), ink_mode="dark")
    sample = extract_letter(path, (0, 0, 40, 32), ink_mode="dark")
    config = ReconstructionRefinementConfig(
        scales=(1.0,),
        max_route_candidates=1,
        max_geometry_changes=0,
        max_width_changes=0,
        max_iterations_per_level=2,
    )
    pyramid_factory = comparison_api.foreground_pyramid
    built_pyramids = []

    def count_pyramid_builds(evidence, *, scales):
        built_pyramids.append(evidence)
        return pyramid_factory(evidence, scales=scales)

    monkeypatch.setattr(comparison_api, "foreground_pyramid", count_pyramid_builds)

    result = compare_extracted_letters(
        reference,
        sample,
        identity=ComparisonIdentity("sample", "reference", "i"),
        reconstruction_config=config,
    )
    assert len(built_pyramids) == 2
    assert built_pyramids[0] is reference.foreground_evidence
    assert built_pyramids[1] is sample.foreground_evidence
    selection = result.route_selection
    assert selection is not None
    assert len(result.reference.route_candidates) == 1
    assert len(result.sample.route_candidates) == 1
    assert selection.reference.reconstruction is not None
    assert selection.sample.reconstruction is not None
    assert selection.reference.reconstruction.ambiguous == selection.reference.ambiguous
    assert selection.sample.reconstruction.ambiguous == selection.sample.ambiguous
    assert "reconstruction_fit_cost" in selection.score.cost_breakdown
    assert "model_complexity_cost" in selection.score.cost_breakdown

    evidence = result.reference.foreground_evidence
    assert evidence is not None
    rendered = reconstruct_route_ink(selection.reference, evidence)
    assert rendered.metrics["renderer"] == "strokeoutline.render_strokes"
    assert np.array_equal(
        rendered.rendered_coverage,
        selection.reference.reconstruction.rendered_coverage,
    )


def _representative_glyph(tmp_path, name: str):
    from PIL import Image, ImageDraw

    from inkgrammar.comparison.api import extract_letter

    scale = 2
    image = Image.new("L", (80 * scale, 80 * scale), 255)
    draw = ImageDraw.Draw(image)
    paths = {
        "l": (((38, 9), (38, 65)),),
        "e": (
            (
                (52, 26),
                (46, 20),
                (36, 17),
                (27, 19),
                (21, 26),
                (19, 35),
                (21, 44),
                (28, 50),
                (39, 51),
                (49, 47),
            ),
            ((20, 34), (44, 34)),
        ),
        "g": (((50, 41), (50, 55), (47, 61), (41, 65), (35, 65)),),
        "i": (((38, 25), (38, 64)),),
        "umlaut": (((20, 24), (20, 45), (24, 52), (32, 56), (41, 55), (49, 49), (50, 25)),),
    }
    if name == "g":
        draw.ellipse((20 * scale, 12 * scale, 54 * scale, 47 * scale), outline=0, width=5 * scale)
    for path in paths[name]:
        draw.line(
            [(round(x * scale), round(y * scale)) for x, y in path],
            fill=0,
            width=5 * scale,
            joint="curve",
        )
    if name == "i":
        draw.ellipse((33 * scale, 8 * scale, 43 * scale, 18 * scale), fill=0)
    elif name == "umlaut":
        draw.ellipse((27 * scale, 7 * scale, 37 * scale, 17 * scale), fill=0)
        draw.ellipse((41 * scale, 7 * scale, 51 * scale, 17 * scale), fill=0)
    image = image.resize((80, 80), Image.Resampling.LANCZOS)
    image_path = tmp_path / f"glyph-{name}.png"
    image.save(image_path)
    return extract_letter(image_path, (0, 0, 80, 80), allow_path_failure=True)


@pytest.mark.parametrize(
    ("name", "expected_components"),
    (("l", 1), ("e", 1), ("g", 1), ("i", 2), ("umlaut", 3)),
)
def test_representative_glyph_reconstruction_keeps_topology_and_bounded_complexity(
    tmp_path,
    name: str,
    expected_components: int,
) -> None:
    from inkgrammar.comparison.reconstruction import render_reconstructed_strokes
    from inkgrammar.comparison.steering import fit_writing_route_candidate

    extracted = _representative_glyph(tmp_path, name)
    evidence = extracted.foreground_evidence
    assert evidence is not None
    assert extracted.diagnostics.topology.component_count == expected_components
    candidate = fit_writing_route_candidate(extracted.route_candidates[0])
    config = ReconstructionRefinementConfig(
        scales=(0.5, 1.0),
        max_route_candidates=1,
        max_geometry_changes=1,
        max_width_changes=1,
        supersampling_coarse=1,
        supersampling_fine=1,
        max_iterations_per_level=3,
    )
    reconstruction = fit_minimum_complexity_reconstruction(
        candidate,
        evidence,
        render_reconstructed_strokes,
        config=config,
    )
    fitted_candidate = replace(candidate, reconstruction=reconstruction)
    diagnostics = reconstruct_route_ink(fitted_candidate, evidence)

    assert reconstruction.topology_preserved
    assert len(reconstruction.strokes) >= expected_components
    assert sum(level.accepted_geometry_changes for level in reconstruction.levels) <= 1
    assert sum(level.accepted_width_changes for level in reconstruction.levels) <= 1
    assert all(
        level.objective_after <= level.objective_before + 1e-9 for level in reconstruction.levels
    )
    assert diagnostics.metrics["intersection_over_union"] >= 0.15
    assert reconstruction.objective_evaluations < 5000
