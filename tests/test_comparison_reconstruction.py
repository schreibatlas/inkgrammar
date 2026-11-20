from __future__ import annotations

from dataclasses import replace

import numpy as np
from rasterregions import ForegroundEvidence

from inkgrammar.comparison.reconstruction import (
    compare_rendered_coverage,
    reconstruct_route_ink,
    render_route_coverage,
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
