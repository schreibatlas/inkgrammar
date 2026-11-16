from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from inkgrammar.dataset.transforms import (
    glyph_units_to_source_px,
    source_px_to_glyph_units,
    standard_frame,
)
from inkgrammar.fitting.objective import symmetric_distance_loss
from inkgrammar.programs.observation import GlyphObservation, prepare_glyph_observation


def test_observation_uses_rasterregions_mask_and_caches_distance_field() -> None:
    image = Image.new("L", (20, 20), 255)
    ImageDraw.Draw(image).rectangle((5, 6, 10, 12), fill=0)

    observation = prepare_glyph_observation(image, mask_mode="grayscale")

    assert isinstance(observation, GlyphObservation)
    assert observation.bbox.as_tuple() == (5, 6, 11, 13)
    assert observation.mask_result is not None
    assert observation.mask_result.method == "grayscale"
    assert observation.mask.dtype == np.dtype(bool)
    assert not observation.mask.flags.writeable
    assert not observation.distance_field.flags.writeable
    assert observation.distance_field[6, 4] == 1.0
    assert observation.image.getpixel((0, 0)) == 255
    assert observation.skeleton_analysis is None
    assert observation.topology_diagnostics is None
    assert prepare_glyph_observation(observation) is observation

    identical_loss = symmetric_distance_loss(
        observation.mask.astype(float),
        observation.mask.astype(float),
        target_distance_field=observation.distance_field,
    )
    assert identical_loss == 0.0


def test_observation_frame_round_trips_pixels_and_model_units() -> None:
    mask = np.zeros((18, 20), dtype=bool)
    mask[4:13, 5:12] = True
    observation = prepare_glyph_observation(mask)
    frame = observation.frame
    source_points = np.array([[5.0, 13.0], [14.0, 4.0], [8.5, 9.25]])

    model_points = source_px_to_glyph_units(source_points, frame)
    round_tripped = glyph_units_to_source_px(model_points, frame)

    np.testing.assert_allclose(round_tripped, source_points, atol=1e-12)
    np.testing.assert_allclose(model_points[:2], [[0.0, 0.0], [1.0, 1.0]], atol=1e-12)
    assert frame == standard_frame((5.0, 13.0), 9.0)


def test_observation_accepts_explicit_mask_and_frame() -> None:
    image = Image.new("L", (16, 12), 255)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:8, 4:10] = 1
    frame = standard_frame((4.0, 8.0), 5.0)

    observation = prepare_glyph_observation(image, mask=mask, frame=frame)

    assert observation.mask_result is None
    assert observation.frame is frame
    assert observation.bbox.as_tuple() == (4, 3, 10, 8)


def test_skeleton_analysis_is_optional_evidence_with_graph_diagnostics() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("skimage")
    mask = np.zeros((41, 41), dtype=bool)
    mask[19:22, 4:37] = True
    mask[4:37, 19:22] = True
    without_skeleton = prepare_glyph_observation(mask)

    observed = prepare_glyph_observation(without_skeleton, analyze_skeleton=True)
    diagnostics = observed.topology_diagnostics

    assert observed.skeleton_analysis is not None
    assert diagnostics is not None
    assert diagnostics.component_count == observed.skeleton_analysis.graph.component_count == 1
    assert diagnostics.node_count == len(observed.skeleton_analysis.graph.nodes)
    assert diagnostics.edge_count == len(observed.skeleton_analysis.graph.edges)
    assert diagnostics.endpoint_count == observed.skeleton_analysis.graph.endpoint_count
    assert diagnostics.junction_count == observed.skeleton_analysis.graph.junction_count
    assert diagnostics.cycle_count == observed.skeleton_analysis.graph.cycle_count
    assert observed.mask is not without_skeleton.mask
    np.testing.assert_array_equal(observed.distance_field, without_skeleton.distance_field)


def test_observation_rejects_empty_and_nonbinary_masks() -> None:
    with pytest.raises(ValueError, match="no foreground"):
        prepare_glyph_observation(np.zeros((8, 8), dtype=bool))
    with pytest.raises(ValueError, match="zero/one"):
        prepare_glyph_observation(
            Image.new("L", (8, 8), 255),
            mask=np.full((8, 8), 2, dtype=np.uint8),
        )
