from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from rasterregions import (
    BBox,
    ForegroundEvidence,
    MaskMode,
    MaskResult,
    foreground_mask,
    tighten_bbox,
)
from skimage.draw import circle_perimeter

from inkgrammar.comparison import (
    CropRect,
    analyze_skeleton,
    crop_image,
)
from inkgrammar.raster import crop_to_ink


def test_crop_rect_keeps_requested_pixels_without_resizing() -> None:
    image = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(image).rectangle((7, 5, 16, 14), fill="blue")

    crop = crop_image(image, (5, 3, 20, 18))

    assert crop.size == (20, 18)
    assert crop.getpixel((1, 1)) == (255, 255, 255)
    assert crop.getpixel((3, 3)) == (0, 0, 255)


def test_crop_rect_rejects_invalid_and_out_of_bounds_rectangles() -> None:
    with pytest.raises(ValueError):
        CropRect(x=0, y=0, width=0, height=2)

    with pytest.raises(ValueError, match="bounds"):
        crop_image(Image.new("L", (8, 8)), (7, 7, 2, 2))


def test_auto_mask_uses_color_to_remove_gray_notebook_rule() -> None:
    image = Image.new("RGB", (100, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.line((4, 35, 95, 35), fill=(180, 180, 180), width=1)
    draw.line((20, 50, 42, 50), fill=(20, 70, 220), width=5)
    draw.line((42, 50, 50, 43), fill=(20, 70, 220), width=5)
    draw.line((50, 43, 58, 50), fill=(20, 70, 220), width=5)

    result = foreground_mask(image)

    assert result.method == "color"
    assert result.mask[50, 30]
    assert not result.mask[35, 30]
    assert result.confidence > 0.5
    assert result.warnings == ()


def test_auto_mask_uses_grayscale_for_black_ink() -> None:
    image = Image.new("RGB", (50, 50), "white")
    ImageDraw.Draw(image).line((25, 5, 25, 44), fill="black", width=3)

    result = foreground_mask(image)

    assert result.method == "grayscale"
    assert result.mask[5:45, 25].all()
    assert result.confidence > 0.9


def test_dark_mode_finds_ink_on_a_gray_paper_background() -> None:
    image = np.full((40, 40), 180, dtype=np.uint8)
    image[8:32, 19:22] = 20

    result = foreground_mask(image, mode="dark")

    assert result.method == "dark"
    assert result.mask[8:32, 20].all()
    assert not result.mask[0, 0]


def test_skeleton_diagnostics_report_endpoints_radius_and_components() -> None:
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:20, 12] = True

    result = analyze_skeleton(mask)

    assert result.topology.endpoint_count == 2
    assert result.topology.junction_count == 0
    assert result.topology.component_count == 1
    assert result.topology.cycle_count == 0
    assert result.radius[10, 12] > 0
    assert result.skeleton.sum() >= 2


def test_skeleton_diagnostics_preserve_a_cycle() -> None:
    mask = np.zeros((31, 31), dtype=bool)
    rows, columns = circle_perimeter(15, 15, 8)
    mask[rows, columns] = True

    result = analyze_skeleton(mask)

    assert result.topology.cycle_count == 1
    assert result.topology.component_count == 1
    assert result.topology.endpoint_count == 0


def test_empty_mask_is_reported_without_fake_topology() -> None:
    result = foreground_mask(Image.new("RGB", (12, 12), "white"))
    diagnostics = analyze_skeleton(result.mask)

    assert result.warnings == ("No foreground pixels were detected.",)
    assert diagnostics.topology.component_count == 0
    assert diagnostics.topology.endpoint_count == 0


def test_transparent_rgba_mask_uses_rasterregions_compositing():
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[:, :, :3] = 0
    rgba[:, :, 3] = 0
    rgba[5:15, 5:15] = (0, 0, 0, 255)

    result = foreground_mask(rgba, mode="grayscale")

    assert isinstance(result, MaskResult)
    assert not result.mask[0, 0]
    assert result.mask[10, 10]


def test_inkgrammar_mask_names_are_external_aliases():
    from rasterregions import MaskMode as RasterMaskMode
    from rasterregions import MaskResult as RasterMaskResult

    from inkgrammar.comparison.models import InkMaskResult, InkMode

    assert InkMaskResult is RasterMaskResult
    assert InkMode is RasterMaskMode
    assert MaskMode is RasterMaskMode


def test_comparison_api_passes_mask_mode_to_rasterregions(tmp_path, monkeypatch):
    from inkgrammar.comparison import api

    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 10] = True
    external_evidence = ForegroundEvidence(
        mask,
        np.where(mask, 1.0, -1.0),
        1.0,
        "grayscale",
    )
    received_modes = []

    def foreground_evidence_stub(image, *, mode):
        received_modes.append(mode)
        return external_evidence

    monkeypatch.setattr(api, "foreground_evidence", foreground_evidence_stub)
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (20, 20), "white").save(image_path)

    extracted = api.extract_letter(image_path, (0, 0, 20, 20), ink_mode="color")

    assert received_modes == ["color"]
    assert np.array_equal(extracted.mask_result.mask, external_evidence.mask)
    assert extracted.mask_result.method == "grayscale"
    assert extracted.diagnostics.analysis is not None


def test_extract_letter_keeps_discrete_walk_and_refined_geometry(tmp_path):
    from inkgrammar.comparison import extract_letter

    image = Image.new("RGB", (52, 52), "white")
    ImageDraw.Draw(image).line((8, 42, 18, 12, 28, 40, 42, 15), fill="black", width=7)
    image_path = tmp_path / "curved.png"
    image.save(image_path)

    extracted = extract_letter(image_path, (0, 0, 52, 52))

    assert extracted.ordered_path is not None
    assert extracted.ordered_path.walk is not None
    assert extracted.diagnostics.analysis is not None
    assert extracted.continuous_path is not None
    assert extracted.continuous_path.source == "skeletonpaths-subpixel-v1"
    assert extracted.continuous_path.boundary_evidence == "rasterregions-decision-field"
    assert extracted.continuous_path.edge_ids == extracted.ordered_path.walk.edge_ids
    assert extracted.resampled_xy.shape == (256, 2)


def test_continuous_reconstruction_failure_preserves_topology_comparison(tmp_path, monkeypatch):
    from skeletonpaths import SubpixelReconstructionError

    from inkgrammar.comparison import api

    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).line((16, 4, 16, 27), fill="black", width=5)
    image_path = tmp_path / "line.png"
    image.save(image_path)

    def fail_refinement(*args, **kwargs):
        raise SubpixelReconstructionError("forced reconstruction failure")

    monkeypatch.setattr(api, "refine_continuous_path", fail_refinement)
    extracted = api.extract_letter(image_path, (0, 0, 32, 32), allow_path_failure=True)
    comparison = api.compare_extracted_letters(
        extracted,
        extracted,
        identity=api.ComparisonIdentity("sample", "reference"),
    )

    assert extracted.ordered_path is not None
    assert extracted.continuous_path is None
    assert extracted.diagnostics.topology.component_count == 1
    assert comparison.alignment is None
    assert any("forced reconstruction failure" in warning for warning in comparison.warnings)


def test_crop_to_ink_uses_rasterregions_bbox_and_preserves_empty_policy():
    mask = np.zeros((30, 40), dtype=bool)
    mask[10:15, 20:25] = True

    cropped = crop_to_ink(mask, pad=2)
    bbox = tighten_bbox(mask, BBox(0, 0, mask.shape[1], mask.shape[0]), pad=2)

    assert np.array_equal(cropped, mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1])
    assert crop_to_ink(np.zeros((3, 5), dtype=bool)).shape == (1, 1)


def test_extract_segmented_glyph_reuses_exact_candidate_mask(tmp_path, monkeypatch):
    from inkgrammar.comparison import api
    from inkgrammar.comparison.segmentation import SegmentedGlyph

    mask = np.zeros((24, 24), dtype=bool)
    mask[3:21, 11:14] = True
    image = Image.new("RGB", (24, 24), "white")
    ImageDraw.Draw(image).rectangle((11, 3, 13, 20), fill="black")
    candidate = SegmentedGlyph(
        glyph_id="sample-000",
        bbox=CropRect(x=4, y=5, width=24, height=24),
        row_index=0,
        column_index=0,
        reading_order=0,
        segmentation_confidence=0.9,
        image=image,
        mask=mask,
        mask_method="dark",
    )
    analysis_calls = 0
    analyze_skeleton = api.analyze_skeleton

    def count_analysis(candidate_mask):
        nonlocal analysis_calls
        analysis_calls += 1
        return analyze_skeleton(candidate_mask)

    def fail_foreground_mask(*args, **kwargs):
        raise AssertionError("segmented extraction must not threshold the candidate image")

    monkeypatch.setattr(api, "analyze_skeleton", count_analysis)
    monkeypatch.setattr(api, "foreground_evidence", fail_foreground_mask)

    extracted = api.extract_segmented_glyph(candidate, source_path=tmp_path / "sample.png")

    assert analysis_calls == 1
    assert np.array_equal(extracted.mask_result.mask, candidate.mask)
    assert extracted.mask_result.method == "dark"
    assert extracted.image is not candidate.image
    assert extracted.ordered_path is not None
