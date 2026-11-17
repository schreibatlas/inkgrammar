from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from inkgrammar.comparison.segmentation import (
    SegmentationError,
    detect_ink_rows,
    segment_reference_gallery,
    segment_sample_glyphs,
)


def _draw_shape(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "e":
        draw.line(
            [
                (x, y + 14),
                (x + 18, y + 14),
                (x + 20, y + 9),
                (x + 17, y + 4),
                (x + 10, y + 2),
                (x + 4, y + 5),
                (x + 1, y + 12),
                (x + 3, y + 20),
                (x + 9, y + 25),
                (x + 17, y + 24),
            ],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 8, y, x + 8, y + 26), fill="black", width=4)
    elif kind == "i":
        draw.line((x + 8, y + 9, x + 8, y + 26), fill="black", width=4)
        draw.ellipse((x + 6, y, x + 10, y + 4), fill="black")
    elif kind == "n":
        draw.line(
            [
                (x, y + 26),
                (x, y + 5),
                (x + 6, y + 2),
                (x + 15, y + 4),
                (x + 19, y + 10),
                (x + 19, y + 26),
            ],
            fill="black",
            width=4,
        )


def _reference_page(path):
    image = Image.new("RGB", (260, 170), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 259, 169), outline="black", width=2)
    for x, kind in ((20, "e"), (75, "l"), (125, "i"), (185, "n")):
        _draw_shape(draw, x, 20, kind)
    for x, kind in ((20, "n"), (80, "e"), (145, "l"), (200, "i")):
        _draw_shape(draw, x, 105, kind)
    image.save(path)
    return path


def _sample_page(path):
    image = Image.new("RGB", (260, 135), "white")
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, 15, 35, "e")
    _draw_shape(draw, 105, 35, "i")
    _draw_shape(draw, 195, 35, "n")
    draw.line((4, 100, 255, 100), fill=(180, 180, 180), width=1)
    draw.point((250, 8), fill="black")
    image.save(path)
    return path


def test_reference_gallery_segments_rows_and_attaches_detached_marks(tmp_path):
    path = _reference_page(tmp_path / "alphabet.png")
    output = tmp_path / "reference-diagnostics"

    gallery = segment_reference_gallery(path, output_dir=output)

    assert len(gallery.candidates) == 8
    assert [candidate.glyph_id for candidate in gallery.candidates] == [
        f"ref-{index:03d}" for index in range(8)
    ]
    assert [candidate.row_index for candidate in gallery.candidates] == [0] * 4 + [1] * 4
    assert [candidate.column_index for candidate in gallery.candidates] == [0, 1, 2, 3] * 2
    dotted = gallery.candidates[2]
    assert dotted.bbox.y <= 20
    assert len(detect_ink_rows(gallery.mask_result.mask)) == 2
    assert (output / "reference-mask.png").is_file()
    assert (output / "reference-candidates.png").is_file()
    manifest = json.loads((output / "reference-candidates.json").read_text())
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
    assert samples[1].bbox.y == 35
    assert all(candidate.segmentation_confidence >= 0.3 for candidate in samples)
    for filename in (
        "sample-mask.png",
        "sample-components.png",
        "sample-boundaries.png",
        "sample-candidates.png",
        "sample-segmentation.json",
    ):
        assert (output / filename).is_file()
    manifest = json.loads((output / "sample-segmentation.json").read_text())
    assert manifest["expected_count"] == manifest["detected_count"] == 3
    assert manifest["status"] == "ok"
    assert any("rule" in warning for warning in manifest["warnings"])
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


def test_alphabet_profile_excludes_title_and_year_and_isolates_all_glyph_rows(tmp_path):
    path = _alphabet_like_page(tmp_path / "synthetic-alphabet.png")
    output = tmp_path / "reference-diagnostics"

    gallery = segment_reference_gallery(path, ink_mode="dark", output_dir=output)

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
