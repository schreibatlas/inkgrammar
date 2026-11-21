from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from inkgrammar.comparison import (
    ComparisonIdentity,
    CropRect,
    WritingDirectionPolicy,
    compare_extracted_letters,
    extract_letter,
)
from inkgrammar.comparison.report import write_comparison_report
from inkgrammar.comparison.routes import (
    generate_writing_route_candidates,
    select_writing_route_pair,
)
from inkgrammar.comparison.skeleton import analyze_skeleton


def _candidates(points: tuple[tuple[int, int], ...], width: int = 4):
    image = Image.new("L", (64, 64), 255)
    ImageDraw.Draw(image).line(points, fill=0, width=width)
    mask = np.asarray(image) < 128
    return generate_writing_route_candidates(analyze_skeleton(mask))


def _lowercase_e_candidates():
    return _candidates(
        (
            (15, 27),
            (34, 27),
            (36, 22),
            (33, 16),
            (25, 14),
            (18, 18),
            (15, 26),
            (17, 34),
            (24, 38),
            (33, 37),
        )
    )


def _x_candidates():
    image = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 10, 50, 50), fill=0, width=3)
    draw.line((10, 50, 50, 10), fill=0, width=3)
    return generate_writing_route_candidates(analyze_skeleton(np.asarray(image) < 128))


def test_left_to_right_policy_resolves_near_tied_orientation() -> None:
    candidates = _candidates(((5, 15), (27, 15)))

    selection = select_writing_route_pair(candidates, candidates)

    for candidate in (selection.reference, selection.sample):
        path = candidate.continuous_strokes[0]
        assert path.xy[-1, 0] > path.xy[0, 0]
        assert candidate.orientation_status == "resolved"
        assert candidate.orientation_evidence == "writing-prior"
        assert candidate.orientation_margin > 0.005
        assert candidate.writing_prior is not None
        assert candidate.writing_prior.direction_cost == 0.0


def test_neutral_policy_keeps_unresolved_orientation_ambiguous() -> None:
    candidates = _candidates(((5, 15), (27, 15)))
    policy = WritingDirectionPolicy(
        script_direction="neutral",
        endpoint_bias_weight=0.0,
        direction_weight=0.0,
        pen_lift_weight=0.0,
        junction_continuity_weight=0.0,
    )

    selection = select_writing_route_pair(candidates, candidates, direction_policy=policy)

    assert selection.ambiguous
    assert selection.reference.orientation_status == "ambiguous"
    assert selection.sample.orientation_status == "ambiguous"


def test_connected_lowercase_e_prefers_one_stroke_with_endpoint_start() -> None:
    candidates = _lowercase_e_candidates()

    selection = select_writing_route_pair(candidates, candidates)

    assert len(selection.reference.continuous_strokes) == 1
    assert selection.reference.hypothesis.pen_lift_count == 0
    assert len(selection.sample.continuous_strokes) == 1
    assert selection.sample.hypothesis.pen_lift_count == 0
    assert selection.reference.writing_prior is not None
    assert selection.reference.writing_prior.start_endpoint_cost < 0.05
    assert (
        selection.reference.continuous_strokes[0].xy[-1, 0]
        > (selection.reference.continuous_strokes[0].xy[0, 0])
    )


def test_x_like_geometry_can_keep_a_multi_stroke_route() -> None:
    candidates = _x_candidates()

    selection = select_writing_route_pair(candidates, candidates)

    assert len(selection.reference.continuous_strokes) == 2
    assert selection.reference.hypothesis.pen_lift_count == 1
    assert selection.reference.writing_prior is not None
    assert selection.reference.writing_prior.pen_lift_cost == 1.0


def test_report_json_includes_selected_writing_prior(tmp_path) -> None:
    image_path = tmp_path / "line.png"
    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).line((5, 15, 27, 15), fill="black", width=3)
    image.save(image_path)
    extracted = extract_letter(image_path, CropRect(x=0, y=0, width=32, height=32))

    result = compare_extracted_letters(
        extracted,
        extracted,
        identity=ComparisonIdentity("sample", "reference", "i"),
        writing_direction_policy=WritingDirectionPolicy(),
    )
    report = write_comparison_report(result, tmp_path / "report")
    writing = json.loads(report.json_path.read_text())["writing"]

    prior = writing["reference"]["writing_prior"]
    assert prior["pen_lift_cost"] == 0.0
    assert prior["direction_cost"] == 0.0
    assert "start_endpoint_cost" in prior
    assert "end_endpoint_cost" in prior
    assert "junction_continuity_cost" in prior
    assert writing["reference"]["orientation"]["status"] == "resolved"
