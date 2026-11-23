from __future__ import annotations

import json

from PIL import Image, ImageDraw
from typer.testing import CliRunner

from inkgrammar.cli import app


def _draw_glyph(draw: ImageDraw.ImageDraw, x: int, kind: str) -> None:
    y = 25
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


def _page(path, glyphs: tuple[str, ...]):
    image = Image.new("RGB", (250, 88), "white")
    draw = ImageDraw.Draw(image)
    for x, kind in zip((15, 95, 175), glyphs, strict=True):
        _draw_glyph(draw, x, kind)
    image.save(path)
    return path


def _reference_page(path):
    image = Image.new("RGB", (280, 90), "white")
    draw = ImageDraw.Draw(image)
    for x, kind in zip((10, 55, 100, 145, 190, 235), ("e", "n", "l") * 2, strict=True):
        _draw_glyph(draw, x, kind)
    image.save(path)
    return path


def test_compare_letter_cli_automatic_mode_writes_batch_summary(tmp_path):
    reference = _reference_page(tmp_path / "reference.png")
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "automatic"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "3",
            "--top-k",
            "4",
            "--verbose",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[compare-letter] 01/13" in result.output
    assert "[compare-letter] 13/13" in result.output
    assert "HTML summary:" in result.output
    assert "JSON summary:" in result.output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sample"]["detected_count"] == 3

    gallery = json.loads((output / "reference-index" / "gallery.json").read_text())
    assert gallery["reference_profile"] == "generic"
    report_markdown = (output / "summary.md").read_text()
    assert "micro-noise removed" in report_markdown
    assert "Detailed reference diagnostics" in report_markdown
    assert "Acceptance decision: accepted" in report_markdown
    assert (output / "segmentation" / "reference-segmentation.md").is_file()
    assert all(1 <= len(match["alternatives"]) <= 3 for match in summary["matches"])
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    extraction = [event for event in events if event["stage"] == "reference_extraction"]
    assert extraction and all("elapsed_seconds" in event["details"] for event in extraction)


def test_compare_letter_cli_reports_exact_count_diagnostics(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "wrong-count"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "4",
            "--output",
            str(output),
        ],
    )

    diagnostic = output / "segmentation" / "sample-boundaries.png"
    assert result.exit_code == 2
    assert "Unable to recover 4 credible" in result.output
    assert "Diagnostics:" in result.output
    assert diagnostic.name in result.output
    assert diagnostic.is_file()


def test_compare_letter_cli_rejects_missing_mixed_and_incomplete_modes(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    base = ["compare-letter", str(reference), str(sample), "--output", str(tmp_path / "out")]
    runner = CliRunner()

    missing = runner.invoke(app, base)
    mixed = runner.invoke(
        app,
        base + ["--letters", "3", "--letter", "e", "--reference-crop", "1,1,20,20"],
    )
    incomplete = runner.invoke(
        app,
        base + ["--letter", "e", "--reference-crop", "1,1,20,20"],
    )

    assert missing.exit_code == 2
    assert "choose --letters N" in missing.output
    assert mixed.exit_code == 2
    assert "cannot be combined" in mixed.output
    assert incomplete.exit_code == 2
    assert "provide --letter" in incomplete.output


def test_compare_letter_help_describes_automatic_controls():
    result = CliRunner().invoke(app, ["compare-letter", "--help"])

    assert result.exit_code == 0
    assert "auto (default)" in result.output
    assert "exactly N visibly" in result.output
    assert "standalone" in result.output
    for option in (
        "--top-k",
        "--reference-profile",
        "--no-progress",
        "--verbose",
        "--segment-only",
    ):
        assert option in result.output


def test_compare_letter_cli_stops_on_rejected_reference_layout(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "e", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "rejected-reference"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "3",
            "--reference-profile",
            "alphabet",
            "--output",
            str(output),
        ],
    )

    decision_path = output / "segmentation" / "reference-segmentation-decision.json"
    assert result.exit_code == 2
    assert "Reference segmentation rejected" in result.output
    assert decision_path.is_file()
    assert not (output / "reference-index" / "gallery.json").exists()
    assert not (output / "segmentation" / "sample-segmentation.json").exists()


def test_compare_letter_cli_segment_only_honors_quiet_mode_and_profiles(tmp_path):
    reference = _page(tmp_path / "reference.png", ("e", "n", "l"))
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "segment-only"

    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "3",
            "--top-k",
            "4",
            "--reference-profile",
            "generic",
            "--segment-only",
            "--no-progress",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[compare-letter]" not in result.output
    assert "HTML diagnostics:" in result.output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["operation"] == "segment-only"
    assert summary["reference"]["reference_profile"] == "generic"
    assert json.loads((output / "run.json").read_text())["status"] == "completed"


def test_compare_letter_cli_reference_failure_reports_public_diagnostics_once(
    tmp_path, monkeypatch
):
    reference = _reference_page(tmp_path / "reference.png")
    sample = _page(tmp_path / "sample.png", ("e", "n", "l"))
    output = tmp_path / "failed-reference"

    def fail_segmentation(*args, **kwargs):
        raise ValueError("synthetic rasterregions failure")

    monkeypatch.setattr(
        "inkgrammar.comparison.segmentation.rr.detect_spaced_region_hypotheses",
        fail_segmentation,
    )
    result = CliRunner().invoke(
        app,
        [
            "compare-letter",
            str(reference),
            str(sample),
            "--letters",
            "3",
            "--output",
            str(output),
        ],
    )

    diagnostic = output / "segmentation" / "reference-segmentation-hypotheses.json"
    assert result.exit_code == 2
    assert diagnostic.name in result.output
    assert result.output.count("Diagnostics:") == 1
    assert diagnostic.is_file()
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    reference_events = [event for event in events if event["stage"] == "reference_mask_rows"]
    assert reference_events[-1]["details"]["failure_phase"] == "rasterregions_detection"
