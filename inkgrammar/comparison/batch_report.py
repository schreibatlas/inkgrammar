from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .matching import (
    AMBIGUITY_ABSOLUTE_MARGIN,
    AMBIGUITY_RELATIVE_MARGIN,
    ASPECT_WEIGHT,
    FILL_WEIGHT,
    RASTER_WEIGHT,
    TOPOLOGY_WEIGHT,
    MatchScore,
)
from .models import CropRect


@dataclass(frozen=True)
class BatchSampleResult:
    sample_id: str
    sample_bbox: CropRect
    sample_confidence: float
    reference_id: str
    reference_label: str | None
    status: str
    segmentation_status: str
    segmentation_warnings: tuple[str, ...]
    score: MatchScore
    alternatives: tuple[MatchScore, ...]
    report_dir: Path
    report_json: Path


@dataclass(frozen=True)
class BatchComparisonReport:
    output_dir: Path
    payload: dict[str, Any]

    @property
    def json_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def markdown_path(self) -> Path:
        return self.output_dir / "summary.md"

    @property
    def html_path(self) -> Path:
        return self.output_dir / "index.html"


def match_score_payload(score: MatchScore) -> dict[str, Any]:
    return {
        "reference_id": score.reference_id,
        "rank": score.rank,
        "cost": score.cost,
        "coarse_cost": score.coarse_cost,
        "refined_cost": score.refined_cost,
        "raster_cost": score.raster_cost,
        "aspect_cost": score.aspect_cost,
        "fill_cost": score.fill_cost,
        "topology_cost": score.topology_cost,
        "centerline_cost": score.centerline_cost,
        "heading_cost": score.heading_cost,
        "fitted_steering_cost": score.fitted_steering_cost,
        "route_structure_cost": score.route_structure_cost,
    }


def match_reason(score: MatchScore, alternatives: tuple[MatchScore, ...]) -> str:
    if not alternatives:
        return "Only available reference candidate."
    runner_up = alternatives[0]
    weights = {
        "raster": ("raster_cost", RASTER_WEIGHT),
        "aspect": ("aspect_cost", ASPECT_WEIGHT),
        "fill": ("fill_cost", FILL_WEIGHT),
        "topology": ("topology_cost", TOPOLOGY_WEIGHT),
        "centerline": ("centerline_cost", 0.45),
        "heading": ("heading_cost", 0.15),
        "fitted steering": ("fitted_steering_cost", 0.15),
        "route structure": ("route_structure_cost", 0.10),
    }
    advantages = []
    for label, (attribute, weight) in weights.items():
        best_value = getattr(score, attribute)
        next_value = getattr(runner_up, attribute)
        if best_value is not None and next_value is not None and best_value < next_value:
            advantages.append((weight * (next_value - best_value), label))
    advantages.sort(reverse=True)
    if not advantages:
        return f"Lowest combined retrieval cost; next candidate is {runner_up.reference_id}."
    features = " and ".join(label for _, label in advantages[:2])
    return (
        f"Ranked ahead of {runner_up.reference_id}; lower {features} costs "
        "were the strongest advantages."
    )


def ambiguity_sources(
    score: MatchScore,
    alternatives: tuple[MatchScore, ...],
    match_status: str,
    segmentation_status: str,
    segmentation_warnings: tuple[str, ...],
    pair_report_json: Path | None = None,
) -> list[dict[str, str]]:
    sources = []
    if match_status == "ambiguous" and alternatives:
        next_cost = alternatives[0].cost
        margin = next_cost - score.cost
        relative_margin = margin / max(abs(score.cost), 1e-9)
        sources.append(
            {
                "source": "match_cost_tie",
                "reason": (
                    f"Best and next retrieval costs differ by {margin:.4g} "
                    f"({relative_margin:.1%}), within the configured ambiguity margin."
                ),
            }
        )
    if segmentation_status == "ambiguous":
        detail = "; ".join(segmentation_warnings[:2])
        sources.append(
            {
                "source": "segmentation_uncertainty",
                "reason": (
                    detail or ("Segmentation reported uncertain sample or reference boundaries.")
                ),
            }
        )
    if pair_report_json is not None and pair_report_json.is_file():
        pair_payload = json.loads(pair_report_json.read_text(encoding="utf-8"))
        writing = pair_payload.get("writing", {})
        if writing.get("route_ambiguous"):
            selected = pair_payload.get("diagnostics", {}).get("selected_route_pair") or {}
            ranks = {
                "reference": selected.get("reference_rank"),
                "sample": selected.get("sample_rank"),
            }
            candidates = pair_payload.get("diagnostics", {}).get("candidate_routes", {})
            hypothesis_ambiguous = any(
                candidate.get("ambiguous")
                for side, rank in ranks.items()
                for candidate in candidates.get(side, ())
                if candidate.get("rank") == rank
            )
            sources.append(
                {
                    "source": (
                        "writing_route_hypothesis"
                        if hypothesis_ambiguous
                        else "writing_route_pair_tie"
                    ),
                    "reason": (
                        "A selected bounded route hypothesis remains ambiguous."
                        if hypothesis_ambiguous
                        else "Alternative joint route-pair interpretations remain near-tied."
                    ),
                }
            )
    return sources


def _ambiguity_text(sources: list[dict[str, str]]) -> str:
    return "; ".join(f"{item['source']}: {item['reason']}" for item in sources) or "None reported."


def _sample_payload(result: BatchSampleResult, output_dir: Path) -> dict[str, Any]:
    reason = match_reason(result.score, result.alternatives)
    ambiguity = ambiguity_sources(
        result.score,
        result.alternatives,
        result.status,
        result.segmentation_status,
        result.segmentation_warnings,
        result.report_json,
    )
    return {
        "sample_id": result.sample_id,
        "sample_bbox": result.sample_bbox.model_dump(mode="json"),
        "segmentation_confidence": result.sample_confidence,
        "reference_id": result.reference_id,
        "reference_label": result.reference_label,
        "status": result.status,
        "match_status": result.status,
        "segmentation_status": result.segmentation_status,
        "match_reason": reason,
        "ambiguity_sources": ambiguity,
        "segmentation_warnings": list(result.segmentation_warnings),
        "cost": result.score.cost,
        "alternatives": [match_score_payload(score) for score in result.alternatives],
        "report_dir": result.report_dir.relative_to(output_dir).as_posix(),
        "report_json": result.report_json.relative_to(output_dir).as_posix(),
        "match_json": f"{result.sample_id}/match.json",
        "sample_image": f"{result.sample_id}/sample.png",
        "matched_reference_image": f"{result.sample_id}/matched-reference.png",
        "alternatives_image": f"{result.sample_id}/alternatives.png",
    }


def write_batch_report(
    output_dir: str | Path,
    *,
    reference_path: str | Path,
    sample_path: str | Path,
    reference_sha256: str,
    expected_count: int,
    reference_cache_hit: bool,
    results: tuple[BatchSampleResult, ...],
    reference_segmentation: dict[str, Any],
) -> BatchComparisonReport:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    samples = [_sample_payload(result, root) for result in results]
    match_ambiguous_count = sum(item["status"] == "ambiguous" for item in samples)
    segmentation_ambiguous_count = sum(
        item["segmentation_status"] == "ambiguous" for item in samples
    )
    payload: dict[str, Any] = {
        "status": ("ambiguous" if match_ambiguous_count or segmentation_ambiguous_count else "ok"),
        "reference": {
            "path": Path(reference_path).name,
            "sha256": reference_sha256,
            "segmentation": reference_segmentation,
        },
        "sample": {
            "path": Path(sample_path).name,
            "expected_count": expected_count,
            "detected_count": len(samples),
        },
        "ambiguous_count": match_ambiguous_count,
        "match_ambiguous_count": match_ambiguous_count,
        "segmentation_ambiguous_count": segmentation_ambiguous_count,
        "reference_cache": {"hit": reference_cache_hit, "key": reference_sha256},
        "matching": {
            "cost_interpretation": "lower retrieval cost ranks ahead; it is not a probability",
            "weights": {
                "raster": RASTER_WEIGHT,
                "aspect": ASPECT_WEIGHT,
                "fill": FILL_WEIGHT,
                "topology": TOPOLOGY_WEIGHT,
            },
            "ambiguity_margin": {
                "absolute": AMBIGUITY_ABSOLUTE_MARGIN,
                "relative": AMBIGUITY_RELATIVE_MARGIN,
            },
        },
        "matches": samples,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    markdown = [
        "# Automatic comparison summary",
        "",
        f"Reference: `{Path(reference_path).name}`",
        f"Sample: `{Path(sample_path).name}` ({expected_count} glyphs requested)",
        (
            f"Reference segmentation: {reference_segmentation['status']} "
            f"({reference_segmentation['reference_profile']} profile, "
            f"{reference_segmentation['row_count']} rows, "
            f"{reference_segmentation['candidate_count']} candidates; "
            f"search truncated: {reference_segmentation['search_truncated']}, "
            f"limit: {reference_segmentation['search_limit']})."
        ),
        f"Status: **{payload['status']}**. Match ambiguities: {match_ambiguous_count}. "
        f"Segmentation-ambiguous samples: {segmentation_ambiguous_count}.",
        "Retrieval costs rank candidates and are not correctness probabilities.",
        "",
        "| Sample | Best reference | Match status | Segmentation status | Cost | "
        "Match reason | Ambiguity sources | Pair report |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for item in samples:
        markdown.append(
            f"| {item['sample_id']} | {item['reference_id']} | {item['status']} "
            f"| {item['segmentation_status']} | {item['cost']:.4f} "
            f"| {item['match_reason']} | {_ambiguity_text(item['ambiguity_sources'])} "
            f"| [{item['report_dir']}/report.html]({item['report_dir']}/report.html) |"
        )
    markdown.extend(["", "## Ranked alternatives", ""])
    for item in samples:
        markdown.append(f"### {item['sample_id']}")
        for alternative in item["alternatives"]:
            markdown.append(
                f"- #{alternative['rank']} `{alternative['reference_id']}` "
                f"cost {alternative['cost']:.4f}"
            )
        markdown.append("")
    (root / "summary.md").write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")

    rows = []
    for item in samples:
        alternatives = "".join(
            f"<li>#{entry['rank']} {html.escape(entry['reference_id'])}: {entry['cost']:.4f}</li>"
            for entry in item["alternatives"]
        )
        report_link = html.escape(item["report_dir"] + "/report.html", quote=True)
        sample_link = html.escape(item["sample_image"], quote=True)
        matched_link = html.escape(item["matched_reference_image"], quote=True)
        alternatives_link = html.escape(item["alternatives_image"], quote=True)
        sample_id = html.escape(item["sample_id"])
        rows.append(
            "<tr>"
            f'<td><a href="{sample_link}"><img src="{sample_link}" width="100" '
            f'alt="{sample_id}"></a></td>'
            f'<td><a href="{matched_link}"><img src="{matched_link}" width="100" '
            f'alt="{html.escape(item["reference_id"])}"></a><br>'
            f"{html.escape(item['reference_id'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item['segmentation_status'])}</td>"
            f"<td>{item['cost']:.4f}</td>"
            f"<td>{html.escape(item['match_reason'])}</td>"
            f"<td>{html.escape(_ambiguity_text(item['ambiguity_sources']))}</td>"
            f'<td><a href="{report_link}">pair report</a></td>'
            f'<td><a href="{alternatives_link}"><img src="{alternatives_link}" '
            f'width="180" alt="ranked alternatives"></a>'
            f"<details><summary>scores</summary><ol>{alternatives}</ol></details></td>"
            "</tr>"
        )
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><meta charset="utf-8">',
            f"<title>Automatic comparison: {html.escape(Path(sample_path).name)}</title>",
            "<body>",
            "<h1>Automatic comparison summary</h1>",
            f"<p>Reference: {html.escape(Path(reference_path).name)}</p>",
            f"<p>Sample: {html.escape(Path(sample_path).name)}; "
            f"{expected_count} glyphs requested; status {payload['status']}.</p>",
            f"<p>Match ambiguities: {match_ambiguous_count}; "
            f"segmentation-ambiguous samples: {segmentation_ambiguous_count}</p>",
            f"<p>Reference segmentation: {reference_segmentation['status']}; "
            f"profile {html.escape(reference_segmentation['reference_profile'])}; "
            f"{reference_segmentation['row_count']} rows, "
            f"{reference_segmentation['candidate_count']} candidates; "
            f"search truncated {reference_segmentation['search_truncated']} "
            f"(limit {reference_segmentation['search_limit']}).</p>",
            "<table><thead><tr><th>Sample</th><th>Best reference</th>"
            "<th>Match status</th><th>Segmentation status</th><th>Cost</th>"
            "<th>Match reason</th><th>Ambiguity sources</th>"
            "<th>Report</th><th>Alternatives</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            "</body></html>",
        ]
    )
    (root / "index.html").write_text(document + "\n", encoding="utf-8")
    return BatchComparisonReport(root, payload)
