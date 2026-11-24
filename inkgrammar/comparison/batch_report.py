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
    CENTERLINE_WEIGHT,
    FILL_WEIGHT,
    FITTED_STEERING_WEIGHT,
    HEADING_WEIGHT,
    RASTER_WEIGHT,
    ROUTE_DESCRIPTOR_WEIGHT,
    ROUTE_STRUCTURE_WEIGHT,
    ROUTE_TOPOLOGY_WEIGHT,
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
    leading_reference_id: str
    accepted_reference_id: str | None
    acceptance_reasons: tuple[str, ...]
    reference_label: str | None
    status: str
    segmentation_status: str
    segmentation_warnings: tuple[str, ...]
    score: MatchScore
    alternatives: tuple[MatchScore, ...]
    search_metadata: dict[str, Any]
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
    selection = score.route_pair_selection
    route_pair = (
        None
        if selection is None
        else {
            "reference_route_rank": selection.score.reference_rank,
            "sample_route_rank": selection.score.sample_rank,
            "ambiguous": selection.ambiguous,
            "eligible": selection.score.eligible,
            "total_cost": selection.score.total_cost,
            "cost_breakdown": selection.score.cost_breakdown,
        }
    )
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
        "route_descriptor_cost": score.route_descriptor_cost,
        "route_topology_cost": score.route_topology_cost,
        "identity_status": score.identity_status,
        "identity_eligible": score.identity_eligible,
        "identity_cost": score.identity_cost,
        "required_rotation_deg": score.required_rotation_deg,
        "orientation_rejection_reason": score.orientation_rejection_reason,
        "stroke_pattern_cost": score.stroke_pattern_cost,
        "missing_evidence_penalty": score.missing_evidence_penalty,
        "route_evidence_status": score.route_evidence_status,
        "route_search_truncated": score.route_search_truncated,
        "route_warning": score.route_warning,
        "route_search_failure": score.route_search_failure,
        "feature_coverage": {
            "available_features": list(score.evidence.available_features),
            "unavailable_features": list(score.evidence.unavailable_features),
        },
        "route_pair_selection": route_pair,
    }


def match_reason(score: MatchScore, alternatives: tuple[MatchScore, ...]) -> str:
    if score.identity_eligible is False:
        rotation = score.required_rotation_deg
        detail = f"Required rotation: {rotation:+.1f} degrees. " if rotation is not None else ""
        reason = score.orientation_rejection_reason or "orientation_ineligible"
        return f"Candidate rejected by orientation gate ({reason}). {detail}"
    if not alternatives:
        return "Only available reference candidate."
    runner_up = alternatives[0]
    weights = {
        "raster": ("raster_cost", RASTER_WEIGHT),
        "aspect": ("aspect_cost", ASPECT_WEIGHT),
        "fill": ("fill_cost", FILL_WEIGHT),
        "topology": ("topology_cost", TOPOLOGY_WEIGHT),
        "centerline": ("centerline_cost", CENTERLINE_WEIGHT),
        "heading": ("heading_cost", HEADING_WEIGHT),
        "fitted steering": ("fitted_steering_cost", FITTED_STEERING_WEIGHT),
        "route structure": ("route_structure_cost", ROUTE_STRUCTURE_WEIGHT),
        "route descriptor": ("route_descriptor_cost", ROUTE_DESCRIPTOR_WEIGHT),
        "route topology": ("route_topology_cost", ROUTE_TOPOLOGY_WEIGHT),
        "stroke pattern": ("stroke_pattern_cost", 1.0),
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
    search_metadata: dict[str, Any] | None = None,
    acceptance_reasons: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    sources = []
    if score.identity_eligible is False:
        rotation = score.required_rotation_deg
        rotation_detail = (
            f" Required rotation: {rotation:+.1f} degrees." if rotation is not None else ""
        )
        sources.append(
            {
                "source": "orientation_ineligible",
                "reason": (
                    f"Candidate rejected by orientation gate: "
                    f"{score.orientation_rejection_reason or 'identity ineligible'}."
                    f"{rotation_detail}"
                ),
            }
        )
    if search_metadata is not None and search_metadata["search_truncated"]:
        unrefined_count = (
            search_metadata["coarse_candidate_count"] - search_metadata["refined_candidate_count"]
        )
        sources.append(
            {
                "source": "reference_search_truncated",
                "reason": (
                    f"Reference refinement reached its limit of "
                    f"{search_metadata['refinement_limit']} candidates with "
                    f"{unrefined_count} unrefined candidates still able to affect the decision."
                ),
            }
        )
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
        directions = [
            writing.get(side, {}).get("orientation") or {} for side in ("reference", "sample")
        ]
        if any(item.get("interpretation") == "unresolved" for item in directions):
            sources.append(
                {
                    "source": "writing_direction_unresolved",
                    "reason": (
                        "Static glyph geometry does not determine historical stroke direction."
                    ),
                }
            )
    acceptance_source_aliases = {
        "reference_search_truncated": {"reference_search_truncated"},
        "retrieval_ambiguity": {"match_cost_tie"},
        "segmentation_uncertainty": {"segmentation_uncertainty"},
        "route_pair_ambiguous": {"writing_route_hypothesis", "writing_route_pair_tie"},
        "required_rotation_exceeds_identity_limit": {"orientation_ineligible"},
    }
    acceptance_reasons_text = {
        "reference_search_truncated": (
            "Reference refinement stopped before all competitive candidates were evaluated."
        ),
        "retrieval_ambiguity": (
            "Several reference candidates remain within the retrieval ambiguity margin."
        ),
        "segmentation_uncertainty": "Sample or reference segmentation is materially uncertain.",
        "missing_essential_route_evidence": (
            "Essential route evidence is unavailable for this match."
        ),
        "route_pair_unavailable": "No joint route-pair selection was available.",
        "route_pair_ambiguous": "The selected route pair remains materially ambiguous.",
    }
    existing_sources = {source["source"] for source in sources}
    for reason in acceptance_reasons:
        aliases = acceptance_source_aliases.get(reason, set())
        if reason not in existing_sources and not aliases.intersection(existing_sources):
            sources.append(
                {
                    "source": reason,
                    "reason": acceptance_reasons_text[reason],
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
        search_metadata=result.search_metadata,
        acceptance_reasons=result.acceptance_reasons,
    )
    return {
        "sample_id": result.sample_id,
        "sample_bbox": result.sample_bbox.model_dump(mode="json"),
        "segmentation_confidence": result.sample_confidence,
        "reference_id": result.reference_id,
        "leading_reference_id": result.leading_reference_id,
        "accepted_reference_id": result.accepted_reference_id,
        "identity_status": (
            "accepted" if result.accepted_reference_id is not None else "candidate_diagnostic"
        ),
        "identity_label": (
            f"Accepted reference: {result.accepted_reference_id}"
            if result.accepted_reference_id is not None
            else f"Leading candidate (not accepted): {result.leading_reference_id}"
        ),
        "acceptance_reasons": list(result.acceptance_reasons),
        "reference_label": result.reference_label,
        "status": result.status,
        "match_status": result.status,
        "segmentation_status": result.segmentation_status,
        "match_reason": reason,
        "ambiguity_sources": ambiguity,
        "segmentation_warnings": list(result.segmentation_warnings),
        "cost": result.score.cost,
        "leading_match": match_score_payload(result.score),
        "alternatives": [match_score_payload(score) for score in result.alternatives],
        "search": result.search_metadata,
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
    refinement_limit: int,
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
    writing_direction_ambiguous_count = sum(
        any(
            source["source"] == "writing_direction_unresolved"
            for source in item["ambiguity_sources"]
        )
        for item in samples
    )
    payload: dict[str, Any] = {
        "status": (
            "ambiguous"
            if match_ambiguous_count
            or segmentation_ambiguous_count
            or writing_direction_ambiguous_count
            else "ok"
        ),
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
        "writing_direction_ambiguous_count": writing_direction_ambiguous_count,
        "reference_cache": {"hit": reference_cache_hit, "key": reference_sha256},
        "matching": {
            "cost_interpretation": "lower retrieval cost ranks ahead; it is not a probability",
            "refinement_limit": refinement_limit,
            "weights": {
                "raster": RASTER_WEIGHT,
                "aspect": ASPECT_WEIGHT,
                "fill": FILL_WEIGHT,
                "topology": TOPOLOGY_WEIGHT,
                "centerline": CENTERLINE_WEIGHT,
                "heading": HEADING_WEIGHT,
                "fitted_steering": FITTED_STEERING_WEIGHT,
                "route_structure": ROUTE_STRUCTURE_WEIGHT,
                "route_descriptor": ROUTE_DESCRIPTOR_WEIGHT,
                "route_topology": ROUTE_TOPOLOGY_WEIGHT,
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
        f"Acceptance decision: {reference_segmentation['decision']['status']}.",
        (
            f"Cleanup: {reference_segmentation['raw_component_count']} raw components; "
            f"{reference_segmentation['removed_page_artifact_count']} page artifacts and "
            f"{reference_segmentation['removed_micro_noise_count']} micro-noise removed; "
            f"{reference_segmentation['retained_secondary_count']} secondary marks retained, "
            f"{reference_segmentation['unassigned_secondary_count']} unassigned."
        ),
        (
            f"Anchor row counts: {reference_segmentation['anchor_row_counts']}; "
            f"selected row counts: {reference_segmentation['selected_row_counts']}; "
            f"foreground ownership valid: {reference_segmentation['foreground_ownership_valid']}."
        ),
        "Detailed reference diagnostics: [segmentation/reference-segmentation.md]("
        "segmentation/reference-segmentation.md).",
        f"Status: **{payload['status']}**. Match ambiguities: {match_ambiguous_count}. "
        f"Segmentation-ambiguous samples: {segmentation_ambiguous_count}; "
        f"writing directions unresolved: {writing_direction_ambiguous_count}.",
        "Retrieval costs rank candidates and are not correctness probabilities.",
        "",
        "| Sample | Reference decision | Match status | Segmentation status | Cost | "
        "Match reason | Ambiguity sources | Pair report |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for item in samples:
        markdown.append(
            f"| {item['sample_id']} | {item['identity_label']} | {item['status']} "
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
            f"{html.escape(item['identity_label'])}</td>"
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
            f"segmentation-ambiguous samples: {segmentation_ambiguous_count}; "
            f"writing directions unresolved: {writing_direction_ambiguous_count}</p>",
            f"<p>Reference segmentation: {reference_segmentation['status']}; "
            f"profile {html.escape(reference_segmentation['reference_profile'])}; "
            f"{reference_segmentation['row_count']} rows, "
            f"{reference_segmentation['candidate_count']} candidates; "
            f"search truncated {reference_segmentation['search_truncated']} "
            f"(limit {reference_segmentation['search_limit']}).</p>",
            f"<p>Acceptance decision: {reference_segmentation['decision']['status']}.</p>",
            f"<p>Cleanup: {reference_segmentation['raw_component_count']} raw components; "
            f"{reference_segmentation['removed_page_artifact_count']} page artifacts and "
            f"{reference_segmentation['removed_micro_noise_count']} micro-noise removed; "
            f"{reference_segmentation['retained_secondary_count']} secondary marks retained, "
            f"{reference_segmentation['unassigned_secondary_count']} unassigned.</p>",
            f"<p>Anchor row counts: {reference_segmentation['anchor_row_counts']}; "
            f"selected row counts: {reference_segmentation['selected_row_counts']}; "
            f"foreground ownership valid: {reference_segmentation['foreground_ownership_valid']}. "
            '<a href="segmentation/reference-segmentation.md">Detailed diagnostics</a></p>',
            "<table><thead><tr><th>Sample</th><th>Reference decision</th>"
            "<th>Match status</th><th>Segmentation status</th><th>Cost</th>"
            "<th>Match reason</th><th>Ambiguity sources</th>"
            "<th>Report</th><th>Alternatives</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            "</body></html>",
        ]
    )
    (root / "index.html").write_text(document + "\n", encoding="utf-8")
    return BatchComparisonReport(root, payload)
