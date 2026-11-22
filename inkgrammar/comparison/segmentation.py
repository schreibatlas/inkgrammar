from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import rasterregions as rr
from PIL import Image, ImageDraw
from rasterregions import Component, MaskMethod, MaskMode, MaskResult, SpacedSegmentationHypotheses
from rasterregions.io import spaced_hypotheses_payload, write_spaced_hypotheses

from .models import CropRect, WritingStructurePolicy


@dataclass(frozen=True)
class SegmentedGlyph:
    glyph_id: str
    bbox: CropRect
    row_index: int
    column_index: int
    reading_order: int
    segmentation_confidence: float
    image: Image.Image
    mask: np.ndarray
    label: str | None = None
    component_indices: tuple[int, ...] = ()
    primary_component_indices: tuple[int, ...] = ()
    secondary_component_indices: tuple[int, ...] = ()
    ambiguous_component_indices: tuple[int, ...] = ()
    secondary_mark_centroids: tuple[tuple[float, float], ...] = ()
    warnings: tuple[str, ...] = ()
    mask_method: MaskMethod = "grayscale"

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (self.bbox.height, self.bbox.width):
            raise ValueError("glyph mask dimensions must match its bounding box")
        if self.image.size != (self.bbox.width, self.bbox.height):
            raise ValueError("glyph image dimensions must match its bounding box")
        if not 0.0 <= self.segmentation_confidence <= 1.0:
            raise ValueError("glyph confidence must be between zero and one")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        object.__setattr__(self, "warnings", tuple(self.warnings))

        object.__setattr__(self, "primary_component_indices", tuple(self.primary_component_indices))
        object.__setattr__(
            self,
            "secondary_component_indices",
            tuple(self.secondary_component_indices),
        )
        object.__setattr__(
            self,
            "ambiguous_component_indices",
            tuple(self.ambiguous_component_indices),
        )
        object.__setattr__(self, "secondary_mark_centroids", tuple(self.secondary_mark_centroids))

    def expected_stroke_count(self, policy: WritingStructurePolicy | None = None) -> int:
        policy = policy or WritingStructurePolicy()
        return policy.preferred_body_strokes + len(self.secondary_component_indices)

    @property
    def component_role_sequence(self) -> tuple[str, ...]:
        primary = set(self.primary_component_indices)
        secondary = set(self.secondary_component_indices)
        ambiguous = set(self.ambiguous_component_indices)
        indices = sorted(set(self.component_indices) | primary | secondary | ambiguous)
        return tuple(
            "primary"
            if index in primary
            else "secondary"
            if index in secondary
            else "ambiguous"
            if index in ambiguous
            else "unassigned"
            for index in indices
        )


ALPHABET_ROW_COUNT_PATTERNS = (
    (12, 8, 6, 11, 10, 5, 10),
    (12, 8, 6, 11, 11, 5, 10),
)
REFERENCE_SEGMENTATION_POLICY_VERSION = "2"
REFERENCE_PROFILE_VERSION = "1"


@dataclass(frozen=True)
class ReferenceSheetProfile:
    name: Literal["alphabet", "generic"] = "generic"
    include_rows: tuple[int, ...] | None = None
    exclude_rows: tuple[int, ...] = ()
    allowed_row_count_patterns: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.include_rows is not None and self.exclude_rows:
            raise ValueError("include_rows and exclude_rows cannot both be set")
        rows = self.include_rows if self.include_rows is not None else self.exclude_rows
        if rows is not None and any(index < 0 for index in rows):
            raise ValueError("row indexes must be nonnegative")
        if self.name not in {"alphabet", "generic"}:
            raise ValueError("reference profile must be 'alphabet' or 'generic'")
        object.__setattr__(
            self,
            "include_rows",
            tuple(self.include_rows) if self.include_rows is not None else None,
        )
        object.__setattr__(self, "exclude_rows", tuple(self.exclude_rows))

        patterns = self.allowed_row_count_patterns
        if patterns is None:
            patterns = ALPHABET_ROW_COUNT_PATTERNS if self.name == "alphabet" else ()
        patterns = tuple(tuple(pattern) for pattern in patterns)
        if any(
            not pattern
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 1
                for count in pattern
            )
            for pattern in patterns
        ):
            raise ValueError("allowed row-count patterns must contain positive integer counts")
        object.__setattr__(self, "allowed_row_count_patterns", patterns)


@dataclass(frozen=True)
class ReferenceCountChangingRepair:
    row_index: int
    anchor_count: int
    selected_count: int
    repair_count: int
    split_anchor_indices: tuple[int, ...] = ()
    merged_anchor_groups: tuple[tuple[int, ...], ...] = ()
    strongly_supported: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_anchor_indices", tuple(self.split_anchor_indices))
        object.__setattr__(
            self,
            "merged_anchor_groups",
            tuple(tuple(group) for group in self.merged_anchor_groups),
        )


@dataclass(frozen=True)
class ReferenceSegmentationDecision:
    profile_name: str
    anchor_row_counts: tuple[int, ...]
    selected_row_counts: tuple[int, ...]
    recognized_layout: tuple[int, ...] | None
    detected_count: int
    accepted_count: int | None
    accepted: bool
    status: Literal["accepted", "rejected"]
    raw_component_count: int
    removed_page_artifact_count: int
    removed_micro_noise_count: int
    retained_secondary_count: int
    unassigned_secondary_count: int
    uncertain_secondary_count: int
    count_changing_repairs: tuple[ReferenceCountChangingRepair, ...]
    search_truncated: bool
    foreground_ownership_valid: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_row_counts", tuple(self.anchor_row_counts))
        object.__setattr__(self, "selected_row_counts", tuple(self.selected_row_counts))
        if self.recognized_layout is not None:
            object.__setattr__(self, "recognized_layout", tuple(self.recognized_layout))
        repairs = tuple(
            repair
            if isinstance(repair, ReferenceCountChangingRepair)
            else ReferenceCountChangingRepair(**repair)
            for repair in self.count_changing_repairs
        )
        object.__setattr__(self, "count_changing_repairs", repairs)
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.accepted != (self.accepted_count is not None):
            raise ValueError("accepted_count must be set exactly for accepted decisions")
        if self.status != ("accepted" if self.accepted else "rejected"):
            raise ValueError("decision status must agree with accepted")

    @property
    def summary(self) -> str:
        if self.accepted:
            return f"Reference segmentation accepted: {self.accepted_count} candidates."
        lines = [
            "Reference segmentation rejected:",
            f"  anchor layout:   {list(self.anchor_row_counts)}",
            f"  selected layout: {list(self.selected_row_counts)}",
        ]
        lines.extend(f"  reason: {reason}" for reason in self.reasons)
        return "\n".join(lines)

@dataclass(frozen=True)
class ReferenceRowDiagnostic:
    row_index: int
    bbox: CropRect
    component_indices: tuple[int, ...]
    candidate_count: int
    median_width: float
    maximum_width: int
    median_height: float
    horizontal_occupancy: float
    median_spacing: float | None
    classification: Literal["glyph", "title", "metadata", "generic", "uncertain"]
    included: bool
    confidence: float
    reason: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageSegmentation:
    source_path: Path
    candidates: tuple[SegmentedGlyph, ...]
    expected_count: int | None = None
    status: Literal["ok", "ambiguous", "failed"] = "ok"
    warnings: tuple[str, ...] = ()
    reference_profile: str | None = None
    rows: tuple[ReferenceRowDiagnostic, ...] = ()
    unassigned_ink_fraction: float = 0.0
    hypothesis_count: int = 1
    search_truncated: bool = False
    search_limit: str | None = None
    anchor_region_count: int | None = None


@dataclass(frozen=True)
class ReferenceGallery:
    source_path: Path
    source_sha256: str
    candidates: tuple[SegmentedGlyph, ...]
    mask_result: MaskResult
    segmentation: PageSegmentation
    decision: ReferenceSegmentationDecision | None = None

    cleanup_diagnostics: dict[str, Any] | None = None

class SegmentationError(ValueError):
    def __init__(
        self,
        message: str,
        diagnostic_path: Path | None = None,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path
        self.details = dict(details or {})


extract_page_ink_mask = rr.foreground_mask
RowBand = rr.RowBand
detect_ink_rows = rr.detect_rows


REFERENCE_SEGMENTATION_VERSION = "6"


def _coerce_reference_profile(
    profile: str | ReferenceSheetProfile,
) -> ReferenceSheetProfile:
    if isinstance(profile, ReferenceSheetProfile):
        return profile
    if profile not in {"alphabet", "generic"}:
        raise ValueError("reference profile must be 'alphabet' or 'generic'")
    return ReferenceSheetProfile(name=profile)


def _classify_reference_rows(
    result: rr.DetectionResult,
    profile: ReferenceSheetProfile,
) -> tuple[
    tuple[ReferenceRowDiagnostic, ...],
    tuple[rr.Region, ...],
    dict[int, tuple[str, ...]],
    bool,
]:
    regions_by_row: dict[int, list[rr.Region]] = {index: [] for index in range(len(result.rows))}
    for region in result.regions:
        regions_by_row.setdefault(region.row_index, []).append(region)

    counts = [len(regions_by_row.get(index, ())) for index in range(len(result.rows))]
    repeated_counts = [count for count in counts if count > 1]
    typical_count = float(median(repeated_counts)) if repeated_counts else 0.0
    minimum_repeated_count = max(2, int(np.ceil(typical_count * 0.5))) if typical_count else 2
    typical_widths = [
        region.bbox.width
        for index in range(len(result.rows))
        if counts[index] >= minimum_repeated_count
        for region in regions_by_row[index]
    ]
    typical_width = float(median(typical_widths)) if typical_widths else 0.0
    row_diagnostics = []
    included_regions = []
    candidate_warnings: dict[int, tuple[str, ...]] = {}
    ambiguous = result.status != "ok"
    diagnostics = result.diagnostics
    unassigned_fraction = diagnostics.unassigned_ink_fraction if diagnostics else 0.0

    for row_index, row in enumerate(result.rows):
        regions = sorted(regions_by_row.get(row_index, ()), key=lambda item: item.reading_order)
        count = len(regions)
        widths = [region.bbox.width for region in regions]
        heights = [region.bbox.height for region in regions]
        median_width = float(median(widths)) if widths else 0.0
        maximum_width = max(widths, default=0)
        median_height = float(median(heights)) if heights else 0.0
        occupancy = min(1.0, sum(widths) / max(1, row.bbox.width))
        spacings = [
            right.bbox.x0 - left.bbox.x1 for left, right in zip(regions, regions[1:], strict=False)
        ]
        median_spacing = float(median(spacings)) if spacings else None
        included = True
        confidence = 1.0
        warnings: list[str] = []

        if profile.name == "generic":
            classification = "generic"
            reason = "The generic profile includes every detected row."
        elif profile.include_rows is not None:
            included = row_index in profile.include_rows
            classification = "glyph" if included else "metadata"
            reason = "Row inclusion was declared by the reference profile."
        elif row_index in profile.exclude_rows:
            included = False
            classification = "metadata"
            reason = "Row exclusion was declared by the reference profile."
        else:
            is_wide_word = count == 1 and maximum_width > max(
                2.5 * typical_width,
                2.5 * row.bbox.height,
            )
            if is_wide_word:
                included = False
                classification = "title"
                confidence = 0.95
                reason = "A single unusually wide region is classified as a title or word row."
            elif count >= minimum_repeated_count:
                classification = "glyph"
                confidence = min(1.0, 0.65 + 0.35 * count / max(typical_count, 1.0))
                reason = "The row matches the repeated glyph-row count distribution."
            elif count == 0 or not typical_count or count / typical_count <= 0.55:
                included = False
                classification = "metadata"
                confidence = 0.9 if count == 0 else 0.8
                reason = "The row is a short outlier from the repeated glyph-row distribution."
            else:
                included = False
                classification = "uncertain"
                confidence = 0.45
                reason = "The row count is near the adaptive repeated-row boundary."
                warnings.append("Reference row inclusion is uncertain.")
                ambiguous = True

        if profile.include_rows is not None or profile.exclude_rows:
            explicitly_selected = (
                row_index in profile.include_rows
                if profile.include_rows is not None
                else row_index not in profile.exclude_rows
            )
            included = explicitly_selected
            classification = "glyph" if included else "metadata"
            reason = "Row inclusion was declared by the reference profile."

        if included and count:
            row_width_median = max(median_width, 1.0)
            for region in regions:
                candidate_messages = []
                aspect_ratio = region.bbox.width / max(1, region.bbox.height)
                if region.bbox.width > max(3.0 * row_width_median, 4.0 * median_height):
                    candidate_messages.append("Candidate is unusually wide relative to its row.")
                if aspect_ratio > 5.0:
                    candidate_messages.append("Candidate aspect ratio is implausibly wide.")
                if region.confidence < 0.3:
                    candidate_messages.append("Candidate boundary confidence is low.")
                if candidate_messages:
                    candidate_warnings[region.reading_order] = tuple(candidate_messages)
                    warnings.extend(candidate_messages)
                    ambiguous = True

        row_diagnostics.append(
            ReferenceRowDiagnostic(
                row_index=row_index,
                bbox=CropRect(
                    x=row.bbox.x0,
                    y=row.bbox.y0,
                    width=row.bbox.width,
                    height=row.bbox.height,
                ),
                component_indices=row.component_indices,
                candidate_count=count,
                median_width=median_width,
                maximum_width=maximum_width,
                median_height=median_height,
                horizontal_occupancy=occupancy,
                median_spacing=median_spacing,
                classification=classification,
                included=included,
                confidence=confidence,
                reason=reason,
                warnings=tuple(warnings),
            )
        )
        if included:
            included_regions.extend(regions)

    if unassigned_fraction > 0.02:
        ambiguous = True
    return tuple(row_diagnostics), tuple(included_regions), candidate_warnings, ambiguous


def _isolated_candidate_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != pixels.shape[:2]:
        raise ValueError("candidate mask dimensions must match its image")
    pixels[~selected] = 255
    return Image.fromarray(pixels)


def _adapt_candidates(
    result: rr.DetectionResult,
    prefix: str,
    regions: tuple[rr.Region, ...] | list[rr.Region] | None = None,
    warnings_by_region: dict[int, tuple[str, ...]] | None = None,
) -> tuple[SegmentedGlyph, ...]:
    selected = result.regions if regions is None else regions
    columns: dict[int, int] = {}
    components_by_index = {component.index: component for component in result.components}
    candidates = []
    for reading_order, region in enumerate(selected):
        column = columns.get(region.row_index, 0)
        columns[region.row_index] = column + 1
        ys, xs = np.nonzero(region.mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bbox = CropRect(
            x=region.bbox.x0 + x0,
            y=region.bbox.y0 + y0,
            width=x1 - x0,
            height=y1 - y0,
        )
        mask = region.mask[y0:y1, x0:x1]
        image = result.image.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
        candidates.append(
            SegmentedGlyph(
                glyph_id=f"{prefix}-{reading_order:03d}",
                bbox=bbox,
                row_index=region.row_index,
                column_index=column,
                reading_order=reading_order,
                segmentation_confidence=region.confidence,
                image=_isolated_candidate_image(image, mask),
                mask=mask,
                component_indices=region.component_indices,
                mask_method=result.mask_result.method,
                primary_component_indices=region.primary_component_indices,
                secondary_component_indices=region.secondary_component_indices,
                ambiguous_component_indices=region.ambiguous_component_indices,
                secondary_mark_centroids=tuple(
                    (
                        (components_by_index[index].center_x - bbox.x) / bbox.width,
                        (components_by_index[index].center_y - bbox.y) / bbox.height,
                    )
                    for index in region.secondary_component_indices
                    if index in components_by_index
                ),
                warnings=(warnings_by_region or {}).get(region.reading_order, ()),
            )
        )
    return tuple(candidates)


def _candidate_record(candidate: SegmentedGlyph) -> dict:
    return {
        "id": candidate.glyph_id,
        "label": candidate.label,
        "row": candidate.row_index,
        "column": candidate.column_index,
        "reading_order": candidate.reading_order,
        "bbox": [
            candidate.bbox.x,
            candidate.bbox.y,
            candidate.bbox.x + candidate.bbox.width,
            candidate.bbox.y + candidate.bbox.height,
        ],
        "dimensions": {"width": candidate.bbox.width, "height": candidate.bbox.height},
        "confidence": round(candidate.segmentation_confidence, 4),
        "component_ids": list(candidate.component_indices),
        "primary_component_ids": list(candidate.primary_component_indices),
        "secondary_component_ids": list(candidate.secondary_component_indices),
        "ambiguous_component_ids": list(candidate.ambiguous_component_indices),
        "secondary_mark_centroids": [list(point) for point in candidate.secondary_mark_centroids],
        "mask_artifact": f"{candidate.glyph_id}-mask.png",
        "crop_artifact": f"{candidate.glyph_id}.png",
        "mask_method": candidate.mask_method,
        "warnings": list(candidate.warnings),
    }


def _write_overlay(
    image: Image.Image,
    candidates: tuple[SegmentedGlyph, ...] | list[SegmentedGlyph],
    output: Path,
) -> None:
    preview = image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for candidate in candidates:
        x0, y0 = candidate.bbox.x, candidate.bbox.y
        x1 = x0 + candidate.bbox.width - 1
        y1 = y0 + candidate.bbox.height - 1
        draw.rectangle((x0, y0, x1, y1), outline=(230, 45, 35), width=2)
        draw.text((x0, max(0, y0 - 12)), candidate.glyph_id, fill=(25, 75, 220))
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, format="PNG")


def _write_page_mask(mask: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output, format="PNG")

def _diagnostics_payload(diagnostics: rr.DetectionDiagnostics | None) -> dict[str, Any]:
    return asdict(diagnostics) if diagnostics is not None else {}


def _write_bbox_overlay(
    image: Image.Image,
    boxes: list[tuple[str, tuple[int, int, int, int]]],
    output: Path,
    color: tuple[int, int, int],
) -> None:
    preview = image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for label, (x0, y0, x1, y1) in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
        draw.text((x0, max(0, y0 - 12)), label, fill=color)
    preview.save(output, format="PNG")


def _write_cleanup_overlay(result: rr.DetectionResult, output: Path) -> None:
    preview = result.image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    assignments = {
        assignment.component_index: assignment.status
        for assignment in (
            result.diagnostics.secondary_mark_assignments if result.diagnostics else ()
        )
    }
    colors = {
        "primary": (35, 155, 55),
        "secondary": (35, 105, 220),
        "secondary-candidate": (35, 105, 220),
        "ambiguous": (230, 80, 200),
        "micro-noise": (225, 35, 35),
        "artifact": (125, 50, 170),
        "noise": (225, 35, 35),
    }
    for decision in result.component_decisions:
        assignment_status = assignments.get(decision.component_index)
        role = decision.role
        color = (
            (245, 185, 20)
            if assignment_status == "unassigned"
            else (230, 80, 200)
            if assignment_status == "ambiguous"
            else colors.get(role, (100, 100, 100))
        )
        box = decision.bbox
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=color, width=2)
        draw.text(
            (box.x0, max(0, box.y0 - 12)),
            f"c{decision.component_index}:{role}",
            fill=color,
        )
    legend = (
        ("primary body", colors["primary"]),
        ("retained detached mark", colors["secondary"]),
        ("removed micro-noise", colors["micro-noise"]),
        ("unassigned secondary", (245, 185, 20)),
    )
    draw.rectangle((3, 3, 188, 67), fill="white", outline="black")
    for index, (label, color) in enumerate(legend):
        draw.text((7, 7 + index * 14), label, fill=color)
    preview.save(output, format="PNG")


def _write_reference_report(
    result: rr.DetectionResult,
    segmentation: PageSegmentation,
    decision: ReferenceSegmentationDecision,
    output: Path,
) -> None:
    diagnostics = result.diagnostics
    search_limit = diagnostics.search_limit if diagnostics is not None else None
    repairs = decision.count_changing_repairs
    lines = [
        "# Reference segmentation",
        "",
        f"- Profile: {decision.profile_name}",
        f"- Raw connected components: {decision.raw_component_count}",
        f"- Page artifacts removed: {decision.removed_page_artifact_count}",
        f"- Micro-noise removed: {decision.removed_micro_noise_count}",
        f"- Plausible detached marks retained: {decision.retained_secondary_count}",
        f"- Unassigned secondary marks: {decision.unassigned_secondary_count}",
        f"- Uncertain secondary marks: {decision.uncertain_secondary_count}",
        f"- Anchor row counts: {list(decision.anchor_row_counts)}",
        f"- Selected row counts: {list(decision.selected_row_counts)}",
        f"- Detected candidates: {decision.detected_count}",
        f"- Accepted candidate count: {decision.accepted_count}",
        f"- Count-changing repairs: {len(repairs)}",
        f"- Search truncated: {'yes' if decision.search_truncated else 'no'}"
        + (f" ({search_limit})" if search_limit else ""),
        f"- Foreground ownership valid: {'yes' if decision.foreground_ownership_valid else 'no'}",
        f"- Status: {decision.status}",
    ]
    if repairs:
        lines.extend(("", "## Count-changing repair evidence", ""))
        lines.extend(f"- {repair}" for repair in repairs)
    if decision.reasons:
        lines.extend(("", "## Blocking reasons", ""))
        lines.extend(f"- {reason}" for reason in decision.reasons)
    if decision.warnings or segmentation.warnings:
        lines.extend(("", "## Warnings", ""))
        lines.extend(f"- {warning}" for warning in (*decision.warnings, *segmentation.warnings))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _write_reference_diagnostics(
    result: rr.DetectionResult,
    segmentation: PageSegmentation,
    hypotheses: rr.SpacedSegmentationHypotheses,
    output_dir: Path,
    decision: ReferenceSegmentationDecision,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(result.clean_mask, output_dir / "reference-mask.png")
    _write_overlay(result.image, segmentation.candidates, output_dir / "reference-candidates.png")
    _write_bbox_overlay(
        result.image,
        [
            (
                f"anchor-{region.reading_order:03d}",
                (region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1),
            )
            for region in result.regions
        ],
        output_dir / "reference-anchor-regions.png",
        (35, 105, 220),
    )
    _write_bbox_overlay(
        result.image,
        [
            (
                candidate.glyph_id,
                (
                    candidate.bbox.x,
                    candidate.bbox.y,
                    candidate.bbox.x + candidate.bbox.width,
                    candidate.bbox.y + candidate.bbox.height,
                ),
            )
            for candidate in segmentation.candidates
        ],
        output_dir / "reference-selected-regions.png",
        (230, 45, 35),
    )
    _write_cleanup_overlay(result, output_dir / "reference-cleanup-overlay.png")

    rows_preview = result.image.convert("RGB")
    rows_draw = ImageDraw.Draw(rows_preview)
    for row in segmentation.rows:
        color = (35, 145, 55) if row.included else (130, 130, 130)
        if row.classification == "uncertain":
            color = (230, 130, 20)
        x0, y0 = row.bbox.x, row.bbox.y
        x1, y1 = x0 + row.bbox.width - 1, y0 + row.bbox.height - 1
        rows_draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        rows_draw.text(
            (x0, max(0, y0 - 12)),
            f"row-{row.row_index:02d} {row.classification} n={row.candidate_count}",
            fill=color,
        )
    rows_preview.save(output_dir / "reference-rows.png", format="PNG")

    components_preview = result.image.convert("RGB")
    components_draw = ImageDraw.Draw(components_preview)
    for component in result.components:
        box = component.bbox
        components_draw.rectangle(
            (box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(30, 130, 210), width=1
        )
    components_preview.save(output_dir / "reference-components.png", format="PNG")

    cell_width, cell_height = 110, 112
    rows_by_index = {row.row_index: row for row in segmentation.rows if row.included}
    candidates_by_row = {
        row_index: [
            candidate for candidate in segmentation.candidates if candidate.row_index == row_index
        ]
        for row_index in rows_by_index
    }
    gallery_width = max(
        cell_width,
        max((len(items) for items in candidates_by_row.values()), default=1) * cell_width,
    )
    gallery_height = max(40, sum(28 + cell_height for _ in rows_by_index))
    gallery = Image.new("RGB", (gallery_width, gallery_height), "white")
    gallery_draw = ImageDraw.Draw(gallery)
    y = 0
    for row_index, row in rows_by_index.items():
        gallery_draw.text(
            (8, y + 5),
            f"row-{row_index:02d} ({row.candidate_count} candidates)",
            fill="black",
        )
        y += 28
        for column, candidate in enumerate(candidates_by_row[row_index]):
            thumbnail = candidate.image.convert("RGB")
            thumbnail.thumbnail((cell_width - 12, cell_height - 28), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - thumbnail.width) // 2
            gallery.paste(thumbnail, (x, y))
            gallery_draw.text(
                (column * cell_width + 6, y + cell_height - 24),
                candidate.glyph_id,
                fill="black",
            )
        y += cell_height
    gallery.save(output_dir / "reference-gallery.png", format="PNG")

    for candidate in segmentation.candidates:
        candidate.image.save(output_dir / f"{candidate.glyph_id}.png", format="PNG")
        Image.fromarray(candidate.mask.astype(np.uint8) * 255).save(
            output_dir / f"{candidate.glyph_id}-mask.png", format="PNG"
        )

    row_records = [
        {
            "row": row.row_index,
            "bbox": row.bbox.model_dump(mode="json"),
            "component_ids": list(row.component_indices),
            "candidate_count": row.candidate_count,
            "median_width": row.median_width,
            "maximum_width": row.maximum_width,
            "median_height": row.median_height,
            "horizontal_occupancy": row.horizontal_occupancy,
            "median_spacing": row.median_spacing,
            "classification": row.classification,
            "included": row.included,
            "excluded_reason": None if row.included else row.reason,
            "confidence": round(row.confidence, 4),
            "warnings": list(row.warnings),
        }
        for row in segmentation.rows
    ]
    payload = {
        "source_image": segmentation.source_path.name,
        "algorithm_version": REFERENCE_SEGMENTATION_VERSION,
        "segmentation_policy_version": REFERENCE_SEGMENTATION_POLICY_VERSION,
        "reference_profile_version": REFERENCE_PROFILE_VERSION,
        "cleanup_policy_version": REFERENCE_SEGMENTATION_POLICY_VERSION,
        "hypothesis_count": segmentation.hypothesis_count,
        "search_truncated": segmentation.search_truncated,
        "reference_profile": segmentation.reference_profile,
        "search_limit": segmentation.search_limit,
        "anchor_region_count": segmentation.anchor_region_count,
        "row_count": len(segmentation.rows),
        "rows": row_records,
        "per_row_candidate_count": {
            str(row.row_index): row.candidate_count for row in segmentation.rows
        },
        "excluded_rows": [row for row in row_records if not row["included"]],
        "candidate_count": len(segmentation.candidates),
        "unassigned_ink_fraction": segmentation.unassigned_ink_fraction,
        "status": segmentation.status,
        "warnings": list(segmentation.warnings),
        "candidates": [_candidate_record(candidate) for candidate in segmentation.candidates],
        "cleanup_diagnostics": _diagnostics_payload(result.diagnostics),
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reference-segmentation-hypotheses.json").write_text(
        json.dumps(spaced_hypotheses_payload(hypotheses), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_spaced_hypotheses(hypotheses, output_dir / "reference-hypotheses")
    decision_payload = asdict(decision)
    decision_payload["summary"] = decision.summary
    (output_dir / "reference-segmentation-decision.json").write_text(
        json.dumps(decision_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reference_report(
        result,
        segmentation,
        decision,
        output_dir / "reference-segmentation.md",
    )


def _reference_ownership_error(result: rr.DetectionResult) -> str | None:
    regions = result.regions
    if result.status == "failed":
        return "Rasterregions marked the reference segmentation as failed."
    if not regions:
        return "Rasterregions returned no reference regions."
    reading_order = [region.reading_order for region in regions]
    if reading_order != sorted(reading_order) or len(set(reading_order)) != len(reading_order):
        return "Reference regions do not have stable reading-order identifiers."

    owned = np.zeros_like(result.clean_mask, dtype=bool)
    for region in regions:
        if not 0 <= region.row_index < len(result.rows):
            return "A reference region does not belong to a detected row."
        if not np.any(region.mask):
            return "A reference region has an empty foreground mask."
        box = region.bbox
        if box.x1 > owned.shape[1] or box.y1 > owned.shape[0]:
            return "A reference region exceeds the page bounds."
        view = owned[box.y0 : box.y1, box.x0 : box.x1]
        if np.any(view & region.mask):
            return "Reference regions have overlapping foreground ownership."
        if np.any(region.mask & ~result.clean_mask[box.y0 : box.y1, box.x0 : box.x1]):
            return "A reference region contains foreground outside the cleaned page mask."
        view |= region.mask

    unassigned = np.asarray(result.unassigned_mask, dtype=bool)
    if np.any(owned & unassigned) or not np.array_equal(owned | unassigned, result.clean_mask):
        return "Reference regions do not conserve cleaned foreground ownership."
    if result.diagnostics is not None:
        actual_fraction = float(unassigned.sum()) / max(1, int(result.clean_mask.sum()))
        if not np.isclose(actual_fraction, result.diagnostics.unassigned_ink_fraction):
            return "Cleaned foreground ownership disagrees with rasterregions diagnostics."
    return None


def _region_row_index(region) -> int:
    if hasattr(region, "row_index"):
        return region.row_index
    return region.frame.row_index


def _region_identity_signature(regions, components) -> tuple:
    component_by_index = {component.index: component for component in components}
    signatures = []
    for region in regions:
        mask = np.zeros(region.mask.shape, dtype=bool)
        for component_index in region.primary_component_indices:
            component = component_by_index[component_index]
            local_y = component.ys - region.bbox.y0
            local_x = component.xs - region.bbox.x0
            inside = (
                (local_y >= 0)
                & (local_y < mask.shape[0])
                & (local_x >= 0)
                & (local_x < mask.shape[1])
            )
            mask[local_y[inside], local_x[inside]] = True
        if not region.primary_component_indices:
            mask = np.asarray(region.mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        signatures.append(
            (
                _region_row_index(region),
                (
                    region.bbox.x0 + x0,
                    region.bbox.y0 + y0,
                    region.bbox.x0 + x1,
                    region.bbox.y0 + y1,
                ),
                tuple(region.primary_component_indices),
                mask[y0:y1, x0:x1].tobytes(),
            )
        )
    return tuple(signatures)


def _strongly_supported_split(row, count_delta: int) -> bool:
    if count_delta <= 0 or count_delta != len(row.split_anchor_indices):
        return False
    if row.repair_count != count_delta or row.merged_anchor_groups or row.truncated_counts:
        return False
    evidence = {
        item.anchor_index: item
        for item in row.split_evidence
        if item.anchor_index in row.split_anchor_indices
    }
    return len(evidence) == count_delta and all(item.accepted for item in evidence.values())


def evaluate_reference_segmentation(
    result: rr.DetectionResult,
    segmentation: PageSegmentation,
    hypotheses: rr.SpacedSegmentationHypotheses,
    profile: ReferenceSheetProfile,
    *,
    ownership_error: str | None = None,
) -> ReferenceSegmentationDecision:
    diagnostics = result.diagnostics
    included_rows = tuple(row for row in segmentation.rows if row.included)
    selected_counts = tuple(row.candidate_count for row in included_rows)
    selected_by_row = {row.row_index: row.candidate_count for row in included_rows}
    package_rows = {
        row.row_index: row for row in diagnostics.row_segmentation
    } if diagnostics is not None else {}
    anchor_counts = tuple(
        package_rows[row.row_index].anchor_count
        if row.row_index in package_rows and package_rows[row.row_index].anchor_count is not None
        else row.candidate_count
        for row in included_rows
    )
    patterns = profile.allowed_row_count_patterns or ()
    selected_layout = next((pattern for pattern in patterns if pattern == selected_counts), None)
    anchor_layout = next((pattern for pattern in patterns if pattern == anchor_counts), None)
    recognized_layout = selected_layout or anchor_layout

    repairs = []
    for row in included_rows:
        package_row = package_rows.get(row.row_index)
        if package_row is None:
            continue
        anchor_count = (
            package_row.anchor_count
            if package_row.anchor_count is not None
            else selected_by_row[row.row_index]
        )
        selected_count = selected_by_row[row.row_index]
        count_delta = selected_count - anchor_count
        if count_delta or package_row.repair_count:
            repairs.append(
                ReferenceCountChangingRepair(
                    row_index=row.row_index,
                    anchor_count=anchor_count,
                    selected_count=selected_count,
                    repair_count=package_row.repair_count,
                    split_anchor_indices=package_row.split_anchor_indices,
                    merged_anchor_groups=package_row.merged_anchor_groups,
                    strongly_supported=_strongly_supported_split(package_row, count_delta),
                )
            )

    assignments = diagnostics.secondary_mark_assignments if diagnostics is not None else ()
    uncertain_assignments = tuple(
        assignment
        for assignment in assignments
        if assignment.status == "ambiguous"
        and (assignment.bbox_expansion_ratio is None or assignment.bbox_expansion_ratio > 1.0)
    )
    retained_secondary = {
        index for region in result.regions for index in region.secondary_component_indices
    }
    retained_secondary.update(
        assignment.component_index
        for assignment in assignments
        if assignment.status == "assigned"
    )
    unassigned_secondary_count = sum(
        assignment.status == "unassigned" for assignment in assignments
    )
    ownership_error = ownership_error or _reference_ownership_error(result)

    reasons = []
    warnings = []
    if not segmentation.candidates:
        reasons.append("no candidate regions")
    if ownership_error is not None:
        reasons.append(ownership_error)
    if any(candidate.ambiguous_component_indices for candidate in segmentation.candidates):
        reasons.append("candidate bounds depend on ambiguous component ownership")
    if uncertain_assignments:
        reasons.append("uncertain secondary ownership can change candidate bounds")
    if any(row.classification == "uncertain" and row.included for row in segmentation.rows):
        reasons.append("reference row inclusion is uncertain")
    if any(candidate.warnings for candidate in segmentation.candidates):
        reasons.append("candidate geometry is uncertain")

    if patterns:
        if recognized_layout is None:
            reasons.append("recognized alphabet layout has no compatible hypothesis")
        elif selected_layout is None:
            reasons.append(
                "selected candidate counts conflict with safe recognized layout "
                f"{list(recognized_layout)}"
            )
    if repairs and any(not repair.strongly_supported for repair in repairs):
        reasons.append("unresolved count-changing split/merge ambiguity")

    selected_signature = _region_identity_signature(result.regions, result.components)
    competing_identity = any(
        _region_identity_signature(page.regions, result.components) != selected_signature
        for page in hypotheses.page_hypotheses[1:]
    )
    if competing_identity:
        reasons.append("segmentation hypotheses disagree on candidate identity")
    identity_unstable = (
        selected_counts != anchor_counts
        and any(not repair.strongly_supported for repair in repairs)
    ) or bool(uncertain_assignments) or competing_identity
    search_truncated = hypotheses.truncated or bool(diagnostics and diagnostics.search_truncated)
    if search_truncated and identity_unstable:
        reasons.append("segmentation search was truncated while candidate identity was unstable")

    component_decisions = result.component_decisions
    raw_component_count = (
        diagnostics.raw_component_count
        if diagnostics is not None
        else len(component_decisions)
    )
    removed_page_artifact_count = sum(
        decision.action == "removed" and decision.role == "artifact"
        for decision in component_decisions
    )
    removed_micro_noise_count = sum(
        decision.action == "removed" and decision.role == "micro-noise"
        for decision in component_decisions
    )
    if removed_micro_noise_count:
        warnings.append(
            f"Rasterregions removed {removed_micro_noise_count} micro-noise components."
        )
    if unassigned_secondary_count:
        warnings.append(
            f"{unassigned_secondary_count} plausible secondary marks remain unassigned."
        )
    if search_truncated and not identity_unstable:
        warnings.append(
            "Segmentation search was truncated, but selected candidate identity is stable."
        )

    accepted = not reasons
    return ReferenceSegmentationDecision(
        profile_name=profile.name,
        anchor_row_counts=anchor_counts,
        selected_row_counts=selected_counts,
        recognized_layout=recognized_layout,
        detected_count=len(segmentation.candidates),
        accepted_count=len(segmentation.candidates) if accepted else None,
        accepted=accepted,
        status="accepted" if accepted else "rejected",
        raw_component_count=raw_component_count,
        removed_page_artifact_count=removed_page_artifact_count,
        removed_micro_noise_count=removed_micro_noise_count,
        retained_secondary_count=len(retained_secondary),
        unassigned_secondary_count=unassigned_secondary_count,
        uncertain_secondary_count=len(uncertain_assignments),
        count_changing_repairs=tuple(repairs),
        search_truncated=search_truncated,
        foreground_ownership_valid=ownership_error is None,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _write_reference_failure_diagnostics(
    source_path: Path,
    error: Exception,
    ink_mode: MaskMode,
    reference_profile: str,
    output_dir: Path,
) -> tuple[str, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source_image:
        image = source_image.convert("RGB")
    mask_result = rr.foreground_mask(source_path, mode=ink_mode)
    rows = rr.detect_rows(mask_result.mask)
    components = rr.components(mask_result.mask)
    _write_page_mask(mask_result.mask, output_dir / "reference-mask.png")

    rows_preview = image.copy()
    rows_draw = ImageDraw.Draw(rows_preview)
    for index, row in enumerate(rows):
        box = row.bbox
        rows_draw.rectangle(
            (box.x0, box.y0, box.x1 - 1, box.y1 - 1),
            outline=(35, 145, 55),
            width=2,
        )
        rows_draw.text(
            (box.x0, max(0, box.y0 - 12)),
            f"row-{index:02d}",
            fill=(35, 145, 55),
        )
    rows_preview.save(output_dir / "reference-rows.png", format="PNG")

    components_preview = image.copy()
    components_draw = ImageDraw.Draw(components_preview)
    for component in components:
        box = component.bbox
        components_draw.rectangle(
            (box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(30, 130, 210), width=1
        )
    components_preview.save(output_dir / "reference-components.png", format="PNG")
    _write_overlay(image, (), output_dir / "reference-candidates.png")
    gallery_preview = image.copy()
    ImageDraw.Draw(gallery_preview).text(
        (8, 8), "Reference segmentation failed", fill=(200, 30, 30)
    )
    gallery_preview.save(output_dir / "reference-gallery.png", format="PNG")

    dependency_diagnostics = getattr(error, "diagnostics", None)
    search_limit = getattr(error, "search_limit", None) or getattr(
        dependency_diagnostics, "search_limit", None
    )
    if search_limit is None and isinstance(error, rr.SegmentationSearchLimitError):
        error_text = str(error).lower()
        if "partition" in error_text:
            search_limit = "row-partition-budget"
        elif "count-candidate" in error_text:
            search_limit = "count-candidate-budget"
    search_truncated = isinstance(error, rr.SegmentationSearchLimitError) or bool(
        getattr(error, "search_truncated", False)
    )
    region_count = getattr(error, "region_count", None)
    anchor_region_count = getattr(error, "anchor_region_count", None)
    if dependency_diagnostics is not None:
        region_count = region_count or getattr(
            dependency_diagnostics, "selected_region_count", None
        )
        anchor_rows = getattr(dependency_diagnostics, "row_segmentation", ())
        if anchor_region_count is None and anchor_rows:
            anchor_region_count = sum(getattr(row, "anchor_count", 0) for row in anchor_rows)

    diagnostic_path = output_dir / "reference-segmentation-hypotheses.json"
    message_lines = [
        "Reference segmentation could not produce a safe gallery.",
        f"Detected {len(rows)} rows and {len(components)} connected components.",
    ]
    if region_count is not None or anchor_region_count is not None:
        message_lines.append(
            f"Detected {region_count or 0} regions and {anchor_region_count or 0} anchor regions."
        )
    if search_truncated:
        message_lines.append(
            f"Search was truncated by {search_limit or 'an unknown search budget'}."
        )
    message_lines.append(f"Rasterregions failed: {error}.")
    message_lines.append(f"Diagnostics: {diagnostic_path}")

    error_record = {"type": type(error).__name__, "message": str(error)}
    row_records = [
        {"row": index, "bbox": row.bbox.as_tuple(), "candidate_count": 0}
        for index, row in enumerate(rows)
    ]
    manifest = {
        "source_image": source_path.name,
        "algorithm_version": REFERENCE_SEGMENTATION_VERSION,
        "segmentation_policy_version": REFERENCE_SEGMENTATION_POLICY_VERSION,
        "reference_profile_version": REFERENCE_PROFILE_VERSION,
        "cleanup_policy_version": REFERENCE_SEGMENTATION_POLICY_VERSION,
        "reference_profile": reference_profile,
        "status": "failed",
        "candidate_count": 0,
        "row_count": len(rows),
        "rows": row_records,
        "per_row_candidate_count": {str(index): 0 for index in range(len(rows))},
        "search_truncated": search_truncated,
        "search_limit": search_limit,
        "anchor_region_count": anchor_region_count,
        "unassigned_ink_fraction": None,
        "component_count": len(components),
        "warnings": [str(error)],
        "dependency_error": error_record,
        "candidates": [],
    }
    payload = {
        "schema_version": 1,
        "segmentation": "spaced",
        "status": "failed",
        "diagnostics": {
            "row_count": len(rows),
            "component_count": len(components),
            "region_count": region_count,
            "anchor_region_count": anchor_region_count,
        },
        "search": {
            "truncated": search_truncated,
            "limit": search_limit,
            "failure_reason": str(error),
        },
        "error": error_record,
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    diagnostic_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hypotheses_dir = output_dir / "reference-hypotheses"
    hypotheses_dir.mkdir(parents=True, exist_ok=True)
    (hypotheses_dir / "failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return (
        "\n".join(message_lines),
        diagnostic_path,
        {
            "failure_phase": "rasterregions_detection",
            "reference_profile": reference_profile,
            "ink_mode": ink_mode,
            "row_count": len(rows),
            "component_count": len(components),
            "region_count": region_count,
            "anchor_region_count": anchor_region_count,
            "search_truncated": search_truncated,
            "search_limit": search_limit,
        },
    )


def _detect_reference_hypotheses(
    source_path: Path,
    ink_mode: MaskMode,
) -> tuple[SpacedSegmentationHypotheses, str | None]:
    try:
        return rr.detect_spaced_region_hypotheses(source_path, mode=ink_mode), None
    except ValueError as error:
        if str(error) != "row hypothesis count and boundaries must match its regions":
            raise

    anchor_result = rr.detect_spaced_regions(source_path, mode=ink_mode)
    anchor_diagnostics = anchor_result.diagnostics
    assert anchor_diagnostics is not None
    row_counts = tuple(row.anchor_count for row in anchor_diagnostics.row_segmentation)
    selected = rr.detect_spaced_regions(source_path, mode=ink_mode, row_counts=row_counts)
    diagnostics = selected.diagnostics
    assert diagnostics is not None
    hypotheses = SpacedSegmentationHypotheses(
        selected=selected,
        row_hypotheses=tuple(() for _ in selected.rows),
        page_hypotheses=(),
        truncated=diagnostics.search_truncated,
    )
    warning = (
        "Rasterregions hypothesis conversion failed its row-boundary invariant; "
        "segmentation was rerun using detected anchor counts."
    )
    return hypotheses, warning


def segment_reference_gallery(
    image_path: str | Path,
    *,
    ink_mode: MaskMode = "auto",
    output_dir: str | Path | None = None,
    reference_profile: str | ReferenceSheetProfile = "generic",
) -> ReferenceGallery:
    """Detect anonymous reference glyphs with an application-level row profile."""
    source_path = Path(image_path)
    profile = _coerce_reference_profile(reference_profile)
    try:
        hypotheses, fallback_warning = _detect_reference_hypotheses(source_path, ink_mode)
    except Exception as error:
        diagnostics_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="inkgrammar-reference-segmentation-"))
        )
        message, diagnostic_path, details = _write_reference_failure_diagnostics(
            source_path, error, ink_mode, profile.name, diagnostics_dir
        )
        raise SegmentationError(message, diagnostic_path, details=details) from error

    result = hypotheses.selected
    override_rows = (
        profile.include_rows if profile.include_rows is not None else profile.exclude_rows
    )
    if override_rows and max(override_rows) >= len(result.rows):
        raise ValueError("reference profile names a row that was not detected")

    ownership_error = _reference_ownership_error(result)
    row_diagnostics, included_regions, candidate_warnings, ambiguous = _classify_reference_rows(
        result, profile
    )
    candidates = (
        []
        if ownership_error
        else list(_adapt_candidates(result, "ref", included_regions, candidate_warnings))
    )
    warnings = list(result.warnings)
    warnings.extend(
        f"Reference row {row.row_index}: {warning}"
        for row in row_diagnostics
        for warning in row.warnings
    )
    if fallback_warning:
        ambiguous = True
        warnings.append(fallback_warning)
    diagnostics = result.diagnostics
    unassigned_fraction = diagnostics.unassigned_ink_fraction if diagnostics else 0.0
    search_truncated = hypotheses.truncated or bool(diagnostics and diagnostics.search_truncated)
    search_limit = diagnostics.search_limit if diagnostics else None
    if search_truncated:
        ambiguous = True
        warnings.append(
            f"Reference segmentation search was truncated by "
            f"{search_limit or 'a bounded search limit'}."
        )
    if unassigned_fraction > 0.02:
        warnings.append(f"Unassigned reference ink fraction is {unassigned_fraction:.3f}.")

    if not candidates and ownership_error is None:
        ownership_error = "No reference glyph candidates were detected."
    if ownership_error is None:
        mask_shapes = []
        for candidate in candidates:
            page_mask = np.zeros(result.clean_mask.shape, dtype=bool)
            x0, y0 = candidate.bbox.x, candidate.bbox.y
            page_mask[y0 : y0 + candidate.bbox.height, x0 : x0 + candidate.bbox.width] = (
                candidate.mask
            )
            mask_shapes.append(page_mask)
        for left in range(len(mask_shapes)):
            if any(np.any(mask_shapes[left] & other) for other in mask_shapes[left + 1 :]):
                ownership_error = "Candidate masks overlap another candidate's assigned foreground."
                warnings.append(ownership_error)
                break

    anchor_region_count = (
        sum(row.anchor_count for row in diagnostics.row_segmentation) if diagnostics else None
    )
    status: Literal["ok", "ambiguous", "failed"]
    if ownership_error:
        status = "failed"
        warnings.append(ownership_error)
    elif ambiguous:
        status = "ambiguous"
    else:
        status = "ok"
    segmentation = PageSegmentation(
        source_path=source_path,
        candidates=tuple(candidates),
        status=status,
        warnings=tuple(dict.fromkeys(warnings)),
        reference_profile=profile.name,
        rows=row_diagnostics,
        unassigned_ink_fraction=unassigned_fraction,
        hypothesis_count=max(len(hypotheses.page_hypotheses), int(fallback_warning is not None)),
        search_truncated=search_truncated,
        search_limit=search_limit,
        anchor_region_count=anchor_region_count,
    )
    decision = evaluate_reference_segmentation(
        result,
        segmentation,
        hypotheses,
        profile,
        ownership_error=ownership_error,
    )

    diagnostics_dir = Path(output_dir) if output_dir is not None else None
    if diagnostics_dir is None and status == "failed":
        diagnostics_dir = Path(tempfile.mkdtemp(prefix="inkgrammar-reference-segmentation-"))
    if diagnostics_dir is not None:
        _write_reference_diagnostics(result, segmentation, hypotheses, diagnostics_dir, decision)
    if status == "failed":
        assert diagnostics_dir is not None
        diagnostic_path = diagnostics_dir / "reference-segmentation-hypotheses.json"
        message_lines = [
            "Reference segmentation could not produce a safe gallery.",
            ownership_error or "Reference ownership is unsafe.",
            f"Detected {len(result.rows)} rows and {len(result.regions)} regions.",
        ]
        if search_truncated:
            message_lines.append(
                f"Search was truncated by {search_limit or 'a bounded search limit'}."
            )
        message_lines.append(f"Diagnostics: {diagnostic_path}")
        raise SegmentationError(
            "\n".join(message_lines),
            diagnostic_path,
            details={
                "failure_phase": "reference_ownership_validation",
                "reference_profile": profile.name,
                "row_count": len(result.rows),
                "region_count": len(result.regions),
                "candidate_count": len(candidates),
                "anchor_region_count": anchor_region_count,
                "search_truncated": search_truncated,
                "search_limit": search_limit,
                "unassigned_ink_fraction": unassigned_fraction,
            },
        )

    return ReferenceGallery(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        candidates=tuple(candidates),
        mask_result=MaskResult(
            mask=result.clean_mask,
            confidence=result.mask_result.confidence,
            method=result.mask_result.method,
            warnings=list(result.warnings),
        ),
        segmentation=segmentation,
        decision=decision,
        cleanup_diagnostics=_diagnostics_payload(result.diagnostics),
    )


def _write_sample_diagnostics(
    image: Image.Image,
    source_name: str,
    mask: np.ndarray,
    components: list[Component],
    candidates: tuple[SegmentedGlyph, ...],
    expected_count: int,
    status: str,
    warnings: list[str],
    boundaries: list[int] | None,
    row_y0: int,
    row_y1: int,
    hypotheses: rr.SpacedSegmentationHypotheses,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(mask, output_dir / "sample-mask.png")
    component_preview = image.convert("RGB")
    component_draw = ImageDraw.Draw(component_preview)
    for component in components:
        component_draw.rectangle(
            (component.bbox.x0, component.bbox.y0, component.bbox.x1 - 1, component.bbox.y1 - 1),
            outline=(30, 130, 210),
            width=1,
        )
    component_preview.save(output_dir / "sample-components.png", format="PNG")
    _write_overlay(image, candidates, output_dir / "sample-candidates.png")
    boundary_preview = image.convert("RGB")
    boundary_draw = ImageDraw.Draw(boundary_preview)
    if boundaries is not None:
        for boundary in boundaries[1:-1]:
            boundary_draw.line((boundary, row_y0, boundary, row_y1), fill=(240, 25, 25), width=2)
    boundary_preview.save(output_dir / "sample-boundaries.png", format="PNG")
    payload = {
        "source_image": source_name,
        "expected_count": expected_count,
        "detected_count": len(candidates),
        "status": status,
        "warnings": warnings,
        "samples": [_candidate_record(candidate) for candidate in candidates],
        "boundaries": boundaries,
        "hypothesis_count": len(hypotheses.page_hypotheses),
        "page_allocations": [
            [row.count for row in page.row_hypotheses] for page in hypotheses.page_hypotheses
        ],
        "search_truncated": hypotheses.truncated,
        "search_limit": (
            hypotheses.selected.diagnostics.search_limit
            if hypotheses.selected.diagnostics
            else None
        ),
    }
    for filename in ("sample-segmentation.json", "sample-candidates.json"):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (output_dir / "sample-hypotheses.json").write_text(
        json.dumps(spaced_hypotheses_payload(hypotheses), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_spaced_hypotheses(hypotheses, output_dir / "sample-hypotheses")


def segment_sample_glyphs(
    image_path: str | Path,
    *,
    expected_count: int,
    ink_mode: MaskMode = "auto",
    output_dir: str | Path | None = None,
) -> list[SegmentedGlyph]:
    """Recover exactly ``expected_count`` glyph regions or raise for failure."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    source_path = Path(image_path)
    hypotheses = rr.detect_spaced_region_hypotheses(
        source_path,
        expected_count=expected_count,
        mode=ink_mode,
    )
    result = hypotheses.selected
    exact_page = (
        hypotheses.page_hypotheses[0]
        if hypotheses.page_hypotheses
        and hypotheses.page_hypotheses[0].count == expected_count
        and hypotheses.page_hypotheses[0].unassigned_ink_fraction == 0.0
        and all(region.body_count > 0 for region in hypotheses.page_hypotheses[0].regions)
        else None
    )
    candidates = _adapt_candidates(result, "sample")
    sample_status = (
        "ok" if exact_page is not None and len(candidates) == expected_count else "failed"
    )
    if output_dir is not None:
        selected_row_index = result.regions[0].row_index if result.regions else 0
        row = result.rows[selected_row_index] if result.rows else None
        boundaries = None
        if result.diagnostics and result.diagnostics.partition_boundaries:
            boundaries = [
                0,
                *(boundary.x for boundary in result.diagnostics.partition_boundaries),
                result.image.width,
            ]
        components = rr.components(result.clean_mask)
        _write_sample_diagnostics(
            result.image,
            source_path.name,
            result.clean_mask,
            components,
            candidates,
            expected_count,
            sample_status,
            list(result.warnings),
            boundaries,
            row.bbox.y0 if row else 0,
            row.bbox.y1 if row else result.image.height,
            hypotheses,
            Path(output_dir),
        )
    if sample_status != "ok" or len(candidates) != expected_count:
        diagnostic = Path(output_dir) / "sample-boundaries.png" if output_dir is not None else None
        raise SegmentationError(
            f"Unable to recover {expected_count} credible letter regions.", diagnostic
        )
    return list(candidates)
