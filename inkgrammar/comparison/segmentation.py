from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np
import rasterregions as rr
from PIL import Image, ImageDraw
from rasterregions import Component, MaskMethod, MaskMode, MaskResult
from rasterregions.io import spaced_hypotheses_payload, write_spaced_hypotheses

from .models import CropRect


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


@dataclass(frozen=True)
class ReferenceSheetProfile:
    name: Literal["alphabet", "generic"] = "alphabet"
    include_rows: tuple[int, ...] | None = None
    exclude_rows: tuple[int, ...] = ()

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



@dataclass(frozen=True)
class ReferenceGallery:
    source_path: Path
    source_sha256: str
    candidates: tuple[SegmentedGlyph, ...]
    mask_result: MaskResult
    segmentation: PageSegmentation


class SegmentationError(ValueError):
    def __init__(self, message: str, diagnostic_path: Path | None = None):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path


extract_page_ink_mask = rr.foreground_mask
RowBand = rr.RowBand
detect_ink_rows = rr.detect_rows


REFERENCE_SEGMENTATION_VERSION = "3"


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
            right.bbox.x0 - left.bbox.x1
            for left, right in zip(regions, regions[1:], strict=False)
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
        image = result.image.crop(
            (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)
        )
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


def _write_reference_diagnostics(
    result: rr.DetectionResult,
    segmentation: PageSegmentation,
    hypotheses: rr.SpacedSegmentationHypotheses,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(result.clean_mask, output_dir / "reference-mask.png")
    _write_overlay(result.image, segmentation.candidates, output_dir / "reference-candidates.png")

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
            candidate
            for candidate in segmentation.candidates
            if candidate.row_index == row_index
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
        "hypothesis_count": segmentation.hypothesis_count,
        "search_truncated": segmentation.search_truncated,
        "reference_profile": segmentation.reference_profile,
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
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reference-segmentation-hypotheses.json").write_text(
        json.dumps(spaced_hypotheses_payload(hypotheses), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def segment_reference_gallery(
    image_path: str | Path,
    *,
    ink_mode: MaskMode = "auto",
    output_dir: str | Path | None = None,
    reference_profile: str | ReferenceSheetProfile = "alphabet",
) -> ReferenceGallery:
    """Detect anonymous reference glyphs with an application-level row profile."""
    source_path = Path(image_path)
    profile = _coerce_reference_profile(reference_profile)
    hypotheses = rr.detect_spaced_region_hypotheses(source_path, mode=ink_mode)
    result = hypotheses.selected
    override_rows = (
        profile.include_rows if profile.include_rows is not None else profile.exclude_rows
    )
    if override_rows and max(override_rows) >= len(result.rows):
        raise ValueError("reference profile names a row that was not detected")

    row_diagnostics, included_regions, candidate_warnings, ambiguous = _classify_reference_rows(
        result, profile
    )
    candidates = list(_adapt_candidates(result, "ref", included_regions, candidate_warnings))
    warnings = list(result.warnings)
    warnings.extend(
        f"Reference row {row.row_index}: {warning}"
        for row in row_diagnostics
        for warning in row.warnings
    )
    diagnostics = result.diagnostics
    unassigned_fraction = diagnostics.unassigned_ink_fraction if diagnostics else 0.0
    if unassigned_fraction > 0.02:
        warnings.append(f"Unassigned reference ink fraction is {unassigned_fraction:.3f}.")

    mask_shapes = []
    for candidate in candidates:
        page_mask = np.zeros(result.clean_mask.shape, dtype=bool)
        x0, y0 = candidate.bbox.x, candidate.bbox.y
        page_mask[y0 : y0 + candidate.bbox.height, x0 : x0 + candidate.bbox.width] = candidate.mask
        mask_shapes.append(page_mask)
    contaminated: set[int] = set()
    for left in range(len(mask_shapes)):
        for right in range(left + 1, len(mask_shapes)):
            if np.any(mask_shapes[left] & mask_shapes[right]):
                contaminated.update((left, right))
    for index in contaminated:
        message = "Candidate mask overlaps another candidate's assigned foreground."
        candidates[index] = replace(
            candidates[index], warnings=(*candidates[index].warnings, message)
        )
        warnings.append(f"{candidates[index].glyph_id}: {message}")
    if contaminated:
        ambiguous = True

    status: Literal["ok", "ambiguous", "failed"]
    if not candidates:
        status = "failed"
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
        hypothesis_count=len(hypotheses.page_hypotheses),
        search_truncated=hypotheses.truncated,
    )
    if output_dir is not None:
        _write_reference_diagnostics(result, segmentation, hypotheses, Path(output_dir))
    if not candidates:
        diagnostic = Path(output_dir) / "reference-candidates.png" if output_dir else None
        raise SegmentationError("No reference glyph candidates were detected.", diagnostic)
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
            [row.count for row in page.row_hypotheses]
            for page in hypotheses.page_hypotheses
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
