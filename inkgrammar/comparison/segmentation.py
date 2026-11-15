from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterregions as rr
from PIL import Image, ImageDraw

from .models import CropRect, InkMaskResult, InkMode


@dataclass(frozen=True)
class Component:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    ys: np.ndarray
    xs: np.ndarray

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class RowBand:
    x0: int
    y0: int
    x1: int
    y1: int
    component_indices: tuple[int, ...]


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


@dataclass(frozen=True)
class PageSegmentation:
    source_path: Path
    candidates: tuple[SegmentedGlyph, ...]
    expected_count: int | None = None
    status: Literal["ok", "ambiguous", "failed"] = "ok"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceGallery:
    source_path: Path
    source_sha256: str
    candidates: tuple[SegmentedGlyph, ...]
    mask_result: InkMaskResult
    segmentation: PageSegmentation


class SegmentationError(ValueError):
    def __init__(self, message: str, diagnostic_path: Path | None = None):
        super().__init__(message)
        self.diagnostic_path = diagnostic_path


def extract_page_ink_mask(image: Image.Image, *, mode: InkMode = "auto") -> InkMaskResult:
    """Adapt rasterregions mask output to InkGrammar's warnings-list result."""
    result = rr.foreground_mask(image, mode=mode)
    return InkMaskResult(
        mask=result.mask,
        confidence=result.confidence,
        method=result.method,
        warnings=list(result.warnings),
    )


def _components(mask: np.ndarray) -> list[Component]:
    return [
        Component(
            index=component.index,
            x0=component.bbox.x0,
            y0=component.bbox.y0,
            x1=component.bbox.x1,
            y1=component.bbox.y1,
            area=component.area,
            ys=component.ys,
            xs=component.xs,
        )
        for component in rr.components(mask)
    ]


def detect_ink_rows(mask: np.ndarray) -> list[RowBand]:
    """Adapt rasterregions row bands to InkGrammar's direct-coordinate model."""
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("page ink mask must be two-dimensional")
    return [
        RowBand(
            x0=row.bbox.x0,
            y0=row.bbox.y0,
            x1=row.bbox.x1,
            y1=row.bbox.y1,
            component_indices=row.component_indices,
        )
        for row in rr.detect_rows(ink)
    ]


def _adapt_candidates(result: rr.DetectionResult, prefix: str) -> tuple[SegmentedGlyph, ...]:
    return tuple(
        SegmentedGlyph(
            glyph_id=f"{prefix}-{region.reading_order:03d}",
            bbox=CropRect(
                x=region.bbox.x0,
                y=region.bbox.y0,
                width=region.bbox.width,
                height=region.bbox.height,
            ),
            row_index=region.row_index,
            column_index=region.column_index,
            reading_order=region.reading_order,
            segmentation_confidence=region.confidence,
            image=region.image,
            mask=region.mask,
        )
        for region in result.regions
    )


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
        "confidence": round(candidate.segmentation_confidence, 4),
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
    image: Image.Image,
    mask: np.ndarray,
    segmentation: PageSegmentation,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(mask, output_dir / "reference-mask.png")
    _write_overlay(image, segmentation.candidates, output_dir / "reference-candidates.png")
    payload = {
        "source_image": segmentation.source_path.name,
        "candidate_count": len(segmentation.candidates),
        "status": segmentation.status,
        "warnings": list(segmentation.warnings),
        "candidates": [_candidate_record(candidate) for candidate in segmentation.candidates],
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def segment_reference_gallery(
    image_path: str | Path,
    *,
    ink_mode: InkMode = "auto",
    output_dir: str | Path | None = None,
) -> ReferenceGallery:
    """Adapt generic raster-region detections into a source-ordered gallery."""
    source_path = Path(image_path)
    result = rr.detect_regions(source_path, mode=ink_mode)
    candidates = _adapt_candidates(result, "ref")
    segmentation = PageSegmentation(
        source_path=source_path,
        candidates=candidates,
        status=result.status,
        warnings=result.warnings,
    )
    if output_dir is not None:
        _write_reference_diagnostics(
            result.image, result.clean_mask, segmentation, Path(output_dir)
        )
    if not candidates:
        raise SegmentationError("No reference glyph candidates were detected.")
    return ReferenceGallery(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        candidates=candidates,
        mask_result=InkMaskResult(
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
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_page_mask(mask, output_dir / "sample-mask.png")
    component_preview = image.convert("RGB")
    component_draw = ImageDraw.Draw(component_preview)
    for component in components:
        component_draw.rectangle(
            (component.x0, component.y0, component.x1 - 1, component.y1 - 1),
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
    }
    for filename in ("sample-segmentation.json", "sample-candidates.json"):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def segment_sample_glyphs(
    image_path: str | Path,
    *,
    expected_count: int,
    ink_mode: InkMode = "auto",
    output_dir: str | Path | None = None,
) -> list[SegmentedGlyph]:
    """Recover exactly ``expected_count`` glyph regions or raise for failure."""
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    source_path = Path(image_path)
    result = rr.detect_regions(
        source_path,
        expected_count=expected_count,
        expected_count_scope="row",
        mode=ink_mode,
    )
    candidates = _adapt_candidates(result, "sample")
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
        components = _components(result.clean_mask)
        _write_sample_diagnostics(
            result.image,
            source_path.name,
            result.clean_mask,
            components,
            candidates,
            expected_count,
            result.status,
            list(result.warnings),
            boundaries,
            row.bbox.y0 if row else 0,
            row.bbox.y1 if row else result.image.height,
            Path(output_dir),
        )
    if result.status != "ok" or len(candidates) != expected_count:
        diagnostic = Path(output_dir) / "sample-boundaries.png" if output_dir is not None else None
        raise SegmentationError(
            f"Unable to recover {expected_count} credible letter regions.", diagnostic
        )
    return list(candidates)
