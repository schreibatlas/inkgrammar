from __future__ import annotations

import hashlib
import json
import shutil
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from rasterregions import MaskMode, MaskResult

from .api import compare_extracted_letters, extract_segmented_glyph
from .batch_report import (
    BatchComparisonReport,
    BatchSampleResult,
    match_score_payload,
    write_batch_report,
)
from .descriptors import GlyphDescriptor
from .matching import (
    DEFAULT_TOP_K,
    MINIMUM_ALTERNATIVES,
    MatchScore,
    ReferenceIndex,
    build_reference_index,
    match_status,
    rank_reference_candidates,
    refine_top_k_matches,
)
from .models import (
    ComparisonIdentity,
    CropRect,
    ExtractedLetter,
    OrderedPath,
)
from .progress import ProgressReporter
from .report import write_comparison_report
from .segmentation import (
    REFERENCE_SEGMENTATION_VERSION,
    PageSegmentation,
    ReferenceGallery,
    ReferenceRowDiagnostic,
    ReferenceSheetProfile,
    SegmentationError,
    SegmentedGlyph,
    segment_reference_gallery,
    segment_sample_glyphs,
)
from .skeleton import diagnostics_from_skeleton

_REFERENCE_CACHE_VERSION = 3
_DESCRIPTOR_SIZE = 96

_REFERENCE_DIAGNOSTICS = (
    "reference-mask.png",
    "reference-rows.png",
    "reference-components.png",
    "reference-candidates.png",
    "reference-candidates.json",
    "reference-segmentation-hypotheses.json",
    "reference-gallery.png",
)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_reference_diagnostics(cache_dir: Path, output_dir: Path) -> None:
    source = cache_dir / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [source / filename for filename in _REFERENCE_DIAGNOSTICS]
    files.extend(sorted(source.glob("ref-*.png")))
    for cached in files:
        if cached.is_file():
            shutil.copy2(cached, output_dir / cached.name)

    hypotheses_dir = source / "reference-hypotheses"
    if hypotheses_dir.is_dir():
        shutil.copytree(
            hypotheses_dir,
            output_dir / "reference-hypotheses",
            dirs_exist_ok=True,
        )

def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.asarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(binary.shape, dtype=np.int64).tobytes())
    digest.update(binary.tobytes())
    return digest.hexdigest()


def _reference_profile_settings(
    profile: str | ReferenceSheetProfile,
) -> dict[str, Any]:
    selected = ReferenceSheetProfile(name=profile) if isinstance(profile, str) else profile
    return {
        "name": selected.name,
        "include_rows": list(selected.include_rows) if selected.include_rows is not None else None,
        "exclude_rows": list(selected.exclude_rows),
        "classifier": "adaptive-repeated-row-geometry-v1",
        "minimum_repeated_row_fraction": 0.5,
        "metadata_row_fraction": 0.55,
        "unassigned_ink_fraction_limit": 0.02,
    }


def _cache_metadata(
    gallery: ReferenceGallery,
    index: ReferenceIndex,
    ink_mode: MaskMode,
    profile: str | ReferenceSheetProfile,
) -> dict[str, Any]:
    profile_settings = _reference_profile_settings(profile)
    rows = [
        {
            "row": row.row_index,
            "bbox": row.bbox.model_dump(mode="json"),
            "component_indices": list(row.component_indices),
            "candidate_count": row.candidate_count,
            "median_width": row.median_width,
            "maximum_width": row.maximum_width,
            "median_height": row.median_height,
            "horizontal_occupancy": row.horizontal_occupancy,
            "median_spacing": row.median_spacing,
            "classification": row.classification,
            "included": row.included,
            "confidence": row.confidence,
            "reason": row.reason,
            "warnings": list(row.warnings),
        }
        for row in gallery.segmentation.rows
    ]
    return {
        "cache_version": _REFERENCE_CACHE_VERSION,
        "descriptor_size": _DESCRIPTOR_SIZE,
        "ink_mode": ink_mode,
        "source_sha256": gallery.source_sha256,
        "reference_profile": profile_settings["name"],
        "reference_profile_settings": profile_settings,
        "row_classification_settings": profile_settings,
        "segmentation_algorithm_version": REFERENCE_SEGMENTATION_VERSION,
        "rasterregions_version": package_version("rasterregions"),
        "mask_mode": ink_mode,
        "mask_confidence": gallery.mask_result.confidence,
        "mask_method": gallery.mask_result.method,
        "mask_warnings": list(gallery.mask_result.warnings),
        "segmentation_status": gallery.segmentation.status,
        "segmentation_warnings": list(gallery.segmentation.warnings),
        "unassigned_ink_fraction": gallery.segmentation.unassigned_ink_fraction,
        "segmentation_hypothesis_count": gallery.segmentation.hypothesis_count,
        "segmentation_search_truncated": gallery.segmentation.search_truncated,
        "segmentation_search_limit": gallery.segmentation.search_limit,
        "anchor_region_count": gallery.segmentation.anchor_region_count,
        "row_count": len(rows),
        "rows": rows,
        "per_row_candidate_counts": {
            str(row.row_index): sum(
                candidate.row_index == row.row_index for candidate in gallery.candidates
            )
            for row in gallery.segmentation.rows
        },
        "candidate_count": len(gallery.candidates),
        "candidates": [
            {
                "id": candidate.glyph_id,
                "bbox": candidate.bbox.model_dump(mode="json"),
                "dimensions": {
                    "width": candidate.bbox.width,
                    "height": candidate.bbox.height,
                },
                "row": candidate.row_index,
                "column": candidate.column_index,
                "reading_order": candidate.reading_order,
                "confidence": candidate.segmentation_confidence,
                "component_indices": list(candidate.component_indices),
                "warnings": list(candidate.warnings),
                "mask_method": candidate.mask_method,
                "label": candidate.label,
                "crop": f"{candidate.glyph_id}.png",
                "mask": f"{candidate.glyph_id}-mask.png",
                "mask_sha256": _mask_sha256(candidate.mask),
            }
            for candidate in gallery.candidates
        ],
        "descriptors": [
            {
                "glyph_id": descriptor.glyph_id,
                "aspect_ratio": descriptor.aspect_ratio,
                "ink_fill": descriptor.ink_fill,
                "component_count": descriptor.component_count,
                "endpoint_count": descriptor.endpoint_count,
                "junction_count": descriptor.junction_count,
                "cycle_count": descriptor.cycle_count,
                "path_available": descriptor.path_available,
            }
            for descriptor in index.descriptors
        ],
    }

def _save_reference_cache(
    cache_dir: Path,
    gallery: ReferenceGallery,
    index: ReferenceIndex,
    ink_mode: MaskMode,
    profile: str | ReferenceSheetProfile,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = cache_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for candidate in gallery.candidates:
        candidate.image.save(crops_dir / f"{candidate.glyph_id}.png", format="PNG")
        Image.fromarray(candidate.mask.astype(np.uint8) * 255).save(
            crops_dir / f"{candidate.glyph_id}-mask.png", format="PNG"
        )
    descriptor_masks = (
        np.stack([descriptor.normalized_mask for descriptor in index.descriptors])
        if index.descriptors
        else np.zeros((0, _DESCRIPTOR_SIZE, _DESCRIPTOR_SIZE), dtype=bool)
    )
    np.savez_compressed(
        cache_dir / "descriptors.npz",
        glyph_ids=np.asarray([item.glyph_id for item in index.descriptors], dtype="U"),
        normalized_masks=descriptor_masks,
        aspect_ratios=np.asarray([item.aspect_ratio for item in index.descriptors]),
        ink_fills=np.asarray([item.ink_fill for item in index.descriptors]),
        component_counts=np.asarray([item.component_count for item in index.descriptors]),
    )
    metadata = _cache_metadata(gallery, index, ink_mode, profile)
    (cache_dir / "gallery.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _load_reference_cache(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    ink_mode: MaskMode,
    profile: str | ReferenceSheetProfile,
) -> tuple[ReferenceGallery, ReferenceIndex, dict[str, Any]]:
    metadata = json.loads((cache_dir / "gallery.json").read_text(encoding="utf-8"))
    profile_settings = _reference_profile_settings(profile)
    if (
        metadata.get("cache_version") != _REFERENCE_CACHE_VERSION
        or metadata.get("descriptor_size") != _DESCRIPTOR_SIZE
        or metadata.get("ink_mode") != ink_mode
        or metadata.get("mask_mode") != ink_mode
        or metadata.get("source_sha256") != source_sha256
        or metadata.get("reference_profile_settings") != profile_settings
        or metadata.get("row_classification_settings") != profile_settings
        or metadata.get("segmentation_algorithm_version") != REFERENCE_SEGMENTATION_VERSION
        or metadata.get("rasterregions_version") != package_version("rasterregions")
    ):
        raise ValueError("reference cache metadata does not match this input")
    if not (cache_dir / "descriptors.npz").is_file():
        raise ValueError("reference descriptor cache is missing")
    if not (cache_dir / "segmentation" / "reference-mask.png").is_file():
        raise ValueError("reference mask cache is missing")
    if len(metadata.get("candidates", ())) != metadata.get("candidate_count"):
        raise ValueError("reference cache candidate count does not match its metadata")

    candidates = []
    crops_dir = cache_dir / "crops"
    for item in metadata["candidates"]:
        with Image.open(crops_dir / item["crop"]) as crop_image:
            image = crop_image.convert("RGB")
        with Image.open(crops_dir / item["mask"]) as crop_mask:
            mask = np.asarray(crop_mask.convert("L"), dtype=np.uint8) > 0
        if _mask_sha256(mask) != item["mask_sha256"]:
            raise ValueError("reference candidate mask does not match its metadata")
        candidates.append(
            SegmentedGlyph(
                glyph_id=item["id"],
                bbox=CropRect.model_validate(item["bbox"]),
                row_index=item["row"],
                column_index=item["column"],
                reading_order=item["reading_order"],
                segmentation_confidence=item["confidence"],
                image=image,
                mask=mask,
                mask_method=item["mask_method"],
                label=item["label"],
                component_indices=tuple(item["component_indices"]),
                warnings=tuple(item["warnings"]),
            )
        )

    with Image.open(cache_dir / "segmentation" / "reference-mask.png") as mask_image:
        page_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    mask_result = MaskResult(
        mask=page_mask,
        confidence=metadata["mask_confidence"],
        method=metadata["mask_method"],
        warnings=list(metadata["mask_warnings"]),
    )
    rows = tuple(
        ReferenceRowDiagnostic(
            row_index=item["row"],
            bbox=CropRect.model_validate(item["bbox"]),
            component_indices=tuple(item["component_indices"]),
            candidate_count=item["candidate_count"],
            median_width=item["median_width"],
            maximum_width=item["maximum_width"],
            median_height=item["median_height"],
            horizontal_occupancy=item["horizontal_occupancy"],
            median_spacing=item["median_spacing"],
            classification=item["classification"],
            included=item["included"],
            confidence=item["confidence"],
            reason=item["reason"],
            warnings=tuple(item["warnings"]),
        )
        for item in metadata["rows"]
    )
    segmentation = PageSegmentation(
        source_path=source_path,
        candidates=tuple(candidates),
        status=metadata["segmentation_status"],
        warnings=tuple(metadata["segmentation_warnings"]),
        reference_profile=metadata["reference_profile"],
        rows=rows,
        unassigned_ink_fraction=metadata["unassigned_ink_fraction"],
        hypothesis_count=metadata["segmentation_hypothesis_count"],
        search_truncated=metadata["segmentation_search_truncated"],
        search_limit=metadata["segmentation_search_limit"],
        anchor_region_count=metadata["anchor_region_count"],
    )
    gallery = ReferenceGallery(
        source_path=source_path,
        source_sha256=source_sha256,
        candidates=tuple(candidates),
        mask_result=mask_result,
        segmentation=segmentation,
    )

    with np.load(cache_dir / "descriptors.npz", allow_pickle=False) as stored:
        descriptors = tuple(
            GlyphDescriptor(
                glyph_id=str(glyph_id),
                normalized_mask=normalized_mask,
                aspect_ratio=float(aspect_ratio),
                ink_fill=float(ink_fill),
                component_count=int(component_count),
                endpoint_count=descriptor_data["endpoint_count"],
                junction_count=descriptor_data["junction_count"],
                cycle_count=descriptor_data["cycle_count"],
                path_available=descriptor_data["path_available"],
            )
            for (
                glyph_id,
                normalized_mask,
                aspect_ratio,
                ink_fill,
                component_count,
                descriptor_data,
            ) in zip(
                stored["glyph_ids"],
                stored["normalized_masks"],
                stored["aspect_ratios"],
                stored["ink_fills"],
                stored["component_counts"],
                metadata["descriptors"],
                strict=True,
            )
        )
    index = ReferenceIndex(gallery=gallery, descriptors=descriptors)
    return gallery, index, metadata


def _reference_extraction_paths(cache_dir: Path, glyph_id: str) -> tuple[Path, Path, Path]:
    extraction_dir = cache_dir / "extractions"
    return (
        extraction_dir / f"{glyph_id}.json",
        extraction_dir / f"{glyph_id}.npz",
        extraction_dir / f"{glyph_id}.png",
    )


def _load_reference_extraction(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    candidate: SegmentedGlyph,
    ink_mode: MaskMode,
) -> ExtractedLetter:
    metadata_path, arrays_path, image_path = _reference_extraction_paths(
        cache_dir, candidate.glyph_id
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["source_sha256"] != source_sha256
        or metadata["reference_id"] != candidate.glyph_id
        or metadata["ink_mode"] != ink_mode
        or metadata["bbox"] != candidate.bbox.model_dump(mode="json")
    ):
        raise ValueError("cached reference extraction does not match the candidate")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        mask = arrays["mask"]
        skeleton = arrays["skeleton"]
        radius = arrays["radius"]
        path_xy = arrays["path_xy"]
        resampled_xy = arrays["resampled_xy"]
    diagnostics = diagnostics_from_skeleton(skeleton, radius)
    path_info = metadata["ordered_path"]
    ordered_path = (
        OrderedPath(
            path_xy,
            covered_edge_count=path_info["covered_edge_count"],
            total_edge_count=path_info["total_edge_count"],
        )
        if path_info is not None
        else None
    )
    with Image.open(image_path) as crop_image:
        image = crop_image.convert("RGB")
    return ExtractedLetter(
        source_path=source_path,
        crop=candidate.bbox,
        image=image,
        mask_result=MaskResult(
            mask=mask,
            confidence=metadata["mask_confidence"],
            method=metadata["mask_method"],
            warnings=metadata["mask_warnings"],
        ),
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=resampled_xy if ordered_path is not None else None,
    )


def _save_reference_extraction(
    cache_dir: Path,
    source_sha256: str,
    ink_mode: MaskMode,
    glyph_id: str,
    extraction: ExtractedLetter,
) -> None:
    metadata_path, arrays_path, image_path = _reference_extraction_paths(cache_dir, glyph_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_path = extraction.ordered_path
    np.savez_compressed(
        arrays_path,
        mask=extraction.mask_result.mask,
        skeleton=extraction.diagnostics.skeleton,
        radius=extraction.diagnostics.radius,
        path_xy=(ordered_path.xy if ordered_path is not None else np.zeros((0, 2))),
        resampled_xy=(
            extraction.resampled_xy if extraction.resampled_xy is not None else np.zeros((0, 2))
        ),
    )
    extraction.image.save(image_path, format="PNG")
    metadata = {
        "source_sha256": source_sha256,
        "ink_mode": ink_mode,
        "reference_id": glyph_id,
        "bbox": extraction.crop.model_dump(mode="json"),
        "mask_confidence": extraction.mask_result.confidence,
        "mask_method": extraction.mask_result.method,
        "mask_warnings": extraction.mask_result.warnings,
        "topology": extraction.diagnostics.topology.model_dump(mode="json"),
        "ordered_path": (
            {
                "covered_edge_count": ordered_path.covered_edge_count,
                "total_edge_count": ordered_path.total_edge_count,
            }
            if ordered_path is not None
            else None
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _get_reference_extraction(
    cache_dir: Path,
    source_path: Path,
    source_sha256: str,
    candidate: SegmentedGlyph,
    ink_mode: MaskMode,
    progress: ProgressReporter,
) -> ExtractedLetter:
    started = perf_counter()
    cache_status = "hit"
    try:
        extraction = _load_reference_extraction(
            cache_dir,
            source_path,
            source_sha256,
            candidate,
            ink_mode,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        cache_status = "miss"
        extraction = extract_segmented_glyph(
            candidate,
            source_path=source_path,
            allow_path_failure=True,
        )
        _save_reference_extraction(
            cache_dir,
            source_sha256,
            ink_mode,
            candidate.glyph_id,
            extraction,
        )
    details: dict[str, Any] = {
        "reference_id": candidate.glyph_id,
        "cache": cache_status,
        "message": f"Reference extraction {candidate.glyph_id}: cache {cache_status}",
    }
    if progress.verbose:
        details["elapsed_seconds"] = round(perf_counter() - started, 3)
        details["message"] += f" ({details['elapsed_seconds']:.3f}s)"
    progress.stage("reference_extraction", details=details)
    return extraction


def _write_reference_index(
    gallery: ReferenceGallery,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous in output_dir.glob("ref-*.png"):
        previous.unlink()
    public_metadata = {
        **metadata,
        "source_image": gallery.source_path.name,
        "cache_hit": metadata.get("cache_hit", False),
    }
    (output_dir / "gallery.json").write_text(
        json.dumps(public_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for candidate in gallery.candidates:
        candidate.image.save(output_dir / f"{candidate.glyph_id}.png", format="PNG")


def _write_reference_manifest(
    gallery: ReferenceGallery,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {
        "source_image": gallery.source_path.name,
        "source_sha256": gallery.source_sha256,
        "algorithm_version": metadata["segmentation_algorithm_version"],
        "reference_profile": metadata["reference_profile"],
        "row_count": metadata["row_count"],
        "rows": metadata["rows"],
        "per_row_candidate_counts": metadata["per_row_candidate_counts"],
        "excluded_rows": [row for row in metadata["rows"] if not row["included"]],
        "candidate_count": len(gallery.candidates),
        "unassigned_ink_fraction": metadata["unassigned_ink_fraction"],
        "status": gallery.segmentation.status,
        "hypothesis_count": metadata["segmentation_hypothesis_count"],
        "search_truncated": metadata["segmentation_search_truncated"],
        "search_limit": metadata["segmentation_search_limit"],
        "anchor_region_count": metadata["anchor_region_count"],
        "warnings": list(gallery.segmentation.warnings),
        "candidates": metadata["candidates"],
    }
    (output_dir / "reference-candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumbnail = image.convert("RGB").copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        thumbnail,
        ((size[0] - thumbnail.width) // 2, (size[1] - thumbnail.height) // 2),
    )
    return canvas


def _write_alternatives(
    sample: SegmentedGlyph,
    ranked: list[MatchScore],
    candidates: dict[str, SegmentedGlyph],
    output_dir: Path,
) -> None:
    cell_width, image_height, cell_height = 220, 110, 142
    sheet_width = cell_width * min(3, len(ranked))
    sheet_height = cell_height * ((len(ranked) + 2) // 3)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, score in enumerate(ranked):
        x = (index % 3) * cell_width
        y = (index // 3) * cell_height
        image = _fit_thumbnail(
            candidates[score.reference_id].image,
            (cell_width - 16, image_height),
        )
        sheet.paste(image, (x + 8, y + 4))
        draw.text(
            (x + 8, y + image_height + 8),
            f"#{score.rank} {score.reference_id} cost={score.cost:.4f}",
            fill="black",
        )
    sheet.save(output_dir / "alternatives.png", format="PNG")
    sample.image.convert("RGB").save(output_dir / "sample.png", format="PNG")


def _assess_segmentation(
    gallery: ReferenceGallery,
    sample: SegmentedGlyph,
    reference: SegmentedGlyph,
) -> tuple[str, tuple[str, ...]]:
    warnings: list[str] = []
    if gallery.segmentation.status != "ok" or gallery.segmentation.warnings:
        warnings.extend(
            f"Reference page: {warning}" for warning in gallery.segmentation.warnings
        )
        if not gallery.segmentation.warnings:
            warnings.append(
                f"Reference page segmentation status is {gallery.segmentation.status}."
            )
    warnings.extend(f"Sample {sample.glyph_id}: {warning}" for warning in sample.warnings)
    warnings.extend(
        f"Reference {reference.glyph_id}: {warning}" for warning in reference.warnings
    )
    if sample.segmentation_confidence < 0.3:
        warnings.append(f"Sample {sample.glyph_id} boundary confidence is low.")
    if reference.segmentation_confidence < 0.3:
        warnings.append(f"Reference {reference.glyph_id} boundary confidence is low.")
    unique_warnings = tuple(dict.fromkeys(warnings))
    return ("ambiguous" if unique_warnings else "ok"), unique_warnings


def _write_match_json(
    output_dir: Path,
    *,
    sample: SegmentedGlyph,
    best: MatchScore,
    ranked: list[MatchScore],
    candidates: dict[str, SegmentedGlyph],
    status: str,
    segmentation_status: str,
    segmentation_warnings: tuple[str, ...],
    pair_report_dir: Path,
) -> None:
    payload = {
        "sample_id": sample.glyph_id,
        "sample_bbox": sample.bbox.model_dump(mode="json"),
        "segmentation_confidence": sample.segmentation_confidence,
        "status": status,
        "match_status": status,
        "segmentation_status": segmentation_status,
        "segmentation_warnings": list(segmentation_warnings),
        "best": {
            **match_score_payload(best),
            "label": candidates[best.reference_id].label,
        },
        "alternatives": [
            {
                **match_score_payload(score),
                "label": candidates[score.reference_id].label,
            }
            for score in ranked[1:]
        ],
        "pair_report_dir": pair_report_dir.name,
    }
    (output_dir / "match.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_segmentation_only_summary(
    output_dir: Path,
    reference: ReferenceGallery,
    expected_count: int,
) -> None:
    segmentation_dir = output_dir / "segmentation"
    reference_manifest = json.loads(
        (segmentation_dir / "reference-candidates.json").read_text(encoding="utf-8")
    )
    sample_manifest = json.loads(
        (segmentation_dir / "sample-segmentation.json").read_text(encoding="utf-8")
    )
    status = (
        "ambiguous"
        if reference.segmentation.status != "ok" or sample_manifest["status"] != "ok"
        else "ok"
    )
    payload = {
        "operation": "segment-only",
        "status": status,
        "reference": reference_manifest,
        "sample": sample_manifest,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = (
        "# Segmentation-only summary\n\n"
        f"Status: **{status}**.\n\n"
        f"Reference profile: `{reference.segmentation.reference_profile}`. "
        f"Candidates: {len(reference.candidates)}.\n\n"
        f"Sample glyphs: {sample_manifest['detected_count']} of {expected_count}.\n\n"
        "Inspect `segmentation/reference-rows.png`, `segmentation/reference-gallery.png`, "
        "and `segmentation/sample-boundaries.png`.\n"
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "index.html").write_text(
        "<!doctype html><html><head>"
        "<meta charset=\"utf-8\"><title>Segmentation diagnostics</title></head>"
        "<body><h1>Segmentation diagnostics</h1><ul>"
        "<li><a href=\"segmentation/reference-rows.png\">Reference rows</a></li>"
        "<li><a href=\"segmentation/reference-candidates.png\">Reference candidates</a></li>"
        "<li><a href=\"segmentation/reference-gallery.png\">Reference gallery</a></li>"
        "<li><a href=\"segmentation/sample-boundaries.png\">Sample boundaries</a></li>"
        "<li><a href=\"summary.json\">Summary JSON</a></li>"
        "</ul></body></html>\n",
        encoding="utf-8",
    )


def compare_letter_page(
    reference_path: str | Path,
    sample_path: str | Path,
    *,
    expected_letters: int,
    output_dir: str | Path,
    reference_ink_mode: MaskMode = "auto",
    sample_ink_mode: MaskMode = "auto",
    top_k: int = DEFAULT_TOP_K,
    reference_profile: str | ReferenceSheetProfile = "alphabet",
    no_progress: bool = False,
    verbose: bool = False,
    segment_only: bool = False,
) -> BatchComparisonReport | None:
    """Segment, retrieve, compare, and report a page of standalone sample glyphs."""
    progress = ProgressReporter(
        output_dir,
        enabled=not no_progress,
        verbose=verbose,
        expected_samples=expected_letters,
    )
    try:
        report = _compare_letter_page_impl(
            reference_path,
            sample_path,
            expected_letters=expected_letters,
            output_dir=output_dir,
            reference_ink_mode=reference_ink_mode,
            sample_ink_mode=sample_ink_mode,
            top_k=top_k,
            reference_profile=reference_profile,
            segment_only=segment_only,
            progress=progress,
        )
    except BaseException as error:
        progress.fail(error)
        raise

    root = Path(output_dir)
    progress.complete(
        {
            "summary_json": str(root / "summary.json"),
            "summary_markdown": str(root / "summary.md"),
            "index_html": str(root / "index.html"),
            "segmentation_directory": str(root / "segmentation"),
            "progress_jsonl": str(progress.progress_path),
            "run_json": str(progress.run_path),
        }
    )
    return report


def _compare_letter_page_impl(
    reference_path: str | Path,
    sample_path: str | Path,
    *,
    expected_letters: int,
    output_dir: str | Path,
    reference_ink_mode: MaskMode,
    sample_ink_mode: MaskMode,
    top_k: int,
    reference_profile: str | ReferenceSheetProfile,
    segment_only: bool,
    progress: ProgressReporter,
) -> BatchComparisonReport | None:
    progress.stage("input_validation", details={"message": "Validate inputs"})
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if expected_letters < 1:
        raise ValueError("expected_letters must be positive")
    _reference_profile_settings(reference_profile)
    reference = Path(reference_path)
    sample_path = Path(sample_path)
    if not reference.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {reference}")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample image does not exist: {sample_path}")

    root = Path(output_dir)
    segmentation_dir = root / "segmentation"
    reference_index_dir = root / "reference-index"
    root.mkdir(parents=True, exist_ok=True)
    progress.stage("reference_hash", details={"message": "Hash reference image"})
    source_sha256 = _source_hash(reference)
    cache_dir = root / ".reference-cache" / source_sha256

    cache_hit = False
    gallery = None
    index = None
    metadata = None
    if segment_only:
        progress.stage(
            "reference_cache",
            details={"cache": "bypassed", "message": "Reference cache bypassed for segment-only"},
        )
    else:
        progress.stage("reference_cache", details={"message": "Load reference cache"})
        try:
            gallery, index, metadata = _load_reference_cache(
                cache_dir,
                reference,
                source_sha256,
                reference_ink_mode,
                reference_profile,
            )
            cache_hit = True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        progress.stage(
            "reference_cache",
            details={
                "cache": "hit" if cache_hit else "miss",
                "message": f"Reference cache: {'hit' if cache_hit else 'miss'}",
            },
        )

    progress.stage(
        "reference_mask_rows",
        details={"message": "Mask reference, detect rows, and verify foreground ownership"},
    )
    if not cache_hit:
        diagnostics_output = segmentation_dir if segment_only else cache_dir / "segmentation"
        try:
            gallery = segment_reference_gallery(
                reference,
                ink_mode=reference_ink_mode,
                reference_profile=reference_profile,
                output_dir=diagnostics_output,
            )
        except SegmentationError as error:
            if not segment_only:
                _copy_reference_diagnostics(cache_dir, segmentation_dir)
            diagnostic = segmentation_dir / "reference-segmentation-hypotheses.json"
            if not diagnostic.is_file():
                diagnostic = segmentation_dir / "reference-candidates.png"
            message = str(error)
            if error.diagnostic_path is not None:
                message = message.replace(str(error.diagnostic_path), str(diagnostic))
            if f"Diagnostics: {diagnostic}" not in message:
                message += f"\nDiagnostics: {diagnostic}"
            failure_phase = error.details.get("failure_phase", "reference_segmentation")
            progress.stage(
                "reference_mask_rows",
                details={
                    **error.details,
                    "failure_phase": failure_phase,
                    "diagnostic_path": str(diagnostic),
                    "message": f"Reference segmentation failed during {failure_phase}.",
                    "error": str(error),
                },
            )
            raise SegmentationError(
                message,
                diagnostic_path=diagnostic,
                details=error.details,
            ) from error

    assert gallery is not None
    progress.stage(
        "reference_segmentation",
        completed=len(gallery.candidates),
        total=len(gallery.candidates),
        details={
            "message": (
                f"Segment reference glyphs: {len(gallery.candidates)} candidates across "
                f"{len(gallery.segmentation.rows)} rows"
            ),
            "status": gallery.segmentation.status,
            "reference_profile": gallery.segmentation.reference_profile,
            "row_count": len(gallery.segmentation.rows),
            "candidate_count": len(gallery.candidates),
            "per_row_candidate_counts": {
                str(row.row_index): sum(
                    candidate.row_index == row.row_index for candidate in gallery.candidates
                )
                for row in gallery.segmentation.rows
            },
            "excluded_row_count": sum(not row.included for row in gallery.segmentation.rows),
            "search_truncated": gallery.segmentation.search_truncated,
            "search_limit": gallery.segmentation.search_limit,
            "anchor_region_count": gallery.segmentation.anchor_region_count,
            "unassigned_ink_fraction": gallery.segmentation.unassigned_ink_fraction,
        }
    )

    if not segment_only:
        if cache_hit:
            progress.stage(
                "reference_descriptors",
                completed=len(index.descriptors),
                total=len(index.descriptors),
                details={"message": f"Load reference descriptors: {len(index.descriptors)}"},
            )
        else:
            index = build_reference_index(gallery)
            progress.stage(
                "reference_descriptors",
                completed=len(index.descriptors),
                total=len(index.descriptors),
                details={"message": f"Build reference descriptors: {len(index.descriptors)}"},
            )
            metadata = _save_reference_cache(
                cache_dir, gallery, index, reference_ink_mode, reference_profile
            )
        metadata = {**metadata, "cache_hit": cache_hit}
        _copy_reference_diagnostics(cache_dir, segmentation_dir)
        _write_reference_index(gallery, metadata, reference_index_dir)
        _write_reference_manifest(gallery, metadata, segmentation_dir)
    else:
        _write_reference_index(
            gallery,
            {
                "cache_hit": False,
                "reference_profile": gallery.segmentation.reference_profile,
                "candidate_count": len(gallery.candidates),
                "candidates": [
                    {
                        "id": candidate.glyph_id,
                        "bbox": candidate.bbox.model_dump(mode="json"),
                        "row": candidate.row_index,
                        "confidence": candidate.segmentation_confidence,
                    }
                    for candidate in gallery.candidates
                ],
            },
            reference_index_dir,
        )

    progress.stage("sample_segmentation", details={"message": "Segment sample glyphs"})
    samples = segment_sample_glyphs(
        sample_path,
        expected_count=expected_letters,
        ink_mode=sample_ink_mode,
        output_dir=segmentation_dir,
    )
    sample_segmentation = json.loads(
        (segmentation_dir / "sample-segmentation.json").read_text(encoding="utf-8")
    )
    progress.stage(
        "sample_segmentation",
        completed=len(samples),
        total=expected_letters,
        details={
            "message": f"Segment sample: {len(samples)}/{expected_letters} glyphs",
            "hypothesis_count": sample_segmentation["hypothesis_count"],
            "search_truncated": sample_segmentation["search_truncated"],
            "search_limit": sample_segmentation["search_limit"],
        },
    )
    if segment_only:
        for _ in samples:
            progress.sample_completed()
        progress.stage(
            "batch_summary",
            details={"message": "Write segmentation summary"},
        )
        _write_segmentation_only_summary(root, gallery, expected_letters)
        return None

    assert index is not None
    effective_top_k = max(top_k, MINIMUM_ALTERNATIVES)
    candidates = {candidate.glyph_id: candidate for candidate in gallery.candidates}
    results = []
    reference_extractions: dict[str, ExtractedLetter] = {}

    for sample_index, sample_candidate in enumerate(samples, start=1):
        sample_extracted = extract_segmented_glyph(
            sample_candidate,
            source_path=sample_path,
            allow_path_failure=True,
        )
        coarse = rank_reference_candidates(
            sample_candidate,
            index,
            top_k=min(effective_top_k, len(gallery.candidates)),
        )
        progress.stage(
            "coarse_ranking",
            completed=sample_index,
            total=len(samples),
            details={
                "message": (
                    f"Compare {sample_candidate.glyph_id}: coarse rank "
                    f"{len(gallery.candidates)} candidates"
                ),
                "ranked_count": len(coarse),
            },
        )
        for coarse_score in coarse:
            reference_id = coarse_score.reference_id
            if reference_id not in reference_extractions:
                reference_extractions[reference_id] = _get_reference_extraction(
                    cache_dir,
                    reference,
                    source_sha256,
                    candidates[reference_id],
                    reference_ink_mode,
                    progress,
                )
        ranked = refine_top_k_matches(
            sample_candidate,
            gallery,
            coarse,
            sample_extracted=sample_extracted,
            reference_extractions=reference_extractions,
        )
        progress.stage(
            "path_refinement",
            completed=len(ranked),
            total=len(coarse),
            details={
                "message": (
                    f"Refine top {len(coarse)} candidates for "
                    f"{sample_candidate.glyph_id}"
                )
            },
        )
        if not ranked:
            raise SegmentationError("No reference candidates are available for matching.")
        best = ranked[0]
        status = match_status(ranked)
        sample_dir = root / sample_candidate.glyph_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        matched_reference = candidates[best.reference_id]
        segmentation_status, segmentation_warnings = _assess_segmentation(
            gallery, sample_candidate, matched_reference
        )
        matched_reference.image.convert("RGB").save(
            sample_dir / "matched-reference.png", format="PNG"
        )
        _write_alternatives(sample_candidate, ranked, candidates, sample_dir)

        reference_extracted = reference_extractions[matched_reference.glyph_id]
        pair_result = compare_extracted_letters(
            reference_extracted,
            sample_extracted,
            identity=ComparisonIdentity(
                sample_id=sample_candidate.glyph_id,
                reference_id=matched_reference.glyph_id,
                label=matched_reference.label,
            ),
        )
        progress.stage(
            "pair_report",
            completed=sample_index,
            total=len(samples),
            details={"message": f"Write {sample_candidate.glyph_id} report"},
        )
        pair_report = write_comparison_report(pair_result, sample_dir)
        _write_match_json(
            sample_dir,
            sample=sample_candidate,
            best=best,
            ranked=ranked,
            candidates=candidates,
            status=status,
            segmentation_status=segmentation_status,
            segmentation_warnings=segmentation_warnings,
            pair_report_dir=pair_report.output_dir,
        )
        results.append(
            BatchSampleResult(
                sample_id=sample_candidate.glyph_id,
                sample_bbox=sample_candidate.bbox,
                sample_confidence=sample_candidate.segmentation_confidence,
                reference_id=matched_reference.glyph_id,
                reference_label=matched_reference.label,
                status=status,
                segmentation_status=segmentation_status,
                segmentation_warnings=segmentation_warnings,
                score=best,
                alternatives=tuple(ranked[1:]),
                report_dir=pair_report.output_dir,
                report_json=pair_report.json_path,
            )
        )
        progress.sample_completed()

    progress.stage("batch_summary", details={"message": "Write batch summary"})
    return write_batch_report(
        root,
        reference_path=reference,
        sample_path=sample_path,
        reference_sha256=source_sha256,
        expected_count=expected_letters,
        results=tuple(results),
        reference_cache_hit=cache_hit,
        reference_segmentation={
            "status": gallery.segmentation.status,
            "reference_profile": gallery.segmentation.reference_profile,
            "row_count": len(gallery.segmentation.rows),
            "candidate_count": len(gallery.candidates),
            "per_row_candidate_counts": {
                str(row.row_index): sum(
                    candidate.row_index == row.row_index for candidate in gallery.candidates
                )
                for row in gallery.segmentation.rows
            },
            "hypothesis_count": gallery.segmentation.hypothesis_count,
            "search_truncated": gallery.segmentation.search_truncated,
            "search_limit": gallery.segmentation.search_limit,
            "anchor_region_count": gallery.segmentation.anchor_region_count,
            "unassigned_ink_fraction": gallery.segmentation.unassigned_ink_fraction,
        },
    )
