from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np
from rasterregions import MaskMode, MaskResult, foreground_evidence
from skeletonpaths import (
    SkeletonAnalysis,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
)

from .crop import coerce_crop_rect, load_crop
from .models import (
    ComparisonIdentity,
    ComparisonResult,
    ContinuousPath,
    CropRect,
    ExtractedLetter,
    OrderedPath,
    SkeletonDiagnostics,
    WritingRouteCandidate,
)
from .path import (
    PathExtractionError,
    align_continuous_paths,
    align_ordered_paths,
    recover_ordered_path,
    resample_path,
)
from .report import WrittenComparisonReport, write_comparison_report
from .routes import generate_writing_route_candidates, select_writing_route_pair
from .segmentation import SegmentedGlyph
from .skeleton import analyze_skeleton, diagnostics_from_analysis
from .steering import fit_writing_route_candidates, steering_profile_distance


def _route_paths(
    diagnostics: SkeletonDiagnostics,
    *,
    decision_field: np.ndarray | None = None,
    allow_path_failure: bool,
) -> tuple[
    tuple[WritingRouteCandidate, ...],
    OrderedPath | None,
    ContinuousPath | None,
    str | None,
]:
    try:
        candidates = generate_writing_route_candidates(
            diagnostics,
            decision_field=decision_field,
        )
        candidate = candidates[0]
        walk = candidate.hypothesis.walks[0]
        ordered_path = OrderedPath(
            xy=walk.xy,
            covered_edge_count=walk.covered_edge_count,
            total_edge_count=walk.total_edge_count,
            walk=walk,
        )
        return candidates, ordered_path, candidate.continuous_strokes[0], None
    except (
        PathExtractionError,
        SubpixelReconstructionError,
        SubpixelReconstructionLimitError,
        ValueError,
    ) as error:
        if not allow_path_failure:
            if isinstance(error, PathExtractionError):
                raise
            raise PathExtractionError(f"Writing-route reconstruction failed: {error}") from error
        ordered_path = None
        if allow_path_failure:
            with suppress(PathExtractionError):
                ordered_path = recover_ordered_path(diagnostics)
        return (), ordered_path, None, f"Writing-route reconstruction failed: {error}"


def _with_selected_route(
    extracted: ExtractedLetter,
    candidate: WritingRouteCandidate,
) -> ExtractedLetter:
    walk = candidate.hypothesis.walks[0]
    path = candidate.continuous_strokes[0]
    return replace(
        extracted,
        ordered_path=OrderedPath(
            xy=walk.xy,
            covered_edge_count=walk.covered_edge_count,
            total_edge_count=walk.total_edge_count,
            walk=walk,
        ),
        continuous_path=path,
        resampled_xy=resample_path(path.xy),
    )


def extract_letter(
    image_path: str | Path,
    crop: CropRect | Sequence[int],
    *,
    ink_mode: MaskMode = "auto",
    allow_path_failure: bool = False,
) -> ExtractedLetter:
    """Crop an image and recover bounded writing-route hypotheses and geometry."""
    source_path = Path(image_path)
    crop_rect = coerce_crop_rect(crop)
    image = load_crop(source_path, crop_rect)
    evidence = foreground_evidence(image, mode=ink_mode)
    mask_result = MaskResult(
        mask=evidence.mask,
        confidence=evidence.confidence,
        method=evidence.method,
        warnings=evidence.warnings,
        threshold=evidence.threshold,
    )
    diagnostics = analyze_skeleton(mask_result.mask)
    route_candidates, ordered_path, continuous_path, geometry_warning = _route_paths(
        diagnostics,
        decision_field=evidence.decision_field,
        allow_path_failure=allow_path_failure,
    )
    return ExtractedLetter(
        source_path=source_path,
        crop=crop_rect,
        image=image,
        mask_result=mask_result,
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=(resample_path(continuous_path.xy) if continuous_path is not None else None),
        continuous_path=continuous_path,
        geometry_warning=geometry_warning,
        route_candidates=route_candidates,
        foreground_evidence=evidence,
    )


def extract_segmented_glyph(
    candidate: SegmentedGlyph,
    *,
    source_path: str | Path,
    allow_path_failure: bool = True,
    analysis: SkeletonAnalysis | None = None,
) -> ExtractedLetter:
    """Analyze a segmented candidate without reloading its source image."""
    diagnostics = (
        diagnostics_from_analysis(analysis)
        if analysis is not None
        else analyze_skeleton(candidate.mask)
    )
    route_candidates, ordered_path, continuous_path, geometry_warning = _route_paths(
        diagnostics,
        allow_path_failure=allow_path_failure,
    )
    return ExtractedLetter(
        source_path=Path(source_path),
        crop=candidate.bbox,
        image=candidate.image.copy(),
        mask_result=MaskResult(
            mask=candidate.mask,
            confidence=candidate.segmentation_confidence,
            method=candidate.mask_method,
            warnings=candidate.warnings,
        ),
        diagnostics=diagnostics,
        ordered_path=ordered_path,
        resampled_xy=(resample_path(continuous_path.xy) if continuous_path is not None else None),
        continuous_path=continuous_path,
        geometry_warning=geometry_warning,
        route_candidates=route_candidates,
    )


def compare_extracted_letters(
    reference: ExtractedLetter,
    sample: ExtractedLetter,
    *,
    identity: ComparisonIdentity,
    smoothing_window: int = 11,
    smoothing_polynomial_order: int = 2,
) -> ComparisonResult:
    """Compare extracted glyphs, using jointly selected routes when available."""
    warnings = [
        f"{side} {extracted.geometry_warning}"
        for side, extracted in (("Reference", reference), ("Sample", sample))
        if extracted.geometry_warning is not None
    ]
    route_selection = None
    if reference.route_candidates and sample.route_candidates:
        reference_candidates = fit_writing_route_candidates(reference.route_candidates)
        sample_candidates = fit_writing_route_candidates(sample.route_candidates)
        reference = replace(reference, route_candidates=reference_candidates)
        sample = replace(sample, route_candidates=sample_candidates)
        route_selection = select_writing_route_pair(
            reference_candidates,
            sample_candidates,
            steering_distance=steering_profile_distance,
        )
        reference = _with_selected_route(reference, route_selection.reference)
        sample = _with_selected_route(sample, route_selection.sample)
        warnings.append(
            "Route order and orientation are inferred from static image geometry; "
            "they are not observed pen traces."
        )
        if route_selection.ambiguous:
            warnings.append("Several plausible writing-route interpretations remain near-tied.")
    if smoothing_window != 11 or smoothing_polynomial_order != 2:
        warnings.append(
            "Legacy smoothing options "
            f"(window={smoothing_window}, polynomial_order={smoothing_polynomial_order}) "
            "differ from defaults (11, 2); they are used only by the ordered-polyline fallback "
            "when continuous geometry is unavailable."
        )
    alignment = None
    if reference.continuous_path is not None and sample.continuous_path is not None:
        alignment = align_continuous_paths(reference.continuous_path, sample.continuous_path)
    elif reference.resampled_xy is not None and sample.resampled_xy is not None:
        alignment = align_ordered_paths(reference.resampled_xy, sample.resampled_xy)

    if alignment is None:
        unavailable = [
            side.lower()
            for side, extracted in (("Reference", reference), ("Sample", sample))
            if extracted.resampled_xy is None
        ]
        warnings.append(
            "Continuous or ordered-path trajectory metrics are unavailable for "
            + " and ".join(unavailable)
            + "; raster and topology comparison remain available."
        )
    else:
        warnings.extend(
            f"{side} skeleton has {extracted.diagnostics.topology.endpoint_count} endpoints; "
            "start and end are inferred from reference matching."
            for side, extracted in (("Reference", reference), ("Sample", sample))
            if extracted.diagnostics.topology.endpoint_count != 2
        )
    return ComparisonResult(
        letter=identity.label,
        reference=reference,
        sample=sample,
        alignment=alignment,
        smoothing_window=smoothing_window,
        smoothing_polynomial_order=smoothing_polynomial_order,
        warnings=warnings,
        identity=identity,
        route_selection=route_selection,
    )


def compare_letter_images(
    reference_path: str | Path,
    sample_path: str | Path,
    *,
    letter: str,
    reference_crop: CropRect | Sequence[int],
    sample_crop: CropRect | Sequence[int],
    output_dir: str | Path,
    reference_ink_mode: MaskMode = "auto",
    sample_ink_mode: MaskMode = "auto",
    smoothing_window: int = 11,
    smoothing_polynomial_order: int = 2,
) -> WrittenComparisonReport:
    """Compare two manually cropped images of the same known letter."""
    if len(letter) != 1:
        raise ValueError("letter label must contain exactly one character")
    reference = extract_letter(reference_path, reference_crop, ink_mode=reference_ink_mode)
    sample = extract_letter(sample_path, sample_crop, ink_mode=sample_ink_mode)
    result = compare_extracted_letters(
        reference,
        sample,
        identity=ComparisonIdentity(
            sample_id=Path(sample_path).stem or "sample",
            reference_id=Path(reference_path).stem or "reference",
            label=letter,
        ),
        smoothing_window=smoothing_window,
        smoothing_polynomial_order=smoothing_polynomial_order,
    )
    return write_comparison_report(result, output_dir)
