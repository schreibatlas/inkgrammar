from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np
from planarcurve.transforms import fit_similarity_transform
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
    WritingDirectionPolicy,
    WritingRouteAlignment,
    WritingRouteCandidate,
)
from .path import (
    PathExtractionError,
    align_continuous_paths,
    align_ordered_paths,
    recover_ordered_path,
    resample_continuous_path,
    resample_path,
    reverse_continuous_path,
    transform_continuous_path,
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
        candidate = next(
            (item for item in candidates if len(item.continuous_strokes) == 1),
            None,
        )
        if candidate is None:
            return candidates, None, None, None
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
    if len(candidate.continuous_strokes) > 1:
        return replace(
            extracted,
            ordered_path=None,
            continuous_path=None,
            resampled_xy=None,
            selected_route=candidate,
        )
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
        selected_route=candidate,
    )


def _align_writing_routes(
    reference: WritingRouteCandidate,
    sample: WritingRouteCandidate,
) -> WritingRouteAlignment:
    best_alignment = None
    for direction in ("forward", "reversed-for-best-match"):
        sample_strokes = sample.continuous_strokes
        if direction == "reversed-for-best-match":
            sample_strokes = tuple(
                reverse_continuous_path(stroke) for stroke in reversed(sample_strokes)
            )
        paired_count = min(len(reference.continuous_strokes), len(sample_strokes))
        reference_samples = tuple(
            resample_continuous_path(stroke, 128).xy
            for stroke in reference.continuous_strokes[:paired_count]
        )
        sample_samples = tuple(
            resample_continuous_path(stroke, 128).xy for stroke in sample_strokes[:paired_count]
        )
        try:
            transform = fit_similarity_transform(
                np.concatenate(reference_samples, axis=0),
                np.concatenate(sample_samples, axis=0),
            )
            aligned_strokes = tuple(
                transform_continuous_path(stroke, transform) for stroke in sample_strokes
            )
            stroke_errors = tuple(
                float(np.mean(np.sum((reference_xy - transform.apply(sample_xy)) ** 2, axis=1)))
                for reference_xy, sample_xy in zip(
                    reference_samples,
                    sample_samples,
                    strict=True,
                )
            )
            stroke_transforms = (transform,) * paired_count
            strategy = "shared"
        except ValueError:
            per_stroke_alignments = tuple(
                align_continuous_paths(
                    reference.continuous_strokes[index],
                    sample_strokes[index],
                )
                for index in range(paired_count)
            )
            stroke_transforms = tuple(item.transform for item in per_stroke_alignments)
            transform = stroke_transforms[0]
            paired_aligned_strokes = []
            for item in per_stroke_alignments:
                assert item.aligned_sample_path is not None
                paired_aligned_strokes.append(item.aligned_sample_path)
            aligned_strokes = (
                *paired_aligned_strokes,
                *(
                    transform_continuous_path(stroke, transform)
                    for stroke in sample_strokes[paired_count:]
                ),
            )
            stroke_errors = tuple(item.mean_squared_error for item in per_stroke_alignments)
            strategy = "per-stroke"
        weights = tuple(
            0.5
            * (
                reference.continuous_strokes[index].arc_length_px
                + sample_strokes[index].arc_length_px
            )
            for index in range(paired_count)
        )
        mean_squared_error = float(
            sum(error * weight for error, weight in zip(stroke_errors, weights, strict=True))
            / max(sum(weights), 1e-12)
        )
        alignment = WritingRouteAlignment(
            transform=transform,
            direction=direction,
            aligned_sample_strokes=aligned_strokes,
            stroke_mean_squared_errors=stroke_errors,
            mean_squared_error=mean_squared_error,
            stroke_transforms=stroke_transforms,
            strategy=strategy,
        )
        if (
            best_alignment is None
            or alignment.mean_squared_error < best_alignment.mean_squared_error
        ):
            best_alignment = alignment
    assert best_alignment is not None
    return best_alignment


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
    writing_direction_policy: WritingDirectionPolicy | None = None,
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
            direction_policy=writing_direction_policy,
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
    route_alignment = None
    if route_selection is not None and (
        len(route_selection.reference.continuous_strokes) > 1
        or len(route_selection.sample.continuous_strokes) > 1
    ):
        route_alignment = _align_writing_routes(
            route_selection.reference,
            route_selection.sample,
        )
    alignment = None
    if reference.continuous_path is not None and sample.continuous_path is not None:
        alignment = align_continuous_paths(reference.continuous_path, sample.continuous_path)
    elif reference.resampled_xy is not None and sample.resampled_xy is not None:
        alignment = align_ordered_paths(reference.resampled_xy, sample.resampled_xy)

    if alignment is None and route_alignment is None:
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
    elif alignment is not None:
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
        route_alignment=route_alignment,
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
    writing_direction_policy: WritingDirectionPolicy | None = None,
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
        writing_direction_policy=writing_direction_policy,
    )
    return write_comparison_report(result, output_dir)
