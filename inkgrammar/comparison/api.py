from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from rasterregions import MaskMode, MaskResult, foreground_evidence, foreground_pyramid
from skeletonpaths import (
    SkeletonAnalysis,
    SubpixelReconstructionError,
    SubpixelReconstructionLimitError,
)

from .crop import coerce_crop_rect, load_crop
from .descriptors import route_descriptor
from .models import (
    ComparisonIdentity,
    ComparisonResult,
    ContinuousPath,
    CropRect,
    ExtractedLetter,
    OrderedPath,
    SkeletonDiagnostics,
    WritingDirectionPolicy,
    WritingRouteCandidate,
    WritingRoutePairSelection,
    WritingStructurePolicy,
)
from .path import (
    PathExtractionError,
    align_continuous_paths,
    align_ordered_paths,
    recover_ordered_path,
    resample_path,
)
from .progress import ProgressReporter
from .reconstruction import render_reconstructed_strokes
from .report import WrittenComparisonReport, write_comparison_report
from .routes import (
    ROUTE_REPORT_LIMIT,
    align_writing_routes,
    evaluate_writing_route_pairs,
    generate_writing_route_candidates,
)
from .segmentation import SegmentedGlyph
from .skeleton import analyze_skeleton, diagnostics_from_analysis
from .steering import (
    fit_writing_route_candidate,
    fit_writing_route_candidates,
    steering_profile_distance,
)

if TYPE_CHECKING:
    from .refinement import ReconstructionRefinementConfig


def _route_paths(
    diagnostics: SkeletonDiagnostics,
    *,
    decision_field: np.ndarray | None = None,
    allow_path_failure: bool,
    expected_strokes: int | None = None,
    structure_policy: WritingStructurePolicy | None = None,
    ambiguous_component_count: int = 0,
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
            expected_strokes=expected_strokes,
            structure_policy=structure_policy,
            ambiguous_component_count=ambiguous_component_count,
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


def _candidate_descriptors(
    extracted: ExtractedLetter,
    candidates: tuple[WritingRouteCandidate, ...],
):
    if not extracted.route_descriptors:
        return tuple(route_descriptor(candidate, extracted) for candidate in candidates)
    descriptors = {
        (candidate.orientation_group, candidate.orientation_variant): descriptor
        for candidate, descriptor in zip(
            extracted.route_candidates,
            extracted.route_descriptors,
            strict=True,
        )
    }
    return tuple(
        descriptors.get(
            (candidate.orientation_group, candidate.orientation_variant),
            route_descriptor(candidate, extracted),
        )
        for candidate in candidates
    )


def _shortlist_route_candidates(
    reference: ExtractedLetter,
    sample: ExtractedLetter,
    config: ReconstructionRefinementConfig,
) -> tuple[tuple[WritingRouteCandidate, ...], tuple[WritingRouteCandidate, ...]]:
    attempt_limit = max(5, 3 * config.max_route_candidates)
    pool_limit = min(
        min(len(reference.route_candidates), len(sample.route_candidates)),
        max(12, 2 * attempt_limit),
    )
    reference_pool = tuple(
        sorted(reference.route_candidates, key=lambda candidate: candidate.score.intrinsic_cost)[
            :pool_limit
        ]
    )
    sample_pool = tuple(
        sorted(sample.route_candidates, key=lambda candidate: candidate.score.intrinsic_cost)[
            :pool_limit
        ]
    )
    preliminary = evaluate_writing_route_pairs(
        reference_pool,
        sample_pool,
        reference_descriptors=_candidate_descriptors(reference, reference_pool),
        sample_descriptors=_candidate_descriptors(sample, sample_pool),
    )
    reference_ranks = []
    sample_ranks = []
    for pair in preliminary.alternatives:
        if pair.reference_rank not in reference_ranks:
            reference_ranks.append(pair.reference_rank)
        if pair.sample_rank not in sample_ranks:
            sample_ranks.append(pair.sample_rank)
        if len(reference_ranks) >= attempt_limit and len(sample_ranks) >= attempt_limit:
            break
    return (
        tuple(reference_pool[rank - 1] for rank in reference_ranks),
        tuple(sample_pool[rank - 1] for rank in sample_ranks),
    )


def _reconstruction_rank(candidate: WritingRouteCandidate) -> tuple[float, float, int, str]:
    reconstruction = candidate.reconstruction
    fit_cost = (
        sum(value for _, value in reconstruction.fit_terms) if reconstruction is not None else 0.0
    )
    complexity_cost = (
        sum(value for _, value in reconstruction.complexity_terms)
        if reconstruction is not None
        else 0.0
    )
    return (
        candidate.score.intrinsic_cost + fit_cost + complexity_cost,
        candidate.score.intrinsic_cost,
        candidate.orientation_group,
        candidate.orientation_variant,
    )


def _reconstruct_route_candidates(
    extracted: ExtractedLetter,
    candidates: tuple[WritingRouteCandidate, ...],
    config: ReconstructionRefinementConfig,
    *,
    side_label: str,
    progress_callback: Callable[..., None] | None = None,
) -> tuple[WritingRouteCandidate, ...]:
    from .refinement import fit_minimum_complexity_reconstruction, seed_reconstructed_strokes

    evidence = extracted.foreground_evidence
    if evidence is None:
        evidence = foreground_evidence(extracted.image, mode=extracted.mask_result.method)
    attempt_limit = min(len(candidates), max(5, 3 * config.max_route_candidates))
    fitted = []
    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.score.intrinsic_cost)[
        :attempt_limit
    ]
    for attempt_index, source in enumerate(ordered_candidates, start=1):
        if progress_callback is not None:
            progress_callback(
                "seed_fitting",
                completed=attempt_index,
                total=attempt_limit,
                details={
                    "message": f"Fit {side_label} route seed {attempt_index}/{attempt_limit}",
                    "side": side_label,
                    "candidate_index": attempt_index,
                    "candidate_count": attempt_limit,
                },
            )
        candidate = fit_writing_route_candidate(source)
        strokes = seed_reconstructed_strokes(candidate)
        if any(stroke.transform.scale * stroke.trajectory.length <= 1e-6 for stroke in strokes):
            continue
        fitted.append(candidate)
        if len(fitted) == config.max_route_candidates:
            break
    if not fitted:
        raise PathExtractionError("no bounded route candidate has a renderable trajectory seed")

    if progress_callback is not None:
        progress_callback(
            "pyramid_construction",
            completed=1,
            total=1,
            details={"message": f"Build {side_label} foreground pyramid"},
        )
    pyramid = foreground_pyramid(evidence, scales=config.scales)
    reconstructed = tuple(
        replace(
            candidate,
            reconstruction=fit_minimum_complexity_reconstruction(
                candidate,
                evidence,
                render_reconstructed_strokes,
                config=config,
                pyramid=pyramid,
                progress_callback=progress_callback,
                candidate_index=candidate_index,
                candidate_count=len(fitted),
            ),
        )
        for candidate_index, candidate in enumerate(fitted, start=1)
    )
    ranked = tuple(sorted(reconstructed, key=_reconstruction_rank))
    if progress_callback is not None:
        progress_callback(
            "candidate_ranking",
            completed=len(ranked),
            total=len(ranked),
            details={
                "message": f"Rank {side_label} reconstructed routes",
                "side": side_label,
                "candidate_count": len(ranked),
                "objective_evaluations": sum(
                    candidate.reconstruction.objective_evaluations
                    for candidate in ranked
                    if candidate.reconstruction is not None
                ),
            },
        )
    return ranked


def extract_letter(
    image_path: str | Path,
    crop: CropRect | Sequence[int],
    *,
    ink_mode: MaskMode = "auto",
    allow_path_failure: bool = False,
    writing_structure_policy: WritingStructurePolicy | None = None,
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
        expected_strokes=(
            writing_structure_policy or WritingStructurePolicy()
        ).preferred_body_strokes,
        structure_policy=writing_structure_policy,
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
        route_descriptors=tuple(route_descriptor(route) for route in route_candidates),
        foreground_evidence=evidence,
    )


def extract_segmented_glyph(
    candidate: SegmentedGlyph,
    *,
    source_path: str | Path,
    allow_path_failure: bool = True,
    analysis: SkeletonAnalysis | None = None,
    writing_structure_policy: WritingStructurePolicy | None = None,
) -> ExtractedLetter:
    """Analyze a segmented candidate without reloading its source image."""
    diagnostics = (
        diagnostics_from_analysis(analysis)
        if analysis is not None
        else analyze_skeleton(candidate.mask)
    )
    route_candidates, ordered_path, continuous_path, geometry_warning = _route_paths(
        diagnostics,
        expected_strokes=candidate.expected_stroke_count(writing_structure_policy),
        structure_policy=writing_structure_policy,
        ambiguous_component_count=len(candidate.ambiguous_component_indices),
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
        route_descriptors=tuple(route_descriptor(route, candidate) for route in route_candidates),
        component_role_sequence=candidate.component_role_sequence,
        secondary_mark_centroids=candidate.secondary_mark_centroids,
    )


def compare_extracted_letters(
    reference: ExtractedLetter,
    sample: ExtractedLetter,
    *,
    identity: ComparisonIdentity,
    smoothing_window: int = 11,
    smoothing_polynomial_order: int = 2,
    writing_direction_policy: WritingDirectionPolicy | None = None,
    route_pair_selection: WritingRoutePairSelection | None = None,
    identity_status: Literal["accepted", "candidate_diagnostic"] = "accepted",
    reconstruction_config: ReconstructionRefinementConfig | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> ComparisonResult:
    """Compare extracted glyphs, using jointly selected routes when available."""
    warnings = [
        f"{side} {extracted.geometry_warning}"
        for side, extracted in (("Reference", reference), ("Sample", sample))
        if extracted.geometry_warning is not None
    ]
    route_selection = route_pair_selection
    if reference.route_candidates and sample.route_candidates:
        if route_selection is None and reconstruction_config is not None:
            if progress_callback is not None:
                progress_callback(
                    "candidate_ranking",
                    completed=0,
                    total=1,
                    details={"message": "Shortlist route candidates"},
                )
            reference_shortlist, sample_shortlist = _shortlist_route_candidates(
                reference, sample, reconstruction_config
            )
            if progress_callback is not None:
                progress_callback(
                    "candidate_ranking",
                    completed=1,
                    total=1,
                    details={"message": "Shortlisted route candidates"},
                )
            reference_candidates = _reconstruct_route_candidates(
                reference,
                reference_shortlist,
                reconstruction_config,
                side_label="reference",
                progress_callback=progress_callback,
            )
            sample_candidates = _reconstruct_route_candidates(
                sample,
                sample_shortlist,
                reconstruction_config,
                side_label="sample",
                progress_callback=progress_callback,
            )
            reference_descriptors = _candidate_descriptors(reference, reference_candidates)
            sample_descriptors = _candidate_descriptors(sample, sample_candidates)
        else:
            reference_sources = tuple(
                sorted(
                    reference.route_candidates,
                    key=lambda candidate: candidate.score.intrinsic_cost,
                )[:ROUTE_REPORT_LIMIT]
            )
            sample_sources = tuple(
                sorted(
                    sample.route_candidates,
                    key=lambda candidate: candidate.score.intrinsic_cost,
                )[:ROUTE_REPORT_LIMIT]
            )
            reference_candidates = fit_writing_route_candidates(reference_sources)
            sample_candidates = fit_writing_route_candidates(sample_sources)
            reference_descriptors = _candidate_descriptors(reference, reference_candidates)
            sample_descriptors = _candidate_descriptors(sample, sample_candidates)
        reference = replace(
            reference,
            route_candidates=reference_candidates,
            route_descriptors=reference_descriptors,
        )
        sample = replace(
            sample,
            route_candidates=sample_candidates,
            route_descriptors=sample_descriptors,
        )
        if route_selection is None:
            if progress_callback is not None:
                progress_callback(
                    "candidate_ranking",
                    completed=0,
                    total=1,
                    details={"message": "Score reference/sample route pairs"},
                )
            route_selection = evaluate_writing_route_pairs(
                reference_candidates,
                sample_candidates,
                steering_distance=steering_profile_distance,
                direction_policy=writing_direction_policy,
                reference_descriptors=reference_descriptors,
                sample_descriptors=sample_descriptors,
            )
            if progress_callback is not None:
                progress_callback(
                    "candidate_ranking",
                    completed=1,
                    total=1,
                    details={
                        "message": "Ranked route pairs",
                        "pair_count": len(route_selection.alternatives),
                    },
                )
    if route_selection is not None and reference.route_candidates and sample.route_candidates:
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
        route_alignment = align_writing_routes(
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
        identity_status=identity_status,
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
    reconstruction_config: ReconstructionRefinementConfig | None = None,
) -> WrittenComparisonReport:
    """Compare two manually cropped images of the same known letter."""
    if len(letter) != 1:
        raise ValueError("letter label must contain exactly one character")
    if reconstruction_config is None:
        from .refinement import ReconstructionRefinementConfig

        reconstruction_config = ReconstructionRefinementConfig(
            scales=(0.5, 1.0),
            max_route_candidates=2,
            max_geometry_changes=1,
            max_width_changes=1,
            supersampling_coarse=1,
            supersampling_fine=2,
            max_iterations_per_level=8,
        )
    progress = ProgressReporter(
        output_dir,
        enabled=True,
        verbose=False,
        expected_samples=1,
    )
    try:
        progress.stage(
            "route_generation",
            completed=1,
            total=2,
            details={"message": "Generate reference route candidates"},
        )
        reference = extract_letter(reference_path, reference_crop, ink_mode=reference_ink_mode)
        progress.stage(
            "route_generation",
            completed=2,
            total=2,
            details={"message": "Generate sample route candidates"},
        )
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
            reconstruction_config=reconstruction_config,
            progress_callback=progress.stage,
        )
        progress.stage(
            "report_rendering",
            details={"message": "Render comparison report"},
        )
        report = write_comparison_report(result, output_dir)
        progress.complete(
            {
                "output_dir": str(report.output_dir),
                "report_json": str(report.json_path),
                "report_markdown": str(report.markdown_path),
                "report_html": str(report.html_path),
                "progress_jsonl": str(progress.progress_path),
                "run_json": str(progress.run_path),
            }
        )
        return report
    except Exception as error:
        progress.fail(error)
        raise
