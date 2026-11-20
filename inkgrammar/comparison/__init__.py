"""Tools for comparing manually selected letter crops."""

from rasterregions import MaskMode, MaskResult
from rasterregions import foreground_mask as extract_ink_mask

from .api import (
    compare_extracted_letters,
    compare_letter_images,
    extract_letter,
    extract_segmented_glyph,
)
from .batch import compare_letter_page
from .batch_report import BatchComparisonReport, BatchSampleResult
from .crop import coerce_crop_rect, crop_image, load_crop
from .descriptors import GlyphDescriptor, canonicalize_mask, glyph_descriptor
from .geometry import (
    DifferentialGeometry,
    continuous_differential_geometry,
    differential_geometry,
    legacy_differential_geometry_from_polyline,
)
from .matching import (
    MatchScore,
    ReferenceIndex,
    build_reference_index,
    coarse_match_cost,
    match_status,
    rank_reference_candidates,
    refine_match,
    refine_top_k_matches,
)
from .metrics import ComparisonAnalysis, analyze_comparison
from .models import (
    AlignmentResult,
    ComparisonIdentity,
    ComparisonResult,
    ContinuousPath,
    CropRect,
    ExtractedLetter,
    InkMaskResult,
    InkMode,
    LetterImageInput,
    OrderedPath,
    SkeletonDiagnostics,
    TopologySummary,
)
from .path import (
    PathExtractionError,
    align_continuous_paths,
    align_ordered_paths,
    fit_similarity_transform,
    recover_ordered_path,
    resample_continuous_path,
    resample_path,
)
from .report import WrittenComparisonReport, write_comparison_report
from .segmentation import (
    PageSegmentation,
    ReferenceGallery,
    ReferenceRowDiagnostic,
    ReferenceSheetProfile,
    RowBand,
    SegmentationError,
    SegmentedGlyph,
    detect_ink_rows,
    extract_page_ink_mask,
    segment_reference_gallery,
    segment_sample_glyphs,
)
from .skeleton import analyze_skeleton

__all__ = [
    "AlignmentResult",
    "ComparisonAnalysis",
    "ComparisonResult",
    "ContinuousPath",
    "compare_extracted_letters",
    "ComparisonIdentity",
    "compare_letter_images",
    "CropRect",
    "DifferentialGeometry",
    "ExtractedLetter",
    "extract_letter",
    "extract_segmented_glyph",
    "InkMaskResult",
    "InkMode",
    "MaskMode",
    "MaskResult",
    "LetterImageInput",
    "OrderedPath",
    "PathExtractionError",
    "SkeletonDiagnostics",
    "TopologySummary",
    "WrittenComparisonReport",
    "align_continuous_paths",
    "align_ordered_paths",
    "analyze_comparison",
    "analyze_skeleton",
    "coerce_crop_rect",
    "crop_image",
    "continuous_differential_geometry",
    "differential_geometry",
    "legacy_differential_geometry_from_polyline",
    "extract_ink_mask",
    "fit_similarity_transform",
    "load_crop",
    "recover_ordered_path",
    "resample_continuous_path",
    "resample_path",
    "write_comparison_report",
    "PageSegmentation",
    "ReferenceGallery",
    "ReferenceRowDiagnostic",
    "ReferenceSheetProfile",
    "RowBand",
    "SegmentationError",
    "SegmentedGlyph",
    "detect_ink_rows",
    "extract_page_ink_mask",
    "segment_reference_gallery",
    "segment_sample_glyphs",
    "GlyphDescriptor",
    "MatchScore",
    "ReferenceIndex",
    "build_reference_index",
    "canonicalize_mask",
    "coarse_match_cost",
    "glyph_descriptor",
    "match_status",
    "rank_reference_candidates",
    "refine_match",
    "refine_top_k_matches",
    "BatchComparisonReport",
    "BatchSampleResult",
    "compare_letter_page",
]
