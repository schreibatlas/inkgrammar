from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image
from planarcurve.transforms import Similarity2D
from pydantic import Field
from rasterregions import ForegroundEvidence, MaskMode, MaskResult
from skeletonpaths import SkeletonAnalysis, SkeletonGraph, SkeletonWalk, StrokeHypothesis

from ..models import StrictModel

if TYPE_CHECKING:
    from planarcurve.fitting import TrajectoryModelSelectionResult

    from .descriptors import RouteDescriptor
    from .refinement import MinimumComplexityReconstruction
InkMode = MaskMode
InkMaskResult = MaskResult


class CropRect(StrictModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class JunctionRegionSummary(StrictModel):
    node_id: int = Field(ge=0)
    xy: tuple[float, float]
    radius_px: float = Field(ge=0)
    support_pixels: list[tuple[int, int]]
    port_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class LetterImageInput(StrictModel):
    path: Path
    crop: CropRect
    ink_mode: MaskMode = "auto"


class TopologySummary(StrictModel):
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    endpoint_count: int = Field(ge=0)
    endpoints: list[tuple[float, float]]
    junction_count: int = Field(ge=0)
    junctions: list[tuple[float, float]]
    junction_regions: list[JunctionRegionSummary] = Field(default_factory=list)
    raw_junction_seed_count: int = Field(ge=0)
    cycle_count: int = Field(ge=0)
    component_count: int = Field(ge=0)


@dataclass(frozen=True)
class OrderedPath:
    xy: np.ndarray
    covered_edge_count: int
    total_edge_count: int
    walk: SkeletonWalk | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("ordered path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("ordered path coordinates must be finite")
        if self.total_edge_count < 1 or self.covered_edge_count < 0:
            raise ValueError("ordered path edge counts are invalid")
        if self.covered_edge_count > self.total_edge_count:
            raise ValueError("covered edge count cannot exceed total edge count")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "xy", points)

    @property
    def coverage(self) -> float:
        return self.covered_edge_count / self.total_edge_count


@dataclass(frozen=True)
class ContinuousPath:
    xy: np.ndarray
    tangent: np.ndarray
    curvature: np.ndarray
    radius_px: np.ndarray
    confidence: np.ndarray
    progress: np.ndarray
    arc_length_px: float
    source: str
    boundary_evidence: str
    route_event_indices: tuple[int, ...] = ()
    route_event_progress: tuple[float, ...] = ()
    edge_ids: tuple[int, ...] = ()
    edge_directions: tuple[Literal["forward", "reverse"], ...] = ()

    def __post_init__(self) -> None:
        points = np.asarray(self.xy, dtype=float)
        tangents = np.asarray(self.tangent, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("continuous path must have shape (N, 2) with N >= 2")
        if tangents.shape != points.shape:
            raise ValueError("continuous path tangents must match its coordinates")
        arrays = {
            "curvature": np.asarray(self.curvature, dtype=float),
            "radius_px": np.asarray(self.radius_px, dtype=float),
            "confidence": np.asarray(self.confidence, dtype=float),
            "progress": np.asarray(self.progress, dtype=float),
        }
        if any(values.shape != (len(points),) for values in arrays.values()):
            raise ValueError("continuous path scalar arrays must match its coordinates")
        if not np.isfinite(points).all() or not np.isfinite(tangents).all():
            raise ValueError("continuous path coordinates and tangents must be finite")
        if any(not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("continuous path scalar arrays must be finite")
        if self.arc_length_px <= 0.0 or not np.isfinite(self.arc_length_px):
            raise ValueError("continuous path arc length must be finite and positive")
        if np.any(arrays["radius_px"] < 0.0):
            raise ValueError("continuous path radii must be nonnegative")
        if np.any((arrays["confidence"] < 0.0) | (arrays["confidence"] > 1.0)):
            raise ValueError("continuous path confidence must be between zero and one")
        if np.any(np.diff(arrays["progress"]) < 0.0):
            raise ValueError("continuous path progress must be nondecreasing")
        if len(self.route_event_indices) != len(self.route_event_progress):
            raise ValueError("route event indices and progress values must match")
        if any(index < 0 or index >= len(points) for index in self.route_event_indices):
            raise ValueError("route event index is outside the continuous path")
        if any(not 0.0 <= value <= 1.0 for value in self.route_event_progress):
            raise ValueError("route event progress must be between zero and one")
        for name, values in (("xy", points), ("tangent", tangents), *arrays.items()):
            owned = np.array(values, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


@dataclass(frozen=True)
class WritingEvent:
    kind: Literal["junction-encounter", "pen-lift", "stroke-start", "stroke-end"]
    progress: float
    junction_node_id: int | None = None
    occurrence: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.progress) or not 0.0 <= self.progress <= 1.0:
            raise ValueError("writing-event progress must be between zero and one")
        if self.kind == "junction-encounter" and self.junction_node_id is None:
            raise ValueError("junction encounters require a semantic junction node")
        if self.occurrence is not None and self.occurrence < 1:
            raise ValueError("junction occurrence numbers must be positive")


@dataclass(frozen=True)
class WritingRouteScore:
    coverage_cost: float
    repeat_cost: float
    lift_cost: float
    continuity_cost: float
    steering_complexity_cost: float
    junction_pairing_cost: float = 0.0
    counterpart_similarity_cost: float | None = None
    structure_prior_cost: float = 0.0

    @property
    def intrinsic_cost(self) -> float:
        return (
            1_000_000.0 * self.coverage_cost
            + 100.0 * self.repeat_cost
            + 10.0 * self.lift_cost
            + 10.0 * self.continuity_cost
            + self.steering_complexity_cost
            + self.junction_pairing_cost
            + self.structure_prior_cost
        )


@dataclass(frozen=True)
class WritingDirectionPolicy:
    script_direction: Literal["left-to-right", "right-to-left", "neutral"] = "left-to-right"
    endpoint_bias_weight: float = 0.2
    direction_weight: float = 0.2
    pen_lift_weight: float = 0.05
    junction_continuity_weight: float = 0.1
    pen_lift_geometry_tolerance: float = 0.1
    ambiguity_margin: float = 0.005

    def __post_init__(self) -> None:
        if self.script_direction not in {"left-to-right", "right-to-left", "neutral"}:
            raise ValueError("unsupported script direction")
        weights = (
            self.endpoint_bias_weight,
            self.direction_weight,
            self.pen_lift_weight,
            self.junction_continuity_weight,
            self.pen_lift_geometry_tolerance,
            self.ambiguity_margin,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("writing direction policy values must be finite and nonnegative")


@dataclass(frozen=True)
class WritingStructurePolicy:
    preferred_body_strokes: int = 1
    detached_mark_stroke_weight: float = 0.25
    unexpected_body_lift_weight: float = 0.25
    side_component_penalty: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.preferred_body_strokes, bool)
            or not isinstance(self.preferred_body_strokes, int)
            or self.preferred_body_strokes < 1
        ):
            raise ValueError("preferred_body_strokes must be a positive integer")
        weights = (
            self.detached_mark_stroke_weight,
            self.unexpected_body_lift_weight,
            self.side_component_penalty,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("writing structure policy weights must be finite and nonnegative")


@dataclass(frozen=True)
class WritingPriorScore:
    pen_lift_cost: float
    start_endpoint_cost: float
    end_endpoint_cost: float
    direction_cost: float
    junction_continuity_cost: float
    total: float


@dataclass(frozen=True)
class WritingRouteCandidate:
    hypothesis: StrokeHypothesis
    continuous_strokes: tuple[ContinuousPath, ...]
    events: tuple[WritingEvent, ...]
    steering_models: tuple[TrajectoryModelSelectionResult, ...]
    score: WritingRouteScore
    ambiguous: bool
    physical_junction_count: int
    cycle_count: int
    orientation_variant: Literal["native", "global-reverse"] = "native"
    orientation_evidence: Literal[
        "graph-order-only", "pair-similarity", "writing-prior", "resolved"
    ] = "graph-order-only"
    orientation_status: Literal["resolved", "ambiguous"] = "ambiguous"
    orientation_margin: float | None = None
    orientation_group: int = 0
    glyph_endpoints: tuple[tuple[float, float], ...] = ()
    writing_prior: WritingPriorScore | None = None
    reconstruction: MinimumComplexityReconstruction | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class WritingRoutePairScore:
    reference_rank: int
    sample_rank: int
    intrinsic_route_cost: float
    aligned_centerline_cost: float
    steering_profile_cost: float
    topology_event_compatibility_cost: float
    total_cost: float
    geometry_cost: float = 0.0
    reconstruction_fit_cost: float = 0.0
    model_complexity_cost: float = 0.0
    writing_prior_cost: float = 0.0
    pen_lift_count: int = 0
    descriptor_matching_cost: float = 0.0

    stroke_pattern_cost: float = 0.0
    required_rotation_deg: float | None = None
    orientation_eligible: bool = True
    orientation_rejection_reason: str | None = None
    eligible: bool = False

    @property
    def cost_breakdown(self) -> dict[str, float]:
        return {
            "intrinsic_route_cost": self.intrinsic_route_cost,
            "aligned_centerline_cost": self.aligned_centerline_cost,
            "steering_profile_cost": self.steering_profile_cost,
            "topology_event_compatibility_cost": self.topology_event_compatibility_cost,
            "writing_prior_cost": self.writing_prior_cost,
            "descriptor_matching_cost": self.descriptor_matching_cost,
            "stroke_pattern_cost": self.stroke_pattern_cost,
            "reconstruction_fit_cost": self.reconstruction_fit_cost,
            "model_complexity_cost": self.model_complexity_cost,
        }

    def __post_init__(self) -> None:
        costs = (*self.cost_breakdown.values(), self.geometry_cost, self.total_cost)
        if any(not np.isfinite(cost) or cost < 0.0 for cost in costs):
            raise ValueError("route-pair costs must be finite and nonnegative")
        if not np.isclose(self.total_cost, sum(self.cost_breakdown.values())):
            raise ValueError("total_cost must equal the sum of named route-pair costs")


@dataclass(frozen=True)
class WritingRoutePairSelection:
    reference: WritingRouteCandidate
    sample: WritingRouteCandidate
    score: WritingRoutePairScore
    ambiguous: bool
    alternatives: tuple[WritingRoutePairScore, ...]


@dataclass(frozen=True)
class IdentityAlignmentPolicy:
    max_rotation_deg: float = 15.0
    rotation_soft_sigma_deg: float = 6.0
    min_scale_ratio: float | None = 0.5
    max_scale_ratio: float | None = 2.0
    allow_hidden_reversal: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_rotation_deg) or not 0.0 <= self.max_rotation_deg <= 180.0:
            raise ValueError("max_rotation_deg must be between 0 and 180")
        if not np.isfinite(self.rotation_soft_sigma_deg) or self.rotation_soft_sigma_deg <= 0.0:
            raise ValueError("rotation_soft_sigma_deg must be finite and positive")
        if self.min_scale_ratio is not None and self.min_scale_ratio <= 0.0:
            raise ValueError("min_scale_ratio must be positive")
        if self.max_scale_ratio is not None and self.max_scale_ratio <= 0.0:
            raise ValueError("max_scale_ratio must be positive")
        if (
            self.min_scale_ratio is not None
            and self.max_scale_ratio is not None
            and self.min_scale_ratio > self.max_scale_ratio
        ):
            raise ValueError("min_scale_ratio cannot exceed max_scale_ratio")


@dataclass(frozen=True)
class AlignmentResult:
    transform: Similarity2D
    direction: Literal["forward", "reversed-for-best-match"]
    aligned_sample_xy: np.ndarray
    mean_squared_error: float
    aligned_sample_path: ContinuousPath | None = None
    required_rotation_deg: float | None = None
    orientation_eligible: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.aligned_sample_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("aligned path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all() or not np.isfinite(self.mean_squared_error):
            raise ValueError("alignment result must be finite")
        if self.aligned_sample_path is not None:
            if self.aligned_sample_path.xy.shape != points.shape:
                raise ValueError("aligned continuous path must match aligned coordinates")
            if not np.allclose(self.aligned_sample_path.xy, points):
                raise ValueError(
                    "aligned continuous path coordinates must match aligned coordinates"
                )
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "aligned_sample_xy", points)


@dataclass(frozen=True)
class WritingRouteAlignment:
    transform: Similarity2D
    direction: Literal["forward", "reversed-for-best-match"]
    aligned_sample_strokes: tuple[ContinuousPath, ...]
    stroke_mean_squared_errors: tuple[float, ...]
    mean_squared_error: float
    stroke_transforms: tuple[Similarity2D, ...] = ()
    strategy: Literal["shared", "per-stroke"] = "shared"

    def __post_init__(self) -> None:
        strokes = tuple(self.aligned_sample_strokes)
        errors = tuple(float(value) for value in self.stroke_mean_squared_errors)
        transforms = tuple(self.stroke_transforms)
        if len(transforms) != len(errors):
            raise ValueError("route alignment must keep one transform per paired stroke")
        if len(errors) > len(strokes) or any(not np.isfinite(value) for value in errors):
            raise ValueError("route alignment stroke errors are invalid")
        if not np.isfinite(self.mean_squared_error):
            raise ValueError("route alignment error must be finite")
        object.__setattr__(self, "aligned_sample_strokes", strokes)
        object.__setattr__(self, "stroke_mean_squared_errors", errors)
        object.__setattr__(self, "stroke_transforms", transforms)


@dataclass(frozen=True)
class SkeletonDiagnostics:
    skeleton: np.ndarray
    radius: np.ndarray
    graph: SkeletonGraph
    topology: TopologySummary
    analysis: SkeletonAnalysis | None = None

    def __post_init__(self) -> None:
        skeleton = np.asarray(self.skeleton, dtype=bool)
        radius = np.asarray(self.radius, dtype=float)
        if skeleton.ndim != 2 or radius.shape != skeleton.shape:
            raise ValueError("skeleton and radius must be matching 2D arrays")
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius map must be finite and nonnegative")
        skeleton = np.array(skeleton, copy=True)
        radius = np.array(radius, copy=True)
        skeleton.setflags(write=False)
        radius.setflags(write=False)
        object.__setattr__(self, "skeleton", skeleton)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class ComparisonIdentity:
    sample_id: str
    reference_id: str
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.reference_id:
            raise ValueError("sample and reference IDs must not be empty")
        if self.label is not None and len(self.label) != 1:
            raise ValueError("letter label must contain exactly one character")


@dataclass(frozen=True)
class ExtractedLetter:
    source_path: Path
    crop: CropRect
    image: Image.Image
    mask_result: MaskResult
    diagnostics: SkeletonDiagnostics
    ordered_path: OrderedPath | None
    resampled_xy: np.ndarray | None
    continuous_path: ContinuousPath | None = None
    geometry_warning: str | None = None
    route_evidence_status: Literal[
        "available", "truncated", "unavailable", "search-truncated-fallback"
    ] = "available"
    route_search_truncated: bool = False
    route_warning: str | None = None
    route_search_failure: dict[str, str | int] | None = None
    route_candidates: tuple[WritingRouteCandidate, ...] = ()
    route_descriptors: tuple[RouteDescriptor, ...] = ()
    foreground_evidence: ForegroundEvidence | None = None
    component_role_sequence: tuple[str, ...] = ()
    secondary_mark_centroids: tuple[tuple[float, float], ...] = ()
    selected_route: WritingRouteCandidate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "component_role_sequence", tuple(self.component_role_sequence))
        object.__setattr__(self, "secondary_mark_centroids", tuple(self.secondary_mark_centroids))
        if any(
            len(centroid) != 2 or not np.isfinite(centroid).all()
            for centroid in self.secondary_mark_centroids
        ):
            raise ValueError("secondary mark centroids must be finite coordinate pairs")
        object.__setattr__(self, "route_candidates", tuple(self.route_candidates))
        object.__setattr__(self, "route_descriptors", tuple(self.route_descriptors))
        if self.route_descriptors and len(self.route_descriptors) != len(self.route_candidates):
            raise ValueError("route descriptors must align with route candidates")
        expected_shape = (self.crop.height, self.crop.width)
        if self.image.size != (self.crop.width, self.crop.height):
            raise ValueError("extracted image dimensions must match its crop rectangle")
        if self.mask_result.mask.shape != expected_shape:
            raise ValueError("ink mask dimensions must match the extracted image")
        if (
            self.foreground_evidence is not None
            and self.foreground_evidence.mask.shape != expected_shape
        ):
            raise ValueError("raster foreground evidence dimensions must match the extracted image")
        if (
            self.selected_route is not None
            and len(self.selected_route.continuous_strokes) > 1
            and any(
                value is not None
                for value in (self.ordered_path, self.resampled_xy, self.continuous_path)
            )
        ):
            raise ValueError("multi-stroke selected routes cannot populate single-path fields")
        if self.ordered_path is None and (
            self.resampled_xy is not None or self.continuous_path is not None
        ):
            raise ValueError("continuous or resampled geometry requires an ordered path")
        if self.continuous_path is not None and self.resampled_xy is None:
            object.__setattr__(self, "resampled_xy", self.continuous_path.xy)
        if self.resampled_xy is None:
            return
        points = np.asarray(self.resampled_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("resampled path must have shape (N, 2) with N >= 2")
        if not np.isfinite(points).all():
            raise ValueError("resampled path coordinates must be finite")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "resampled_xy", points)


@dataclass(frozen=True)
class ComparisonResult:
    letter: str | None
    reference: ExtractedLetter
    sample: ExtractedLetter
    alignment: AlignmentResult | None
    smoothing_window: int = 11
    smoothing_polynomial_order: int = 2
    warnings: list[str] = field(default_factory=list)
    identity: ComparisonIdentity | None = None
    identity_status: Literal["accepted", "candidate_diagnostic"] = "accepted"
    route_selection: WritingRoutePairSelection | None = None
    route_alignment: WritingRouteAlignment | None = None

    def __post_init__(self) -> None:
        identity = self.identity
        if identity is None:
            if self.letter is None:
                raise ValueError("comparison identity or letter label is required")
            identity = ComparisonIdentity("sample", "reference", self.letter)
            object.__setattr__(self, "identity", identity)
        elif self.letter is not None and identity.label != self.letter:
            raise ValueError("comparison letter and identity label must match")
        else:
            object.__setattr__(self, "letter", identity.label)
        has_paths = self.reference.resampled_xy is not None and self.sample.resampled_xy is not None
        if self.route_alignment is not None and self.route_selection is None:
            raise ValueError("route alignment requires a selected writing route pair")
        if self.alignment is not None:
            if not has_paths:
                raise ValueError("alignment requires ordered paths for both inputs")
            if len(self.reference.resampled_xy) != len(self.sample.resampled_xy):
                raise ValueError("reference and sample paths must have the same sample count")
            if len(self.alignment.aligned_sample_xy) != len(self.reference.resampled_xy):
                raise ValueError("aligned sample path must match the reference sample count")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing window must be odd and at least three")
        if not 1 <= self.smoothing_polynomial_order < self.smoothing_window:
            raise ValueError("smoothing polynomial order is invalid")
        object.__setattr__(self, "warnings", list(self.warnings))
