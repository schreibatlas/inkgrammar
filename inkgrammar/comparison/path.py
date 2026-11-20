from __future__ import annotations

import numpy as np
from planarcurve.polyline import polyline_length, resample_polyline_by_arclength
from planarcurve.transforms import Similarity2D, fit_similarity_transform
from skeletonpaths import (
    RefinedSkeletonGeometry,
    SkeletonWalk,
    SubpixelReconstructionError,
    covering_walk,
    refine_graph_geometry,
    refined_walk,
)

from .models import AlignmentResult, ContinuousPath, OrderedPath, SkeletonDiagnostics


class PathExtractionError(ValueError):
    """Raised when the skeleton cannot provide a complete ordered path."""


def recover_ordered_path(diagnostics: SkeletonDiagnostics) -> OrderedPath:
    graph = diagnostics.graph
    if not graph.edges:
        raise PathExtractionError("skeleton has no traversable edges")
    if graph.component_count != 1:
        raise PathExtractionError(f"ordered walk covers 0 of {len(graph.edges)} skeleton edges")

    endpoints = sorted(
        (node for node in graph.nodes if node.degree == 1),
        key=lambda node: (node.xy[0], node.xy[1]),
    )
    if len(endpoints) >= 2:
        start, end = endpoints[0], endpoints[-1]
    else:
        start = end = min(graph.nodes, key=lambda node: (node.xy[0], node.xy[1]))
    walk = covering_walk(graph, start_node=start.id, end_node=end.id)
    if len(walk.xy) < 2 or polyline_length(walk.xy) <= 0.0:
        raise PathExtractionError("ordered skeleton path has no finite positive arc length")
    return OrderedPath(
        walk.xy,
        walk.covered_edge_count,
        walk.total_edge_count,
        walk,
    )


def _semantic_junction_transition_indices(
    diagnostics: SkeletonDiagnostics,
    walk: SkeletonWalk,
    refined_graph: RefinedSkeletonGeometry,
    progress: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    semantic_node_ids = {region.node_id for region in diagnostics.graph.junction_regions}
    edge_lookup = {edge.edge_id: edge for edge in refined_graph.edges}
    transitions = []
    offset = 0

    def append_event(node_id: int, sample_index: int) -> None:
        if node_id not in semantic_node_ids:
            return
        if any(
            existing_node == node_id
            and np.isclose(progress[existing_index], progress[sample_index], atol=1e-12)
            for existing_node, existing_index in transitions
        ):
            return
        transitions.append((node_id, sample_index))

    append_event(walk.start_node, 0)
    for index, (edge_id, direction) in enumerate(
        zip(walk.edge_ids, walk.edge_directions, strict=True)
    ):
        if index:
            previous_edge = diagnostics.graph.edges[walk.edge_ids[index - 1]]
            current_edge = diagnostics.graph.edges[edge_id]
            previous_direction = walk.edge_directions[index - 1]
            previous_node = previous_edge.b if previous_direction == "forward" else previous_edge.a
            current_node = current_edge.a if direction == "forward" else current_edge.b
            if previous_node != current_node:
                raise PathExtractionError("walk edge sequence does not meet at a shared graph node")
            append_event(previous_node, offset)
        offset += len(edge_lookup[edge_id].xy)
    if offset != len(progress):
        raise SubpixelReconstructionError("refined walk samples do not match its edge sequence")
    append_event(walk.end_node, len(progress) - 1)
    return tuple(transitions)


def refine_continuous_walk(
    diagnostics: SkeletonDiagnostics,
    walk: SkeletonWalk,
    *,
    decision_field: np.ndarray | None = None,
    refined_graph: RefinedSkeletonGeometry | None = None,
) -> ContinuousPath:
    if diagnostics.analysis is None:
        raise PathExtractionError("skeleton analysis is unavailable")
    if refined_graph is None:
        refined_graph = refine_graph_geometry(
            diagnostics.analysis,
            decision_field=decision_field,
        )
    refined = refined_walk(refined_graph, walk)
    xy = np.asarray(refined.xy, dtype=float)
    if len(xy) < 2:
        raise PathExtractionError("refined covering walk has fewer than two samples")
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    sampled_length = float(np.sum(segment_lengths))
    if sampled_length <= 0.0:
        raise PathExtractionError("refined covering walk has no positive arc length")
    progress = np.clip(
        np.concatenate(([0.0], np.cumsum(segment_lengths))) / sampled_length,
        0.0,
        1.0,
    )
    transitions = _semantic_junction_transition_indices(
        diagnostics,
        walk,
        refined_graph,
        progress,
    )
    boundary_evidence = (
        "rasterregions-decision-field"
        if decision_field is not None
        else "skeletonpaths-binary-distance-fallback"
    )
    event_indices = tuple(index for _, index in transitions)
    return ContinuousPath(
        xy=xy,
        tangent=refined.tangent,
        curvature=refined.curvature,
        radius_px=refined.radius_px,
        confidence=refined.confidence,
        progress=progress,
        arc_length_px=refined.length_px,
        source="skeletonpaths-subpixel-v1",
        boundary_evidence=boundary_evidence,
        route_event_indices=event_indices,
        route_event_progress=tuple(float(progress[index]) for index in event_indices),
        edge_ids=refined.edge_ids,
        edge_directions=refined.edge_directions,
    )


def refine_continuous_path(
    diagnostics: SkeletonDiagnostics,
    ordered_path: OrderedPath,
    *,
    decision_field: np.ndarray | None = None,
) -> ContinuousPath:
    if ordered_path.walk is None:
        raise PathExtractionError("covering-walk provenance is unavailable")
    return refine_continuous_walk(
        diagnostics,
        ordered_path.walk,
        decision_field=decision_field,
    )


def resample_path(points_xy: np.ndarray, sample_count: int = 256) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must have shape (N, 2) with N >= 2")
    if not np.isfinite(points).all():
        raise ValueError("path coordinates must be finite")
    if sample_count < 2:
        raise ValueError("sample count must be at least two")
    if polyline_length(points) <= 0.0:
        raise ValueError("path must have positive arc length")
    return resample_polyline_by_arclength(points, sample_count)


def align_ordered_paths(reference_xy: np.ndarray, sample_xy: np.ndarray) -> AlignmentResult:
    reference = np.asarray(reference_xy, dtype=float)
    sample = np.asarray(sample_xy, dtype=float)
    if reference.shape != sample.shape:
        raise ValueError("reference and sample paths must have matching sample counts")

    candidates = []
    for direction, oriented_sample in (
        ("forward", sample),
        ("reversed-for-best-match", sample[::-1]),
    ):
        transform = fit_similarity_transform(reference, oriented_sample)
        aligned = transform.apply(oriented_sample)
        error = float(np.mean(np.sum((reference - aligned) ** 2, axis=1)))
        candidates.append((error, direction, transform, aligned))
    error, direction, transform, aligned = min(candidates, key=lambda candidate: candidate[0])
    return AlignmentResult(transform, direction, aligned, error)


def resample_continuous_path(
    path: ContinuousPath,
    sample_count: int = 256,
) -> ContinuousPath:
    if sample_count < 2:
        raise ValueError("sample count must be at least two")
    progress = np.linspace(0.0, 1.0, sample_count)

    def interpolate(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.interp(progress, path.progress, values)
        return np.column_stack(
            [np.interp(progress, path.progress, values[:, axis]) for axis in range(values.shape[1])]
        )

    xy = interpolate(path.xy)
    tangent = interpolate(path.tangent)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm <= 1e-12):
        raise ValueError("continuous path has an undefined sampled tangent")
    tangent = tangent / tangent_norm[:, None]
    event_indices = tuple(
        int(np.argmin(np.abs(progress - event_progress)))
        for event_progress in path.route_event_progress
    )
    return ContinuousPath(
        xy=xy,
        tangent=tangent,
        curvature=interpolate(path.curvature),
        radius_px=interpolate(path.radius_px),
        confidence=interpolate(path.confidence),
        progress=progress,
        arc_length_px=path.arc_length_px,
        source=path.source,
        boundary_evidence=path.boundary_evidence,
        route_event_indices=event_indices,
        route_event_progress=path.route_event_progress,
        edge_ids=path.edge_ids,
        edge_directions=path.edge_directions,
    )


def reverse_continuous_path(path: ContinuousPath) -> ContinuousPath:
    indexed_events = sorted(
        (len(path.xy) - 1 - index, 1.0 - progress)
        for index, progress in zip(
            path.route_event_indices,
            path.route_event_progress,
            strict=True,
        )
    )
    return ContinuousPath(
        xy=path.xy[::-1],
        tangent=-path.tangent[::-1],
        curvature=-path.curvature[::-1],
        radius_px=path.radius_px[::-1],
        confidence=path.confidence[::-1],
        progress=1.0 - path.progress[::-1],
        arc_length_px=path.arc_length_px,
        source=path.source,
        boundary_evidence=path.boundary_evidence,
        route_event_indices=tuple(index for index, _ in indexed_events),
        route_event_progress=tuple(value for _, value in indexed_events),
        edge_ids=path.edge_ids[::-1],
        edge_directions=tuple(
            "reverse" if direction == "forward" else "forward"
            for direction in path.edge_directions[::-1]
        ),
    )


def transform_continuous_path(
    path: ContinuousPath,
    transform: Similarity2D,
) -> ContinuousPath:
    angle = np.deg2rad(transform.rotation_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return ContinuousPath(
        xy=transform.apply(path.xy),
        tangent=path.tangent @ rotation.T,
        curvature=path.curvature / transform.scale,
        radius_px=path.radius_px * transform.scale,
        confidence=path.confidence,
        progress=path.progress,
        arc_length_px=path.arc_length_px * transform.scale,
        source=path.source,
        boundary_evidence=path.boundary_evidence,
        route_event_indices=path.route_event_indices,
        route_event_progress=path.route_event_progress,
        edge_ids=path.edge_ids,
        edge_directions=path.edge_directions,
    )


def align_continuous_paths(
    reference: ContinuousPath,
    sample: ContinuousPath,
    sample_count: int = 256,
) -> AlignmentResult:
    reference_samples = resample_continuous_path(reference, sample_count)
    sample_samples = resample_continuous_path(sample, sample_count)
    candidates = []
    for direction, oriented_sample in (
        ("forward", sample_samples),
        ("reversed-for-best-match", reverse_continuous_path(sample_samples)),
    ):
        transform = fit_similarity_transform(reference_samples.xy, oriented_sample.xy)
        aligned_path = transform_continuous_path(oriented_sample, transform)
        error = float(np.mean(np.sum((reference_samples.xy - aligned_path.xy) ** 2, axis=1)))
        candidates.append((error, direction, transform, aligned_path))
    error, direction, transform, aligned_path = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return AlignmentResult(
        transform=transform,
        direction=direction,
        aligned_sample_xy=aligned_path.xy,
        mean_squared_error=error,
        aligned_sample_path=aligned_path,
    )
