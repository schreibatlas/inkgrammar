from __future__ import annotations

import numpy as np
from planarcurve.polyline import polyline_length, resample_polyline_by_arclength
from planarcurve.transforms import fit_similarity_transform
from skeletonpaths import covering_walk

from .models import AlignmentResult, OrderedPath, SkeletonDiagnostics


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
    return OrderedPath(walk.xy, walk.covered_edge_count, walk.total_edge_count)


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
