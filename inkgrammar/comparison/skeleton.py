from __future__ import annotations

import numpy as np
from skeletonpaths import SkeletonAnalysis, SkeletonGraph, analyze_mask, graph_from_skeleton

from .models import JunctionRegionSummary, SkeletonDiagnostics, TopologySummary


def _diagnostics_from_graph(
    skeleton: np.ndarray,
    radius: np.ndarray,
    graph: SkeletonGraph,
    analysis: SkeletonAnalysis | None = None,
) -> SkeletonDiagnostics:
    endpoints = [node.xy for node in graph.nodes if node.degree == 1]
    junction_regions = graph.junction_regions
    topology = TopologySummary(
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        endpoint_count=len(endpoints),
        endpoints=endpoints,
        junction_count=len(junction_regions),
        junctions=[region.xy for region in junction_regions],
        junction_regions=[
            JunctionRegionSummary(
                node_id=region.node_id,
                xy=region.xy,
                radius_px=region.radius_px,
                support_pixels=list(region.support_pixels),
                port_count=region.port_count,
                confidence=region.confidence,
            )
            for region in junction_regions
        ],
        raw_junction_seed_count=graph.raw_junction_seed_count,
        cycle_count=graph.cycle_count,
        component_count=graph.component_count,
    )
    return SkeletonDiagnostics(skeleton, radius, graph, topology, analysis)


def diagnostics_from_skeleton(
    skeleton: np.ndarray,
    radius: np.ndarray,
) -> SkeletonDiagnostics:
    """Build topology diagnostics from persisted skeleton and radius rasters."""
    skeleton = np.asarray(skeleton, dtype=bool)
    radius = np.asarray(radius, dtype=float)
    graph = graph_from_skeleton(skeleton, radius=radius)
    return _diagnostics_from_graph(skeleton, radius, graph)


def diagnostics_from_analysis(analysis: SkeletonAnalysis) -> SkeletonDiagnostics:
    """Build diagnostics while preserving a prepared SkeletonPaths analysis."""
    return _diagnostics_from_graph(
        analysis.skeleton,
        analysis.radius,
        analysis.graph,
        analysis,
    )


def analyze_skeleton(mask: np.ndarray) -> SkeletonDiagnostics:
    ink = np.array(mask, dtype=bool, copy=True)
    if ink.ndim != 2:
        raise ValueError("ink mask must be a 2D array")
    return diagnostics_from_analysis(analyze_mask(ink))
