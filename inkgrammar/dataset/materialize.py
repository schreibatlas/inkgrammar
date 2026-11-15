"""Deterministic dataset materialization.

Turns a dataset recipe plus authorized private sources into a materialized
dataset root (the brief's section 13 layout): per-page extraction output,
raw/clean masks with cleanup records, normalized previews, derived
skeleton/centerline/topology artifacts with algorithm versions, and a
canonical ``index.json``. All artifact paths are relative to the dataset root
and all JSON is written canonically (sorted keys), so two clean runs produce
byte-identical semantic artifacts.

Materialization combines automatic candidates with review patches into the
effective dataset; stale patches are reported and never applied.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rasterregions import components as raster_components
from skeletonpaths import (
    SkeletonGraph,
    SkeletonWalk,
    analyze_mask,
    covering_walk,
    graph_from_skeleton,
)

from ..io import load_layout
from .layouts import load_region_layout
from .models import ObservationKind, ObservationManifest
from .preprocess import (
    PreprocessParams,
    estimate_metric_guides,
    isolate_target_ink,
    preprocess_crop,
)
from .project import DatasetPage, DatasetProjectBundle, load_observation_manifest, verify_sources
from .review import apply_review_patch, load_review_patches
from .transforms import standard_frame

DATASET_INDEX_FORMAT = "inkgrammar-materialized-dataset"
EXTRACTION_VERSION = 1
SOURCE_VERIFICATION_FORMAT = "inkgrammar-source-verification"
DERIVE_ALGORITHMS = {
    "skeleton": "skeletonpaths-analyze-mask-v1+spur-prune-3px-v1",
    "centerline": "skeletonpaths-covering-walk-v1+leftmost-latin-v1",
    "topology": "skeletonpaths-pixel-graph-v1",
}
MIN_SPUR_LENGTH_PX = 3

# Normalized preview canvas convention (preview only, not stored geometry):
# glyph x = 0 at canvas x 16, baseline (glyph y = 0) at canvas y 72, one
# glyph unit (x-height when known) is 32 canvas px, y upward against image y.
PREVIEW_SIZE = (128, 96)
PREVIEW_X_ORIGIN = 16
PREVIEW_BASELINE = 72
PREVIEW_UNIT = 32


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> str:
    """Write canonical JSON and return its SHA-256."""
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return _sha256_bytes(payload.encode("utf-8"))


def _write_mask(path: Path, mask: np.ndarray) -> str:
    """Write a binary mask PNG and return its SHA-256."""
    buffer = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


# --- derived artifacts (section 28: candidates, not source truth) ---


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _left_to_right_covering_walk(graph: SkeletonGraph) -> SkeletonWalk:
    endpoints = sorted(
        (node for node in graph.nodes if node.degree == 1),
        key=lambda node: (node.xy[0], node.xy[1]),
    )
    if len(endpoints) >= 2:
        start, end = endpoints[0], endpoints[-1]
    else:
        start = end = min(graph.nodes, key=lambda node: (node.xy[0], node.xy[1]))
    return covering_walk(graph, start_node=start.id, end_node=end.id)


def derive_artifacts(clean_mask: np.ndarray) -> dict[str, dict]:
    """Compute derived skeleton/centerline/topology artifacts with versions.

    These are derived computational candidates, never source truth; each
    artifact records the algorithm and version that produced it.
    """
    analysis = analyze_mask(clean_mask, min_spur_length=MIN_SPUR_LENGTH_PX)
    skeleton = analysis.skeleton
    graph = analysis.graph
    pixel_count = int(skeleton.sum())
    endpoints = [[int(node.xy[0]), int(node.xy[1])] for node in graph.nodes if node.degree == 1]
    junctions = [[int(node.xy[0]), int(node.xy[1])] for node in graph.nodes if node.degree > 2]

    candidates = []
    components = raster_components(skeleton)
    for component in components:
        component_skeleton = np.zeros_like(skeleton)
        component_skeleton[component.ys, component.xs] = True
        component_graph = graph_from_skeleton(component_skeleton, radius=analysis.radius)
        if not component_graph.edges:
            continue
        walk = _left_to_right_covering_walk(component_graph)
        points = [[int(x), int(y)] for x, y in walk.xy]
        length = float(np.linalg.norm(np.diff(walk.xy, axis=0), axis=1).sum())
        candidates.append(
            {
                "points": points,
                "start": points[0],
                "end": points[-1],
                "length_px": round(length, 4),
                "stitched": graph.component_count == 1,
            }
        )
    candidates.sort(key=lambda candidate: candidate["length_px"], reverse=True)

    return {
        "skeleton": {
            "data": _npy_bytes(skeleton),
            "format": "npy",
            "algorithm": DERIVE_ALGORITHMS["skeleton"],
        },
        "centerline": {
            "data": {"algorithm": DERIVE_ALGORITHMS["centerline"], "candidates": candidates},
            "format": "json",
            "algorithm": DERIVE_ALGORITHMS["centerline"],
        },
        "topology": {
            "data": {
                "algorithm": DERIVE_ALGORITHMS["topology"],
                "pixel_count": pixel_count,
                "components": graph.component_count,
                "endpoints": endpoints,
                "junctions": junctions,
                "cycle_rank": graph.cycle_count,
            },
            "format": "json",
            "algorithm": DERIVE_ALGORITHMS["topology"],
        },
    }


# --- normalized preview ---


def render_normalized_preview(
    clean_mask: np.ndarray, *, baseline_px: float, unit_px: float
) -> Image.Image:
    height, width = clean_mask.shape
    canvas_y, canvas_x = np.mgrid[0 : PREVIEW_SIZE[1], 0 : PREVIEW_SIZE[0]]
    source_x = (canvas_x - PREVIEW_X_ORIGIN) / PREVIEW_UNIT * unit_px
    source_y = baseline_px - (PREVIEW_BASELINE - canvas_y) / PREVIEW_UNIT * unit_px
    sample_x = np.rint(source_x).astype(int)
    sample_y = np.rint(source_y).astype(int)
    valid = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    canvas = np.zeros(PREVIEW_SIZE[::-1], dtype=bool)
    canvas[valid] = clean_mask[sample_y[valid], sample_x[valid]]
    return Image.fromarray((canvas * 255).astype(np.uint8))


def _preview_frame(mask: np.ndarray, metrics: dict | None) -> tuple[float, float]:
    """Baseline and unit (px) for the preview: metric candidates first."""
    rows = np.flatnonzero(mask.any(axis=1))
    ink_bottom = float(rows.max()) if rows.size else 0.0
    ink_height = float(rows.max() - rows.min() + 1) if rows.size else 1.0
    if metrics and metrics.get("baseline_y") is not None:
        baseline_px = float(metrics["baseline_y"]) * mask.shape[0]
        for key in ("x_height_y", "cap_height_y"):
            if metrics.get(key) is not None:
                return baseline_px, abs(baseline_px - float(metrics[key]) * mask.shape[0])
        return baseline_px, ink_height
    return ink_bottom, ink_height


# --- materialization ---


def _glyph_class(label: str) -> str:
    if len(label) == 1 and label.islower():
        return "lowercase"
    if len(label) == 1 and label.isupper():
        return "uppercase"
    return "mixed"


def _locator_basename(locator: str) -> str:
    """Basename of a crop locator such as ``local-only:alphabet/xx.png``."""
    return locator.rsplit("/", 1)[-1].split(":")[-1]


def _page_dir_name(bundle: DatasetProjectBundle, page: DatasetPage) -> str:
    stem = Path(page.layout).stem
    return stem.removeprefix(f"{bundle.project.id}-")


def _relations_from_region(record: dict) -> dict:
    relations = {}
    if record.get("members"):
        relations["member_labels"] = record["members"]
    if record.get("category_hint"):
        relations["connection_category"] = record["category_hint"]
    sequence = record.get("sequence") or []
    if record.get("kind") == "connection-pair" and len(sequence) == 2:
        relations["left_label"] = sequence[0]
        relations["right_label"] = sequence[1]
        relations["sequence_text"] = record.get("label") or "".join(sequence)
    elif record.get("kind") in ("word", "sequence") and record.get("label"):
        relations["sequence_text"] = record["label"]
    return relations


_EXTRACTORS = {
    "labeled-rows-v1": "extract_labeled_page",
    "region-layout-v1": "extract_region_page",
    "annotated-region-v1": "extract_annotated_region_page",
}


def _run_extractor(image_path: Path, layout, output_dir: Path, extractor: str) -> list[dict]:
    from .. import page as page_module

    extract = getattr(page_module, _EXTRACTORS[extractor])
    return extract(image_path, layout, output_dir)["detections"]


def _row_centers(
    layout, detection: dict, source_size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...], tuple[float, float]] | None:
    """Reviewed cell center and the row's other label centers, in crop pixels."""
    row = next((row for row in layout.rows if row.id == detection["row"]), None)
    if row is None or row.centers is None:
        return None
    width, height = source_size
    bbox = detection["bbox"]
    row_y = (row.y[0] + row.y[1]) / 2 * height
    points = [(center * width - bbox[0], row_y - bbox[1]) for center in row.centers]
    own = detection["index"]
    neighbors = tuple(point for index, point in enumerate(points) if index != own)
    left_fence = (points[own - 1][0] + points[own][0]) / 2 if own > 0 else -1.0
    right_fence = (
        (points[own][0] + points[own + 1][0]) / 2 if own + 1 < len(points) else width + 1.0
    )
    return points[own], neighbors, (left_fence, right_fence)


def _frame_from_metrics(
    source_bbox: list[int], crop_size: tuple[int, int], metrics: dict | None
) -> dict | None:
    """Candidate ObservationFrame in source pixels from metric guides."""
    baseline_y = (metrics or {}).get("baseline_y")
    x_height_y = (metrics or {}).get("x_height_y")
    if baseline_y is None or x_height_y is None:
        return None
    x_height_px = (baseline_y - x_height_y) * crop_size[1]
    if x_height_px <= 0:
        return None
    frame = standard_frame(
        origin_source_px=(
            float(source_bbox[0]),
            float(source_bbox[1]) + baseline_y * crop_size[1],
        ),
        x_height_px=float(x_height_px),
    )
    return frame.model_dump(mode="json")


def _write_overlay(
    output_dir: Path,
    rel_path: str,
    crop_image: Image.Image,
    clean_mask: np.ndarray,
    kept_mask: np.ndarray,
) -> str:
    """Isolation review overlay: kept ink green, excluded ink red, over crop."""
    gray = np.asarray(crop_image.convert("L"))
    overlay = np.stack([gray, gray, gray], axis=-1)
    overlay[kept_mask] = (60, 180, 60)
    overlay[clean_mask & ~kept_mask] = (200, 40, 40)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    payload = buffer.getvalue()
    path = output_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def materialize_dataset(
    bundle: DatasetProjectBundle,
    output_dir: str | Path,
    *,
    review_dir: str | Path | None = None,
    params: PreprocessParams | None = None,
) -> dict:
    """Materialize the dataset described by ``bundle`` into ``output_dir``.

    Returns the canonical index dict (also written to ``index.json``).
    """
    output_dir = Path(output_dir)
    params = params or PreprocessParams()
    if review_dir is None:
        review_dir = bundle.private_root.parent / "annotations" / bundle.project.id / "reviews"
    review_dir = Path(review_dir)

    verification = verify_sources(bundle)
    if not verification.ok:
        problems = {
            check.source_id: check.problems for check in verification.checks if check.problems
        }
        raise ValueError(f"source verification failed: {problems}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "sources" / "verified.json",
        {
            "format": SOURCE_VERIFICATION_FORMAT,
            "format_version": 1,
            "sources": [check.model_dump() for check in verification.checks],
        },
    )

    manifest: ObservationManifest = load_observation_manifest(bundle.observation_manifest_path)
    source_hashes = {source.source_id: source.sha256 for source in manifest.sources}
    manifest_items = {
        _locator_basename(item.crop_locator): item.model_dump(mode="json")
        for item in manifest.items
        if item.crop_locator
    }
    patches = load_review_patches(review_dir)
    prefix = bundle.project.id.replace("-", "")

    observations: dict[str, dict] = {}
    for page in bundle.project.pages:
        source = next(s for s in manifest.sources if s.source_id == page.source_id)
        image_path = bundle.private_root / source.filename
        page_dir_name = _page_dir_name(bundle, page)
        page_dir = output_dir / page_dir_name
        if page.extractor == "labeled-rows-v1":
            layout = load_layout(bundle.layout_path(page))
        else:
            layout = load_region_layout(bundle.layout_path(page))
        detections = _run_extractor(image_path, layout, page_dir, page.extractor)

        for detection in detections:
            record = _observation_record(
                prefix=prefix,
                page=page,
                layout=layout,
                detection=detection,
                manifest_items=manifest_items,
                output_dir=output_dir,
                page_dir_name=page_dir_name,
                params=params,
                source_size=(source.width_px, source.height_px),
            )
            if record["id"] in observations:
                raise ValueError(f"duplicate observation id: {record['id']}")
            patch = patches.get(record["id"])
            if patch is None:
                record["review_patch"] = {"status": "none"}
            else:
                record, _ = apply_review_patch(
                    record, patch, source_sha256=source_hashes[page.source_id]
                )
            observations[record["id"]] = record

    index = {
        "format": DATASET_INDEX_FORMAT,
        "format_version": 1,
        "model_id": bundle.project.id,
        "source_hashes": source_hashes,
        "extraction_version": EXTRACTION_VERSION,
        "preprocess": params.model_dump(mode="json"),
        "observations": observations,
    }
    _write_json(output_dir / "index.json", index)
    return index


def _observation_record(
    *,
    prefix: str,
    page: DatasetPage,
    layout,
    detection: dict,
    manifest_items: dict[str, dict],
    output_dir: Path,
    page_dir_name: str,
    params: PreprocessParams,
    source_size: tuple[int, int],
) -> dict:
    matched = manifest_items.get(_locator_basename(detection["crop"]))
    label = detection.get("label") or (matched or {}).get("label") or ""
    if "row" in detection:
        kind = (matched or {}).get("kind", "glyph-isolated")
        region_id = f"{detection['row']}:{detection['index']}"
        source_bbox = (matched or {}).get("source_bbox") or detection["bbox"]
        target_crop = detection["crop"]
        confidence = detection.get("confidence")
        relations = (matched or {}).get("relations") or {}
    else:
        kind = detection["kind"]
        region_id = detection["region"]
        ink = detection.get("ink_region")
        source_bbox = (ink or detection).get("bbox")
        target_crop = (ink or {}).get("crop") or detection["crop"]
        confidence = detection.get("confidence")
        relations = _relations_from_region(detection)
    if kind not in ObservationKind.__args__:
        raise ValueError(f"region kind is not an observation kind: {kind}")

    crop_rel = f"{page_dir_name}/{target_crop}"
    crop_image = Image.open(output_dir / crop_rel).convert("RGB")
    result = preprocess_crop(
        crop_image, params, glyph_class=_glyph_class(label) if label else "mixed"
    )
    target_mask = result.clean_mask
    metrics = result.metrics.model_dump(mode="json")
    isolation = None
    centers = _row_centers(layout, detection, source_size) if "row" in detection else None
    if centers is not None:
        center, neighbor_centers, fence = centers
        target_mask, proposal = isolate_target_ink(
            result.clean_mask, center=center, neighbor_centers=neighbor_centers
        )
        # cell ownership fence: cursive neighbors touch, so keep only ink
        # between the midpoints to the adjacent label centers
        columns = np.arange(target_mask.shape[1])
        inside = (columns >= fence[0]) & (columns < fence[1])
        target_mask = target_mask & inside[None, :]
        metrics = estimate_metric_guides(
            target_mask, glyph_class=_glyph_class(label) if label else "mixed"
        ).model_dump(mode="json")
        isolation = {"status": "pending", **proposal.model_dump(mode="json")}
    basename = Path(target_crop).stem
    masks_dir = f"{page_dir_name}/masks"
    derived_dir = f"{page_dir_name}/derived/{basename}"
    artifacts = {
        "crop": crop_rel,
        "raw_mask": f"{masks_dir}/{basename}__raw.png",
        "clean_mask": f"{masks_dir}/{basename}__clean.png",
        "cleanup": f"{masks_dir}/{basename}__cleanup.json",
        "fitting_mask": f"{masks_dir}/{basename}__clean.png",
        "normalized": f"{page_dir_name}/normalized/{basename}.png",
    }
    if isolation is not None:
        artifacts["fitting_mask"] = f"{masks_dir}/{basename}__isolation.png"
        artifacts["isolation_mask"] = f"{masks_dir}/{basename}__isolation.png"
        artifacts["isolation_overlay"] = f"{page_dir_name}/isolation/{basename}__overlay.png"
    artifact_hashes = {
        "crop": _sha256_bytes((output_dir / crop_rel).read_bytes()),
        "raw_mask": _write_mask(output_dir / artifacts["raw_mask"], result.raw_mask),
        "clean_mask": _write_mask(output_dir / artifacts["clean_mask"], result.clean_mask),
        "cleanup": _write_json(
            output_dir / artifacts["cleanup"], result.cleanup.model_dump(mode="json")
        ),
    }
    if isolation is not None:
        artifact_hashes["isolation_mask"] = _write_mask(
            output_dir / artifacts["isolation_mask"], target_mask
        )
        artifact_hashes["isolation_overlay"] = _write_overlay(
            output_dir, artifacts["isolation_overlay"], crop_image, result.clean_mask, target_mask
        )

    baseline_px, unit_px = _preview_frame(target_mask, metrics)
    preview = render_normalized_preview(target_mask, baseline_px=baseline_px, unit_px=unit_px)
    preview_path = output_dir / artifacts["normalized"]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_bytes = io.BytesIO()
    preview.save(preview_bytes, format="PNG")
    preview_path.write_bytes(preview_bytes.getvalue())
    artifact_hashes["normalized"] = _sha256_bytes(preview_bytes.getvalue())

    derived = {}
    for name, artifact in derive_artifacts(target_mask).items():
        if artifact["format"] == "npy":
            rel_path = f"{derived_dir}/{name}.npy"
            (output_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (output_dir / rel_path).write_bytes(artifact["data"])
            digest = _sha256_bytes(artifact["data"])
        else:
            rel_path = f"{derived_dir}/{name}.json"
            digest = _write_json(output_dir / rel_path, artifact["data"])
        derived[name] = {
            "path": rel_path,
            "sha256": digest,
            "algorithm": artifact["algorithm"],
        }

    record = {
        "id": (matched or {}).get("id") or f"{prefix}-{region_id}",
        "kind": kind,
        "label": label,
        "unicode": (matched or {}).get("unicode") or _unicode_of(label, kind),
        "source_id": page.source_id,
        "source_bbox": list(source_bbox),
        "label_source": (matched or {}).get("label_source") or "declared-layout-region",
        "review_status": (matched or {}).get("review_status") or "pending",
        "fitting_status": (matched or {}).get("fitting_status") or "pending",
        "isolation_status": (matched or {}).get("isolation_status")
        or ("pending" if isolation is not None else None),
        "extraction": {
            "extractor": page.extractor,
            "extractor_version": EXTRACTION_VERSION,
            "threshold_policy": params.threshold_policy,
            "source_bbox_policy": "declared-row-sequence" if "row" in detection else "declared",
            "layout_id": layout.id,
            "layout_region_id": region_id,
            "detection_confidence": confidence,
        },
        "metrics": metrics,
        "frame": _frame_from_metrics(source_bbox, crop_image.size, metrics),
        "isolation": isolation,
        "relations": relations,
        "detection": detection,
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "derived": derived,
    }
    if matched:
        record["crop_locator"] = matched.get("crop_locator")
        record["notes"] = matched.get("notes")
    return record


def _unicode_of(label: str, kind: str) -> list[str]:
    if kind.startswith("glyph") and label:
        return [f"U+{ord(ch):04X}" for ch in label]
    return []
