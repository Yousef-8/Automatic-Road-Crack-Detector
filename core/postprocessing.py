"""
core/postprocessing.py
======================
Post-processing filters applied after YOLO inference.

The pipeline matches the final experiment setup from the research notebooks
(Section 14 of the Kaggle notebook):

    Stage 1 — Confidence filtering    (per-class threshold)
    Stage 2 — Minimum area filtering  (per-class minimum box area)
    Stage 3 — Aspect-ratio filtering  (class-specific elongation rules)
    Stage 4 — NMS                     (per-image, per-class IoU NMS)

Public API:
    apply_postprocessing(detections) → filtered list of detection dicts

The input and output use the same dict schema as `core.inference.run_inference`
so the two modules compose cleanly in `pipelines.experiment_runner`.

Why operate on lists of dicts instead of DataFrames here?
    The Streamlit app works with single images at a time, so the volume is
    small (typically < 30 detections). List comprehensions are fast enough
    and avoid a pandas dependency in the hot path. DataFrames are only used
    in the UI's display table, where they're constructed from the final list.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import settings as S


# =============================================================================
# Public API
# =============================================================================

def apply_postprocessing(detections: List[Dict]) -> List[Dict]:
    """
    Run all four post-processing stages on a list of detection dicts.

    Args:
        detections: Raw YOLO detections from `core.inference.run_inference`.

    Returns:
        Filtered list.  May be empty if all detections are removed.
    """
    filtered = _validate_and_clean(detections)
    filtered = _filter_confidence(filtered)
    filtered = _filter_area(filtered)
    filtered = _filter_aspect_ratio(filtered)
    filtered = _apply_nms(filtered)
    return filtered


# =============================================================================
# Stage implementations
# =============================================================================

def _validate_and_clean(detections: List[Dict]) -> List[Dict]:
    """
    Stage 0 (matches Kaggle 'validate_and_clean_predictions'): drop invalid
    boxes before any threshold filtering. Removes degenerate geometry that
    would otherwise pass later filters by accident.
    """
    cleaned = []
    for d in detections:
        if d["x2"] <= d["x1"] or d["y2"] <= d["y1"]:
            continue
        if d["box_width"] <= 0 or d["box_height"] <= 0 or d["box_area"] <= 0:
            continue
        cleaned.append(d)
    return cleaned

def _filter_confidence(detections: List[Dict]) -> List[Dict]:
    """
    Stage 1: remove detections below the per-class confidence threshold.
    """
    return [
        d for d in detections
        if d["confidence"] >= S.CONF_THRESHOLDS.get(d["class_name"], 0.30)
    ]


def _filter_area(detections: List[Dict]) -> List[Dict]:
    """
    Stage 2: remove detections whose bounding box is smaller than the
    per-class minimum area.  Tiny boxes are almost always noise or partial
    detections at the image border.
    """
    return [
        d for d in detections
        if d["box_area"] >= S.MIN_AREA.get(d["class_name"], 500.0)
    ]


def _filter_aspect_ratio(detections: List[Dict]) -> List[Dict]:
    """
    Stage 3: class-specific aspect-ratio filtering.

    For elongated crack classes (longitudinal, transverse) boxes that are
    nearly square are likely false positives — real cracks are elongated.
    Alligator crack, other corruption, and pothole are compact by nature,
    so their minimum aspect ratio is 1.0 (no shape constraint).

    Large boxes (area ≥ LARGE_BOX_AREA_KEEP) are never filtered by this rule
    because big defects should always be reported.
    """
    kept = []
    for d in detections:
        # Protect large boxes regardless of shape
        if d["box_area"] >= S.LARGE_BOX_AREA_KEEP:
            kept.append(d)
            continue

        ar = _aspect_ratio(d["box_width"], d["box_height"])
        min_ar = S.CLASS_MIN_ASPECT_RATIO.get(d["class_name"], 1.0)
        if ar >= min_ar:
            kept.append(d)

    return kept


def _apply_nms(detections: List[Dict]) -> List[Dict]:
    """
    Stage 4: per-image, per-class Non-Maximum Suppression.

    Greedy algorithm (same as in the notebook):
        1. Sort by confidence descending.
        2. Keep the highest-confidence box.
        3. Remove all remaining boxes with IoU ≥ NMS_IOU_THRESHOLD.
        4. Repeat until no boxes remain.
    """
    if not detections:
        return []

    # Group by (image_name, class_name) — NMS is applied within each group
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for d in detections:
        key = (d.get("image_name", ""), d["class_name"])
        groups.setdefault(key, []).append(d)

    kept: List[Dict] = []
    for group in groups.values():
        kept.extend(_nms_group(group))

    return kept


# =============================================================================
# Internal helpers
# =============================================================================

def _aspect_ratio(width: float, height: float) -> float:
    """max(w, h) / min(w, h).  Returns 0.0 for degenerate boxes."""
    mn = min(width, height)
    return (max(width, height) / mn) if mn > 0 else 0.0


def _iou(a: Dict, b: Dict) -> float:
    """Intersection-over-Union for two detection dicts."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    union_area = a["box_area"] + b["box_area"] - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _nms_group(group: List[Dict]) -> List[Dict]:
    """Greedy NMS within a single (image, class) group."""
    remaining = sorted(group, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best, d) < S.NMS_IOU_THRESHOLD]

    return kept
