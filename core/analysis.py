"""
core/analysis.py
================
Image-level summary statistics computed from a list of detections.

Public API:
    compute_image_analysis(detections, image_shape) → dict

The returned dict is used both for the Streamlit metric cards and for the
on-image summary banner drawn by `core.visualization`.

Severity heuristic (from the research notebook, Section 10.23):
    NONE     — no detections
    LOW      — density < 1% and ≤ 1 detection
    MODERATE — density < 3% and ≤ 3 detections
    HIGH     — density < 6% and ≤ 5 detections
    SEVERE   — everything else
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import settings as S


def compute_image_analysis(
    detections: List[Dict],
    image_shape: Tuple[int, ...],
) -> Dict:
    """
    Compute image-level summary from a list of detection dicts.

    Args:
        detections:  List of detection dicts (may be empty).
        image_shape: OpenCV image shape — (H, W, C) or (H, W).

    Returns:
        Dict with keys:
            total_detections  int
            class_counts      dict[str, int]
            avg_confidence    float
            largest_box_area  float
            total_box_area    float
            damage_density    float   (% of image area)
            dominant_type     str
            severity          str     (NONE / LOW / MODERATE / HIGH / SEVERE)
    """
    h, w = image_shape[:2]
    image_area = float(h * w)

    total = len(detections)

    if total == 0:
        return {
            "total_detections": 0,
            "class_counts":     {cls: 0 for cls in S.CLASS_NAMES},
            "avg_confidence":   0.0,
            "largest_box_area": 0.0,
            "total_box_area":   0.0,
            "damage_density":   0.0,
            "dominant_type":    "None",
            "severity":         "NONE",
        }

    # Class-wise counts (include all known classes with 0 for missing ones)
    class_counts: Dict[str, int] = {cls: 0 for cls in S.CLASS_NAMES}
    for d in detections:
        cls = d["class_name"]
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = class_counts.get(cls, 0) + 1

    avg_conf = float(sum(d["confidence"] for d in detections) / total)
    largest  = float(max(d["box_area"]  for d in detections))
    total_area = float(sum(d["box_area"] for d in detections))
    damage_density = (total_area / image_area * 100.0) if image_area > 0 else 0.0

    # Dominant class = class with highest count
    dominant_type = max(class_counts, key=lambda c: class_counts[c])
    if class_counts[dominant_type] == 0:
        dominant_type = "None"

    # Severity heuristic (matches notebook section 10.23)
    if damage_density < 1.0 and total <= 1:
        severity = "LOW"
    elif damage_density < 3.0 and total <= 3:
        severity = "MODERATE"
    elif damage_density < 6.0 and total <= 5:
        severity = "HIGH"
    else:
        severity = "SEVERE"

    return {
        "total_detections": total,
        "class_counts":     class_counts,
        "avg_confidence":   avg_conf,
        "largest_box_area": largest,
        "total_box_area":   total_area,
        "damage_density":   damage_density,
        "dominant_type":    dominant_type,
        "severity":         severity,
    }
