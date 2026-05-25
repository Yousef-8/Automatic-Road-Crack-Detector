"""
core/visualization.py
=====================
Drawing utilities: bounding boxes, summary banner, optional ROI enhancement.

Public API:
    draw_enhanced_system_output(image, detections, apply_roi_enhancement)
        → (annotated_bgr: np.ndarray, analysis: dict)

Internal helpers (not called directly from outside this module):
    _draw_boxes(image, detections)
    _draw_banner(image, analysis)
    _enhance_crack_roi_visual(image, x1, y1, x2, y2)

All functions follow the BGR uint8 convention used throughout this codebase.
The input `image` is never mutated — every function works on copies.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import settings as S
from core.analysis import compute_image_analysis
from core.crack_measurement import measure_and_draw_crack


# =============================================================================
# Public API
# =============================================================================

def draw_enhanced_system_output(
    image: np.ndarray,
    detections: List[Dict],
    apply_roi_enhancement: bool = False,
    apply_crack_measurement: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Produce the final annotated image and compute the analysis summary.

    Pipeline:
        1. (Optional) crack measurement + skeleton overlay on crack boxes
        2. (Optional) ROI visual enhancement (darkening) on crack boxes
        3. Draw colour-coded bounding boxes + labels
        4. Draw top summary banner
        5. Compute and return analysis dict

    Args:
        image:                BGR uint8 array. NOT mutated.
        detections:           List of detection dicts (may be empty).
        apply_roi_enhancement: If True, runs the per-ROI darkening algorithm
                              inside each detected crack box.
        apply_crack_measurement: If True, segments+skeletonizes each crack box,
                              overlays the centreline, and records per-crack
                              length/width measurements in the analysis dict
                              under "crack_measurements". Visual + descriptive
                              only — does NOT affect detection metrics.

    Returns:
        (annotated_image, analysis_dict)
    """
    output = image.copy()
    measurements: List[Dict] = []

    # ---- Step 1: optional crack measurement + skeleton overlay ----
    if apply_crack_measurement:
        for d in detections:
            if d["class_name"] in S.CRACK_CLASSES:
                color = S.CLASS_COLORS.get(d["class_name"], (0, 255, 255))
                output, m = measure_and_draw_crack(
                    output, d["x1"], d["y1"], d["x2"], d["y2"], color=color
                )
                if m is not None:
                    m["class_name"] = d["class_name"]
                    measurements.append(m)

    # ---- Step 2: optional per-ROI crack enhancement (darkening) ----
    if apply_roi_enhancement:
        for d in detections:
            if d["class_name"] in S.CRACK_CLASSES:
                output = _enhance_crack_roi_visual(
                    output, d["x1"], d["y1"], d["x2"], d["y2"]
                )

    # ---- Step 3: draw boxes and labels ----
    output = _draw_boxes(output, detections)

    # ---- Step 4: compute analysis ----
    analysis = compute_image_analysis(detections, image.shape)
    analysis["crack_measurements"] = measurements

    # NOTE: the on-image "ROAD DAMAGE ANALYSIS" banner was intentionally
    # removed. The same summary is rendered as metric cards beneath the image
    # in the UI (see ui/components.render_summary_metrics), so drawing it on
    # the image was redundant and obscured the photo.

    return output, analysis


# =============================================================================
# Drawing helpers
# =============================================================================

def _draw_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw class-coloured bounding boxes and confidence labels."""
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cls  = d["class_name"]
        conf = d["confidence"]
        color = S.CLASS_COLORS.get(cls, (0, 255, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {conf:.2f}"
        text_y = max(y1 - 8, 18)
        cv2.putText(
            out, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return out


def _draw_banner(image: np.ndarray, analysis: Dict) -> np.ndarray:
    """
    Draw a semi-transparent dark banner at the top of the image containing
    three lines of summary text.
    """
    out = image.copy()
    banner_h = 95
    h, w = out.shape[:2]

    # Draw dark rectangle over banner area
    banner = out.copy()
    cv2.rectangle(banner, (0, 0), (w, banner_h), (25, 25, 25), -1)
    out = cv2.addWeighted(banner, 0.85, out, 0.15, 0)

    # Line 0: title
    cv2.putText(
        out, "ROAD DAMAGE ANALYSIS",
        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Line 1: totals
    line1 = (
        f"Total Defects: {analysis['total_detections']} | "
        f"Severity: {analysis['severity']} | "
        f"Damage Density: {analysis['damage_density']:.2f}%"
    )
    cv2.putText(
        out, line1,
        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (230, 230, 230), 1, cv2.LINE_AA,
    )

    # Line 2: dominant type + confidence
    line2 = (
        f"Dominant Type: {analysis['dominant_type']} | "
        f"Avg Conf: {analysis['avg_confidence']:.2f} | "
        f"Largest Box: {analysis['largest_box_area']:.0f} px"
    )
    cv2.putText(
        out, line2,
        (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (230, 230, 230), 1, cv2.LINE_AA,
    )

    # Line 3: per-class counts
    ordered = [
        "longitudinal crack", "transverse crack",
        "alligator crack", "other corruption", "pothole",
    ]
    parts = [
        f"{cls[:12]}: {analysis['class_counts'].get(cls, 0)}"
        for cls in ordered
    ]
    cv2.putText(
        out, " | ".join(parts),
        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (210, 210, 210), 1, cv2.LINE_AA,
    )

    return out


# =============================================================================
# ROI visual enhancement (optional)
# =============================================================================

def _enhance_crack_roi_visual(
    image: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    gaussian_ksize: int = 5,
    blackhat_kernel_size: int = 9,
    crack_mask_thresh_percentile: int = 88,
    darken_strength: float = 0.38,
    background_smooth_alpha: float = 0.35,
) -> np.ndarray:
    """
    Make crack pixels inside the ROI darker and more visually prominent.

    Algorithm (from notebook Section 10.12):
        1. Grayscale + Gaussian smooth
        2. Black-hat morphology to detect dark crack structures
        3. Percentile threshold to create a crack emphasis mask
        4. Morphological clean-up + connected-component filtering
        5. Darken masked pixels; smooth background lightly
        6. Mild unsharp-mask to restore crack edge contrast

    This is purely visual — it does NOT affect detection confidence.
    """
    output = image.copy()
    h, w = output.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return output

    roi = output[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return output

    # Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)

    # Black-hat
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (blackhat_kernel_size, blackhat_kernel_size)
    )
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)

    # Percentile threshold → crack mask
    thresh_val = float(np.percentile(blackhat, crack_mask_thresh_percentile))
    crack_mask = (blackhat >= thresh_val).astype(np.uint8) * 255

    # Morphological clean-up
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, ck, iterations=1)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN,  ck, iterations=1)

    # Connected-component filtering: keep elongated, reasonably-sized blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        crack_mask, connectivity=8
    )
    filtered_mask = np.zeros_like(crack_mask)
    for lbl in range(1, num_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        wc     = stats[lbl, cv2.CC_STAT_WIDTH]
        hc     = stats[lbl, cv2.CC_STAT_HEIGHT]
        if min(wc, hc) == 0:
            continue
        ar = max(wc, hc) / min(wc, hc)
        if area >= 12 and ar >= 1.5:
            filtered_mask[labels == lbl] = 255

    # Smooth background, darken crack pixels
    smooth_bgr = cv2.GaussianBlur(roi, (5, 5), 0)
    roi_clean  = cv2.addWeighted(
        roi, 1.0 - background_smooth_alpha,
        smooth_bgr, background_smooth_alpha, 0
    )

    mask_float = filtered_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)  # soft edges

    enhanced = roi_clean.astype(np.float32)
    darkening = 1.0 - darken_strength * mask_float[..., None]
    enhanced  = np.clip(enhanced * darkening, 0, 255).astype(np.uint8)

    # Mild unsharp-mask for edge clarity
    blur_small = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    enhanced   = cv2.addWeighted(enhanced, 1.08, blur_small, -0.08, 0)

    output[y1:y2, x1:x2] = enhanced
    return output
