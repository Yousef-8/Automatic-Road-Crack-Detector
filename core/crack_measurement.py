"""
core/crack_measurement.py
=========================
Quantitative crack analysis via classical image-processing operators.

This module elevates the post-detection visualization from a cosmetic
darkening into a *measurement*: inside each detected crack box it segments
the crack, reduces it to a 1-pixel centreline (skeleton), and measures the
crack's LENGTH (skeleton extent) and AVERAGE WIDTH (distance transform).

The techniques used are all standard Digital Image Processing operators:
    - Adaptive thresholding              (segmentation,         G&W Ch. 10)
    - Black-hat morphology               (morphological proc.,  G&W Ch. 9)
    - Morphological opening/closing      (morphological proc.,  G&W Ch. 9)
    - Connected-component analysis       (region labelling,     G&W Ch. 9)
    - Morphological thinning/skeleton    (morphological proc.,  G&W Ch. 9)
    - Distance transform                 (morphological proc.,  G&W Ch. 9)

IMPORTANT — this does NOT affect detection metrics. It runs after detection
and after box filtering, on pixels inside an already-accepted box. It only
changes the annotated image and adds descriptive per-crack measurements.

Units: measurements are in PIXELS. Without camera calibration they cannot be
converted to millimetres, so the report should present them as relative /
pixel-domain quantities.

Public API:
    measure_and_draw_crack(image_bgr, x1, y1, x2, y2, color) -> (image, dict|None)
        Returns the image with a skeleton overlay drawn inside the box, and a
        measurement dict {length_px, avg_width_px, max_width_px, area_px} or
        None if no crack structure was found.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Skeletonization — prefer cv2.ximgproc, fall back to morphological thinning
# =============================================================================

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Reduce a binary mask (0/255) to a 1-pixel-wide skeleton (0/255).

    Tries cv2.ximgproc.thinning (Zhang-Suen, from opencv-contrib). If that is
    unavailable, uses a classical morphological thinning loop (erosion +
    opening), which is the textbook skeleton-by-morphology algorithm.
    """
    # Preferred: opencv-contrib Zhang-Suen thinning
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        try:
            return ximgproc.thinning(mask)
        except Exception:
            pass  # fall through to morphological method

    # Fallback: morphological skeleton (Gonzalez & Woods, Ch. 9)
    skel = np.zeros_like(mask)
    work = mask.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Bound iterations to avoid any pathological infinite loop
    for _ in range(200):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return skel


# =============================================================================
# Crack mask segmentation inside a single ROI
# =============================================================================

def _segment_crack_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Build a clean binary crack mask (0/255) for one ROI using classical IP.

    Steps:
        grayscale -> CLAHE -> Gaussian blur -> black-hat -> adaptive threshold
        -> morphological close+open -> connected-component cleanup.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Local contrast so faint cracks survive thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reduce asphalt speckle
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Black-hat isolates dark thin structures (cracks are dark on light road)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, bh_kernel)

    # Adaptive threshold copes with uneven lighting across the ROI
    mask = cv2.adaptiveThreshold(
        blackhat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        21, -3,
    )

    # Reconnect gaps, then remove speckle
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    # Keep only crack-like components. We judge "crack-like" by EXTENT
    # (filled_area / bounding_box_area) rather than bounding-box aspect ratio:
    # a thin crack — even a diagonal one whose bounding box is square — fills
    # only a small fraction of its bounding box, so it has LOW extent. Compact
    # blobs (noise, fills) have high extent. This correctly accepts diagonal
    # cracks that an aspect-ratio test would wrongly reject.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    roi_area = roi_bgr.shape[0] * roi_bgr.shape[1]
    for lbl in range(1, num):
        area = stats[lbl, cv2.CC_STAT_AREA]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        bbox_area = w * h
        if bbox_area == 0:
            continue
        extent = area / float(bbox_area)
        bbox_ar = max(w, h) / max(1, min(w, h))
        # crack-like = thin (low extent) OR clearly elongated bounding box;
        # plus size sanity (not speck, not whole-ROI fill)
        is_thin = extent <= 0.45
        is_elongated = bbox_ar >= 1.4
        size_ok = area >= 15 and area <= 0.6 * roi_area
        if size_ok and (is_thin or is_elongated):
            clean[labels == lbl] = 255

    return clean


# =============================================================================
# Public API
# =============================================================================

def measure_and_draw_crack(
    image_bgr: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Segment, skeletonize, measure, and overlay the crack inside one box.

    Args:
        image_bgr: full BGR image. NOT mutated (works on a copy).
        x1..y2:    box coordinates (any numeric type).
        color:     BGR colour for the skeleton overlay.

    Returns:
        (output_image, measurement_dict_or_None)
        measurement_dict keys: length_px, avg_width_px, max_width_px, area_px
    """
    output = image_bgr.copy()
    h, w = output.shape[:2]

    xa = max(0, min(int(x1), w - 1))
    ya = max(0, min(int(y1), h - 1))
    xb = max(0, min(int(x2), w))
    yb = max(0, min(int(y2), h))
    if xb <= xa or yb <= ya:
        return output, None

    roi = output[ya:yb, xa:xb]
    if roi.size == 0:
        return output, None

    # ---- Segment crack mask ----
    mask = _segment_crack_mask(roi)
    if cv2.countNonZero(mask) == 0:
        return output, None  # no crack structure found

    # ---- Distance transform → width ----
    # For each crack pixel, distanceTransform gives distance to nearest
    # background pixel. Along the skeleton, that distance ≈ half the local
    # crack width, so width ≈ 2 × distance.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # ---- Skeleton → centreline and length ----
    skel = _skeletonize(mask)
    skel_pixels = cv2.countNonZero(skel)
    if skel_pixels == 0:
        return output, None

    # Length ≈ number of skeleton pixels (1-px centreline).
    length_px = float(skel_pixels)

    # Width from distance transform sampled along the skeleton.
    skel_bool = skel > 0
    widths = dist[skel_bool] * 2.0
    avg_width_px = float(np.mean(widths)) if widths.size else 0.0
    max_width_px = float(np.max(widths)) if widths.size else 0.0

    area_px = float(cv2.countNonZero(mask))

    # ---- Overlay skeleton on the ROI (dilated slightly for visibility) ----
    skel_vis = cv2.dilate(skel, np.ones((2, 2), np.uint8), iterations=1)
    roi_overlay = roi.copy()
    roi_overlay[skel_vis > 0] = color
    output[ya:yb, xa:xb] = roi_overlay

    measurement = {
        "length_px":    round(length_px, 1),
        "avg_width_px": round(avg_width_px, 1),
        "max_width_px": round(max_width_px, 1),
        "area_px":      round(area_px, 1),
    }
    return output, measurement
