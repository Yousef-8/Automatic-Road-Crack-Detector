"""
core/preprocessing.py
=====================
EXACT port of the Kaggle training preprocessing pipeline (`preprocess_image_mild`).

This is the multi-cue crack-saliency enhancement used to build the
PREPROCESSED ("RDD2022-Enhanced") training dataset. Reproducing it verbatim
here guarantees that the app's PREPROCESSED pipelines feed the model exactly
the same kind of image it was trained on — so app results match the Kaggle
experiments.

Pipeline (see preprocess_image_mild at the bottom):
    Step 1  Mild CLAHE on LAB L-channel, blended with original luminance
    Step 2  Light edge-preserving (bilateral) smoothing
    Step 3  Multi-cue crack saliency map (black-hat + Sobel + morph-grad + Canny)
    Step 4  Explicit crack-layer enhancement (darken + local contrast in L)
    Step 5  Crack-guided local sharpening (only where saliency is high)
    Step 6  Final blend with the original image (realism preserved)

Input/output convention:
    `preprocess_image_mild(bgr_img)` takes a BGR uint8 array and returns a
    BGR uint8 array (the enhanced image). A second return value carries the
    intermediate diagnostic maps used for report figures.

    The app calls `preprocess_for_pipeline(bgr_img)` which returns ONLY the
    BGR image (drops the diagnostics) so the pipeline runner stays simple.

NOTE: This is verbatim from the Kaggle notebook (Section 5B / 3A). The
parameter values are intentionally preserved exactly; do not "tidy" them, as
any change would break consistency with the trained model.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------
# Helper 1: Normalize array to [0, 1]
# ---------------------------------------------------------
def normalize_to_unit(x):
    """Normalize an array to [0, 1] as float32."""
    x = x.astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)


# ---------------------------------------------------------
# Helper 2: Mild CLAHE on LAB luminance only
# ---------------------------------------------------------
def apply_mild_clahe_lab(rgb_img, clip_limit=1.3, tile_grid_size=(8, 8), blend_alpha=0.35):
    """
    Apply CLAHE only to the LAB L channel, then blend with the original
    luminance to keep the output natural.
    """
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    l_blend = cv2.addWeighted(l, 1.0 - blend_alpha, l_clahe, blend_alpha, 0)

    lab_out = cv2.merge((l_blend, a, b))
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    return rgb_out, l, l_clahe, l_blend


# ---------------------------------------------------------
# Helper 3: Light edge-preserving smoothing
# ---------------------------------------------------------
def light_edge_preserving_smoothing(rgb_img, d=5, sigma_color=18, sigma_space=18):
    """Lighter bilateral filtering to preserve thin crack structures."""
    return cv2.bilateralFilter(rgb_img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ---------------------------------------------------------
# Helper 4: Multi-scale black-hat crack detector
# ---------------------------------------------------------
def multi_scale_blackhat(gray_img, kernel_sizes=(3, 5, 7, 9, 11, 13)):
    """
    Compute multi-scale black-hat maps to detect dark thin crack-like
    structures of different widths.
    """
    blackhat_maps = {}
    accum = np.zeros_like(gray_img, dtype=np.float32)

    for k in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bh = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
        blackhat_maps[f"blackhat_{k}"] = bh
        accum += bh.astype(np.float32)

    combined = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    return combined, blackhat_maps


# ---------------------------------------------------------
# Helper 5: Morphological gradient
# ---------------------------------------------------------
def morphological_gradient(gray_img, kernel_size=3):
    """Morphological gradient emphasizes local structural boundaries."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
    return grad


# ---------------------------------------------------------
# Helper 6: Sobel magnitude
# ---------------------------------------------------------
def compute_sobel_magnitude(gray_img):
    """Sobel gradient magnitude map."""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel_mag


# ---------------------------------------------------------
# Helper 7: Canny edges
# ---------------------------------------------------------
def compute_canny_edges(gray_img, low_thresh=50, high_thresh=130):
    """Canny edge detector for crack-support cues and diagnostics."""
    return cv2.Canny(gray_img, low_thresh, high_thresh)


# ---------------------------------------------------------
# Helper 8: Laplacian absolute map
# ---------------------------------------------------------
def compute_laplacian_abs(gray_img):
    """Laplacian absolute response for fine-detail diagnostics."""
    lap = cv2.Laplacian(gray_img, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_abs = np.clip(lap_abs, 0, 255).astype(np.uint8)
    return lap_abs


# ---------------------------------------------------------
# Helper 9: Build crack saliency map
# ---------------------------------------------------------
def build_crack_saliency_map(gray_img):
    """Build a crack saliency map from multiple crack-related cues."""
    crack_map_bh, blackhat_maps = multi_scale_blackhat(gray_img, kernel_sizes=(3, 5, 7, 9, 11, 13))
    sobel_mag = compute_sobel_magnitude(gray_img)
    morph_grad = morphological_gradient(gray_img, kernel_size=3)
    canny_edges = compute_canny_edges(gray_img, low_thresh=50, high_thresh=130)

    bh_n = normalize_to_unit(crack_map_bh)
    sobel_n = normalize_to_unit(sobel_mag)
    morph_n = normalize_to_unit(morph_grad)
    canny_n = normalize_to_unit(canny_edges)

    canny_support = cv2.dilate((canny_n * 255).astype(np.uint8),
                               np.ones((3, 3), np.uint8),
                               iterations=1)
    canny_support = normalize_to_unit(canny_support)

    # Slightly stronger black-hat dominance
    saliency = (
        0.66 * bh_n +
        0.16 * sobel_n +
        0.10 * morph_n +
        0.08 * canny_support
    )

    # Slightly stronger crack emphasis than before
    saliency = np.power(np.clip(saliency, 0, 1), 0.72)

    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (5, 5), 0)
    saliency = normalize_to_unit(saliency)
    saliency_u8 = np.uint8(np.clip(saliency * 255.0, 0, 255))

    aux_maps = {
        "crack_map_blackhat": crack_map_bh,
        "sobel_mag": sobel_mag,
        "morph_grad": morph_grad,
        "canny_edges": canny_edges,
        "saliency_u8": saliency_u8,
        **blackhat_maps
    }

    return saliency, aux_maps


# ---------------------------------------------------------
# Helper 10: Explicit crack-layer enhancement
# ---------------------------------------------------------
def apply_explicit_crack_enhancement(rgb_img,
                                     saliency_map,
                                     gray_reference,
                                     crack_darken_strength=38,
                                     crack_contrast_strength=0.32):
    """
    Crack enhancement that injects a crack-emphasis layer into luminance.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat_refined = cv2.morphologyEx(gray_reference, cv2.MORPH_BLACKHAT, kernel)
    blackhat_refined_f = blackhat_refined.astype(np.float32) / 255.0

    # Slightly favor refined black-hat a bit more
    crack_layer = 0.60 * saliency_map + 0.40 * blackhat_refined_f
    crack_layer = np.clip(crack_layer, 0, 1)

    crack_layer = np.power(crack_layer, 0.65)
    crack_layer = cv2.GaussianBlur(crack_layer.astype(np.float32), (5, 5), 0)
    crack_layer = normalize_to_unit(crack_layer)

    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Slightly stronger darkening
    l_new = l - crack_darken_strength * crack_layer

    # Slightly stronger local contrast
    l_centered = l_new - 128.0
    l_new = l_new + crack_contrast_strength * crack_layer * l_centered

    l_new = np.clip(l_new, 0, 255)

    lab_out = cv2.merge((l_new, a, b)).astype(np.uint8)
    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    crack_layer_u8 = np.uint8(np.clip(crack_layer * 255.0, 0, 255))
    blackhat_refined_u8 = np.uint8(np.clip(blackhat_refined_f * 255.0, 0, 255))

    return rgb_out, crack_layer_u8, blackhat_refined_u8


# ---------------------------------------------------------
# Helper 11: Crack-guided local sharpening
# ---------------------------------------------------------
def crack_guided_local_sharpen(rgb_img, saliency_map, sigma=1.0, amount=1.25, saliency_gain=1.0):
    """Sharpen mainly where crack saliency is high."""
    blurred = cv2.GaussianBlur(rgb_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp_full = cv2.addWeighted(rgb_img, 1.0 + amount, blurred, -amount, 0)
    sharp_full = np.clip(sharp_full, 0, 255).astype(np.uint8)

    saliency_local = np.clip(saliency_map * saliency_gain, 0, 1)
    saliency_3c = np.repeat(saliency_local[:, :, None], 3, axis=2)

    out = (
        rgb_img.astype(np.float32) * (1.0 - 1.00 * saliency_3c) +
        sharp_full.astype(np.float32) * (1.00 * saliency_3c)
    )

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sharp_full


# ---------------------------------------------------------
# Helper 12: Final blend with original image
# ---------------------------------------------------------
def blend_with_original(rgb_original, rgb_enhanced, original_weight=0.35, enhanced_weight=0.65):
    """Final blend that preserves realism but lets crack enhancement show."""
    blended = cv2.addWeighted(rgb_original, original_weight, rgb_enhanced, enhanced_weight, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# ---------------------------------------------------------
# Main preprocessing pipeline (VERBATIM from Kaggle)
# ---------------------------------------------------------
def preprocess_image_mild(bgr_img):
    """
    Crack-focused preprocessing pipeline used to build the PREPROCESSED
    training dataset. Verbatim from the Kaggle notebook (Section 5B / 3A).

    Args:
        bgr_img: BGR uint8 array (cv2 convention).

    Returns:
        (final_bgr, intermediate_dict)
        final_bgr: enhanced BGR uint8 image (this is what YOLO receives).
        intermediate_dict: diagnostic maps (for report figures; unused by app).
    """
    # Step 0: Convert input
    rgb_original = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Step 1: Mild CLAHE on luminance only
    rgb_clahe, l_original, l_clahe, l_blend = apply_mild_clahe_lab(
        rgb_original,
        clip_limit=1.3,
        tile_grid_size=(8, 8),
        blend_alpha=0.35
    )

    # Step 2: Light edge-preserving smoothing
    rgb_smooth = light_edge_preserving_smoothing(
        rgb_clahe,
        d=5,
        sigma_color=18,
        sigma_space=18
    )

    # Step 3: Crack saliency map
    gray_smooth = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)
    saliency_map, aux_maps = build_crack_saliency_map(gray_smooth)

    # Step 4: Explicit crack-layer enhancement
    rgb_crack_enhanced, crack_layer_u8, blackhat_refined_u8 = apply_explicit_crack_enhancement(
        rgb_smooth,
        saliency_map=saliency_map,
        gray_reference=gray_smooth,
        crack_darken_strength=38,
        crack_contrast_strength=0.32
    )

    # Step 5: Crack-guided local sharpening
    rgb_crack_sharp, rgb_sharp_full = crack_guided_local_sharpen(
        rgb_crack_enhanced,
        saliency_map=saliency_map,
        sigma=1.0,
        amount=1.25,
        saliency_gain=1.0
    )

    # Step 6: Final blend with original image
    rgb_final = blend_with_original(
        rgb_original,
        rgb_crack_sharp,
        original_weight=0.35,
        enhanced_weight=0.65
    )

    # Step 7: Final diagnostics (for report figures only)
    gray_final = cv2.cvtColor(rgb_final, cv2.COLOR_RGB2GRAY)
    canny_final = compute_canny_edges(gray_final, low_thresh=50, high_thresh=130)
    sobel_final = compute_sobel_magnitude(gray_final)
    laplacian_final = compute_laplacian_abs(gray_final)

    # Convert back to BGR for saving / inference
    final_bgr = cv2.cvtColor(rgb_final, cv2.COLOR_RGB2BGR)

    intermediate_dict = {
        "rgb_original": rgb_original,
        "l_original": l_original,
        "l_clahe": l_clahe,
        "l_blend": l_blend,
        "rgb_clahe": rgb_clahe,
        "rgb_smooth": rgb_smooth,
        **aux_maps,
        "saliency_u8": np.uint8(np.clip(saliency_map * 255.0, 0, 255)),
        "blackhat_refined_u8": blackhat_refined_u8,
        "crack_layer_u8": crack_layer_u8,
        "rgb_crack_enhanced": rgb_crack_enhanced,
        "rgb_sharp_full": rgb_sharp_full,
        "rgb_crack_sharp": rgb_crack_sharp,
        "rgb_final": rgb_final,
        "canny_final": canny_final,
        "sobel_final": sobel_final,
        "laplacian_final": laplacian_final,
    }

    return final_bgr, intermediate_dict


# ---------------------------------------------------------
# Thin wrapper for the app pipeline runner
# ---------------------------------------------------------
def preprocess_for_pipeline(bgr_img: np.ndarray) -> np.ndarray:
    """
    App-facing wrapper: run the full Kaggle preprocessing and return ONLY the
    enhanced BGR image (drop the diagnostic maps).

    Args:
        bgr_img: BGR uint8 array. NOT mutated.

    Returns:
        Enhanced BGR uint8 image, ready for YOLO inference.
    """
    if bgr_img is None or bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError(
            f"preprocess_for_pipeline: expected BGR uint8 (H,W,3), "
            f"got shape {None if bgr_img is None else bgr_img.shape}"
        )
    final_bgr, _ = preprocess_image_mild(bgr_img)
    return final_bgr
