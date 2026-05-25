"""
core/inference.py
=================
Model loading and YOLO inference.

Two public functions:

    load_model(model_kind)          → YOLO instance (cached)
    run_inference(model, image, …)  → list of detection dicts

Why cache models with @st.cache_resource?
    YOLOv8 model loading reads the .pt file, builds the network, and moves
    weights to device. On CPU this takes ~2–4 seconds. Caching means the
    overhead is paid once per session, not once per image upload.

    `@st.cache_resource` is the correct Streamlit primitive here (not
    `@st.cache_data`) because a YOLO model is a stateful object, not
    serialisable data.

Detection dict schema (matches the Kaggle notebook columns):

    {
        "image_name":  str,
        "class_id":    int,
        "class_name":  str,
        "confidence":  float,
        "x1":          float,   # pixel coordinates (absolute)
        "y1":          float,
        "x2":          float,
        "y2":          float,
        "box_width":   float,
        "box_height":  float,
        "box_area":    float,
        "aspect_ratio": float,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st
from ultralytics import YOLO

from config import settings as S


# =============================================================================
# Model loading — cached per session
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_kind: str) -> YOLO:
    """
    Load and cache a YOLO model for the given kind.

    Args:
        model_kind: "raw" or "prep"

    Returns:
        Loaded YOLO model (cached — loaded only once per session).

    Raises:
        ValueError:       if `model_kind` is not "raw" or "prep".
        FileNotFoundError: if the .pt weights file doesn't exist.
    """
    if model_kind == "raw":
        weights_path: Path = S.RAW_MODEL_WEIGHTS
    elif model_kind == "prep":
        weights_path: Path = S.PREP_MODEL_WEIGHTS
    else:
        raise ValueError(
            f"load_model: expected 'raw' or 'prep', got {model_kind!r}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Place the trained .pt file there before running the app."
        )

    model = YOLO(str(weights_path))
    return model


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model: YOLO,
    image_bgr: np.ndarray,
    image_name: str = "image.jpg",
) -> List[Dict]:
    """
    Run YOLO inference on a single BGR image and return structured detections.

    Args:
        model:      Loaded YOLO model (from `load_model`).
        image_bgr:  BGR uint8 NumPy array.  NOT mutated.
        image_name: Filename to embed in each detection dict (cosmetic only).

    Returns:
        List of detection dicts (empty list if no detections).
        Each dict matches the schema documented at the top of this file.

    Notes:
        - YOLO's built-in NMS is applied internally at INFER_IOU threshold.
        - Boxes are returned in absolute pixel coordinates.
        - The aspect_ratio field is max(w,h)/min(w,h); 0.0 for degenerate boxes.
    """
    results = model.predict(
        source=image_bgr,
        imgsz=S.INFER_IMGSZ,
        conf=S.INFER_CONF,
        iou=S.INFER_IOU,
        device=S.INFER_DEVICE,
        verbose=False,
    )

    result = results[0]
    detections: List[Dict] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs      = result.boxes.conf.cpu().numpy()
    class_ids  = result.boxes.cls.cpu().numpy().astype(int)

    for box, score, cls_id in zip(boxes_xyxy, confs, class_ids):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        aspect_ratio = (max(w, h) / min(w, h)) if min(w, h) > 0 else 0.0

        cls_name = (
            S.CLASS_NAMES[cls_id]
            if 0 <= cls_id < len(S.CLASS_NAMES)
            else f"class_{cls_id}"
        )

        detections.append({
            "image_name":   image_name,
            "class_id":     int(cls_id),
            "class_name":   cls_name,
            "confidence":   float(score),
            "x1":           x1,
            "y1":           y1,
            "x2":           x2,
            "y2":           y2,
            "box_width":    w,
            "box_height":   h,
            "box_area":     area,
            "aspect_ratio": aspect_ratio,
        })

    return detections
