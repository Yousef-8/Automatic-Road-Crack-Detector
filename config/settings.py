"""
config/settings.py
==================
Central configuration for the Road Crack Detection application.

Environment-specific values (model paths, device) are loaded from a .env
file in the project root via python-dotenv.  Research parameters (thresholds,
class definitions, pipeline names) are hard-coded here — they are not secrets
and belong in version control so the experiment is reproducible.

Setup:
    1. Copy .env.example to .env in the project root.
    2. Fill in your model paths and preferred device.
    3. Never commit .env to git (it is listed in .gitignore).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# Load .env from the project root (two levels up from this file)
# =============================================================================

_BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

# =============================================================================
# Paths — read from .env, fall back to sensible defaults
# =============================================================================

def _resolve(env_key: str, default: str) -> Path:
    """
    Read a path from the environment.  Relative paths are resolved against
    the project root so they work regardless of the working directory.
    """
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else _BASE_DIR / p


MODELS_DIR: Path        = _BASE_DIR / "models"
RAW_MODEL_WEIGHTS: Path = _resolve("RAW_MODEL_WEIGHTS",  "models/raw_best.pt")
PREP_MODEL_WEIGHTS: Path = _resolve("PREP_MODEL_WEIGHTS", "models/prep_best.pt")

# =============================================================================
# Inference settings
# =============================================================================

INFER_IMGSZ: int  = 640
INFER_CONF: float = 0.25        # Minimum confidence for YOLO to emit a box
INFER_IOU: float  = 0.45        # IoU threshold for YOLO's built-in NMS
INFER_DEVICE: str = os.getenv("INFER_DEVICE", "cpu")

# =============================================================================
# Upload limit
# =============================================================================

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))

# =============================================================================
# Class definitions (must match the YOLO training label order)
# =============================================================================

CLASS_NAMES: list[str] = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]

# Crack-only classes (used by aspect-ratio filter and ROI enhancement)
CRACK_CLASSES: set[str] = {
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
}

# BGR colors for bounding boxes (OpenCV convention)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "longitudinal crack": (255, 0,   0),    # blue
    "transverse crack":   (0,   255, 0),    # green
    "alligator crack":    (0,   0,   255),  # red
    "other corruption":   (0,   255, 255),  # yellow
    "pothole":            (255, 0,   255),  # magenta
}

# =============================================================================
# Post-processing thresholds
# =============================================================================

CONF_THRESHOLDS: dict[str, float] = {
    "longitudinal crack": 0.30,
    "transverse crack":   0.30,
    "alligator crack":    0.25,
    "other corruption":   0.35,
    "pothole":            0.35,
}

MIN_AREA: dict[str, float] = {
    "longitudinal crack": 500.0,
    "transverse crack":   500.0,
    "alligator crack":    800.0,
    "other corruption":   700.0,
    "pothole":            800.0,
}

CLASS_MIN_ASPECT_RATIO: dict[str, float] = {
    "longitudinal crack": 2.0,
    "transverse crack":   2.0,
    "alligator crack":    1.0,
    "other corruption":   1.0,
    "pothole":            1.0,
}

LARGE_BOX_AREA_KEEP: float = 20_000.0
NMS_IOU_THRESHOLD: float   = 0.50

# =============================================================================
# Preprocessing parameters
# =============================================================================

CLAHE_CLIP_LIMIT: float        = 2.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
GAMMA: float                   = 0.85
DENOISE_KSIZE: int             = 3
SHARPEN_ALPHA: float           = 1.5
SHARPEN_BETA: float            = -0.5

# =============================================================================
# Pipeline identifiers
# =============================================================================

PIPELINE_RAW: str       = "RAW → YOLO"
PIPELINE_PREP: str      = "PREPROCESSED → YOLO"
PIPELINE_RAW_POST: str  = "RAW → YOLO → Post-processing"
PIPELINE_PREP_POST: str = "PREPROCESSED → YOLO → Post-processing"

PIPELINE_NAMES: tuple[str, ...] = (
    PIPELINE_RAW,
    PIPELINE_PREP,
    PIPELINE_RAW_POST,
    PIPELINE_PREP_POST,
)

# =============================================================================
# UI / App metadata
# =============================================================================

APP_TITLE: str     = "Road Crack Detection"
APP_SUBTITLE: str  = "YOLOv8s · RDD2022 · Master's Project"
APP_PAGE_ICON: str = "🛣️"

SUPPORTED_UPLOAD_EXTS: tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp")
