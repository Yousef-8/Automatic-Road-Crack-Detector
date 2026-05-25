"""
pipelines/experiment_runner.py
==============================
Single entry point for all four experiment pipelines.

The four experiments from the project:
    1. RAW          -> YOLO
    2. PREPROCESSED -> YOLO
    3. RAW          -> YOLO -> Post-processing
    4. PREPROCESSED -> YOLO -> Post-processing

Why one dispatcher, not four functions:
    The UI just needs to know "which pipeline did the user pick" — it
    shouldn't care about which preprocessing step ran or which model loaded.
    A single `run_pipeline(image, pipeline_name)` keeps the UI ignorant of
    the internals, and makes adding a 5th experiment a one-line change here.

What this module does:
    1. Routes to the right model (raw vs prep weights)
    2. Optionally applies the preprocessing pipeline
    3. Runs YOLO inference
    4. Optionally applies post-processing
    5. Renders the annotated output and computes the analysis dict
    6. Returns a single result object for the UI to display

What it does NOT do:
    - Touch the filesystem (no reading or writing files)
    - Import Streamlit (so unit tests and CLI scripts can use it)
    - Mutate the input image
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import settings as S
from core.preprocessing import preprocess_for_pipeline
from core.inference import load_model, run_inference
from core.postprocessing import apply_postprocessing
from core.visualization import draw_enhanced_system_output


# =============================================================================
# Result container
# =============================================================================

@dataclass
class PipelineResult:
    """
    Everything the UI needs from one pipeline run.

    Why a dataclass and not a tuple/dict:
        - The UI accesses these fields by name (`result.annotated_image`)
          which is more readable than `result[2]` or `result["annotated_image"]`.
        - Type hints flow through to IDE autocomplete in app.py.
        - Easy to add fields later (e.g., timing breakdowns) without breaking
          existing callers.
    """

    pipeline_name: str
    """Which pipeline ran. One of `config.settings.PIPELINE_NAMES`."""

    image_input: np.ndarray
    """The image actually fed to YOLO (BGR uint8). For RAW pipelines this
    is the user's upload. For PREP pipelines this is the preprocessed image."""

    image_original: np.ndarray
    """The user's original upload, untouched (BGR uint8). The UI shows
    this in a 'before' panel for all four pipelines."""

    annotated_image: np.ndarray
    """`image_input` with boxes + banner drawn on top (BGR uint8).
    Ready to display."""

    detections: List[Dict]
    """Detection dicts AFTER any post-processing for this pipeline.
    Same schema as `core.inference.run_inference` output."""

    analysis: Dict
    """Image-level summary from `core.analysis.compute_image_analysis`.
    Used for sidebar metric cards."""

    detections_pre_postprocessing: Optional[List[Dict]] = None
    """Raw YOLO output before post-processing was applied. None for the two
    pipelines without a post-processing stage. Useful for the optional
    'before/after filtering' UI panel."""

    timings: Dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds per stage: 'preprocess', 'inference',
    'postprocess', 'visualize', 'total'. Stages skipped by the chosen
    pipeline simply don't appear in the dict."""


# =============================================================================
# Public API
# =============================================================================

def run_pipeline(
    image_bgr: np.ndarray,
    pipeline_name: str,
    image_name: str = "uploaded.jpg",
    apply_roi_enhancement: bool = False,
    apply_crack_measurement: bool = False,
) -> PipelineResult:
    """
    Run one of the four experiment pipelines on a single image.

    Args:
        image_bgr:
            BGR uint8 image as returned by `cv2.imread` or by decoding a
            Streamlit upload. NOT mutated.
        pipeline_name:
            One of:
                - `S.PIPELINE_RAW`        ("RAW -> YOLO")
                - `S.PIPELINE_PREP`       ("PREPROCESSED -> YOLO")
                - `S.PIPELINE_RAW_POST`   ("RAW -> YOLO -> Post-processing")
                - `S.PIPELINE_PREP_POST`  ("PREPROCESSED -> YOLO -> Post-processing")
        image_name:
            Filename to embed in detection dicts (cosmetic only).
        apply_roi_enhancement:
            If True, the visualization step also runs the per-ROI darkening
            algorithm on detected cracks. Off by default — adds ~50 ms per
            crack box on CPU and isn't required for the report.

    Returns:
        A `PipelineResult` carrying the annotated image, detections, and
        analysis dict.

    Raises:
        ValueError: if `pipeline_name` isn't recognized.
        FileNotFoundError: if the required `.pt` weights are missing.
    """
    if pipeline_name not in S.PIPELINE_NAMES:
        raise ValueError(
            f"run_pipeline: unknown pipeline {pipeline_name!r}. "
            f"Expected one of {list(S.PIPELINE_NAMES)}."
        )
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"run_pipeline: expected 3-channel BGR image, "
            f"got {None if image_bgr is None else image_bgr.shape}."
        )

    # Decompose the pipeline name into the two boolean flags it controls.
    # This is the only place that knows the routing rules — every downstream
    # step just looks at `use_preprocessing` / `use_postprocessing`.
    use_preprocessing, use_postprocessing = _parse_pipeline_name(pipeline_name)

    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # ---- Stage 1: Preprocessing (pipelines 2 and 4 only) ----
    if use_preprocessing:
        t0 = time.perf_counter()
        # Returns a NEW BGR uint8 array — does not mutate the input.
        image_to_infer = preprocess_for_pipeline(image_bgr)
        timings["preprocess"] = time.perf_counter() - t0
    else:
        # RAW pipelines: feed the original image straight to YOLO.
        # Note: we don't copy here — `run_inference` doesn't mutate its input,
        # and we need the original for the UI's "before" panel anyway.
        image_to_infer = image_bgr

    # ---- Stage 2: Model loading + inference ----
    # Pick the model that matches the input type:
    #   - PREPROCESSED pipelines (2, 4) use weights trained on preprocessed images
    #   - RAW pipelines (1, 3) use weights trained on raw images
    model_kind = "prep" if use_preprocessing else "raw"
    model = load_model(model_kind)

    t0 = time.perf_counter()
    detections_raw = run_inference(model, image_to_infer, image_name=image_name)
    timings["inference"] = time.perf_counter() - t0

    # ---- Stage 3: Post-processing (pipelines 3 and 4 only) ----
    if use_postprocessing:
        t0 = time.perf_counter()
        detections_final = apply_postprocessing(detections_raw)
        timings["postprocess"] = time.perf_counter() - t0
        detections_pre_post = detections_raw  # save the unfiltered set
    else:
        detections_final = detections_raw
        detections_pre_post = None

    # ---- Stage 4: Visualization + analysis ----
    # Draw on a copy of the image that was actually fed to YOLO. For RAW
    # pipelines this is the original; for PREP pipelines this is the
    # preprocessed image (so what the user sees is what the model saw).
    #
    # Match the Kaggle notebook's VISUAL_MODE = "enhanced_roi_boxes": ROI
    # crack enhancement is ON for post-processing pipelines (3 and 4), which
    # is how the experiment output images were generated. The user can still
    # force it on for RAW/PREP-only pipelines via the sidebar toggle.
    roi_on = apply_roi_enhancement or use_postprocessing

    t0 = time.perf_counter()
    annotated, analysis = draw_enhanced_system_output(
        image_to_infer,
        detections_final,
        apply_roi_enhancement=roi_on,
        apply_crack_measurement=apply_crack_measurement,
    )
    timings["visualize"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_total_start

    return PipelineResult(
        pipeline_name=pipeline_name,
        image_input=image_to_infer,
        image_original=image_bgr,
        annotated_image=annotated,
        detections=detections_final,
        analysis=analysis,
        detections_pre_postprocessing=detections_pre_post,
        timings=timings,
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _parse_pipeline_name(pipeline_name: str) -> tuple[bool, bool]:
    """
    Map a pipeline name to (use_preprocessing, use_postprocessing).

    Centralizing this here means everything else in the file is data-driven
    by two booleans — no string comparisons sprinkled through the code.
    """
    use_preprocessing = pipeline_name in (S.PIPELINE_PREP, S.PIPELINE_PREP_POST)
    use_postprocessing = pipeline_name in (S.PIPELINE_RAW_POST, S.PIPELINE_PREP_POST)
    return use_preprocessing, use_postprocessing


# =============================================================================
# Convenience: run all four pipelines on the same image
# =============================================================================

def run_all_pipelines(
    image_bgr: np.ndarray,
    image_name: str = "uploaded.jpg",
    apply_roi_enhancement: bool = False,
    apply_crack_measurement: bool = False,
) -> Dict[str, PipelineResult]:
    """
    Run all four experiments on the same image. Useful for the optional
    "compare all four" view in the UI, and for the report figures.

    Models are loaded once each (cached in `core.inference`), so this is
    not 4x the cost of a single pipeline — preprocessing runs twice
    (pipelines 2 and 4) and inference runs four times.

    Returns:
        Dict keyed by pipeline name (same strings as `S.PIPELINE_NAMES`).
    """
    return {
        name: run_pipeline(
            image_bgr,
            pipeline_name=name,
            image_name=image_name,
            apply_roi_enhancement=apply_roi_enhancement,
            apply_crack_measurement=apply_crack_measurement,
        )
        for name in S.PIPELINE_NAMES
    }