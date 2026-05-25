"""
ui/components.py
================
Reusable Streamlit display blocks. Each function takes data and renders —
nothing here calls into the model or pipeline code.

Why split UI into components.py + app.py:
    `app.py` is the page layout and event flow. Pulling each visual block
    into a named function here keeps `app.py` short and readable, and means
    you can rearrange the layout without rewriting the rendering logic.

Public API:
    render_header()
    render_pipeline_picker(default_index=0) -> str
    render_options_panel() -> dict
    render_summary_metrics(analysis)
    render_detection_table(detections)
    render_image_pair(original_bgr, annotated_bgr, original_caption, annotated_caption)
    render_timings(timings)
    render_class_legend()
    render_no_detections_message()
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import settings as S


# =============================================================================
# Top of page
# =============================================================================

def render_header() -> None:
    """App title + one-line subtitle. Called once at the top of the page."""
    st.title(f"{S.APP_PAGE_ICON} {S.APP_TITLE}")
    st.caption(S.APP_SUBTITLE)


# =============================================================================
# Sidebar: pipeline picker + options
# =============================================================================

def render_pipeline_picker(default_index: int = 0) -> str:
    """
    Sidebar dropdown for the four experiment pipelines.

    Returns:
        The selected pipeline name (one of `S.PIPELINE_NAMES`).
    """
    return st.sidebar.selectbox(
        "Experiment pipeline",
        options=S.PIPELINE_NAMES,
        index=default_index,
        help=(
            "RAW -> YOLO: feed the upload directly to the RAW-trained model.\n"
            "PREPROCESSED: apply CLAHE + crack-saliency enhancement, then "
            "feed to the PREP-trained model.\n"
            "Add Post-processing for confidence/area/aspect/NMS filtering."
        ),
    )


def render_options_panel() -> Dict[str, bool]:
    """
    Sidebar controls for optional features. Returns a dict of flags so the
    main page can pass them to the pipeline runner.
    """
    st.sidebar.markdown("### Options")

    apply_roi = st.sidebar.checkbox(
        "Apply ROI crack enhancement",
        value=False,
        help=(
            "Darken crack-like pixels inside detected boxes. Adds a few "
            "hundred milliseconds per image. Off by default."
        ),
    )

    show_original = st.sidebar.checkbox(
        "Show original alongside",
        value=True,
        help="Display the user's upload next to the annotated output.",
    )

    show_timings = st.sidebar.checkbox(
        "Show timings",
        value=False,
        help="Per-stage wall-clock seconds.",
    )

    crack_measurement = st.sidebar.checkbox(
        "Measure cracks (skeleton + length/width)",
        value=False,
        help=(
            "Segment each detected crack, trace its centreline, and measure "
            "length and average width in pixels. Visual + descriptive only — "
            "does not change detection metrics."
        ),
    )

    return {
        "apply_roi_enhancement": apply_roi,
        "show_original": show_original,
        "show_timings": show_timings,
        "apply_crack_measurement": crack_measurement,
    }


# =============================================================================
# Summary metrics (top of results panel)
# =============================================================================

def render_summary_metrics(analysis: Dict) -> None:
    """
    Render the analysis dict as a row of Streamlit metric cards.
    """
    # Top row: counts and severity
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Defects", analysis["total_detections"])
    c2.metric("Severity", analysis["severity"])
    c3.metric("Damage Density", f"{analysis['damage_density']:.2f}%")
    c4.metric("Avg Confidence", f"{analysis['avg_confidence']:.2f}")

    # Second row: dominant + largest
    c5, c6 = st.columns(2)
    c5.metric("Dominant Type", analysis["dominant_type"])
    c6.metric("Largest Box", f"{int(analysis['largest_box_area']):,} px²")


# =============================================================================
# Detection table (one row per detected box)
# =============================================================================

def render_detection_table(detections: List[Dict]) -> None:
    """
    Render the per-detection list as a sortable Streamlit dataframe.
    Hidden when there are no detections.
    """
    if not detections:
        return

    rows = []
    for i, d in enumerate(detections, start=1):
        rows.append({
            "#": i,
            "Class": d["class_name"],
            "Confidence": round(d["confidence"], 3),
            "Width (px)": int(d["box_width"]),
            "Height (px)": int(d["box_height"]),
            "Area (px²)": int(d["box_area"]),
            "Aspect Ratio": round(d.get("aspect_ratio", 0.0), 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# Image display
# =============================================================================

def render_image_pair(
    original_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    original_caption: str = "Original",
    annotated_caption: str = "Annotated",
) -> None:
    """
    Show original and annotated images side by side. Both inputs are BGR
    (the convention used everywhere else in this codebase) — we convert
    here to RGB once for Streamlit.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
            caption=original_caption,
            use_container_width=True,
        )
    with col2:
        st.image(
            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            caption=annotated_caption,
            use_container_width=True,
        )


def render_single_image(annotated_bgr: np.ndarray, caption: str = "Annotated output") -> None:
    """Show one full-width annotated image."""
    st.image(
        cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        caption=caption,
        use_container_width=True,
    )


# =============================================================================
# Optional panels
# =============================================================================

def render_timings(timings: Dict[str, float]) -> None:
    """Per-stage seconds. Hidden behind the 'Show timings' checkbox."""
    if not timings:
        return
    st.markdown("**Timings (seconds)**")
    rows = [{"Stage": k, "Seconds": f"{v:.3f}"} for k, v in timings.items()]
    st.table(pd.DataFrame(rows))


def render_class_legend() -> None:
    """Color legend for the bounding-box colors used on the annotated image."""
    st.markdown("**Class colors**")
    # Build a small inline HTML legend. One swatch + label per class.
    items = []
    for cls in S.CLASS_NAMES:
        b, g, r = S.CLASS_COLORS.get(cls, (0, 255, 0))
        # The CLASS_COLORS dict stores BGR (for OpenCV); convert to RGB for HTML.
        rgb = f"rgb({r},{g},{b})"
        items.append(
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"background:{rgb};margin-right:6px;vertical-align:middle;"
            f"border:1px solid #888;'></span>{cls}"
        )
    legend_html = " &nbsp; ".join(items)
    st.markdown(legend_html, unsafe_allow_html=True)


def render_no_detections_message() -> None:
    """Friendly message when YOLO + filters return zero boxes."""
    st.info(
        "No road damage detected on this image (with the current pipeline "
        "and filters). Try a different pipeline — switching off post-processing "
        "or using the RAW model often surfaces more candidate detections."
    )


def render_crack_measurements(analysis: Dict) -> None:
    """
    Render the per-crack measurement table (length/width in pixels), if any
    measurements were produced by the crack-measurement step.
    """
    measurements = analysis.get("crack_measurements") or []
    if not measurements:
        return

    st.markdown("### Crack Measurements")
    st.caption(
        "Centreline length and width measured in pixels via morphological "
        "skeletonization and distance transform. Descriptive only — does not "
        "affect detection metrics."
    )
    rows = []
    for i, m in enumerate(measurements, start=1):
        rows.append({
            "#": i,
            "Class": m.get("class_name", ""),
            "Length (px)": m.get("length_px", 0.0),
            "Avg Width (px)": m.get("avg_width_px", 0.0),
            "Max Width (px)": m.get("max_width_px", 0.0),
            "Crack Area (px²)": m.get("area_px", 0.0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Full detailed summary tables (bottom of page)
# =============================================================================

# Class display order for the summary table columns
CLASS_ORDER = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole",
]


def _match_measurement(detection: Dict, measurements: List[Dict]) -> Dict:
    """
    Best-effort pairing of a detection with its crack measurement.

    Measurements are produced per crack-class detection in the same order the
    detections are iterated, so we match by class name in sequence. Returns an
    empty dict if no measurement is available for this detection.
    """
    cls = detection["class_name"]
    for m in measurements:
        if not m.get("_used") and m.get("class_name") == cls:
            m["_used"] = True
            return m
    return {}


def render_detailed_detection_table(
    detections: List[Dict],
    analysis: Dict,
    pipeline_name: str,
) -> None:
    """
    Full per-detection table combining EVERYTHING known about each detection:
    pipeline, image-level severity, class, confidence, full box geometry, and
    (if available) the crack measurement (length / avg width / max width /
    crack area) for that detection.

    One row per detection. This is the comprehensive bottom-of-page table.
    """
    if not detections:
        return

    # Work on a copy of measurements so the "_used" flag doesn't leak out
    measurements = [dict(m) for m in (analysis.get("crack_measurements") or [])]
    severity = analysis.get("severity", "")

    rows = []
    for i, d in enumerate(detections, start=1):
        m = _match_measurement(d, measurements)
        rows.append({
            "#": i,
            "Pipeline": pipeline_name,
            "Severity": severity,
            "Crack Type": d["class_name"],
            "Confidence": round(d["confidence"], 3),
            "X1": int(d["x1"]),
            "Y1": int(d["y1"]),
            "X2": int(d["x2"]),
            "Y2": int(d["y2"]),
            "Box W (px)": int(d["box_width"]),
            "Box H (px)": int(d["box_height"]),
            "Box Area (px²)": int(d["box_area"]),
            "Aspect Ratio": round(d.get("aspect_ratio", 0.0), 2),
            "Crack Length (px)": m.get("length_px", "—"),
            "Avg Width (px)": m.get("avg_width_px", "—"),
            "Max Width (px)": m.get("max_width_px", "—"),
            "Crack Area (px²)": m.get("area_px", "—"),
        })

    st.markdown("### Full Detection & Measurement Table")
    st.caption(
        "One row per detected defect, combining detection geometry, confidence, "
        "and (where available) crack measurements. Crack measurements are present "
        "only for crack-type classes when the measurement option is enabled."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_pipeline_summary_table(
    analysis: Dict,
    pipeline_name: str,
) -> None:
    """
    Image-level summary table (single row): pipeline, severity, totals,
    density, dominant type, average confidence, and per-class counts.

    Complements the per-detection table above by giving the whole-image
    picture in one line.
    """
    counts = analysis.get("class_counts", {})
    row = {
        "Pipeline": pipeline_name,
        "Severity": analysis.get("severity", ""),
        "Total Defects": analysis.get("total_detections", 0),
        "Damage Density (%)": round(analysis.get("damage_density", 0.0), 2),
        "Dominant Type": analysis.get("dominant_type", "None"),
        "Avg Confidence": round(analysis.get("avg_confidence", 0.0), 3),
        "Largest Box (px²)": int(analysis.get("largest_box_area", 0.0)),
    }
    # Per-class counts as their own columns
    for cls in CLASS_ORDER:
        row[f"{cls.split()[0].capitalize()}"] = counts.get(cls, 0)

    st.markdown("### Pipeline Summary Table")
    st.caption(
        "Whole-image summary for the selected pipeline, including per-class "
        "defect counts."
    )
    st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)


# =============================================================================
# Footer
# =============================================================================

def render_footer() -> None:
    """Small print at the bottom of the page."""
    st.divider()
    st.caption(
        "Road Crack Detection — YOLOv8s | Master's Project, "
        "Advanced Image Processing. Trained on RDD2022 with five damage "
        "classes (longitudinal, transverse, alligator, other corruption, pothole)."
    )