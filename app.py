"""
app.py
======
Streamlit entry point for the Road Crack Detection application.

Run with:
    streamlit run app.py

Two view modes (sidebar toggle):
    Single Pipeline  — pick one of four pipelines, see detailed results
    Compare All 4    — run all four pipelines on the same image and display
                       them side-by-side in a 2×2 grid with per-pipeline
                       metrics below each image
"""

from __future__ import annotations

import io
from typing import Dict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import settings as S
from pipelines.experiment_runner import run_pipeline, run_all_pipelines, PipelineResult
from ui import components as C


# =============================================================================
# Page configuration — must be first Streamlit call
# =============================================================================

st.set_page_config(
    page_title=S.APP_TITLE,
    page_icon=S.APP_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Helpers
# =============================================================================

def _decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile to a BGR uint8 NumPy array."""
    img_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _check_models_present() -> bool:
    """Verify both .pt files exist before the user does anything."""
    missing = []
    if not S.RAW_MODEL_WEIGHTS.exists():
        missing.append(str(S.RAW_MODEL_WEIGHTS))
    if not S.PREP_MODEL_WEIGHTS.exists():
        missing.append(str(S.PREP_MODEL_WEIGHTS))
    if missing:
        st.error(
            "**Trained model weights are missing.**\n\n"
            "Place `raw_best.pt` and `prep_best.pt` in the `models/` folder.\n\n"
            "Expected paths:\n" + "\n".join(f"- `{p}`" for p in missing)
        )
        return False
    return True


def _size_ok(uploaded_file) -> bool:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > S.MAX_UPLOAD_MB:
        st.error(
            f"This image is {size_mb:.1f} MB, which exceeds the "
            f"{S.MAX_UPLOAD_MB} MB limit. Please upload a smaller image."
        )
        return False
    return True


# =============================================================================
# Single-pipeline view (original behaviour)
# =============================================================================

def _run_single(image_bgr: np.ndarray, uploaded_file, options: Dict) -> None:
    pipeline_name = options["pipeline_name"]

    with st.spinner(f"Running: {pipeline_name} …"):
        try:
            result = run_pipeline(
                image_bgr,
                pipeline_name=pipeline_name,
                image_name=uploaded_file.name,
                apply_roi_enhancement=options["apply_roi_enhancement"],
                apply_crack_measurement=options.get("apply_crack_measurement", False),
            )
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    st.subheader(f"Results — {result.pipeline_name}")

    if options["show_original"]:
        C.render_image_pair(
            original_bgr=result.image_original,
            annotated_bgr=result.annotated_image,
            original_caption=f"Original — {uploaded_file.name}",
            annotated_caption="Annotated output",
        )
    else:
        C.render_single_image(result.annotated_image)

    st.markdown("### Summary")
    C.render_summary_metrics(result.analysis)

    if result.detections:
        st.markdown("### Detections")
        C.render_detection_table(result.detections)
    else:
        C.render_no_detections_message()

    # Per-crack measurements (only present if the option was enabled)
    C.render_crack_measurements(result.analysis)

    if options["show_timings"]:
        st.markdown("### Timings")
        C.render_timings(result.timings)

    # ---- Full detailed tables at the bottom of the page ----
    st.divider()
    C.render_pipeline_summary_table(result.analysis, result.pipeline_name)
    C.render_detailed_detection_table(
        result.detections, result.analysis, result.pipeline_name
    )


# =============================================================================
# Compare-all-4 view
# =============================================================================

# Short labels that fit inside a column header
_PIPELINE_SHORT = {
    S.PIPELINE_RAW:       "① RAW → YOLO",
    S.PIPELINE_PREP:      "② PREP → YOLO",
    S.PIPELINE_RAW_POST:  "③ RAW → YOLO → Post",
    S.PIPELINE_PREP_POST: "④ PREP → YOLO → Post",
}

# Pipeline descriptions shown under each column header
_PIPELINE_DESC = {
    S.PIPELINE_RAW:       "Raw image fed directly to the RAW-trained model. No preprocessing, no filtering.",
    S.PIPELINE_PREP:      "Crack-saliency enhancement applied first, then fed to the PREP-trained model.",
    S.PIPELINE_RAW_POST:  "RAW → YOLO output refined by confidence, area, aspect-ratio and NMS filters.",
    S.PIPELINE_PREP_POST: "Preprocessing + YOLO + full detection-refinement filter chain.",
}


def _render_pipeline_card(
    col,
    result: PipelineResult,
    pipeline_name: str,
    show_timings: bool = True,
) -> None:
    """Render one pipeline's annotated image + compact metrics inside a column."""
    import pandas as pd
    with col:
        st.markdown(f"**{_PIPELINE_SHORT[pipeline_name]}**")
        st.caption(_PIPELINE_DESC[pipeline_name])

        # Annotated image
        st.image(
            cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        # Compact metric row
        a = result.analysis
        m1, m2, m3 = st.columns(3)
        m1.metric("Defects",  a["total_detections"])
        m2.metric("Severity", a["severity"])
        m3.metric("Density",  f"{a['damage_density']:.1f}%")

        m4, m5 = st.columns(2)
        m4.metric("Avg Conf",     f"{a['avg_confidence']:.2f}")
        m5.metric("Dominant",     a["dominant_type"])

        # Per-class count bar (compact table, no index)
        counts = {
            cls: a["class_counts"].get(cls, 0)
            for cls in S.CLASS_NAMES
        }
        df = pd.DataFrame(
            [{"Class": cls, "Count": cnt} for cls, cnt in counts.items() if cnt > 0]
        )
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True, height=35 + 35 * len(df))
        else:
            st.caption("No detections.")

        # Crack measurements (only present if the option was enabled)
        measurements = a.get("crack_measurements") or []
        if measurements:
            mrows = [{
                "Class": m.get("class_name", ""),
                "Len (px)": m.get("length_px", 0.0),
                "Avg W (px)": m.get("avg_width_px", 0.0),
            } for m in measurements]
            st.caption("Crack measurements")
            st.dataframe(pd.DataFrame(mrows), use_container_width=True,
                         hide_index=True, height=35 + 35 * len(mrows))

        # Timing badge
        if show_timings:
            total_t = result.timings.get("total")
            if total_t is not None:
                st.caption(f"⏱ {total_t:.2f} s")


def _run_compare(image_bgr: np.ndarray, uploaded_file, options: dict) -> None:
    apply_roi = options.get("apply_roi_enhancement", False)
    apply_meas = options.get("apply_crack_measurement", False)
    show_original = options.get("show_original", True)
    show_timings = options.get("show_timings", True)

    with st.spinner("Running all 4 pipelines — this may take 10–30 s on CPU …"):
        try:
            all_results = run_all_pipelines(
                image_bgr,
                image_name=uploaded_file.name,
                apply_roi_enhancement=apply_roi,
                apply_crack_measurement=apply_meas,
            )
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"One or more pipelines failed: {e}")
            return

    st.subheader("4-Pipeline Comparison")
    st.caption(
        f"Image: **{uploaded_file.name}** — same image, four different experiment configurations."
    )

    # ---- Original image full-width at top (respects the show-original option) ----
    if show_original:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Original input image",
            use_container_width=True,
        )
        st.divider()

    # ---- Row 1: pipelines 1 and 2 ----
    st.markdown("#### Without Post-processing")
    col1, col2 = st.columns(2)
    _render_pipeline_card(col1, all_results[S.PIPELINE_RAW],  S.PIPELINE_RAW,  show_timings)
    _render_pipeline_card(col2, all_results[S.PIPELINE_PREP], S.PIPELINE_PREP, show_timings)

    st.divider()

    # ---- Row 2: pipelines 3 and 4 ----
    st.markdown("#### With Post-processing")
    col3, col4 = st.columns(2)
    _render_pipeline_card(col3, all_results[S.PIPELINE_RAW_POST],  S.PIPELINE_RAW_POST,  show_timings)
    _render_pipeline_card(col4, all_results[S.PIPELINE_PREP_POST], S.PIPELINE_PREP_POST, show_timings)

    st.divider()

    # ---- Summary comparison table ----
    st.markdown("#### Side-by-side Metrics")
    import pandas as pd
    rows = []
    for pname in S.PIPELINE_NAMES:
        a = all_results[pname].analysis
        t = all_results[pname].timings.get("total", float("nan"))
        row = {
            "Pipeline":        _PIPELINE_SHORT[pname],
            "Defects":         a["total_detections"],
            "Severity":        a["severity"],
            "Density (%)":     round(a["damage_density"], 2),
            "Avg Confidence":  round(a["avg_confidence"],  3),
            "Dominant Type":   a["dominant_type"],
        }
        if show_timings:
            row["Time (s)"] = round(t, 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    C.render_header()

    if not _check_models_present():
        st.stop()

    # ---- Sidebar ----
    st.sidebar.header("Input")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a road image",
        type=list(S.SUPPORTED_UPLOAD_EXTS),
        help=f"Supported: {', '.join(S.SUPPORTED_UPLOAD_EXTS)}. Max {S.MAX_UPLOAD_MB} MB.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### View Mode")
    view_mode = st.sidebar.radio(
        "Display",
        options=["Single Pipeline", "Compare All 4"],
        index=0,
        help=(
            "Single Pipeline: choose one pipeline and see full detail.\n"
            "Compare All 4: run every pipeline on the same image and view them together."
        ),
    )

    st.sidebar.divider()

    if view_mode == "Single Pipeline":
        pipeline_name = C.render_pipeline_picker(default_index=3)
        options = C.render_options_panel()
        options["pipeline_name"] = pipeline_name
    else:
        # Compare mode now exposes the SAME options as single mode so the two
        # views behave consistently. "show_original" is handled differently
        # here (the original is always shown once at the top), but the flag is
        # still read so the panel is identical.
        options = C.render_options_panel()

    st.sidebar.divider()
    C.render_class_legend()

    # ---- Main area ----
    if uploaded_file is None:
        st.info(
            "Upload a road image from the sidebar to begin. "
            "Use **Compare All 4** mode to view all experiment pipelines at once."
        )
        C.render_footer()
        return

    if not _size_ok(uploaded_file):
        return

    try:
        image_bgr = _decode_uploaded_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        return

    if view_mode == "Single Pipeline":
        _run_single(image_bgr, uploaded_file, options)
    else:
        _run_compare(image_bgr, uploaded_file, options)

    C.render_footer()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
