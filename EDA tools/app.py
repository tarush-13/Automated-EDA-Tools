"""Smart Data Quality Analyzer & Automated EDA Tool — Streamlit dashboard.

Run with:  streamlit run app.py

Pipeline (multi-tab, sidebar navigation):
  Upload → Profile → Quality → Cleaning → EDA → Insights → Score → Report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import APP_NAME, APP_VERSION, CLEANING_METHODS_LABELS, OUTLIER_ACTIONS_LABELS

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Imports (deferred is not required, but keep stage imports obvious)
# --------------------------------------------------------------------------- #
from load import load_dataset
from profile import detect_column_types, profile_dataset
from quality import (
    missingness_heatmap,
    plot_outliers,
    run_quality_analysis,
)
from clean import run_cleaning
from eda import run_eda
from visualize import (
    histogram,
    boxplot,
    scatter_matrix,
    bar_chart,
    scatter,
    corr_heatmap,
    auto_generate_charts,
)
from insights import generate_insights
from score import compute_quality_score, DIMENSION_WEIGHTS, score_delta_after_cleaning
from report import build_report_data, generate_html_report, generate_pdf_report


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def init_state():
    defaults = {
        "raw_df": None,
        "current_df": None,
        "load_result": None,
        "column_types": None,
        "profile": None,
        "quality": None,
        "eda": None,
        "insights": None,
        "score_before": None,
        "score_after": None,
        "cleaning_log": None,
        "cleaning_preview": None,
        "report_data": None,
        "active_page": "Upload",
        "file_name": "Uploaded dataset",
        "auto_clean_done": False,
        "processed_upload": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def require_dataset() -> bool:
    if st.session_state.raw_df is None:
        st.info("👆 Please load a dataset on the **Upload** tab first.")
        return False
    return True


def rerun():
    for key in ["score_before", "score_after", "insights", "report_data", "quality", "eda", "profile",
                "current_df", "cleaning_log", "cleaning_preview", "column_types_cleaned"]:
        st.session_state[key] = None
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = [
    "📤 Upload",
    "🔍 Profile",
    "⚠️ Quality",
    "🧹 Cleaning",
    "📈 EDA",
    "💡 Insights",
    "🎯 Score",
    "📄 Report",
]

with st.sidebar:
    st.markdown(f"# {APP_NAME}")
    st.caption(f"v{APP_VERSION}")
    st.divider()
    page = st.radio("Pipeline navigation", PAGES, label_visibility="collapsed")
    st.session_state["active_page"] = page

    st.divider()
    if st.session_state.raw_df is not None:
        st.metric("Rows loaded", f"{len(st.session_state.raw_df):,}")
        st.metric("Columns", len(st.session_state.raw_df.columns))
        if st.session_state.cleaning_log is not None:
            st.info(f"Cleaning actions logged: **{len(st.session_state.cleaning_log)}**")
        if st.button("🔄 Reset analysis (re-run full pipeline)"):
            rerun()

    if st.session_state.load_result is not None:
        st.caption(
            f"File: {st.session_state.load_result.file_name} · "
            f"{st.session_state.load_result.file_type.upper()}"
        )
        if st.session_state.load_result.truncated:
            st.warning("Truncated to 200k rows for analysis.")


# --------------------------------------------------------------------------- #
# Upload tab
# --------------------------------------------------------------------------- #
def tab_upload():
    st.header("📤 Upload dataset")
    st.caption("Accepted formats: **CSV**, **XLSX**, **JSON** — encoding/delimiter auto-detected.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Choose a file", type=["csv", "txt", "tsv", "xlsx", "xls", "json"],
            help="Larger files are loaded lazily in chunks (max 200k rows analysed).",
        )
    with col2:
        use_sample = st.selectbox("Or load a bundled sample", ["", "Mixed dataset", "Numeric-heavy", "Categorical-heavy"])
    with col3:
        st.write("")
        st.write("")
        sample_btn = st.button("Load sample", type="primary", disabled=not use_sample)

    if uploaded is not None:
        # Avoid reprocessing the same file on every Streamlit rerun (the widget
        # stays truthy) — guard on (name, size) so it runs once per upload.
        upload_key = (uploaded.name, getattr(uploaded, "size", 0))
        if st.session_state.get("processed_upload") != upload_key:
            with st.spinner("Loading & sniffing encoding/delimiter..."):
                result = load_dataset(uploaded)
            st.session_state.load_result = result
            st.session_state.raw_df = result.df.copy()
            st.session_state.auto_clean_done = False
            st.session_state.column_types = detect_column_types(result.df)
            st.session_state.file_name = result.file_name
            st.session_state.processed_upload = upload_key
            _clear_downstream()
            st.session_state["score_before"] = _compute_score_snapshot()
            st.success(f"Loaded {len(result.df):,} rows × {len(result.df.columns)} columns from `{result.file_name}`")
            _show_load_metadata(result)
            st.session_state["profile"] = profile_dataset(result.df, st.session_state.column_types)

    if sample_btn and use_sample:
        df = _load_bundled_sample(use_sample)
        if df is not None:
            from load import _BytesLikeReadable as _BLR

            st.session_state.load_result = load_dataset(_BLR(_to_csv_bytes(df), name=f"{use_sample.replace(' ', '_').lower()}.csv"))
            st.session_state.raw_df = df.copy()
            st.session_state.auto_clean_done = False
            st.session_state.processed_upload = (f"{use_sample}.csv", None)
            st.session_state.column_types = detect_column_types(df)
            st.session_state.file_name = f"{use_sample}.csv"
            _clear_downstream()
            st.session_state["profile"] = profile_dataset(df, st.session_state.column_types)
            st.session_state["score_before"] = _compute_score_snapshot()
            st.success(f"Sample **{use_sample}** loaded ({len(df):,} rows × {len(df.columns)} cols).")

    if st.session_state.raw_df is None:
        st.markdown("---")
        st.info("No data loaded yet. Drop a file on the left, or pick a bundled sample to try it instantly.")


def _clear_downstream():
    for k in ["quality", "eda", "insights", "score_after", "cleaning_log",
              "cleaning_preview", "report_data", "column_types_cleaned",
              "current_df", "profile"]:
        st.session_state[k] = None


def _compute_score_snapshot():
    df = st.session_state.raw_df
    if df is None:
        return None
    column_types = st.session_state.column_types
    quality = run_quality_analysis(df, column_types)
    return compute_quality_score(
        df,
        quality["missing"],
        quality["duplicates"]["count"],
        column_types,
        quality["type_issues"],
        quality["categorical"],
        quality["validation"],
        quality["outliers"],
    )


def _show_load_metadata(result):
    with st.expander("Load details (auto-detected)"):
        st.json({
            "file": result.file_name,
            "type": result.file_type,
            "encoding": result.encoding,
            "delimiter": result.delimiter,
            "rows_loaded": len(result.df),
            "original_lines": result.original_rows,
            "chunks_processed": result.chunk_count,
            "truncated": result.truncated,
        })


# --------------------------------------------------------------------------- #
# Profile tab
# --------------------------------------------------------------------------- #
def tab_profile():
    st.header("🔍 Dataset profiling")
    if not require_dataset():
        return
    df = st.session_state.raw_df
    if st.session_state.profile is None:
        with st.spinner("Profiling dataset..."):
            st.session_state.profile = profile_dataset(df, st.session_state.column_types)
    profile = st.session_state.profile

    mcol = st.columns(6)
    metrics = [
        ("Rows", f"{profile.n_rows:,}"),
        ("Columns", profile.n_cols),
        ("Memory", f"{profile.memory_mb:.1f} MB"),
        ("Duplicates", f"{profile.duplicated_count:,}"),
        ("Missing cells", f"{int(profile.missing_by_col.sum()):,}"),
        ("Completeness", f"{100 * (1 - profile.missing_by_col.sum() / df.size) if df.size else 0:.1f}%"),
    ]
    for c, (label, val) in zip(mcol, metrics):
        c.metric(label, val)

    st.subheader("Detected column types")
    counts = {
        k: int(v)
        for k, v in {
            "Numerical": profile.n_numeric,
            "Categorical": profile.n_categorical,
            "Datetime": profile.n_datetime,
            "Free text": profile.n_text,
            "Boolean": profile.n_boolean,
        }.items()
    }
    st.bar_chart(pd.Series(counts))

    st.subheader("Column type detection (with reasoning)")
    type_rows = pd.DataFrame(
        {
            "column": list(profile.column_types.keys()),
            "detected_kind": [t.kind for t in profile.column_types.values()],
            "confidence": [t.confidence for t in profile.column_types.values()],
            "reason": ["; ".join(t.reasons) for t in profile.column_types.values()],
            "inferred_dtype": [t.infer_dtype for t in profile.column_types.values()],
        }
    )
    st.dataframe(type_rows, use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(profile.summary, use_container_width=True, height=400)

    st.subheader("Data preview")
    st.dataframe(df.head(50), use_container_width=True)


# --------------------------------------------------------------------------- #
# Quality tab
# --------------------------------------------------------------------------- #
def ensure_quality():
    if st.session_state.quality is None:
        with st.spinner("Running data quality analysis..."):
            st.session_state.quality = run_quality_analysis(
                st.session_state.raw_df, st.session_state.column_types
            )
    return st.session_state.quality


def tab_quality():
    st.header("⚠️ Data quality analysis")
    if not require_dataset():
        return
    df = st.session_state.raw_df
    quality = ensure_quality()

    tabs = st.tabs(["Missing values", "Duplicates", "Type consistency", "Outliers", "Categorical", "Validation"])

    # ---- Missing ----
    with tabs[0]:
        mr = quality["missing"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Missing cells", f"{mr.total_missing:,}")
        c2.metric("Overall missing", f"{mr.overall_pct:.2f}%")
        c3.metric("Missing columns", f"{(mr.by_column['missing'] > 0).sum()}")
        st.plotly_chart(missingness_heatmap(df), use_container_width=True)
        st.dataframe(mr.by_column, use_container_width=True)

    # ---- Duplicates ----
    with tabs[1]:
        d = quality["duplicates"]
        c1, c2 = st.columns(2)
        c1.metric("Duplicate rows", f"{d['count']:,}")
        c2.metric("Share", f"{d['share']:.2f}%")
        if d["count"]:
            with st.expander(f"Preview duplicated rows ({len(d['preview'])} shown)"):
                st.dataframe(d["preview"], use_container_width=True)
            with st.expander("Preview after removing duplicates"):
                st.dataframe(d["deduplicated_preview"], use_container_width=True)
        else:
            st.success("No fully duplicated rows found.")

    # ---- Type consistency ----
    with tabs[2]:
        ti = quality["type_issues"]
        if ti.empty:
            st.success("✅ No type-consistency anomalies detected.")
        else:
            st.dataframe(ti, use_container_width=True)

    # ---- Outliers ----
    with tabs[3]:
        outliers = quality["outliers"]
        method = st.radio("Method", ["IQR", "Z-score", "Both"], horizontal=True)
        visible = [o for o in outliers if method in ("Both", o.method)]
        if not visible:
            st.info("No numeric columns with enough rows to test.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "column": o.column, "method": o.method,
                            "lower_bound": round(o.bounds[0], 3) if o.bounds[0] is not None else None,
                            "upper_bound": round(o.bounds[1], 3) if o.bounds[1] is not None else None,
                            "outliers": o.n_outliers, "share_%": o.share_pct,
                        }
                        for o in visible
                    ]
                ),
                use_container_width=True,
            )
            st.plotly_chart(plot_outliers(df, st.session_state.column_types, visible), use_container_width=True)

    # ---- Categorical ----
    with tabs[4]:
        cat = quality["categorical"]
        if cat:
            for cname, catres in cat.items():
                with st.expander(f"{cname} — {catres.n_unique} unique, {catres.suggestion or 'no major issue'}"):
                    if not catres.top_values.empty:
                        st.dataframe(catres.top_values, use_container_width=True)
                    if catres.rare_categories is not None and not catres.rare_categories.empty:
                        st.markdown("**Rare categories (<1%)**")
                        st.dataframe(catres.rare_categories, use_container_width=True)
                    if catres.inconsistent_labels is not None and not catres.inconsistent_labels.empty:
                        st.markdown("**Inconsistent labels**")
                        st.dataframe(catres.inconsistent_labels, use_container_width=True)
        else:
            st.info("No categorical columns detected.")

    # ---- Validation ----
    with tabs[5]:
        val = quality["validation"]
        if val:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "column": v.column, "check": v.check, "severity": v.severity,
                            "failures": v.failures, "share_%": v.share_pct,
                            "examples": v.examples[:120], "detail": v.detail,
                        }
                        for v in val
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.success("✅ No validation rule violations detected.")


# --------------------------------------------------------------------------- #
# Cleaning tab
# --------------------------------------------------------------------------- #
def tab_cleaning():
    st.header("🧹 Interactive data cleaning")
    if not require_dataset():
        return
    df = st.session_state.raw_df
    column_types = st.session_state.column_types

    st.caption("Every action is logged and can be undone by re-running. Changes apply to a **working copy**.")

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("1 · Missing values")
        missing_by_col = df.isna().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0].index.tolist()
        if not cols_with_missing:
            st.success("No missing values in any column.")
        else:
            imputation = {}
            constants = {}
            for col in cols_with_missing:
                n_miss = int(missing_by_col[col])
                c1, c2 = st.columns([2, 1.4])
                with c1:
                    method = st.selectbox(
                        f"{col} ({n_miss} missing)",
                        list(CLEANING_METHODS_LABELS.keys()),
                        format_func=lambda m, _c=col: CLEANING_METHODS_LABELS[m],
                        key=f"imp_{col}",
                    )
                with c2:
                    const = None
                    if method == "constant":
                        const = st.text_input("Constant", value="", key=f"const_{col}")
                imputation[col] = method
                if method == "constant":
                    constants[col] = const

        st.subheader("2 · Duplicates")
        c1, c2 = st.columns(2)
        with c1:
            drop_dups = st.checkbox("Remove duplicate rows", value=df.duplicated().sum() > 0, key="drop_dups")
        dup_cols = ["__all__"] + list(df.columns)
        with c2:
            dedup_subset = st.multiselect(
                "Consider subset (optional)", list(df.columns),
                help="Leave empty to compare the whole row.",
            )

        st.subheader("3 · Data types")
        correct_types = st.checkbox("Auto-correct dtypes (string → numeric/datetime)", value=True, key="fix_types")

        st.subheader("4 · Categorical labels")
        standardise = st.checkbox("Standardise categorical values (strip/case + fuzzy merge)", value=False, key="std_cats")
        fuzzy_threshold = st.slider("Fuzzy merge threshold", 60, 100, 88, disabled=not standardise)

        st.subheader("5 · Outliers")
        outlier_action = st.selectbox(
            "Handling strategy",
            ["", "flag", "cap", "remove"],
            format_func=lambda a: "" if not a else OUTLIER_ACTIONS_LABELS[a],
        )

    with right:
        st.subheader("Working copy preview")
        if df is not None:
            st.dataframe(df.head(15), use_container_width=True)

    st.divider()
    run_col, reset_col = st.columns([1, 3])
    with run_col:
        apply_btn = st.button("▶️ Apply cleaning", type="primary", use_container_width=True)
    with reset_col:
        pass

    if apply_btn:
        with st.spinner("Cleaning in progress..."):
            cleaned, log, stats = run_cleaning(
                df,
                column_types,
                imputation=imputation if cols_with_missing else None,
                constants=constants,
                drop_dups=drop_dups,
                dedup_subset=None if not dedup_subset or "__all__" in dedup_subset else dedup_subset,
                correct_types=correct_types,
                standardise_cats=standardise,
                fuzzy_threshold=fuzzy_threshold,
                outlier_action=outlier_action,
            )
        st.session_state.current_df = cleaned
        st.session_state.cleaning_log = log
        if st.session_state.current_df is not None:
            st.session_state.eda = None
            st.session_state.insights = None
            st.session_state.report_data = None
            st.session_state["column_types_cleaned"] = detect_column_types(cleaned)
            from score import compute_quality_score as _cq

            cleaned_types = st.session_state["column_types_cleaned"]
            cq = run_quality_analysis(cleaned, cleaned_types)
            st.session_state.quality = cq
            st.session_state.score_after = _cq(
                cleaned, cq["missing"], cq["duplicates"]["count"], cleaned_types,
                cq["type_issues"], cq["categorical"], cq["validation"], cq["outliers"],
            )

        st.success("Cleaning complete — see audit log below. Use the Report tab to export.")

    if st.session_state.cleaning_log is not None:
        st.subheader("Cleaning log (audit trail)")
        st.dataframe(st.session_state.cleaning_log.to_dataframe(), use_container_width=True)

        st.subheader("Before vs After")
        b = st.session_state.raw_df.shape
        a = st.session_state.current_df.shape if st.session_state.current_df is not None else b
        cols = st.columns(3)
        cols[0].metric("Rows", f"{b[0]:,} → {a[0]:,}")
        cols[1].metric("Columns", f"{b[1]:,} → {a[1]:,}")
        cols[2].metric("Missing cells", f"{int(st.session_state.raw_df.isna().sum().sum()):,} → "
                                        f"{int(st.session_state.current_df.isna().sum().sum()):,}")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download cleaned data (CSV)",
                data=_df_to_csv(st.session_state.current_df),
                file_name=f"{Path(st.session_state.file_name).stem}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download cleaning log (CSV)",
                data=_df_to_csv(st.session_state.cleaning_log.to_dataframe()),
                file_name="cleaning_log.csv",
                mime="text/csv",
                use_container_width=True,
            )


# --------------------------------------------------------------------------- #
# EDA tab
# --------------------------------------------------------------------------- #
def ensure_eda():
    df = _analysis_df()
    if st.session_state.eda is None:
        column_types = st.session_state.get("column_types_cleaned") or st.session_state.column_types
        with st.spinner("Running automated EDA..."):
            st.session_state.eda = run_eda(df, column_types)
    return st.session_state.eda


def _analysis_df():
    return st.session_state.current_df if st.session_state.current_df is not None else st.session_state.raw_df


def _analysis_types():
    return st.session_state.get("column_types_cleaned") or st.session_state.column_types


def tab_eda():
    st.header("📈 Automated EDA")
    if not require_dataset():
        return
    df = _analysis_df()
    column_types = _analysis_types()
    eda = ensure_eda()

    num_tab, cat_tab, bi_tab, auto_tab = st.tabs(
        ["Numerical columns", "Categorical columns", "Relationships", "Auto-generated charts"]
    )

    with num_tab:
        if eda.numeric:
            for n in eda.numeric:
                with st.expander(f"{n.column} — {n.distribution_label}"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.plotly_chart(histogram(df, n.column, bins=n.histogram_bins), use_container_width=True)
                    with c2:
                        st.dataframe(
                            pd.DataFrame(n.stats.items(), columns=["statistic", "value"]),
                            use_container_width=True, height=300,
                        )
        else:
            st.info("No numeric columns to analyse.")

        if len(eda.numeric_cols) >= 2:
            st.subheader("Numeric comparisons")
            st.plotly_chart(boxplot(df, eda.numeric_cols[:10]), use_container_width=True)

    with cat_tab:
        if eda.categorical:
            for c in eda.categorical:
                with st.expander(f"{c.column} — {c.distribution_label} ({c.n_unique} unique)"):
                    st.dataframe(c.top, use_container_width=True)
                    st.plotly_chart(bar_chart(df, c.column), use_container_width=True)
        else:
            st.info("No categorical columns to analyse.")

    with bi_tab:
        if eda.numeric_cols:
            st.subheader("Correlation matrix (numeric)")
            st.plotly_chart(corr_heatmap(eda.correlation), use_container_width=True)
            if len(eda.numeric_cols) > 2:
                st.plotly_chart(scatter_matrix(df, eda.numeric_cols), use_container_width=True)
        if eda.bivariate:
            st.subheader("Detected relationships")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"type": r.type, "x": r.x, "y": r.y, "metric": round(r.metric, 3),
                         "metric_name": r.metric_name, "detail": r.detail}
                        for r in eda.bivariate
                    ]
                ),
                use_container_width=True,
            )
        xcol, ycol, ccol = st.columns(3)
        with xcol:
            x = st.selectbox("X", eda.numeric_cols or [""], key="x_scatter")
        with ycol:
            y = st.selectbox("Y", eda.numeric_cols or [""], index=1 if len(eda.numeric_cols) > 1 else 0, key="y_scatter")
        with ccol:
            color = st.selectbox("Color by (optional)", [None] + eda.categorical_cols, key="c_scatter")
        if x and y:
            st.plotly_chart(scatter(df, x, y, color), use_container_width=True)

    with auto_tab:
        st.subheader("Auto-selected charts")
        charts = auto_generate_charts(df, column_types, eda)
        for i, chart in enumerate(charts):
            st.markdown(f"**{chart['title']}**")
            st.caption(chart["reason"])
            st.plotly_chart(chart["fig"], use_container_width=True, key=f"auto_{i}")


# --------------------------------------------------------------------------- #
# Insights tab
# --------------------------------------------------------------------------- #
def ensure_insights():
    if st.session_state.insights is None:
        with st.spinner("Generating insights..."):
            df = _analysis_df()
            quality = ensure_quality()
            eda = ensure_eda()
            st.session_state.insights = generate_insights(
                df, st.session_state.profile, quality, eda,
                missing_report_=quality["missing"],
                cleaning_log=st.session_state.cleaning_log,
            )
    return st.session_state.insights


def tab_insights():
    st.header("💡 Automated insights")
    if not require_dataset():
        return
    report = ensure_insights()

    issues = [i for i in report.insights if i.severity in ("warning", "error")]
    st.progress(1 - len(issues) / max(1, len(report.insights)), text="Overall health indicator")

    if issues:
        with st.expander("⚠️ Top issues needing attention", expanded=True):
            for i in issues[:8]:
                st.markdown(f"- **[{i.severity.upper()}]** {i.text}")
    else:
        st.success("No significant issues detected.")

    st.subheader("All findings")
    for i in report.insights:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[i.severity]
        st.markdown(f"{icon} **({i.category})** {i.text}")


# --------------------------------------------------------------------------- #
# Score tab
# --------------------------------------------------------------------------- #
def _score_now(df, column_types):
    q = run_quality_analysis(df, column_types)
    return compute_quality_score(
        df, q["missing"], q["duplicates"]["count"], column_types,
        q["type_issues"], q["categorical"], q["validation"], q["outliers"],
    ), q


def tab_score():
    st.header("🎯 Data quality score")
    if not require_dataset():
        return

    df = _analysis_df()
    column_types = _analysis_types()
    if st.session_state.score_after is None:
        st.session_state.score_after, _ = _score_now(df, column_types)

    score = st.session_state.score_after
    before = st.session_state.get("score_before")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score.overall,
            delta={"reference": before.overall if before else 0},
            title={"text": f"Overall — {score.grade}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _grade_color(score.grade)},
                "steps": [
                    {"range": [0, 50], "color": "#f8d7da"},
                    {"range": [50, 65], "color": "#fff3cd"},
                    {"range": [65, 80], "color": "#d1ecf1"},
                    {"range": [80, 100], "color": "#d4edda"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.85, "value": score.overall},
            },
        ))
        fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[score.dimensions[k] for k in DIMENSION_WEIGHTS] + [score.dimensions["completeness"]],
            theta=list(DIMENSION_WEIGHTS) + ["completeness"],
            fill="toself", name="Score",
            line_color="#2a9df4",
        ))
        if before:
            fig.add_trace(go.Scatterpolar(
                r=[before.dimensions[k] for k in DIMENSION_WEIGHTS] + [before.dimensions["completeness"]],
                theta=list(DIMENSION_WEIGHTS) + ["completeness"],
                fill="toself", name="Before cleaning", opacity=0.35,
                line_color="#e74c3c",
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=350, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader(f"Overall score: {score.overall}/100 ({score.grade})")
        st.caption("Weights — " + ", ".join(f"{k}: {v}" for k, v in DIMENSION_WEIGHTS.items()))
        st.markdown("#### Dimension breakdown")
        dim_df = pd.DataFrame(
            {
                "dimension": list(score.dimensions.keys()),
                "score": list(score.dimensions.values()),
                "detail": [score.details.get(k, "") for k in score.dimensions],
            }
        )
        if before:
            dim_df["improvement"] = [f"{score_delta_after_cleaning(before, score)[k]:+.1f}" for k in score.dimensions]
        st.dataframe(dim_df, use_container_width=True)

    if before is not None and st.session_state.current_df is not None:
        delta = score_delta_after_cleaning(before, score)
        st.subheader("Improvement from cleaning")
        st.write({k: f"{v:+.1f}" for k, v in delta.items()})


def _grade_color(grade: str) -> str:
    return {
        "Excellent": "#28a745", "Good": "#7fb800", "Fair": "#f0b429",
        "Poor": "#e76f51", "Critical": "#c1121f",
    }.get(grade, "#2a9df4")


# --------------------------------------------------------------------------- #
# Report tab
# --------------------------------------------------------------------------- #
def build_report():
    if st.session_state.report_data is not None:
        return st.session_state.report_data
    if not require_dataset():
        return None
    df = _analysis_df()
    quality = ensure_quality()
    eda = ensure_eda()
    insights = ensure_insights()
    score = st.session_state.score_after or st.session_state.score_before

    outlier_summary = pd.DataFrame(
        [
            {"column": o.column, "method": o.method, "n_outliers": o.n_outliers, "share_%": o.share_pct}
            for o in quality["outliers"]
        ]
    )
    data = build_report_data(
        file_name=st.session_state.file_name,
        df=df,
        profile_result=st.session_state.profile,
        quality_result=quality,
        outlier_summary=outlier_summary,
        insight_report=insights,
        quality_score=score,
        cleaning_log=st.session_state.cleaning_log,
        eda_result=eda,
        is_cleaned=st.session_state.cleaning_log is not None,
    )
    st.session_state.report_data = data
    return data


def tab_report():
    st.header("📄 Report")
    if not require_dataset():
        return
    data = build_report()

    dl1, dl2, dl3 = st.columns(3)
    html_bytes = generate_html_report(data)
    with dl1:
        st.download_button(
            "⬇️ Download HTML report",
            data=html_bytes,
            file_name=f"{Path(st.session_state.file_name).stem}_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl2:
        try:
            pdf_bytes = generate_pdf_report(data)
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=f"{Path(st.session_state.file_name).stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF export failed ({exc}) — HTML is available.", icon="⚠️")
    with dl3:
        st.download_button(
            "⬇️ Download cleaned dataset",
            data=_df_to_csv(_analysis_df()),
            file_name=f"{Path(st.session_state.file_name).stem}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### HTML preview")
    st.components.v1.html(html_bytes, height=700, scrolling=True)


# --------------------------------------------------------------------------- #
# Sample dataset generation
# --------------------------------------------------------------------------- #
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _load_bundled_sample(name: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if name == "Numeric-heavy":
        n = 5000
        return pd.DataFrame({
            "sales": pd.Series(rng.normal(5000, 1200, n)).clip(lower=0).apply(lambda x: round(x, 2)),
            "profit_margin": pd.Series(rng.normal(18, 7, n)).clip(lower=0).apply(lambda x: round(x, 2)),
            "units_sold": rng.poisson(340, n),
            "days_to_ship": rng.integers(1, 60, n),
            "return_rate": pd.Series(rng.normal(6, 3, n)).clip(0, 40).round(2),
            "customer_age": rng.integers(18, 85, n),
            "order_value": pd.Series(rng.exponential(90, n)).apply(lambda x: round(x, 2)),
        })
    if name == "Categorical-heavy":
        n = 4000
        regions = rng.choice(["North", "South", "East", "West", "north", "NORTH", "south ", "East Coast"], size=n, p=[.2, .18, .15, .12, .1, .08, .1, .07])
        segments = rng.choice(["Premium", "Standard", "Budget", "budget", "standard", "PREMIUM"], size=n, p=[.25, .3, .2, .1, .1, .05])
        products = rng.choice(["Widget A", "Widget B", "Gadget C", "Gizmo D", "Gadget c"], size=n, p=[.35, .25, .2, .12, .08])
        return pd.DataFrame({
            "region": regions,
            "customer_segment": segments,
            "product": products,
            "channel": rng.choice(["Online", "Retail", "Partner", "Wholesale"], size=n, p=[.4, .3, .2, .1]),
            "repeat_customer": rng.choice([True, False], size=n, p=[.45, .55]),
            "tier": rng.choice(["A", "B", "C"], size=n),
        })
    # Mixed dataset
    n = 1500
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    mixed = pd.DataFrame({
        "transaction_id": [f"TRX-{1000 + i}" for i in rng.permutation(n)],
        "date": dates,
        "amount": pd.Series(rng.normal(250, 90, n)).clip(lower=5).round(2),
        "category": rng.choice(["Food", "Travel", "Utilities", "Entertainment"], size=n, p=[.4, .25, .2, .15]),
        "city": rng.choice(["New York", "Chicago", "LA", "Houston", "new york", "CHICAGO"], size=n, p=[.3, .25, .2, .1, .1, .05]),
        "email": [f"user{i % 120}@example.com" for i in range(n)],
        "age": rng.integers(18, 70, n),
        "is_fraud": rng.choice([True, False], size=n, p=[.04, .96]),
    })
    # sprinkle some dirty data
    mixed.loc[0:20, "age"] = "abc"
    mixed.loc[21:30, "email"] = ["not-an-email"] * 10
    mixed.loc[31:40, "amount"] = np.nan
    mixed.loc[41:50, "date"] = np.nan
    return mixed


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    st.title(APP_NAME)
    renderers = {
        "📤 Upload": tab_upload,
        "🔍 Profile": tab_profile,
        "⚠️ Quality": tab_quality,
        "🧹 Cleaning": tab_cleaning,
        "📈 EDA": tab_eda,
        "💡 Insights": tab_insights,
        "🎯 Score": tab_score,
        "📄 Report": tab_report,
    }
    renderers[st.session_state["active_page"]]()

    # pipeline progress: which stages are done
    progress = []
    if st.session_state.raw_df is not None:
        progress.append("Upload")
    for name, flag in [
        ("Profile", st.session_state.profile is not None),
        ("Quality", st.session_state.quality is not None),
        ("Cleaning", st.session_state.cleaning_log is not None),
        ("EDA", st.session_state.eda is not None),
        ("Insights", st.session_state.insights is not None),
        ("Score", (st.session_state.score_after or st.session_state.score_before) is not None),
        ("Report", st.session_state.report_data is not None),
    ]:
        if flag:
            progress.append(name)
    if progress:
        st.divider()
        st.caption("Pipeline progress: " + " → ".join(progress))


def _df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


if __name__ == "__main__":
    main()