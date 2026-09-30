"""Auto-generated Plotly visualisations.

Charts are selected automatically based on the detected column types:

- numeric        -> histogram + box plot (+ optional scatter when paired)
- categorical    -> bar / pie chart
- datetime       -> time-series line chart
- boolean        -> donut chart
- numeric x all  -> scatter/correlation matrix
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE
from profile import ColumnTypeResult


def _palette() -> List[str]:
    return px.colors.qualitative.Plotly


# --------------------------------------------------------------------------- #
# Single-column charts
# --------------------------------------------------------------------------- #
def histogram(df: pd.DataFrame, col: str, bins: Optional[int] = None, show_box: bool = True) -> go.Figure:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if bins is None:
        bins = 20
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                        vertical_spacing=0.06, subplot_titles=(f"{col} distribution", "Box plot"))
    fig.add_trace(go.Histogram(x=series.values, nbinsx=bins, name="histogram", marker_color="#2a9df4"), row=1, col=1)
    fig.add_trace(go.Box(x=series.values, name="box", marker_color="#e67e22", boxmean=True), row=2, col=1)
    fig.update_layout(
        title=f"Distribution of {col}",
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=False,
        margin=dict(l=60, r=20, t=70, b=40),
    )
    return fig


def boxplot(df: pd.DataFrame, numeric_cols: List[str], by: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        fig.add_trace(go.Box(y=series.values, name=col, boxmean="sd"))
    fig.update_layout(
        title=f"Box plots of numeric columns" + (f" grouped by {by}" if by else ""),
        template=PLOTLY_TEMPLATE,
        height=max(400, 80 * len(numeric_cols)),
        margin=dict(l=60, r=20, t=60, b=80),
    )
    if len(numeric_cols) > 8:
        fig.update_xaxes(tickangle=45)
    return fig


def grouped_box(df: pd.DataFrame, numeric_col: str, cat_col: str, top_n: int = 10) -> go.Figure:
    sample = df
    if len(df) > 50_000:
        sample = df.sample(50_000, random_state=42)
    top = sample[cat_col].dropna().value_counts().head(top_n).index
    sub = sample[sample[cat_col].isin(top)]
    fig = px.box(sub, x=cat_col, y=numeric_col, color=cat_col,
                 title=f"{numeric_col} by {cat_col}")
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=440)
    fig.update_xaxes(tickangle=35)
    return fig


def bar_chart(df: pd.DataFrame, col: str, top_n: int = 15, horizontal: bool = True) -> go.Figure:
    series = df[col].dropna()
    vc = series.value_counts().head(top_n)
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(x=vc.values[::-1], y=vc.index[::-1], orientation="h",
                             marker_color=_palette()[:len(vc)]))
        fig.update_layout(height=45 * min(len(vc), top_n) + 160)
    else:
        fig.add_trace(go.Bar(x=vc.index, y=vc.values, marker_color=_palette()[:len(vc)]))
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=440)
    fig.update_layout(
        title=f"Frequency of {col}",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=120, r=20, t=60, b=60),
    )
    return fig


def pie_chart(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    series = df[col].dropna()
    vc = series.value_counts()
    top = vc.head(top_n)
    other = vc[top_n:].sum() if len(vc) > top_n else 0
    if other > 0:
        top = pd.concat([top, pd.Series({"Other": other})])
    fig = px.pie(top, names=top.index, values=top.values,
                 title=f"Share of {col}",
                 hole=0.35 if len(top) > 2 else 0.0)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=440,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig


def donut_boolean(df: pd.DataFrame, col: str) -> go.Figure:
    vc = df[col].dropna().value_counts()
    labels = [str(k) for k in vc.index]
    fig = px.pie(vc, names=labels, values=vc.values, hole=0.55, title=f"Boolean breakdown: {col}")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
    return fig


def time_series(df: pd.DataFrame, date_col: str, value_col: Optional[str] = None,
                agg: str = "mean") -> go.Figure:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    tmp = pd.DataFrame({"date": dates})
    if value_col is None:
        # count by day
        daily = tmp.dropna().groupby(pd.Grouper(key="date", freq="D")).size().reset_index(name="count")
        fig = px.line(daily, x="date", y="count", markers=True, title=f"Records per day by {date_col}")
    else:
        tmp["value"] = pd.to_numeric(df[value_col], errors="coerce")
        tmp = tmp.dropna()
        freq = "D" if len(tmp) > 40 else "AS"
        grouped = tmp.groupby(pd.Grouper(key="date", freq=freq))["value"].agg(agg).reset_index()
        fig = px.line(grouped, x="date", y="value", markers=True, title=f"{value_col} over time ({agg})")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
    return fig


# --------------------------------------------------------------------------- #
# Pairwise charts
# --------------------------------------------------------------------------- #
def scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
            sample_limit: int = 8_000) -> go.Figure:
    sample = df.sample(min(len(df), sample_limit), random_state=42) if len(df) > sample_limit else df
    # A column detected as "numerical" may still hold non-numeric strings, so
    # coerce both axes and drop rows we cannot cast (keeps color aligned).
    sub = pd.DataFrame({
        "_x": pd.to_numeric(sample[x], errors="coerce"),
        "_y": pd.to_numeric(sample[y], errors="coerce"),
    })
    if color and color in sample.columns:
        sub["_color"] = sample[color].values
        fig_color = "_color"
    else:
        fig_color = None
    sub = sub.dropna(subset=["_x", "_y"])
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No numeric values in common for {y} vs {x}",
            showarrow=False,
            font=dict(size=14, color="#888888"),
        )
    else:
        fig = px.scatter(sub, x="_x", y="_y", color=fig_color,
                         opacity=0.65, trendline="ols",
                         title=f"{y} vs {x}" + (f", colored by {color}" if color else ""))
        fig.update_layout(xaxis_title=x, yaxis_title=y)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=460)
    return fig


def corr_heatmap(corr: pd.DataFrame, title: str = "Correlation matrix") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.round(3).values,
        x=list(corr.columns),
        y=list(corr.columns),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=max(400, 55 * len(corr.columns)),
        margin=dict(l=120, r=20, t=60, b=120),
    )
    return fig


def scatter_matrix(df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 6) -> go.Figure:
    if len(numeric_cols) > max_cols:
        # pick the most spread ones (highest variance share)
        variances = df[numeric_cols].apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
        cols = list(variances.head(max_cols).index)
    else:
        cols = numeric_cols
    sample = df.sample(min(len(df), 5_000), random_state=42)
    fig = px.scatter_matrix(sample[cols], dimensions=cols, opacity=0.4)
    fig.update_layout(title="Scatter matrix (numeric columns)", template=PLOTLY_TEMPLATE,
                      height=800, width=900)
    fig.update_traces(diagonal_visible=False)
    return fig


# --------------------------------------------------------------------------- #
# Automatic chart selection engine
# --------------------------------------------------------------------------- #
def auto_generate_charts(
    df: pd.DataFrame,
    column_types: Dict[str, ColumnTypeResult],
    eda_result,
    n_charts: int = 8,
) -> List[Dict[str, object]]:
    """Choose the most informative charts automatically from the data.

    Returns a list of ``{"title", "fig", "reason"}`` dictionaries.
    """
    charts: List[Dict[str, object]] = []
    numeric_cols = eda_result.numeric_cols
    cat_cols = [c for c in eda_result.categorical_cols if c in df.columns]
    datetime_cols = eda_result.datetime_cols
    bool_cols = [c for c, t in column_types.items() if t.kind == "boolean"]

    # 1) Top numeric histogram (most skewed first is interesting)
    if eda_result.numeric:
        ordered = sorted(eda_result.numeric, key=lambda n: abs(n.skew), reverse=True)
        n = ordered[0]
        charts.append({
            "title": f"Histogram • {n.column}",
            "fig": histogram(df, n.column, bins=n.histogram_bins),
            "reason": f"Numeric column; distribution classified as '{n.distribution_label}'",
        })

    # 2) Second numeric histogram / scatter matrix if many numerics
    if len(eda_result.numeric) >= 2:
        charts.append({
            "title": "Scatter matrix • numeric",
            "fig": scatter_matrix(df, numeric_cols),
            "reason": "Multiple numeric columns — multivariate view",
        })

    # 3) Box plot comparison of numerics
    if len(numeric_cols) >= 2:
        charts.append({
            "title": "Box plots • numeric comparison",
            "fig": boxplot(df, numeric_cols[:10]),
            "reason": "Compare scale/outliers across numeric columns",
        })

    # 4) Categorical frequency (largest category presence)
    if cat_cols:
        # pick the categorical with highest relevance (fewer unique but not trivial)
        best_cat = _pick_interesting_cat(df, cat_cols)
        charts.append({
            "title": f"Bar chart • {best_cat}",
            "fig": bar_chart(df, best_cat),
            "reason": "Categorical column — ranks the most frequent values",
        })
        # pie for a second category if present
        remaining = [c for c in cat_cols if c != best_cat]
        if remaining and df[remaining[0]].nunique(dropna=True) <= 20:
            charts.append({
                "title": f"Pie chart • {remaining[0]}",
                "fig": pie_chart(df, remaining[0]),
                "reason": "Categorical column with modest cardinality — share view",
            })

    # 5) Boolean donut
    if bool_cols:
        charts.append({
            "title": f"Boolean donut • {bool_cols[0]}",
            "fig": donut_boolean(df, bool_cols[0]),
            "reason": "Boolean distribution",
        })

    # 6) Time series
    if datetime_cols:
        date_col = datetime_cols[0]
        val = numeric_cols[0] if numeric_cols else None
        charts.append({
            "title": f"Time series • {date_col}",
            "fig": time_series(df, date_col, val),
            "reason": "Datetime column detected — trend over time",
        })

    # 7) Correlation heatmap
    if not eda_result.correlation.empty:
        charts.append({
            "title": "Correlation heatmap",
            "fig": corr_heatmap(eda_result.correlation),
            "reason": "Pairwise linear relationships between numeric columns",
        })

    # 8) Strongest bivariate relationship
    if eda_result.strong_relationships:
        r = eda_result.strong_relationships[0]
        fig = None
        if r.type == "num-num" and r.x in df.columns and r.y in df.columns:
            fig = scatter(df, r.x, r.y)
        elif r.type == "num-cat" and r.y in df.columns and r.x in df.columns:
            fig = grouped_box(df, r.y, r.x)
        elif r.type == "cat-cat":
            # stacked bar
            fig = stacked_bar(df, r.x, r.y)
        if fig is not None:
            charts.append({
                "title": f"Top relationship • {r.x} vs {r.y}",
                "fig": fig,
                "reason": f"{r.metric_name} = {r.metric:.2f} ({r.detail})",
            })

    # Trim to requested count, keep the strongest/most relevant
    return charts[:n_charts]


def stacked_bar(df: pd.DataFrame, x: str, y: str, top_x: int = 8, top_y: int = 6) -> go.Figure:
    top_x_vals = df[x].dropna().value_counts().head(top_x).index
    sub = df[df[x].isin(top_x_vals)]
    ct = pd.crosstab(sub[x], sub[y])
    if ct.shape[1] > top_y:
        totals = ct.sum().sort_values(ascending=False)
        keep = totals.head(top_y).index
        ct = ct[keep]
    fig = go.Figure(data=[go.Bar(name=str(c), y=ct.index, x=ct[c], orientation="h") for c in ct.columns])
    fig.update_layout(barmode="stack", title=f"{x} composition by {y}", template=PLOTLY_TEMPLATE,
                      height=420, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _pick_interesting_cat(df: pd.DataFrame, cat_cols: List[str]) -> str:
    best, score = cat_cols[0], -1
    for c in cat_cols:
        n = df[c].nunique(dropna=True)
        # Interesting = enough categories to show structure but not free-form text
        if 2 <= n <= 12:
            s = 100
        elif n < 2:
            s = 0
        elif n <= 50:
            s = np.clip(50, 0, 60)
        else:
            s = 20
        if s > score:
            score = s
            best = c
    return best