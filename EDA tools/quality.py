"""Data quality analysis.

Produces:

- missing-value report (+ Plotly missingness heatmap)
- duplicate-count report with preview of the duplicated rows
- type-consistency checks (mis-typed values in otherwise numeric/date columns)
- outlier detection (IQR and Z-score) with per-column bounds
- categorical analysis (cardinality, rare categories, inconsistent casing)
- validation checks (email, phone, ranges, dates) + logical/cross-column rules
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import PLOTLY_TEMPLATE, VALID_EMAIL_PATTERN, VALID_TEL_PATTERN


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
@dataclass
class MissingReport:
    total_cells: int
    total_missing: int
    overall_pct: float
    by_column: pd.DataFrame  # column, missing, missing_%
    completeness_pct: float

    @property
    def completeness(self) -> float:
        return 100.0 - self.overall_pct


def missing_report(df: pd.DataFrame) -> MissingReport:
    missing = df.isna().sum()
    total = df.size
    by_col = pd.DataFrame(
        {
            "column": df.columns,
            "missing": missing.values,
            "missing_%": np.round(100 * missing.values / len(df), 2),
        }
    ).sort_values("missing", ascending=False).reset_index(drop=True)
    return MissingReport(
        total_cells=int(total),
        total_missing=int(missing.sum()),
        overall_pct=round(100 * missing.sum() / total, 2) if total else 0.0,
        by_column=by_col,
        completeness_pct=round(100 * (1 - missing.sum() / total), 2) if total else 100.0,
    )


def missingness_heatmap(df: pd.DataFrame, sample_limit: int = 10_000) -> go.Figure:
    """Plotly binary heatmap of missingness (row sampling for huge frames)."""
    sample = df
    if len(df) > sample_limit:
        sample = df.sample(sample_limit, random_state=42)
    mask = sample.isna().astype(int)
    if mask.empty or mask.shape[1] == 0:
        return go.Figure()
    # Simpler: build a matrix figure manually.
    fig = go.Figure(
        data=go.Heatmap(
            z=mask.T.values,
            x=[str(i) for i in range(mask.shape[0])],
            y=list(mask.columns),
            colorscale=[[0, "#2c3e50"], [1, "#e74c3c"]],
            showscale=False,
            hovertemplate="Row %{x}<br>Column %{y}<br>Missing: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missingness heatmap (rows sampled)" if len(df) > sample_limit else "Missingness heatmap",
        template=PLOTLY_TEMPLATE,
        height=max(300, 60 * min(mask.shape[1], 30)),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def duplicate_report(df: pd.DataFrame) -> Dict[str, object]:
    duplicated = df.duplicated()
    count = int(duplicated.sum())
    dup_rows = df[duplicated]
    subset = df.drop_duplicates()
    return {
        "count": count,
        "share": round(100 * count / len(df), 2) if len(df) else 0.0,
        "preview": dup_rows.head(20),
        "after_shape": (len(df) - count, len(df.columns)),
        "deduplicated_preview": subset.head(20),
    }


# --------------------------------------------------------------------------- #
# Type consistency
# --------------------------------------------------------------------------- #
def type_consistency_checks(df: pd.DataFrame, column_types: Dict, sample_limit: int = 30_000) -> pd.DataFrame:
    """Detect mis-typed cells: non-numeric text inside numeric columns, etc."""
    rows: List[Dict] = []
    for col, info in column_types.items():
        series = df[col]
        if info.kind == "numerical":
            nums = pd.to_numeric(series, errors="coerce")
            bad = series.notna() & nums.isna()
            if bad.sum():
                rows.append(
                    {
                        "column": col,
                        "detected_as": "numeric",
                        "anomalous_count": int(bad.sum()),
                        "anomalous_share": round(100 * bad.mean(), 2),
                        "examples": _sample_examples(series[bad]),
                        "suggestion": "Convert to numeric after cleaning / inspection",
                    }
                )
        elif info.kind == "datetime":
            parsed = pd.to_datetime(series.dropna(), errors="coerce")
            if parsed.isna().sum():
                rows.append(
                    {
                        "column": col,
                        "detected_as": "datetime",
                        "anomalous_count": int(parsed.isna().sum()),
                        "anomalous_share": round(100 * parsed.isna().mean(), 2),
                        "examples": _sample_examples(series[parsed.isna().index]),
                        "suggestion": "Normalise date formats",
                    }
                )
    return pd.DataFrame(rows, columns=[
        "column", "detected_as", "anomalous_count", "anomalous_share", "examples", "suggestion"
    ])


def _sample_examples(idx_like, n: int = 5) -> str:
    try:
        vals = list(pd.Series(idx_like).astype(str).unique())[:n]
        return ", ".join(vals)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Outliers (IQR + Z-score)
# --------------------------------------------------------------------------- #
OUTLIER_MIN_ROWS = 6


@dataclass
class OutlierResult:
    column: str
    method: str
    bounds: Tuple[Optional[float], Optional[float]]
    n_outliers: int
    share_pct: float
    outlier_idx: Optional[np.ndarray]

    @property
    def unique_key(self):
        return (self.column, self.method)


def detect_outliers(
    df: pd.DataFrame,
    column_types: Dict,
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
    sample_limit: int = 100_000,
) -> List[OutlierResult]:
    """Detect outliers on numeric columns using IQR and Z-score."""
    results: List[OutlierResult] = []
    for col, info in column_types.items():
        if info.kind != "numerical":
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < OUTLIER_MIN_ROWS:
            continue
        # IQR method
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        mask = (series < low) | (series > high)
        results.append(
            OutlierResult(
                column=col,
                method="IQR",
                bounds=(float(low), float(high)),
                n_outliers=int(mask.sum()),
                share_pct=round(100 * mask.mean(), 2),
                outlier_idx=np.where(mask)[0],
            )
        )
        # Z-score method
        mu, sigma = series.mean(), series.std(ddof=0)
        if sigma > 0:
            z = np.abs((series - mu) / sigma)
            zmask = z > z_threshold
            results.append(
                OutlierResult(
                    column=col,
                    method="Z-score",
                    bounds=(float(mu - z_threshold * sigma), float(mu + z_threshold * sigma)),
                    n_outliers=int(zmask.sum()),
                    share_pct=round(100 * zmask.mean(), 2),
                    outlier_idx=np.where(zmask)[0],
                )
            )
    return results


def plot_outliers(df: pd.DataFrame, column_types: Dict, outliers: List[OutlierResult]) -> go.Figure:
    """Box-plot style outliers figure: dots outside the whiskers."""
    numeric_cols = [c for c, i in column_types.items() if i.kind == "numerical"]
    if not numeric_cols:
        return go.Figure()
    box_data = []
    for col in numeric_cols:
        valid = pd.to_numeric(df[col], errors="coerce").dropna()
        box_data.append(go.Box(
            y=valid.values,
            name=col,
            boxpoints="outliers",
            marker=dict(color="#2a9df4", outliercolor="#e74c3c", opacity=0.85),
            line=dict(color="#2a9df4", width=1),
            hovertemplate="%{y:.3f}<extra>" + col + "</extra>",
        ))
    fig = go.Figure(data=box_data)
    fig.update_layout(
        title="Outlier visualisation (dots outside whiskers are candidates)",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    if len(numeric_cols) > 8:
        fig.update_xaxes(tickangle=45)
    return fig


# --------------------------------------------------------------------------- #
# Categorical analysis
# --------------------------------------------------------------------------- #
@dataclass
class CategoryAnalysis:
    column: str
    n_unique: int
    cardinality_ratio: float
    top_values: pd.DataFrame
    rare_categories: pd.DataFrame
    inconsistent_labels: pd.DataFrame  # same normalized label -> different raw labels
    suggestion: str = ""


def analyze_categorical(df: pd.DataFrame, col: str, rare_threshold: float = 0.01) -> CategoryAnalysis:
    series = df[col].dropna()
    if series.empty:
        return CategoryAnalysis(col, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    vc = series.value_counts()
    n_unique = int(len(vc))
    cardinality_ratio = n_unique / max(1, len(df))

    top_vals = vc.head(10).reset_index()
    top_vals.columns = ["value", "count"]
    top_vals["percentage"] = round(100 * top_vals["count"] / len(df), 2)

    rare = vc[vc / len(df) < rare_threshold].head(10).reset_index()
    rare.columns = ["value", "count"]
    rare["percentage"] = round(100 * rare["count"] / len(df), 2)

    inconsistent = _find_inconsistent(df[col])

    suggestion = ""
    if cardinality_ratio > 0.95 and (column_is_texty(df[col])):
        suggestion = "Looks like free text / unique IDs: treat as text, not category."
    elif inconsistent is not None and not inconsistent.empty:
        suggestion = "Similar labels differ in casing/whitespace — consider standardising."
    elif n_unique > 50:
        suggestion = "High cardinality category — consider grouping rare values."

    return CategoryAnalysis(
        column=col,
        n_unique=n_unique,
        cardinality_ratio=round(cardinality_ratio, 3),
        top_values=top_vals,
        rare_categories=rare,
        inconsistent_labels=inconsistent if inconsistent is not None and not inconsistent.empty else pd.DataFrame(),
        suggestion=suggestion,
    )


def column_is_texty(series: pd.Series) -> bool:
    vals = series.dropna().astype(str)
    sample = vals.sample(min(5000, len(vals)), random_state=1)
    return sample.str.len().mean() > 50 or vals.nunique() > 0.9 * max(1, len(vals))


def _find_inconsistent(series: pd.Series) -> Optional[pd.DataFrame]:
    vals = series.dropna().astype(str).str.strip()
    if vals.empty:
        return None
    normalized = vals.str.lower().str.replace(r"\s+", " ", regex=True)
    tmp = pd.DataFrame({"raw": vals.values, "norm": normalized.values})
    groups = tmp.groupby("norm")["raw"].unique()
    groups = groups[groups.apply(len) > 1]
    if groups.empty:
        return None
    rows = [
        {"normalised": k, "variants": ", ".join(sorted(v)[:5]), "n_variants": len(v)}
        for k, v in groups.items()
    ]
    return pd.DataFrame(rows).sort_values("n_variants", ascending=False).head(10)


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    column: str
    check: str
    severity: str  # info | warning | error
    failures: int
    share_pct: float
    examples: str
    detail: str = ""


def run_validation(df: pd.DataFrame, column_types: Dict) -> List[ValidationResult]:
    """Range / format / ID / logical checks."""
    results: List[ValidationResult] = []
    for col, info in column_types.items():
        series = df[col]

        if info.kind == "numerical":
            nums = pd.to_numeric(series, errors="coerce")
            finite = nums[np.isfinite(nums)]
            if finite.empty:
                continue
            # Infer plausible bounds from distribution.
            lo, hi = finite.quantile([0.01, 0.99])
            iqr = finite.quantile(0.75) - finite.quantile(0.25)
            hard_lo, hard_hi = lo - 10 * max(iqr, 1e-9), hi + 10 * max(iqr, 1e-9)
            outside = (finite < hard_lo) | (finite > hard_hi)
            if outside.sum():
                results.append(
                    ValidationResult(
                        col, "value range", "warning", int(outside.sum()),
                        round(100 * outside.mean(), 2),
                        _sample_examples(finite[outside])[:200],
                        f"Values far outside the plausible range [{hard_lo:.2f}, {hard_hi:.2f}]",
                    )
                )
            negatives = finite < 0
            if negatives.sum():
                results.append(
                    ValidationResult(
                        col, "negative value", "info", int(negatives.sum()),
                        round(100 * negatives.mean(), 2),
                        _sample_examples(finite[negatives])[:200],
                        "Non-negative expected",
                    )
                )

        elif info.kind == "datetime":
            parsed = pd.to_datetime(series, errors="coerce")
            if (parsed > pd.Timestamp.now()).any():
                bad = parsed > pd.Timestamp.now()
                results.append(
                    ValidationResult(
                        col, "future date", "info", int(bad.sum()),
                        round(100 * bad.mean(), 2),
                        _sample_examples(series[bad])[:200],
                        "Dates in the future found",
                    )
                )

        elif info.kind == "categorical" or info.kind == "text":
            lower = col.lower()
            if "email" in lower:
                bad = series.dropna().astype(str).str.strip().apply(
                    lambda v: not bool(re.match(VALID_EMAIL_PATTERN, v, re.IGNORECASE))
                )
                if bad.any():
                    results.append(
                        ValidationResult(
                            col, "email format", "warning", int(bad.sum()),
                            round(100 * bad.mean(), 2),
                            _sample_examples(series.dropna()[bad])[:200],
                            "Values failing email regex",
                        )
                    )
            elif any(k in lower for k in ("phone", "tel", "mobile", "contact")):
                bad = series.dropna().astype(str).str.strip().apply(
                    lambda v: not bool(re.match(VALID_TEL_PATTERN, v))
                )
                if bad.any():
                    results.append(
                        ValidationResult(
                            col, "phone format", "warning", int(bad.sum()),
                            round(100 * bad.mean(), 2),
                            _sample_examples(series.dropna()[bad])[:200],
                            "Values failing phone pattern",
                        )
                    )
            elif any(k in lower for k in ("id", "code", "sku", "number")):
                dups = series.dropna().duplicated()
                if dups.any():
                    results.append(
                        ValidationResult(
                            col, "ID uniqueness", "error", int(dups.sum()),
                            round(100 * dups.mean(), 2),
                            _sample_examples(series.dropna()[dups])[:200],
                            "Duplicate identifiers found — IDs should be unique",
                        )
                    )

    return results


def validate_logical(df: pd.DataFrame) -> List[ValidationResult]:
    """Cross-column logical consistency checks (duration > 0, start <= end, etc)."""
    results: List[ValidationResult] = []
    numeric_cols = list(df.columns)
    # start/end pairs
    for start, end in _date_range_pairs(df):
        try:
            s = pd.to_datetime(df[start], errors="coerce")
            e = pd.to_datetime(df[end], errors="coerce")
            invalid = (s.notna() & e.notna() & (s > e))
            if invalid.sum():
                results.append(
                    ValidationResult(
                        f"{start} vs {end}", "start<=end", "warning",
                        int(invalid.sum()), round(100 * invalid.mean(), 2),
                        "",
                        f"Rows where {start} is after {end}",
                    )
                )
        except Exception:
            pass
    return results


def _date_range_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    cols = [c.lower() for c in df.columns]
    pairs: List[Tuple[str, str]] = []
    for i, start in enumerate(cols):
        for j, end in enumerate(cols):
            if i == j:
                continue
            if ("start" in start and "end" in end) and start.replace("start", "") == end.replace("end", ""):
                pairs.append((df.columns[i], df.columns[j]))
            elif "from" in start and "to" in end and start.replace("from", "") == end.replace("to", ""):
                pairs.append((df.columns[i], df.columns[j]))
    return pairs


# --------------------------------------------------------------------------- #
# Master quality inspection entry point
# --------------------------------------------------------------------------- #
def run_quality_analysis(
    df: pd.DataFrame,
    column_types: Dict,
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
) -> Dict[str, object]:
    """Run the complete quality inspection suite and bundle the results."""
    return {
        "missing": missing_report(df),
        "duplicates": duplicate_report(df),
        "type_issues": type_consistency_checks(df, column_types),
        "outliers": detect_outliers(df, column_types, iqr_factor, z_threshold),
        "categorical": {
            col: analyze_categorical(df, col)
            for col, info in column_types.items()
            if info.kind in ("categorical", "text")
        },
        "validation": run_validation(df, column_types) + validate_logical(df),
    }