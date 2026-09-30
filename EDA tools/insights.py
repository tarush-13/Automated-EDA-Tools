"""Plain-language insight generation.

Consumes the profiling / quality / EDA results and writes sentences that a data
analyst would produce, plus a short list of data issues needing attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from eda import EDAResult


@dataclass
class Insight:
    category: str  # structure | distribution | relationship | anomaly | data_issue
    severity: str  # info | warning | error
    text: str


@dataclass
class InsightReport:
    insights: List[Insight] = field(default_factory=list)

    @property
    def warnings(self) -> List[Insight]:
        return [i for i in self.insights if i.severity in ("warning", "error")]

    @property
    def findings(self) -> List[Insight]:
        return self.insights


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt(x, digits: int = 2) -> str:
    try:
        v = float(x)
        if v != v:  # NaN
            return "N/A"
        return f"{v:,.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x) -> str:
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _corr_words(r: float) -> str:
    a = abs(r)
    if a >= 0.85:
        return "very strong"
    if a >= 0.65:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "very weak"


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def structural_insights(profile, missing_report_) -> List[Insight]:
    out: List[Insight] = []
    out.append(Insight(
        "structure", "info",
        f"The dataset has {profile.n_rows:,} rows and {profile.n_cols} columns "
        f"({profile.n_numeric} numeric, {profile.n_categorical} categorical, "
        f"{profile.n_datetime} datetime, {profile.n_text} free-text, {profile.n_boolean} boolean).",
    ))
    if profile.duplicated_count:
        out.append(Insight(
            "data_issue", "warning",
            f"{profile.duplicated_count:,} fully duplicated rows ({_pct(profile.duplicated_count / profile.n_rows)}) "
            f"will skew all downstream metrics if left untouched.",
        ))
    if missing_report_ is not None and missing_report_.overall_pct > 1:
        out.append(Insight(
            "data_issue", "warning" if missing_report_.overall_pct > 5 else "info",
            f"Overall missingness is {missing_report_.overall_pct:.1f}% across {missing_report_.total_missing:,} cells.",
        ))
    return out


def distribution_insights(eda: EDAResult) -> List[Insight]:
    out: List[Insight] = []
    for n in eda.numeric:
        skew = n.skew
        if not np.isnan(skew):
            if abs(skew) >= 1.5:
                out.append(Insight(
                    "distribution", "warning",
                    f"'{n.column}' is highly skewed ({skew:+.2f}) — {n.distribution_label}. "
                    f"Consider log/square-root transforms for models.",
                ))
            elif abs(skew) >= 0.5:
                out.append(Insight(
                    "distribution", "info",
                    f"'{n.column}' is mildly skewed ({skew:+.2f}).",
                ))
        stats = n.stats
        if stats.get("cv") and float(stats["cv"]) > 2:
            out.append(Insight(
                "distribution", "info",
                f"'{n.column}' has very high relative dispersion (CV={_fmt(stats['cv'])}) — "
                f"values are spread over many orders of magnitude.",
            ))
    for c in eda.categorical:
        if "dominated" in c.distribution_label:
            out.append(Insight(
                "distribution", "info",
                f"'{c.column}' is dominated by a single category ({_pct(c.dominant_share / 100)} of rows). "
                f"Check whether ≤{c.n_unique} categories are truly meaningful.",
            ))
        elif "constant" in c.distribution_label:
            out.append(Insight(
                "distribution", "info",
                f"'{c.column}' is constant — it carries no discriminating information.",
            ))
    return out


def relationship_insights(eda: EDAResult) -> List[Insight]:
    out: List[Insight] = []
    for r in eda.strong_relationships:
        if r.type == "num-num":
            out.append(Insight(
                "relationship", "info",
                f"{_corr_words(r.metric).capitalize()} {r.direction} correlation ({r.metric:+.2f}) between "
                f"'{r.x}' and '{r.y}'. "
                + ("This may indicate redundancy." if abs(r.metric) > 0.9 else ""),
            ))
        elif r.type == "num-cat":
            out.append(Insight(
                "relationship", "info",
                f"'{r.y}' varies strongly with '{r.x}' (eta={r.metric:.2f}): {r.detail.lower()}.",
            ))
        elif r.type == "cat-cat":
            out.append(Insight(
                "relationship", "info",
                f"'{r.x}' and '{r.y}' are strongly associated (Cramér's V={r.metric:.2f}) — "
                f"they may encode overlapping information.",
            ))
    return out


def quality_insights(quality_result) -> List[Insight]:
    out: List[Insight] = []
    # Missing per-column
    top_missing = quality_result["missing"].by_column
    if top_missing is not None and not top_missing.empty:
        cols_missing = top_missing[top_missing["missing_%"] > 10]
        for _, row in cols_missing.head(5).iterrows():
            out.append(Insight(
                "data_issue", "warning",
                f"Column '{row['column']}' is missing {_fmt(row['missing_%'])} of its values "
                f"({row['missing']:,} cells).",
            ))

    # Type issues
    for _, row in quality_result["type_issues"].head(5).iterrows():
        out.append(Insight(
            "data_issue", "warning",
            f"'{row['column']}' contains {row['anomalous_count']} values that don't parse as "
            f"{row['detected_as']} (e.g. '{row['examples']}').",
        ))

    # Outliers
    for o in quality_result["outliers"]:
        if o.method == "IQR" and o.n_outliers and o.share_pct >= 2:
            out.append(Insight(
                "anomaly", "warning",
                f"IQR method flags {o.n_outliers} outliers ({_pct(o.share_pct / 100)}) in '{o.column}' "
                f"outside [{_fmt(o.bounds[0])}, {_fmt(o.bounds[1])}].",
            ))

    # Categorical inconsistencies
    for cat_col, cat in quality_result["categorical"].items():
        if cat.inconsistent_labels is not None and not cat.inconsistent_labels.empty:
            first = cat.inconsistent_labels.iloc[0]
            out.append(Insight(
                "data_issue", "warning",
                f"'{cat_col}' has inconsistent labels: '{first['variants']}' look like the same value "
                f"({first['n_variants']} variants). Standardise casing before analysis.",
            ))

    # Validation
    for v in quality_result["validation"]:
        if v.severity == "error":
            out.append(Insight(
                "data_issue", "error",
                f"{v.check} check failed on '{v.column}': {v.failures} violations ({_pct(v.share_pct / 100)}). {v.detail}",
            ))
        elif v.severity == "warning":
            out.append(Insight(
                "data_issue", "info",
                f"{v.check} check on '{v.column}' found {v.failures} suspicious values ({_pct(v.share_pct / 100)}).",
            ))
    return out


def time_insights(eda: EDAResult, df: pd.DataFrame) -> List[Insight]:
    out: List[Insight] = []
    for dcol in eda.datetime_cols:
        dates = pd.to_datetime(df[dcol], errors="coerce")
        if dates.notna().sum() < 2:
            continue
        span = dates.max() - dates.min()
        days = span.days
        if days > 0:
            freq = dates.notna().sum() / max(1, days)
            density = "high" if freq >= 100 else ("medium" if freq >= 1 else "sparse")
            out.append(Insight(
                "structure", "info",
                f"'{dcol}' spans {days} days ({dates.min().date()} → {dates.max().date()}) with "
                f"{density} data density (~{freq:.1f} records/day).",
            ))
    return out


# --------------------------------------------------------------------------- #
# Master generator
# --------------------------------------------------------------------------- #
def generate_insights(
    df: pd.DataFrame,
    profile,
    quality_result,
    eda: EDAResult,
    missing_report_=None,
    cleaning_log=None,
) -> InsightReport:
    insights: List[Insight] = []
    insights += structural_insights(profile, missing_report_ or quality_result["missing"])
    insights += distribution_insights(eda)
    insights += relationship_insights(eda)
    insights += quality_insights(quality_result)
    insights += time_insights(eda, df)

    if cleaning_log and len(cleaning_log) > 2:
        insights.append(Insight(
            "structure", "info",
            f"Cleaning applied {len(cleaning_log) - 2} transformations — see the cleaning log for details.",
        ))

    report = InsightReport(insights)
    report.insights = sorted(report.insights, key=lambda i: {"error": 0, "warning": 1, "info": 2}[i.severity])
    return report


def summarize_top_issues(insight_report: InsightReport, top: int = 6) -> List[str]:
    """Plain-text bullets for quickly communicating what needs attention."""
    issues = [i for i in insight_report.insights if i.severity in ("warning", "error")]
    if not issues:
        return ["No critical issues detected. Dataset looks reasonably clean."]
    return [i.text for i in issues[:top]]