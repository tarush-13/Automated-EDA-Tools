"""Data quality score (0-100).

Scores five dimensions and blends them into one overall score:

- completeness  : how many cells are filled
- consistency   : type consistency, categorical label consistency
- validity      : passes validation rules (ranges, formats, logical checks)
- uniqueness    : freedom from duplicate rows / repeated IDs
- accuracy      : approx. share of values that look sane (parseable, in-range, non-outlier share)

Weights (sum = 1.0): completeness 0.25, consistency 0.2, validity 0.2,
uniqueness 0.15, accuracy 0.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

DIMENSION_WEIGHTS = {
    "completeness": 0.25,
    "consistency": 0.20,
    "validity": 0.20,
    "uniqueness": 0.15,
    "accuracy": 0.20,
}


@dataclass
class QualityScore:
    overall: float
    dimensions: Dict[str, float]
    details: Dict[str, str] = field(default_factory=dict)
    grade: str = ""

    def __post_init__(self):
        self.grade = grade_for(self.overall)


def grade_for(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 65:
        return "Fair"
    if score >= 50:
        return "Poor"
    return "Critical"


# --------------------------------------------------------------------------- #
# Individual dimensions
# --------------------------------------------------------------------------- #
def _completeness(df: pd.DataFrame, missing) -> tuple:
    pct = missing.completeness_pct  # 0-100
    if pct >= 99:
        shortfall = 0
    elif pct >= 95:
        shortfall = 5
    elif pct >= 90:
        shortfall = 12
    elif pct >= 75:
        shortfall = 25
    else:
        shortfall = 50
    score = max(0.0, 100 - shortfall)
    return score, f"{pct:.1f}% of cells are filled"


def _uniqueness(df: pd.DataFrame, duplicated_count: int) -> tuple:
    dup_share = duplicated_count / max(1, len(df))
    score = max(0.0, 100 * (1 - dup_share))
    # further penalise near-duplicate repeat groups
    if duplicated_count and dup_share > 0.05:
        score -= 10 * min(dup_share, 0.5)
    return max(0.0, score), (
        f"{100 * (1 - dup_share):.1f}% of rows are unique"
    )


def _validity(df: pd.DataFrame, validation_results: List) -> tuple:
    if not validation_results:
        return 100.0, "No validation rule violations found"
    failures = sum(v.failures for v in validation_results if v.severity in ("warning", "error"))
    weighted = sum(
        v.failures * (1.5 if v.severity == "error" else 1.0)
        for v in validation_results
    )
    total_checks = max(1, len(df) * len(validation_results))
    violation_share = weighted / max(1, len(df))
    base = max(0.0, 100 - 40 * min(violation_share, 2.5))
    return base, f"{failures} rule violations across {len(validation_results)} checks"


def _consistency(df: pd.DataFrame, column_types, type_issues: pd.DataFrame, categorical_analysis: Dict) -> tuple:
    penalties = 0.0

    # mis-typed cells
    if type_issues is not None and not type_issues.empty:
        for _, row in type_issues.iterrows():
            penalties += min(row["anomalous_share"] * 0.6, 30)

    # inconsistent categorical labels
    n_inconsistent_cols = 0
    for cat in categorical_analysis.values():
        if cat.inconsistent_labels is not None and not cat.inconsistent_labels.empty:
            n_inconsistent_cols += 1
    if n_inconsistent_cols:
        penalties += min(5 + 6 * n_inconsistent_cols, 35)

    # conflicting dtypes across columns
    return max(0.0, 100 - penalties), (
        f"{len(type_issues) if type_issues is not None else 0} columns with type anomalies; "
        f"{n_inconsistent_cols} categorical columns with inconsistent labels"
    )


def _accuracy(df: pd.DataFrame, outliers, column_types) -> tuple:
    numeric_cols = [c for c, t in column_types.items() if t.kind == "numerical"]
    total_cells = 0
    suspicious = 0
    seen_columns = set()
    for o in outliers:
        if o.method != "IQR":
            continue
        if o.column in seen_columns:
            continue
        seen_columns.add(o.column)
        suspicious += o.n_outliers
        total_cells += max(1, len(df))
    if total_cells == 0:
        total_cells = max(1, df.size)
    outlier_share = suspicious / total_cells
    score = max(0.0, 100 - 60 * min(outlier_share, 1.0))
    return score, (
        f"{suspicious} IQR-outlier cells in {len(seen_columns)} numeric columns"
        if seen_columns else "No outlier cells detected"
    )


# --------------------------------------------------------------------------- #
# Master calculator
# --------------------------------------------------------------------------- #
def compute_quality_score(
    df: pd.DataFrame,
    missing,
    duplicated_count: int,
    column_types: Dict,
    type_issues: pd.DataFrame,
    categorical_analysis: Dict,
    validation_results: List,
    outliers,
    weights: Dict[str, float] | None = None,
) -> QualityScore:
    """Compute all dimensions + overall score."""
    if len(df) == 0:
        return QualityScore(0.0, {k: 0.0 for k in DIMENSION_WEIGHTS}, {k: "Empty dataframe" for k in DIMENSION_WEIGHTS})

    completeness, comp_detail = _completeness(df, missing)
    uniqueness, uni_detail = _uniqueness(df, duplicated_count)
    validity, val_detail = _validity(df, validation_results)
    consistency, con_detail = _consistency(df, column_types, type_issues, categorical_analysis)
    accuracy, acc_detail = _accuracy(df, outliers, column_types)

    dimensions = {
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "validity": round(validity, 1),
        "consistency": round(consistency, 1),
        "accuracy": round(accuracy, 1),
    }
    w = weights or DIMENSION_WEIGHTS
    overall = round(
        sum(dimensions[k] * w[k] for k in dimensions), 1
    )
    details = {
        "completeness": comp_detail,
        "uniqueness": uni_detail,
        "validity": val_detail,
        "consistency": con_detail,
        "accuracy": acc_detail,
    }
    return QualityScore(overall, dimensions, details)


def score_delta_after_cleaning(before: QualityScore, after: QualityScore) -> Dict[str, float]:
    """Improvement per dimension (after - before)."""
    return {k: round(after.dimensions[k] - before.dimensions[k], 1) for k in before.dimensions}