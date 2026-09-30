"""Dataset profiling: size, dtypes, summary statistics and column-type detection.

``detect_column_types`` decides whether each column is numerical, categorical,
datetime, text or boolean. The result is reused by quality / eda / visualize /
insights modules so every stage agrees on column semantics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import CATEGORY_MAX_UNIQUE_DISPLAY

BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f", "true.", "false."}


@dataclass
class ColumnTypeResult:
    """Detected semantics + evidence for a single column."""

    column: str
    kind: str  # numerical | categorical | datetime | text | boolean
    confidence: float  # 0..1
    reasons: List[str] = field(default_factory=list)
    infer_dtype: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"ColumnTypeResult(column={self.column!r}, kind={self.kind!r}, confidence={self.confidence:.2f})"


def _sample_values(df: pd.DataFrame, col: str, n: int = 20_000) -> pd.Series:
    if len(df) > n:
        return df[col].sample(n=n, random_state=42)
    return df[col]


def _is_boolish(series: pd.Series) -> bool:
    vals = series.dropna()
    if vals.empty:
        return False
    if pd.api.types.is_bool_dtype(vals):
        return True
    sample = vals.astype(str).str.strip().str.lower()
    if sample.size > 8000:
        sample = sample.sample(8000, random_state=42)
    unique = set(sample.unique())
    return len(unique) <= 2 and unique.issubset(BOOL_VALUES)


def _looks_numeric_strings(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip()
    if vals.empty:
        return False
    sample = vals if len(vals) <= 8000 else vals.sample(8000, random_state=42)
    replaced = sample.str.replace(",", "").str.replace("$", "").str.replace("%", "")
    # If most values look numeric after removing currency/percent markers => numeric-ish.
    numeric_mask = replaced.str.replace(r"[-+]?[\d.]+", "", regex=True).str.len() == 0
    return numeric_mask.mean() > 0.75


def detect_column_types(df: pd.DataFrame, sample_limit: int = 20_000) -> Dict[str, ColumnTypeResult]:
    """Return a dict mapping column name -> ColumnTypeResult."""
    results: Dict[str, ColumnTypeResult] = {}
    for col in df.columns:
        series = df[col]
        reasons: List[str] = []
        kind = "text"
        confidence = 0.5
        infer_dtype = str(series.dtype)

        # 1) Boolean by explicit dtype or low-cardinality true/false strings.
        if _is_boolish(series):
            kind, reasons, confidence = "boolean", ["Low cardinality and values map to true/false"], 0.95
        # 2) Real datetime dtype.
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind, reasons, confidence = "datetime", ["Parsed as datetime dtype"], 1.0
        # 3) Numeric pandas dtypes.
        elif pd.api.types.is_numeric_dtype(series):
            kind, reasons, confidence = "numerical", ["Numeric dtype"], 0.95
        else:
            vals = _sample_values(df, col, n=sample_limit).dropna()
            if vals.empty:
                kind, reasons, confidence = "text", ["Column is entirely empty"], 0.4
                infer_dtype = "empty"
            elif _looks_numeric_strings(vals):
                kind, reasons, confidence = "numerical", ["Values look numeric but stored as text"], 0.75
                infer_dtype = "object (numeric-like)"
            else:
                # Try to coerce to datetime.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        parsed = pd.to_datetime(vals, errors="coerce")
                        parsed_share = parsed.notna().mean()
                    except Exception:
                        parsed_share = 0.0
                unique = vals.nunique()
                if parsed_share > 0.8:
                    kind, reasons, confidence = "datetime", [f"{parsed_share:.0%} values parse as dates"], 0.9
                elif unique <= CATEGORY_MAX_UNIQUE_DISPLAY and unique <= max(2, len(vals) * 0.05):
                    kind, reasons, confidence = "categorical", [f"Only {unique} unique values"], 0.85
                elif unique > 100:
                    kind, reasons, confidence = "text", [f"High cardinality ({unique} unique)"], 0.7
                else:
                    kind, reasons, confidence = "categorical", [f"{unique} unique values"], 0.8

        results[col] = ColumnTypeResult(
            column=col,
            kind=kind,
            confidence=confidence,
            reasons=reasons,
            infer_dtype=infer_dtype,
        )
    return results


def infer_dtypes_from_types(types: Dict[str, ColumnTypeResult]) -> Dict[str, str]:
    """Map detected kinds onto pandas dtypes to use when parsing."""
    mapping = {
        "numerical": "float64",
        "categorical": "category",
        "datetime": "object",
        "text": "object",
        "boolean": "bool",
    }
    return {col: mapping[r.kind] for col, r in types.items()}


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
@dataclass
class ProfileResult:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    n_boolean: int
    memory_mb: float
    size_estimate_mb: float
    dtype_counts: Dict[str, int]
    summary: pd.DataFrame
    column_types: Dict[str, ColumnTypeResult]
    duplicated_count: int
    missing_by_col: pd.Series


def profile_dataset(df: pd.DataFrame, column_types: Dict[str, ColumnTypeResult]) -> ProfileResult:
    """Compute dataset-level statistics used across the app."""
    if df.empty:
        raise ValueError("Cannot profile an empty dataframe")

    memory_bytes = df.memory_usage(deep=True).sum()
    size_estimate = None
    try:
        import io

        buffer = io.BytesIO()
        df.head(1000).to_csv(buffer, index=False)
        size_estimate = buffer.tell() / (1_000_000_000 if size_estimate is None else 1_000_000)
    except Exception:
        size_estimate = memory_bytes / 1_000_000

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "inferred_dtype": [str(df[c].dtype) for c in df.columns],
            "count": [int(df[c].notna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_%": [round(100 * df[c].isna().mean(), 2) for c in df.columns],
        }
    )
    summary = _append_stats(summary, df, column_types)

    kind_counts = {k: 0 for k in ["numerical", "categorical", "datetime", "text", "boolean"]}
    for types in column_types.values():
        kind_counts[types.kind] += 1

    return ProfileResult(
        n_rows=len(df),
        n_cols=len(df.columns),
        n_numeric=kind_counts["numerical"],
        n_categorical=kind_counts["categorical"],
        n_datetime=kind_counts["datetime"],
        n_text=kind_counts["text"],
        n_boolean=kind_counts["boolean"],
        memory_mb=memory_bytes / 1_000_000,
        size_estimate_mb=size_estimate,
        dtype_counts=dict(df.dtypes.astype(str).value_counts()),
        summary=summary,
        column_types=column_types,
        duplicated_count=int(df.duplicated().sum()),
        missing_by_col=df.isna().sum(),
    )


def _append_stats(
    summary: pd.DataFrame, df: pd.DataFrame, column_types: Dict[str, ColumnTypeResult]
) -> pd.DataFrame:
    """Attach numeric stats for numeric columns; extra info for others."""
    n = len(summary)
    idx = summary.index
    # object dtype so we can hold floats, datetimes and strings in one frame
    summary["mean"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["std"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["min"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["max"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["skew"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["kurtosis"] = pd.Series([np.nan] * n, index=idx, dtype=object)
    summary["first"] = ""
    summary["last"] = ""

    for idx, col in enumerate(df.columns):
        series = df[col]
        if column_types[col].kind == "numerical":
            nums = pd.to_numeric(series, errors="coerce")
            valid = nums.notna() & series.notna()
            if valid.sum() > 0:
                s = nums[valid]
                summary.at[idx, "mean"] = round(s.mean(), 3) if not s.empty else np.nan
                summary.at[idx, "std"] = round(s.std(), 4) if len(s) > 1 else np.nan
                summary.at[idx, "min"] = s.min()
                summary.at[idx, "max"] = s.max()
                if len(s) > 3:
                    summary.at[idx, "skew"] = round(float(s.skew()), 3)
                    summary.at[idx, "kurtosis"] = round(float(s.kurt()), 3)
        elif column_types[col].kind == "categorical":
            vals = series.dropna()
            if not vals.empty:
                summary.at[idx, "mean"] = vals.value_counts().index[0]  # most frequent
        elif column_types[col].kind == "datetime":
            dates = pd.to_datetime(series, errors="coerce")
            if dates.notna().any():
                summary.at[idx, "min"] = dates.min()
                summary.at[idx, "max"] = dates.max()
        elif column_types[col].kind == "boolean":
            vals = series.dropna()
            if not vals.empty:
                summary.at[idx, "mean"] = round(vals.astype(int).mean(), 3)
        if not series.empty:
            first_valid = series.dropna()
            if not first_valid.empty:
                summary.at[idx, "first"] = str(first_valid.iloc[0])
                summary.at[idx, "last"] = str(first_valid.iloc[-1])
    return summary


def unique_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    """Top-N value counts for a column (used by categorical analysis)."""
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    counts["percentage"] = round(100 * counts["count"] / len(df), 2)
    return counts.head(top_n)