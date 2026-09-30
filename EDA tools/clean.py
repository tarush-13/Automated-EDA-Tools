"""Data cleaning with a transparent action log.

Every transformation appends a record to ``CleaningLog`` so the user can see
exactly what happened, and the log feeds into the report generator.

Supported operations:

- missing-value handling (drop / mean / median / mode / ffill / bfill / constant)
- duplicate removal (with before/after previews)
- dtype correction (string -> numeric / datetime)
- categorical label standardisation (strip, case-fold, fuzzy merge via rapidfuzz)
- outlier handling (cap / remove / flag)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    HAS_RAPIDFUZZ = False


@dataclass
class CleaningAction:
    """A single, auditable cleaning step."""

    timestamp: str
    operation: str
    columns: str
    detail: str
    rows_affected: int = 0
    status: str = "applied"

    def to_dict(self) -> dict:
        return asdict(self)


class CleaningLog:
    """Accumulates ``CleaningAction`` records for transparency + reporting."""

    def __init__(self) -> None:
        self.actions: List[CleaningAction] = []

    def record(
        self,
        operation: str,
        columns: str,
        detail: str,
        rows_affected: int = 0,
    ) -> None:
        self.actions.append(
            CleaningAction(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation=operation,
                columns=columns,
                detail=detail,
                rows_affected=rows_affected,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.actions])

    def __len__(self) -> int:
        return len(self.actions)

    def __bool__(self) -> bool:
        return bool(self.actions)


# --------------------------------------------------------------------------- #
# 1) Missing values
# --------------------------------------------------------------------------- #
def impute_missing(
    df: pd.DataFrame,
    col: str,
    method: str,
    column_types: Dict,
    *,
    constant: Optional[str] = None,
) -> pd.DataFrame:
    """Impute/fill missing values in ``col`` using ``method``."""
    out = df.copy()
    series = out[col]
    if series.notna().all():
        return out, 0
    n_missing = int(series.isna().sum())

    method = method.lower()
    if method in ("mean", "median", "mode"):
        if method == "mean":
            value = pd.to_numeric(series, errors="coerce").mean()
        elif method == "median":
            value = pd.to_numeric(series, errors="coerce").median()
        else:
            value = series.mode().iloc[0] if not series.mode().empty else np.nan
        out[col] = series.fillna(value)
    elif method == "ffill":
        out[col] = series.ffill()
    elif method == "bfill":
        out[col] = series.bfill()
    elif method == "constant":
        out[col] = series.fillna(constant)
    elif method in ("drop", None) or method == "":
        out = out.dropna(subset=[col])
    else:
        raise ValueError(f"Unknown method: {method}")
    return out, n_missing


# --------------------------------------------------------------------------- #
# 2) Duplicates
# --------------------------------------------------------------------------- #
def drop_duplicates(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=list(subset) if subset is not None else None)
    return out, before - len(out)


# --------------------------------------------------------------------------- #
# 3) Dtype correction
# --------------------------------------------------------------------------- #
def correct_dtypes(df: pd.DataFrame, column_types: Dict) -> Tuple[pd.DataFrame, CleaningLog]:
    """Coerce columns to intended pandas dtypes given the detected kinds."""
    out = df.copy()
    log = CleaningLog()
    for col, kind in column_types.items():
        if col not in out.columns:
            continue
        if kind == "numerical":
            numeric = pd.to_numeric(out[col], errors="coerce")
            changed = int(out[col].notna().sum() - numeric.notna().sum())
            out[col] = numeric
            if changed:
                log.record("dtype-correction", col, f"Coerced to numeric; {changed} unparseable values became NaN", changed)
        elif kind == "datetime":
            parsed = pd.to_datetime(out[col], errors="coerce")
            changed = int(out[col].notna().sum() - parsed.notna().sum())
            out[col] = parsed
            if changed:
                log.record("dtype-correction", col, f"Coerced to datetime; {changed} unparseable values became NaN", changed)
        elif kind == "boolean":
            out[col] = _coerce_bool(out[col])
            log.record("dtype-correction", col, "Coerced to boolean")
        elif kind == "categorical":
            out[col] = out[col].astype("category") if out[col].notna().any() else out[col]
    return out, log


def _coerce_bool(series: pd.Series) -> pd.Series:
    def to_bool(v):
        if pd.isna(v):
            return np.nan
        if pd.api.types.is_bool_dtype(type(v)):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("true", "yes", "y", "1", "t"):
            return True
        if s in ("false", "no", "n", "0", "f"):
            return False
        return np.nan

    return series.map(to_bool)


# --------------------------------------------------------------------------- #
# 4) Categorical standardisation
# --------------------------------------------------------------------------- #
def standardise_categories(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    strip: bool = True,
    lower: bool = True,
    merge_fuzzy: bool = True,
    threshold: int = 88,
) -> Tuple[pd.DataFrame, CleaningLog]:
    """Clean categorical labels: strip/case-fold then fuzzy-merge near-duplicates."""
    out = df.copy()
    log = CleaningLog()
    for col in columns:
        if col not in out.columns:
            continue
        s = out[col].astype("object").copy()
        orig_repr = s.fillna("__NA__").astype(str)
        if strip:
            s = s.str.strip()
        if lower:
            s = s.str.lower()
        if merge_fuzzy and HAS_RAPIDFUZZ:
            mapping = _build_fuzzy_map(s, threshold)
            if mapping:
                s = s.map(mapping).fillna(s)
        new_repr = s.fillna("__NA__").astype(str)
        out[col] = s
        changed = int((orig_repr != new_repr).sum())
        if changed:
            log.record(
                "categorical-standardisation",
                col,
                f"Normalised {changed} labels (strip/case{' + fuzzy merge' if merge_fuzzy and HAS_RAPIDFUZZ else ''})",
                changed,
            )
    return out, log


def _build_fuzzy_map(series: pd.Series, threshold: int) -> Dict[str, str]:
    """Group near-duplicate labels and return {label: canonical_label} map.

    Labels already lowercased/stripped. The shortest label in a cluster becomes
    the representative (usually the cleanest form).
    """
    norm = series.dropna().astype(str)
    if norm.empty:
        return {}
    cleaned = norm.str.strip().str.lower()
    unique = sorted(set(cleaned.tolist()), key=lambda s: (len(s), s))
    if len(unique) <= 1:
        return {}
    representatives: list = []
    mapping: Dict[str, str] = {}
    for label in unique:
        matched = None
        for rep in representatives:
            if rep == label:
                matched = rep
                break
            score = fuzz.ratio(label, rep)
            if score >= threshold:
                matched = rep
                break
        if matched is not None:
            mapping[label] = matched
        else:
            representatives.append(label)
            mapping[label] = label
    return {k: v for k, v in mapping.items() if k != v}


# --------------------------------------------------------------------------- #
# 5) Outlier handling
# --------------------------------------------------------------------------- #
def handle_outliers(
    df: pd.DataFrame,
    outliers,
    action: str,
    column_types: Dict,
    *,
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, CleaningLog, List]:
    """Apply cap / remove / flag per action across all detected outliers."""
    out = df.copy()
    log = CleaningLog()
    stats: List[dict] = []
    if not outliers:
        return out, log, stats

    if action == "flag":
        handled_keys = set()
        for r in outliers:
            if r.unique_key in handled_keys:
                continue
            handled_keys.add(r.unique_key)
            col = r.column
            idx = r.outlier_idx
            if idx is None or len(idx) == 0:
                continue
            flag_col = f"{col}_is_outlier"
            if flag_col in out.columns:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            valid_positions = out.index[numeric.notna().values]
            mask_row = np.zeros(len(out), dtype=bool)
            if len(valid_positions) > 0:
                safe_idx = np.asarray(idx).clip(max=len(valid_positions) - 1)
                mask_row[valid_positions[safe_idx]] = True
            out[flag_col] = False
            out.loc[mask_row, flag_col] = True
            stats.append({"method": r.method, "column": col, "action": "flag", "affected": int(mask_row.sum())})
        log.record(
            "outlier-flag",
            ", ".join(r.column for r in outliers),
            "Added *_is_outlier columns flagging detected outliers",
            int(sum(s["affected"] for s in stats)),
        )

    elif action == "cap":
        for r in outliers:
            col = r.column
            if r.method != "IQR":
                continue
            low, high = r.bounds
            if low is None or high is None:
                continue
            series = pd.to_numeric(out[col], errors="coerce")
            before = int(((series < low) | (series > high)).sum())
            out.loc[series < low, col] = low
            out.loc[series > high, col] = high
            stats.append({"method": "IQR", "column": col, "action": "cap", "affected": before})
        log.record("outlier-cap", ", ".join(r.column for x in stats), "Winsorized values outside IQR bounds to nearest fence")

    elif action == "remove":
        drop_mask = pd.Series(False, index=out.index)
        for r in outliers:
            if r.method != "IQR" or r.outlier_idx is None:
                continue
            col = r.column
            series = pd.to_numeric(out[col], errors="coerce")
            valid = series.notna()
            idx = r.outlier_idx.clip(max=len(valid[valid]) - 1) if valid.sum() else np.array([])
            positions = out.index[valid.values]
            if len(idx):
                drop_mask.iloc[positions[idx]] = True
        before = len(out)
        out = out[~drop_mask]
        stats.append({"action": "remove", "column": "all", "affected": int(before - len(out))})
        log.record("outlier-remove", "all", f"Removed {before - len(out)} rows flagged as IQR outliers", before - len(out))
    return out, log, stats


# --------------------------------------------------------------------------- #
# Master cleaning pipeline
# --------------------------------------------------------------------------- #
def run_cleaning(
    df_original: pd.DataFrame,
    column_types: Dict,
    *,
    imputation: Optional[Dict[str, str]] = None,
    constants: Optional[Dict[str, str]] = None,
    drop_dups: bool = False,
    dedup_subset: Optional[List[str]] = None,
    correct_types: bool = True,
    standardise_cats: bool = False,
    fuzzy_threshold: int = 88,
    outlier_action: Optional[str] = None,
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, CleaningLog, List]:
    """Execute the full, configurable cleaning pipeline and log each step."""
    from quality import detect_outliers

    out = df_original.copy()
    log = CleaningLog()
    log.record("start", "", f"Began cleaning on {len(out)} rows x {len(out.columns)} columns")

    # 1) duplicates
    if drop_dups:
        out, removed = drop_duplicates(out, dedup_subset)
        log.record("duplicate-removal", ", ".join(dedup_subset or ["all"]), f"Removed {removed} duplicate rows", removed)

    # 2) missing-value imputation
    imputation = imputation or {}
    constants = constants or {}
    for col, method in imputation.items():
        if col not in out.columns:
            continue
        if method in ("drop", ""):
            n_missing = int(out[col].isna().sum())
            out = out.dropna(subset=[col])
            log.record("missing-drop", col, f"Dropped {n_missing} rows with missing {col}", n_missing)
        else:
            constant = constants.get(col)
            try:
                out, n_missing = impute_missing(out, col, method, column_types, constant=constant)
                log.record("missing-impute", col, f"Imputed {n_missing} missing values using '{method}'", n_missing)
            except Exception as exc:
                log.record("missing-impute", col, f"Could not impute using '{method}': {exc}", 0)

    # 3) dtype correction
    if correct_types:
        out, dtype_log = correct_dtypes(out, {c: t for c, t in column_types.items() if c in out.columns})
        log.actions.extend(dtype_log.actions)

    # 4) categorical standardisation
    if standardise_cats:
        cat_cols = [c for c, t in column_types.items() if t.kind in ("categorical",) and c in out.columns]
        if cat_cols:
            out, cat_log = standardise_categories(out, cat_cols, threshold=fuzzy_threshold)
            log.actions.extend(cat_log.actions)

    # 5) outliers
    stats: List[dict] = []
    if outlier_action:
        outliers = detect_outliers(out, {c: t for c, t in column_types.items() if c in out.columns}, iqr_factor, z_threshold)
        out, out_log, stats = handle_outliers(out, outliers, outlier_action, column_types, iqr_factor=iqr_factor, z_threshold=z_threshold)
        log.actions.extend(out_log.actions)

    log.record("end", "", f"Cleaning complete: {len(out)} rows x {len(out.columns)} columns")
    return out, log, stats


def pre_clean_type_correction(df: pd.DataFrame, column_types: Dict) -> pd.DataFrame:
    """Read/display-only type coercion used for previews before the user commits."""
    out = df.copy()
    for col, info in column_types.items():
        if info.kind == "numerical":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif info.kind == "datetime":
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out