"""Automated Exploratory Data Analysis.

Separates the analysis path for numerical and categorical columns and returns a
structured ``EDAResult`` containing distribution summaries, frequency counts,
bivariate/multivariate relationship discovery and correlation matrices —

i.e. the *facts* that the visualisation and insight modules render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class NumericEDA:
    column: str
    stats: pd.Series
    distribution_label: str  # normal / skewed-left / skewed-right / multimodal / uniform-ish
    histogram_bins: int

    @property
    def skew(self) -> float:
        return float(self.stats.get("skew", np.nan) or 0.0)


@dataclass
class CategoricalEDA:
    column: str
    n_unique: int
    top: pd.DataFrame
    cardinality: float
    dominant_share: float  # % held by most frequent value
    distribution_label: str  # balanced / dominated / long-tail


@dataclass
class BivariateResult:
    type: str  # num-num | num-cat | cat-cat
    x: str
    y: str
    metric: float
    metric_name: str
    direction: str = ""
    detail: str = ""


@dataclass
class EDAResult:
    numeric: List[NumericEDA]
    categorical: List[CategoricalEDA]
    bivariate: List[BivariateResult]
    correlation: pd.DataFrame
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    text_cols: List[str]

    @property
    def strong_relationships(self) -> List[BivariateResult]:
        out = []
        for r in self.bivariate:
            if abs(r.metric) >= 0.6:
                out.append(r)
        return out


# --------------------------------------------------------------------------- #
# Numerical EDA
# --------------------------------------------------------------------------- #
def numeric_analysis(df: pd.DataFrame, col: str) -> NumericEDA:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No numeric data in column {col}")
    stats = series.describe()
    stats["variance"] = series.var()
    stats["skew"] = series.skew() if len(series) > 2 else np.nan
    stats["kurtosis"] = series.kurt() if len(series) > 3 else np.nan
    stats["median"] = series.median()
    stats["mode"] = series.mode().iloc[0] if not series.mode().empty else np.nan
    stats["range"] = series.max() - series.min()
    stats["cv"] = series.std() / series.mean() if series.mean() else np.nan

    label = classify_distribution(series, stats)
    bins = _optimal_bins(series)
    return NumericEDA(column=col, stats=stats, distribution_label=label, histogram_bins=bins)


def classify_distribution(series: pd.Series, stats: pd.Series) -> str:
    skew = stats.get("skew")
    if np.isnan(skew) or skew is None:
        return "unknown"
    skew = float(skew)
    kurt = float(stats.get("kurtosis", 0) or 0)
    if abs(skew) < 0.3 and abs(kurt) < 0.5:
        return "roughly normal / symmetric"
    if abs(skew) < 0.3:
        return "symmetric with heavy tails" if kurt > 1 else "symmetric"
    if skew > 1.0:
        return "strongly skewed right (right tail)"
    if skew < -1.0:
        return "strongly skewed left (left tail)"
    return "mildly skewed right" if skew > 0 else "mildly skewed left"


def _optimal_bins(series: pd.Series) -> int:
    n = len(series)
    if n < 30:
        return max(5, n // 5)
    # Freedman–Diaconis rule
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = max(q3 - q1, 1e-9)
    width = 2.0 * iqr / (n ** (1 / 3))
    bins = int(np.ceil((series.max() - series.min()) / width)) if width > 0 else 10
    return int(np.clip(bins, 5, 50))


# --------------------------------------------------------------------------- #
# Categorical EDA
# --------------------------------------------------------------------------- #
def categorical_analysis(df: pd.DataFrame, col: str, top_n: int = 15) -> CategoricalEDA:
    series = df[col].dropna()
    if series.empty:
        return CategoricalEDA(col, 0, pd.DataFrame(), 0.0, 0.0, "empty")
    vc = series.value_counts()
    n_unique = int(len(vc))
    top = vc.head(top_n).reset_index()
    top.columns = ["value", "count"]
    top["percentage"] = round(100 * top["count"] / len(df), 2)
    cardinality = n_unique / len(df)
    dominant_share = 100 * vc.iloc[0] / len(df)

    if n_unique <= 1:
        label = "constant (single value)"
    elif dominant_share > 70:
        label = "dominated by one category"
    elif cardinality > 0.4 and n_unique > 40:
        label = "high-cardinality / long-tail"
    elif vc.iloc[0] / vc.max() < 1.5 and n_unique > 4:
        label = "roughly balanced"
    else:
        label = "concentrated in few categories"
    return CategoricalEDA(col, n_unique, top, round(cardinality, 3), round(dominant_share, 2), label)


# --------------------------------------------------------------------------- #
# Bivariate / multivariate discovery
# --------------------------------------------------------------------------- #
def bivariate_analysis(
    df: pd.DataFrame,
    column_types: Dict,
    max_combos: int = 120,
) -> List[BivariateResult]:
    numeric_cols = [c for c, t in column_types.items() if t.kind == "numerical"]
    cat_cols = [c for c, t in column_types.items() if t.kind in ("categorical", "boolean")]
    results: List[BivariateResult] = []
    sampled = df.sample(min(len(df), 50_000), random_state=42) if len(df) > 50_000 else df

    # num-num Pearson correlations
    matrix = sampled[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = matrix.corr().abs()
    pairs = _top_corr_pairs(corr, max_pairs=10)
    for a, b, r in pairs:
        results.append(
            BivariateResult(
                "num-num", a, b, r, "Pearson corr (abs)",
                direction="positive" if matrix[a].corr(matrix[b]) > 0 else "negative",
                detail=_describe_corr(r),
            )
        )

    # num-cat: ANOVA-style eta
    if numeric_cols and cat_cols:
        eta = _eta_correlation(sampled, numeric_cols, cat_cols)
        for (num, cat), val in eta.items():
            if val >= 0.5:
                results.append(
                    BivariateResult(
                        "num-cat", cat, num, val, "Eta (association)",
                        detail=f"Category of {cat} explains ~{val:.0%} of variance in {num}",
                    )
                )

    # cat-cat: Cramér's V
    if len(cat_cols) >= 2:
        cramer = _cramers_v_batch(sampled, cat_cols, max_pairs=int(max_combos / 3))
        for (a, b), val in cramer.items():
            if val >= 0.4:
                results.append(
                    BivariateResult("cat-cat", a, b, val, "Cramér's V",
                                    detail=f"Strong association between categories {a} and {b}")
                )

    results.sort(key=lambda r: abs(r.metric), reverse=True)
    return results[:max_combos]


def _top_corr_pairs(corr: pd.DataFrame, max_pairs: int = 10) -> List[Tuple[str, str, float]]:
    cols = list(corr.columns)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if not np.isnan(v):
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max_pairs]


def _eta_correlation(df: pd.DataFrame, numeric_cols, cat_cols, max_cats: int = 30) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    for cat in cat_cols:
        series = df[cat].astype("category")
        if series.cat.categories.size > max_cats:
            continue
        for num in numeric_cols:
            try:
                val = _eta(df[num], series)
                if not np.isnan(val):
                    out[(num, cat)] = val
            except Exception:
                continue
    return out


def _eta(y: pd.Series, x: pd.Series) -> float:
    """Eta squared (association strength) for numeric y grouped by categorical x."""
    y = pd.to_numeric(y, errors="coerce")
    tmp = pd.DataFrame({"y": y, "x": x})
    tmp = tmp.dropna()
    if len(tmp) < 10 or tmp["x"].nunique() < 2:
        return np.nan
    grand = tmp["y"].var()
    if grand == 0:
        return np.nan
    groups = tmp.groupby("x", observed=True)["y"].apply(lambda s: len(s) * ((s.mean() - tmp["y"].mean()) ** 2)).sum()
    explained = groups / (len(tmp) * grand)
    return float(np.sqrt(np.clip(explained, 0, 1)))


def _cramers_v_batch(df: pd.DataFrame, cat_cols, max_pairs: int) -> Dict[tuple, float]:
    import itertools

    out: Dict[tuple, float] = {}
    cols = list(itertools.combinations(cat_cols, 2))
    cols = cols[:max_pairs]
    for a, b in cols:
        try:
            v = cramers_v(df[a], df[b])
            if not np.isnan(v):
                out[(a, b)] = v
        except Exception:
            continue
    return out


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(a.dropna(), b.dropna())
    if min(ct.shape) < 2 or ct.size == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.to_numpy().sum()
    denom = n * (min(ct.shape) - 1)
    if denom == 0:
        return np.nan
    return float(np.sqrt(chi2 / denom))


# --------------------------------------------------------------------------- #
# Correlation + master function
# --------------------------------------------------------------------------- #
def _describe_corr(r: float) -> str:
    """Plain-language description for an absolute correlation strength."""
    a = abs(r)
    direction = "positive" if r > 0 else "negative"
    if a >= 0.85:
        strength = "very strong"
    elif a >= 0.65:
        strength = "strong"
    elif a >= 0.4:
        strength = "moderate"
    elif a >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    return f"{strength} {direction} linear relationship"


def compute_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    matrix = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return matrix.corr()


def run_eda(df: pd.DataFrame, column_types: Dict) -> EDAResult:
    """Main EDA entry point."""
    numeric_cols = [c for c, t in column_types.items() if t.kind == "numerical"]
    cat_cols = [c for c, t in column_types.items() if t.kind in ("categorical", "boolean")]
    datetime_cols = [c for c, t in column_types.items() if t.kind == "datetime"]
    text_cols = [c for c, t in column_types.items() if t.kind == "text"]

    numeric_eda = []
    for c in numeric_cols:
        try:
            numeric_eda.append(numeric_analysis(df, c))
        except Exception:
            continue

    categorical_eda = []
    for c in cat_cols:
        categorical_eda.append(categorical_analysis(df, c))

    bivariate = bivariate_analysis(df, column_types)
    corr = compute_correlation(df, numeric_cols)

    return EDAResult(
        numeric=numeric_eda,
        categorical=categorical_eda,
        bivariate=bivariate,
        correlation=corr,
        numeric_cols=numeric_cols,
        categorical_cols=cat_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
    )