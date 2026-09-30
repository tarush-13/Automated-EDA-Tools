"""End-to-end smoke tests for every pipeline module.

Run:  python tests/test_pipeline.py

Exercises load/profile/quality/clean/eda/visualize/insights/score/report on
numeric-heavy, categorical-heavy and mixed sample datasets.
"""

import io
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from load import load_dataset, _BytesLikeReadable  # noqa: E402
from profile import detect_column_types, profile_dataset  # noqa: E402
from quality import run_quality_analysis, missingness_heatmap, plot_outliers  # noqa: E402
from clean import run_cleaning  # noqa: E402
from eda import run_eda  # noqa: E402
from visualize import auto_generate_charts, corr_heatmap, scatter  # noqa: E402
from insights import generate_insights  # noqa: E402
from score import compute_quality_score, DIMENSION_WEIGHTS  # noqa: E402
from report import build_report_data, generate_html_report, generate_pdf_report  # noqa: E402


# --------------------------------------------------------------------------- #
# Sample datasets
# --------------------------------------------------------------------------- #
def sample_numeric():
    rng = np.random.default_rng(7)
    n = 2000
    return pd.DataFrame({
        "sales": pd.Series(rng.normal(5000, 1500, n)).clip(lower=0).round(2),
        "units": rng.poisson(200, n),
        "days_to_ship": rng.integers(1, 90, n),
        "return_rate": pd.Series(rng.normal(6, 4, n)).clip(0, 50).round(2),
        "age": rng.integers(18, 90, n),
    })


def sample_categorical():
    rng = np.random.default_rng(7)
    n = 1500
    return pd.DataFrame({
        "region": rng.choice(["North", "South", "north", "NORTH", "south ", "East"], size=n,
                             p=[.25, .2, .12, .1, .1, .23]),
        "segment": rng.choice(["Premium", "Standard", "budget", "standard", "Budget"], size=n,
                              p=[.3, .3, .15, .12, .13]),
        "channel": rng.choice(["Online", "Retail", "Partner"], size=n, p=[.5, .3, .2]),
        "repeat": rng.choice([True, False], size=n, p=[.4, .6]),
    })


def sample_mixed():
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame({
        "txn_id": [f"TX-{i:05d}" for i in rng.permutation(n)],
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "amount": pd.Series(rng.normal(250, 90, n)).clip(lower=5).round(2),
        "category": rng.choice(["Food", "Travel", "Utilities", "Entertainment"], size=n, p=[.4, .25, .2, .15]),
        "city": rng.choice(["NYC", "LA", "CHI", "nyc", "HOU"], size=n, p=[.3, .25, .2, .15, .1]),
        "email": [f"user{i % 100}@example.com" for i in range(n)],
        "age": rng.integers(18, 70, n),
    })
    df["age"] = df["age"].astype(object)
    df.loc[0:20, "age"] = "invalid"
    df.loc[21:30, "email"] = ["bad-email"] * 10
    df.loc[31:45, "amount"] = np.nan
    df.loc[46:60, "date"] = np.nan
    return df


DATASETS = {
    "numeric-heavy": sample_numeric,
    "categorical-heavy": sample_categorical,
    "mixed": sample_mixed,
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as exc:
        FAIL.append((name, exc))
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def _load_via_api(df, suffix=".csv"):
    blob = df.to_csv(index=False).encode("utf-8")
    return load_dataset(_BytesLikeReadable(blob, name=f"sample{suffix}"))


def full_pipeline(name, df_builder, suffix=".csv"):
    print(f"\n=== Pipeline: {name} ===")

    def s1():
        result = _load_via_api(df_builder(), suffix)
        assert result.df.shape[0] > 0 and result.df.shape[1] > 0
    check("load", s1)

    def s2():
        df = df_builder()
        types = detect_column_types(df)
        assert set(types.keys()) == set(df.columns)
        profile = profile_dataset(df, types)
        assert profile.n_rows == len(df) and len(profile.summary) == len(df.columns)
        assert sum([profile.n_numeric, profile.n_categorical, profile.n_datetime,
                    profile.n_text, profile.n_boolean]) == len(df.columns)
    check("profile", s2)

    def s3():
        df = df_builder()
        types = detect_column_types(df)
        q = run_quality_analysis(df, types)
        assert q["missing"] is not None
        assert isinstance(q["duplicates"]["count"], int)
        assert isinstance(q["type_issues"], pd.DataFrame)
        assert isinstance(q["outliers"], list)
        assert isinstance(q["categorical"], dict)
        assert isinstance(q["validation"], list)
        missingness_heatmap(df)
        plot_outliers(df, types, q["outliers"])
    check("quality", s3)

    def s4():
        df = df_builder()
        types = detect_column_types(df)
        cols_with_missing = [c for c in df.columns if df[c].isna().any()]
        imputation = {c: "median" for c in cols_with_missing[:2]}
        cleaned, log, stats = run_cleaning(
            df, types, imputation=imputation, drop_dups=True, correct_types=True,
            standardise_cats=True, outlier_action="flag",
        )
        assert len(log.actions) >= 2, "expected start+end log entries"
        assert cleaned.shape[0] <= df.shape[0]
        assert isinstance(cleaned, pd.DataFrame)
    check("cleaning", s4)

    def s5():
        df = df_builder()
        types = detect_column_types(df)
        eda = run_eda(df, types)
        assert isinstance(eda.numeric, list)
        assert isinstance(eda.categorical, list)
        charts = auto_generate_charts(df, types, eda)
        assert isinstance(charts, list)
        corr_heatmap(eda.correlation)
    check("eda+visualize", s5)

    def s5b():
        # Regression: a column mis-detected as "numerical" (>75% numeric-looking
        # strings) must not crash scatter() when plotting the dirty values.
        df = pd.DataFrame({
            "a": ["10", "20", "30", "40"] * 20 + ["abc"] * 20,
            "b": ["1", "2", "3", "4"] * 25,
            "cat": ["x", "y"] * 50,
        })
        types = detect_column_types(df)
        assert types["a"].kind == "numerical", "dirty column should still be detected as numerical"
        eda = run_eda(df, types)
        fig = scatter(df, "a", "b")
        assert fig is not None
        # Auto-generated charts must also tolerate the dirty numeric column.
        auto_generate_charts(df, types, eda)
    check("eda+visualize_scatter_dirty_numeric", s5b)

    def s6():
        df = df_builder()
        types = detect_column_types(df)
        q = run_quality_analysis(df, types)
        eda = run_eda(df, types)
        profile = profile_dataset(df, types)
        report = generate_insights(df, profile, q, eda, missing_report_=q["missing"])
        assert len(report.insights) >= 1
        assert any(len(i.text) > 0 for i in report.insights)
    check("insights", s6)

    def s7():
        df = df_builder()
        types = detect_column_types(df)
        q = run_quality_analysis(df, types)
        score = compute_quality_score(
            df, q["missing"], q["duplicates"]["count"], types,
            q["type_issues"], q["categorical"], q["validation"], q["outliers"],
        )
        assert 0 <= score.overall <= 100
        assert set(score.dimensions.keys()) == set(DIMENSION_WEIGHTS.keys())
    check("score", s7)

    def s8():
        df = df_builder()
        types = detect_column_types(df)
        q = run_quality_analysis(df, types)
        eda = run_eda(df, types)
        profile = profile_dataset(df, types)
        insights = generate_insights(df, profile, q, eda, missing_report_=q["missing"])
        score = compute_quality_score(df, q["missing"], q["duplicates"]["count"], types,
                                      q["type_issues"], q["categorical"], q["validation"], q["outliers"])
        data = build_report_data(
            file_name=f"{name}.csv", df=df, profile_result=profile, quality_result=q,
            outlier_summary=pd.DataFrame({"column": ["a"], "n_outliers": [0]}),
            insight_report=insights, quality_score=score, eda_result=eda,
        )
        html = generate_html_report(data)
        assert len(html) > 1000
        pdf = generate_pdf_report(data)
        assert len(pdf) > 1000 and pdf[:4] == b"%PDF" and pdf.rstrip().endswith(b"%%EOF")
    check("report", s8)


if __name__ == "__main__":
    for name, builder in DATASETS.items():
        full_pipeline(name, builder)

    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASS)}  FAILED: {len(FAIL)}")
    for name, exc in FAIL:
        print(f"  FAIL {name}: {exc}")
    raise SystemExit(1 if FAIL else 0)