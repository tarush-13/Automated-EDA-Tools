# Smart Data Quality Analyzer & Automated EDA Tool

An end-to-end, modular data quality + automated EDA web application built with
**Python** and **Streamlit**. Upload any CSV / XLSX / JSON dataset and the app
runs the full pipeline: loading → profiling → quality analysis → interactive
cleaning → EDA → visualisation → insights → quality score → downloadable report.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed URL (default http://localhost:8501). No data handy? Use
the bundled samples in the Upload tab (*Numeric-heavy*, *Categorical-heavy*,
*Mixed dataset*).

## Pipeline stages

| Tab | What it does | Module |
|-----|--------------|--------|
| **Upload** | Accepts CSV/XLSX/JSON, auto-detects encoding + delimiter, lazy chunked loading for large files | `load.py` |
| **Profile** | Rows/cols/dtypes, memory, summary stats, automatic column-type detection (numerical / categorical / datetime / text / boolean) | `profile.py` |
| **Quality** | Missing values (+ heatmap), duplicates, type consistency, IQR & Z-score outliers, categorical analysis (cardinality, rare categories, inconsistent labels), format/range/ID validation, cross-column logical checks | `quality.py` |
| **Cleaning** | Interactive, per-column missing-value handling, duplicate removal, dtype auto-correction, fuzzy categorical standardisation, outlier cap/remove/flag — every step logged | `clean.py` |
| **EDA** | Separate numeric vs categorical analyses, distributions, frequency counts, bivariate (correlation / Eta / Cramér's V) & multivariate detection | `eda.py` |
| **Insights** | Plain-language findings + data issues needing attention | `insights.py` |
| **Score** | 0–100 quality score across completeness, consistency, validity, uniqueness, accuracy with gauge + radar | `score.py` |
| **Report** | Downloadable PDF + HTML summary and cleaned dataset | `report.py` |

## Architecture

Each stage lives in its own module and can be tested independently before being
wired into the Streamlit dashboard (`app.py`):

```
load.py       dataset loading (encoding/delimiter sniffing, chunking)
profile.py    profiling + column-type detection
quality.py    data quality analysis suite
clean.py      cleaning pipeline with audit log
eda.py        automated EDA computations
visualize.py  auto-selected Plotly charts
insights.py   natural-language insight generation
score.py      data quality scoring
report.py     PDF / HTML report generation
app.py        Streamlit multi-tab dashboard
config.py     shared constants
```

### Drive the pipeline from Python (headless)

```python
import pandas as pd
from load import load_dataset
from profile import detect_column_types, profile_dataset
from quality import run_quality_analysis
from clean import run_cleaning
from eda import run_eda
from insights import generate_insights
from score import compute_quality_score

df = pd.read_csv("data.csv")                      # or load_dataset("data.csv")
types = detect_column_types(df)
profile = profile_dataset(df, types)
quality = run_quality_analysis(df, types)
cleaned, log, stats = run_cleaning(df, types,
                                   imputation={"age": "median"},
                                   drop_dups=True,
                                   outlier_action="cap")
eda = run_eda(cleaned, types)
score = compute_quality_score(cleaned, quality["missing"],
                              quality["duplicates"]["count"], types,
                              quality["type_issues"], quality["categorical"],
                              quality["validation"], quality["outliers"])
```

## Testing

Run the smoke tests that exercise every module on all three sample datasets:

```bash
python tests/test_pipeline.py
```

The test script generates the sample datasets itself and checks that each
pipeline stage runs and returns plausible results.

## Notes

- Files larger than ~200k rows are **sampled** for the interactive analysis to
  keep the UI responsive.
- PDF export uses reportlab (no extra system dependencies); the HTML report
  embeds the full analytics for rich viewing/printing.
- Fuzzy categorical standardisation uses `rapidfuzz` if installed.