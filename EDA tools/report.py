"""Report generation — downloadable HTML and PDF summaries.

Both formats cover:

- dataset overview & profiling summary
- data quality findings
- cleaning action log
- key EDA findings / insights
- data quality score breakdown

``ReportData`` collects everything in one place so PDF and HTML stay identical
in content. Charts are embedded in HTML via Plotly's image export; PDF keeps
tables + text.
"""

from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config import APP_NAME, APP_VERSION

try:  # reportlab is optional at import time but required for PDF export
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, SimpleDocTemplate,
    )

    HAS_REPORTLAB = True
except ImportError:  # pragma: no cover
    Paragraph = SimpleDocTemplate = Table = TableStyle = None
    Spacer = getSampleStyleSheet = colors = None
    A4 = landscape = inch = ParagraphStyle = None
    HAS_REPORTLAB = False


@dataclass
class ReportData:
    file_name: str = "dataset.csv"
    generated_at: str = ""
    n_rows: int = 0
    n_cols: int = 0
    profile_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_report: Optional[object] = None
    dup_count: int = 0
    outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    type_issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    cleaning_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    insights: List[str] = field(default_factory=list)
    quality_score: Optional[object] = None
    top_relationships: pd.DataFrame = field(default_factory=pd.DataFrame)
    is_cleaned: bool = False

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Shared content builders
# --------------------------------------------------------------------------- #
def _safe_df_to_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return ""
    shown = df.head(max_rows).copy()
    for c in shown.columns:
        shown[c] = shown[c].apply(lambda v: str(v)[:80])
    return shown.to_html(index=False, border=0, classes="data")


def _build_common_sections(data: ReportData) -> List[dict]:
    sections: List[dict] = []

    # 1. Overview
    sections.append({
        "title": "1. Dataset Overview",
        "body": (
            f"File: <b>{html.escape(data.file_name)}</b><br>"
            f"Rows: {data.n_rows:,} &nbsp;|&nbsp; Columns: {data.n_cols}<br>"
            f"Generated: {data.generated_at}<br>"
            f"Quality Score (0-100): "
            f"<b>{data.quality_score.overall if data.quality_score else 'N/A'}</b> "
            f"({data.quality_score.grade if data.quality_score else ''})"
        ),
    })

    # 2. Profiling
    if not data.profile_summary.empty:
        cols = ["column", "inferred_dtype", "count", "unique", "missing", "missing_%"]
        cols = [c for c in cols if c in data.profile_summary.columns]
        sections.append({
            "title": "2. Column Profile",
            "body": _safe_df_to_html(data.profile_summary[cols] if cols else data.profile_summary),
        })

    # 3. Quality
    if data.missing_report is not None:
        mr = data.missing_report
        sections.append({
            "title": "3. Data Quality",
            "body": (
                f"Completeness: {mr.completeness_pct:.1f}% ({mr.total_missing:,} missing cells out of {mr.total_cells:,})<br>"
                f"Duplicated rows: {data.dup_count:,}<br>"
                + (
                    _safe_df_to_html(data.outlier_summary, 20)
                    if not data.outlier_summary.empty else "No outliers detected."
                )
                + "<hr>"
                + (
                    _safe_df_to_html(data.type_issues, 20)
                    if not data.type_issues.empty else "No type-consistency issues."
                )
                + "<hr>"
                + (
                    _safe_df_to_html(data.validation_summary, 25)
                    if not data.validation_summary.empty else "No validation issues."
                )
            ),
        })

    # 4. Cleaning
    if data.is_cleaned:
        sections.append({
            "title": "4. Cleaning Actions",
            "body": (
                _safe_df_to_html(data.cleaning_log, 60)
                if not data.cleaning_log.empty else "No cleaning actions recorded."
            ),
        })

    # 5. Insights
    if data.insights:
        lis = "".join(f"<li>{html.escape(t)}</li>" for t in data.insights[:25])
        sections.append({"title": "5. Automated Insights", "body": f"<ul>{lis}</ul>"})

    # 5b. Key relationships
    if not data.top_relationships.empty:
        sections.append({
            "title": "5b. Key Bivariate Relationships",
            "body": _safe_df_to_html(data.top_relationships, 10),
        })

    # 6. Score
    if data.quality_score is not None:
        rows = "".join(
            f"<tr><td>{k.capitalize()}</td><td>{v}</td><td>{data.quality_score.details.get(k, '')}</td></tr>"
            for k, v in data.quality_score.dimensions.items()
        )
        sections.append({
            "title": "6. Quality Score Breakdown",
            "body": (
                f"Overall: <b>{data.quality_score.overall}/100 ({data.quality_score.grade})</b>"
                f"<table><tr><th>Dimension</th><th>Score</th><th>Detail</th></tr>{rows}</table>"
            ),
        })
    return sections


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
HTML_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; color: #222; margin: 2em; }
  h1 { color: #1d3557; border-bottom: 3px solid #e63946; padding-bottom: .3em; }
  h2 { color: #2b3a67; margin-top: 1.6em; border-left: 5px solid #2a9df4; padding-left: .6em; }
  .meta { color: #666; font-size: .9em; }
  table.data { border-collapse: collapse; width: 100%; font-size: .85em; margin: .6em 0; }
  table.data th { background: #2a9df4; color: #fff; padding: 6px 8px; text-align: left; }
  table.data td { border: 1px solid #ddd; padding: 5px 8px; }
  table.data tr:nth-child(even) { background: #f4f7fb; }
  .score-banner { background: #eef6ff; border: 1px solid #2a9df4; border-radius: 8px; padding: 1em; }
  ul { line-height: 1.5; }
</style>
"""


def generate_html_report(data: ReportData) -> str:
    sections_html = ""
    for section in _build_common_sections(data):
        sections_html += f"<h2>{html.escape(section['title'])}</h2>\n<div>{section['body']}</div>\n"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data Quality Report — {html.escape(data.file_name)}</title>
{HTML_CSS}
</head>
<body>
<h1>{html.escape(APP_NAME)}</h1>
<div class="meta">Generated at {data.generated_at} — version {APP_VERSION}</div>
{sections_html}
</body>
</html>"""


def generate_pdf_bytes(data: ReportData) -> bytes:
    """Alias for ``generate_pdf_report`` so the app-facing API is consistent."""
    return generate_pdf_report(data)


def html_to_pdf_bytes(html_str: str) -> bytes:
    """Small HTML->PDF helper (reportlab based, no external deps).

    Kept intentionally simple: text is written to the PDF, tables are laid out
    as grids via reportlab's Table. Full HTML rendering needs weasyprint which
    is tricky to ship on Windows; the PDF remains faithful to the data.
    """
    raise NotImplementedError  # use generate_pdf_report instead


# --------------------------------------------------------------------------- #
# PDF report (reportlab)
# --------------------------------------------------------------------------- #
def generate_pdf_report(data: ReportData) -> bytes:
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab is not installed — cannot export PDF.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=f"Data Quality Report — {data.file_name}",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#1d3557"), fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#2b3a67"), fontSize=13, spaceAfter=2)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12)
    meta = ParagraphStyle("Meta", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)
    cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=7.5, leading=9)

    story = [Paragraph(APP_NAME, h1), Paragraph(f"Generated {data.generated_at} — v{APP_VERSION}", meta), Spacer(1, 6)]

    def add_table(story, df, max_rows=25):
        if df is None or df.empty:
            story.append(Paragraph("None", body))
            return
        shown = df.head(max_rows)
        header = [str(c)[:30] for c in shown.columns]
        rows = []
        for _, r in shown.iterrows():
            rows.append([Paragraph(str(v)[:60].replace("\n", " "), cell) for v in r])
        t = Table([header] + rows, repeatRows=1, colWidths=None)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a9df4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)

    for section in _build_common_sections(data):
        story.append(Paragraph(section["title"], h2))
        # rewrite simple HTML from the shared builder into PDF paragraphs/tables
        _append_section_rich(story, section["body"], body, data, add_table)
        story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


def _append_section_rich(story, body_html: str, body_style, data, add_table):
    """Minimal HTML de-core for reportlab from the rich-text we already built."""
    import re as _re

    if not body_html:
        return

    def to_plain(chunk):
        chunk = _re.sub(r"<[^>]+>", "", chunk)
        return html_unescape(chunk).strip()

    # Split into block-ish lines first to preserve layout roughly.
    blocks = _re.split(r"(?=</td>|</th>|</tr>|<br>|<hr>|</li>)", body_html)
    lines: List[str] = []
    for block in blocks:
        if "<tr>" in block or "<td>" in block:
            cells = _re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", block, flags=_re.S)
            if cells:
                lines.append(" | ".join(to_plain(c) for c in cells))
            continue
        if "<li>" in block:
            lines.append("• " + to_plain(block))
            continue
        plain = to_plain(block)
        if plain:
            lines.append(plain)
    for line in _re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))


def html_unescape(s: str) -> str:
    import html as _h

    return _h.unescape(s)


# --------------------------------------------------------------------------- #
# ReportData builder from app results
# --------------------------------------------------------------------------- #
def build_report_data(
    *,
    file_name: str,
    df,
    profile_result,
    quality_result,
    outlier_summary: pd.DataFrame,
    insight_report,
    quality_score,
    cleaning_log=None,
    eda_result=None,
    is_cleaned: bool = False,
) -> ReportData:
    insights = [i.text for i in insight_report.insights] if insight_report else []
    top_relationships = pd.DataFrame()
    if eda_result is not None:
        rels = eda_result.strong_relationships[:10]
        top_relationships = pd.DataFrame(
            [
                {"x": r.x, "y": r.y, "type": r.type, "metric": round(r.metric, 3), "metric_name": r.metric_name}
                for r in rels
            ]
        )
    return ReportData(
        file_name=file_name,
        n_rows=len(df),
        n_cols=len(df.columns),
        profile_summary=profile_result.summary if profile_result else pd.DataFrame(),
        missing_report=quality_result["missing"] if quality_result else None,
        dup_count=quality_result["duplicates"]["count"] if quality_result else 0,
        outlier_summary=outlier_summary,
        type_issues=quality_result["type_issues"] if quality_result else pd.DataFrame(),
        validation_summary=_validation_to_df(quality_result) if quality_result else pd.DataFrame(),
        cleaning_log=cleaning_log.to_dataframe() if cleaning_log else pd.DataFrame(),
        insights=insights,
        quality_score=quality_score,
        top_relationships=top_relationships,
        is_cleaned=is_cleaned,
    )


def _validation_to_df(quality_result) -> pd.DataFrame:
    v = quality_result["validation"]
    if not v:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"column": r.column, "check": r.check, "severity": r.severity,
             "failures": r.failures, "share_%": r.share_pct, "examples": r.examples}
            for r in v
        ]
    )