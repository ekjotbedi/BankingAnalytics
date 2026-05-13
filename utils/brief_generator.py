"""
utils/brief_generator.py
------------------------
Auto-generates the executive brief from cluster statistics.

Reads from: output_generated/clustered_clients.csv
Writes to: output_generated/executive_brief.md
           output_generated/executive_brief.pdf

Called automatically by run_pipeline.py.
"""

import os
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable,
)
# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR      = os.path.join(PROJECT_ROOT, "output_generated")
CLUSTERED    = os.path.join(OUT_DIR, "clustered_clients.csv")
MD_PATH      = os.path.join(OUT_DIR, "executive_brief.md")
PDF_PATH     = os.path.join(OUT_DIR, "executive_brief.pdf")

# Segment narratives — filled with real stats to produce readable prose
NARRATIVES = {
    "High-Growth": (
        "High-Growth clients demonstrate consistently elevated transaction volumes "
        "and a strong positive net cash flow, indicating healthy revenue generation. "
        "Their high transaction frequency suggests deep engagement with commercial "
        "banking products."
    ),
    "At-Risk": (
        "At-Risk clients show a meaningful decline in recent transaction frequency and "
        "a high number of days since last activity. Their net flow has deteriorated, "
        "signalling potential cash flow stress or migration to a competitor. "
    ),
    "Seasonal": (
        "Seasonal clients exhibit significant Q4 volume spikes relative to the rest of "
        "the year, consistent with retail, hospitality, and agricultural cycles. "
    ),
    "Stable": (
        "Stable clients maintain predictable, low-volatility transaction patterns. "
        "While not high-growth, their reliability makes them cost-efficient to serve. "
    ),
    "Outlier": (
        "Outlier clients display atypical transaction patterns that do not conform to "
        "any identified segment. This may indicate unique business models or accounts "
        "requiring compliance review. "
    ),
}

def compute_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """aggregate key metrics per segment"""
    stats = (
        df.groupby("segment")
        .agg(
            count =("client_id", "count"),
            avg_vol =("avg_monthly_volume", "mean"),
            avg_freq =("avg_monthly_txn_count", "mean"),
            avg_net_flow =("net_flow", "mean"),
            pct_at_risk  =("at_risk_flag", "mean"),
        )
        .reset_index()
    )
    stats["avg_vol"] = stats["avg_vol"].round(0).astype(int)
    stats["avg_freq"] = stats["avg_freq"].round(1)
    stats["avg_net_flow"] = stats["avg_net_flow"].round(0).astype(int)
    stats["pct_at_risk"] = (stats["pct_at_risk"] * 100).round(1)
    return stats

def build_markdown(df: pd.DataFrame, stats: pd.DataFrame) -> str:
    """Build the full Markdown brief as a string."""
    today = datetime.today().strftime("%B %d, %Y")
    total = len(df)
    lines = [
        "# Client Segment Intelligence Brief",
        f"**Date:** {today}  ",
        f"**Clients analysed:** {total:,}  ",
        f"**Data window:** January 2023 – December 2024",
        "", "---", "",
        "## Executive Summary", "",
        f"Using K-Means clustering and DBSCAN applied to {total:,} commercial banking "
        f"clients, we identified **{stats.shape[0]} behavioural segments** from 8 "
        f"engineered features (volume, frequency, net flow, recency, seasonality).", "",
    ]
    for _, r in stats.iterrows():
        lines.append(
            f"- **{r['segment']}** ({r['count']:,} clients): "
            f"avg volume ${r['avg_vol']:,}/mo | {r['avg_freq']} txns/mo | "
            f"{r['pct_at_risk']}% at-risk"
        )
    lines += ["", "---", "", "## Segment Profiles & Recommendations", ""]
    for _, r in stats.iterrows():
        seg = r["segment"]
        narrative = NARRATIVES.get(seg, "").replace("<b>", "**").replace("</b>", "**")
        lines += [
            f"### {seg}",
            f"*{r['count']:,} clients — {r['count']/total*100:.1f}% of portfolio*", "",
            f"| Metric | Value |", f"|--------|-------|",
            f"| Avg monthly volume | ${r['avg_vol']:,} |",
            f"| Avg txns / month   | {r['avg_freq']} |",
            f"| Avg net flow       | ${r['avg_net_flow']:,} |",
            f"| At-risk %          | {r['pct_at_risk']}% |", "",
            narrative, "",
        ]
    lines += [
        "---", "",
        "## Methodology", "",
        "1. **Data:** 12,000 synthetic transactions across 500 Canadian SME clients.",
        "2. **ETL:** Python + SQLite. SQL CTEs computed per-client KPIs.",
        "3. **Scaling:** StandardScaler (mean=0, std=1).",
        "4. **Reduction:** PCA to 2 components for visualisation.",
        "5. **Clustering:** K-Means + DBSCAN outliers.",
        "", "---"
    ]
    return "\n".join(lines)


def build_pdf(df: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Generate a professional PDF brief using ReportLab's Platypus engine."""
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(PDF_PATH, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    # custom styles
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                 fontSize=20, textColor=colors.HexColor("#003168"), spaceAfter=6)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                 fontSize=13, textColor=colors.HexColor("#003168"),
                                 spaceBefore=14, spaceAfter=4)
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"],
                                 fontSize=11, textColor=colors.HexColor("#003168"),
                                 spaceBefore=10, spaceAfter=3)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=14, spaceAfter=8)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"],
                                 fontSize=9, textColor=colors.grey, spaceAfter=2)

    today = datetime.today().strftime("%B %d, %Y")
    total = len(df)
    story = []   # list of flowables rendered top-to-bottom

    # title block
    story.append(Paragraph("Client Segment Intelligence Brief", title_style))
    story.append(Paragraph(f"Commercial Banking Analytics  |  {today}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#003168")))
    story.append(Spacer(1, 0.15*inch))

    # summary table
    story.append(Paragraph("Segment Overview", h2_style))
    tbl_data = [["Segment", "Clients", "Avg Monthly Vol", "Avg Txns/Mo", "Net Flow", "At-Risk %"]]
    for _, r in stats.iterrows():
        tbl_data.append([r["segment"], f"{r['count']:,}", f"${r['avg_vol']:,}",
                         str(r["avg_freq"]), f"${r['avg_net_flow']:,}", f"{r['pct_at_risk']}%"])
    tbl = Table(tbl_data, hAlign="LEFT",
                colWidths=[1.2*inch, 0.7*inch, 1.3*inch, 1.1*inch, 1.1*inch, 0.9*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#003168")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2*inch))

    # per-segment narratives
    story.append(Paragraph("Segment Profiles & Recommendations", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.1*inch))
    for _, r in stats.iterrows():
        seg = r["segment"]
        narrative = NARRATIVES.get(seg, "")
        story.append(Paragraph(
            f"{seg}  <font size='9' color='grey'>({r['count']:,} clients — "
            f"{r['count']/total*100:.1f}%)</font>", h3_style
        ))
        story.append(Paragraph(narrative, body_style))

    doc.build(story)
    print(f"[✓] PDF brief saved → {PDF_PATH}")


def generate_brief() -> None:
    """Entry point — generates both Markdown and PDF briefs."""
    if not os.path.exists(CLUSTERED):
        raise FileNotFoundError(
            f"Clustered data not found at:\n  {CLUSTERED}\nRun run_pipeline.py first."
        )
    df = pd.read_csv(CLUSTERED)
    stats = compute_segment_stats(df)

    md = build_markdown(df, stats)
    with open(MD_PATH, "w") as f:
        f.write(md)
    print(f"[✓] Markdown brief saved → {MD_PATH}")

    build_pdf(df, stats)

if __name__ == "__main__":
    print("Generating executive brief…")
    generate_brief()
    print("Done.")
