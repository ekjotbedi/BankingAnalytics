# Customer Transaction Behaviour Clustering
### A Full-Stack Data Analytics Portfolio Project

> Built to demonstrate skills in **Python · SQL · Machine Learning · Streamlit · Business Intelligence**  
> Targeted at: *RBC Commercial Banking — Data Analyst / Client Insights Analyst Co-op 2026*

---

## What This Project Does

This project ingests synthetic Canadian SME (Small & Medium Enterprise) bank transaction data, builds a SQL-powered analytics pipeline, applies unsupervised machine learning to segment clients by behaviour, and surfaces findings through an interactive Streamlit dashboard and auto-generated executive PDF brief.

**The 4 client segments identified:**

| Segment | Behaviour | Business Action |
|---------|-----------|-----------------|
| **High-Growth** | High volume, high frequency, positive net flow | Offer credit facilities & treasury products |
| **At-Risk** | Declining activity, high days since last txn | Proactive relationship manager outreach |
| **Seasonal** | Q4 volume spikes, high volatility | Pre-emptive seasonal credit line offers |
| **Stable** | Consistent, low-volatility activity | Automate servicing, introduce digital products |

---

## Project Structure

```
rbc_transaction_clustering/
│
├── data/
│   ├── generate_data.py        # Step 1: synthetic data → SQLite
│   ├── transactions.db         # SQLite database (generated)
│   ├── features.csv            # ML feature matrix (generated)
│   ├── elbow_scores.csv        # K-selection scores (generated)
│   └── clustered_clients.csv   # Labelled clients (generated)
│
├── sql/
│   └── analytics_queries.sql   # Step 2: SQL CTEs & views for feature engineering
│
├── utils/
│   ├── etl.py                  # Step 2: runs SQL views, builds feature DataFrame
│   └── brief_generator.py      # Step 4: Markdown + PDF executive brief
│
├── ml/
│   └── cluster.py              # Step 3: StandardScaler → PCA → K-Means → DBSCAN
│
├── app/
│   └── dashboard.py            # Streamlit interactive dashboard
│
├── output/
│   ├── executive_brief.md      # Markdown brief (generated)
│   └── executive_brief.pdf     # PDF brief (generated)
│
├── run_pipeline.py             # Master script: runs all 4 steps in order
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```

This runs all 4 steps and generates all output files (~30–60 seconds).

### 3. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data generation | Python · Faker | 12,000 synthetic CAD SME transactions |
| Storage | SQLite | Relational database — clients + transactions tables |
| Analytics | SQL (CTEs, window functions) | Per-client KPI feature engineering |
| ETL | Python · Pandas | Extract → clean → outlier cap → save |
| ML — scaling | Scikit-learn StandardScaler | Normalise features before clustering |
| ML — reduction | Scikit-learn PCA | 2D projection for visualisation |
| ML — clustering | Scikit-learn K-Means | Primary segment assignment |
| ML — outliers | Scikit-learn DBSCAN | Detect atypical accounts |
| Dashboard | Streamlit · Plotly | Interactive web app with filters + charts |
| Reporting | ReportLab | Auto-generated executive PDF brief |
| Documentation | Markdown | README + brief template |

---

## Features

### Interactive Streamlit Dashboard
- Sidebar filters: segment, sector, client tier
- KPI cards: total clients, at-risk count, avg monthly volume, avg net flow
- PCA scatter plot: visual cluster map (hover for client details)
- Elbow / silhouette chart: model validation transparency
- Segment KPI bar charts: volume, frequency, net flow by segment
- Client lookup table: sortable, filterable
- Portfolio composition donut chart
- Download buttons: filtered CSV + executive PDF

### SQL Analytics Layer
- `v_client_summary`: 8-feature per-client aggregation using CTEs
- `v_seasonality`: Q4 vs annual spend index
- `v_at_risk_signal`: recent activity drop detector (< 60% of quarterly baseline)
- Window functions, JULIANDAY recency scoring, NULLIF safe division

### ML Pipeline
- Elbow analysis: K=2 through K=9 with inertia + silhouette scores
- Optimal K auto-selected by maximum silhouette score
- DBSCAN eps=1.5, min_samples=5 for outlier labelling
- Segment labels written back to SQLite for SQL querying

---

## Resume Bullet Points

> Copy-paste these into your resume under this project:

- Engineered a **full-stack data pipeline** ingesting 12,000+ synthetic SME transactions into SQLite, processing them with SQL CTEs and window functions to extract 8 behavioural KPIs per client
- Applied **K-Means + DBSCAN** unsupervised clustering (scikit-learn) with PCA dimensionality reduction and silhouette-score model selection to segment 500 commercial banking clients into 4 actionable profiles
- Built an **interactive Streamlit dashboard** (Plotly charts, sidebar filters, KPI cards) enabling relationship managers to explore segments, identify at-risk clients, and download executive reports
- Auto-generated an **executive PDF brief** (ReportLab) translating cluster statistics into business narratives and recommendations — mirroring analyst deliverables in a commercial banking environment

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Python (Pandas, NumPy) | All pipeline steps |
| SQL (CTEs, window functions, views) | `sql/analytics_queries.sql` |
| Machine Learning (unsupervised) | `ml/cluster.py` |
| Data pipeline / ETL | `utils/etl.py` |
| Business Intelligence / dashboards | `app/dashboard.py` |
| Data storytelling | `utils/brief_generator.py` |
| Version control ready | Clean modular structure |

---

*Data is 100% synthetic — generated using the Faker library. No real client data was used.*