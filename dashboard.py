"""
dashboard.py
------------
Streamlit interactive dashboard for the Customer Transaction Behaviour Clustering.

Sections:
  1. KPI summary cards (total clients, segments, at-risk count, avg monthly volume)
  2. PCA scatter plot — visual cluster map
  3. Elbow chart — model validation
  4. Segment bar charts — KPI comparison across segments
  5. Client lookup table — filterable, sortable client list
  6. Executive brief download button

Run:
    python -m streamlit run dashboard.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # project root folder

# including project roots and utils/ to Python path so imports work
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

OUTPUT_DIR   = os.path.join(BASE_DIR, "output_generated")
CLUSTERED    = os.path.join(OUTPUT_DIR, "clustered_clients.csv")   # ML-labelled clients
ELBOW_CSV    = os.path.join(OUTPUT_DIR, "elbow_scores.csv")        # K selection scores
PDF_PATH     = os.path.join(OUTPUT_DIR, "executive_brief.pdf")     # executive brief PDF

SEGMENT_COLORS = {
    "High-Growth": "#033772",
    "Stable":      "#138F29",
    "Seasonal":    "#F4A300",
    "At-Risk":     "#D21010",
    "Outlier":     "#4C5458"
}


# loading data
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read clustered_clients.csv and elbow_scores.csv into DataFrames.
    """
    df    = pd.read_csv(CLUSTERED)
    elbow = pd.read_csv(ELBOW_CSV)
    return df, elbow


def run_pipeline_if_needed() -> None:
    """
    if the pipeline hasn't been run yet, this function runs it automatically.
    checks if clustered_clients.csv exists in output_generated/ folder
    if user already ran the run_pipeline.py command the function will do nothing.
    """
    if not os.path.exists(CLUSTERED):
        st.info("First run detected — generating data and running ML pipeline…")

        # import from project root and utils
        from data import generate_clients, generate_transactions, seed_database
        from utils.etl import run_etl
        from cluster import run_clustering
        from utils.brief_generator import generate_brief

        # running all the pipeline steps in order
        clients_df = generate_clients(500)
        txns_df    = generate_transactions(clients_df, 12_000)
        seed_database(clients_df, txns_df)
        run_etl()
        run_clustering()
        generate_brief()

        st.success("Pipeline complete — loading dashboard!")
        st.rerun() # refreshes the page


# page configuration
st.set_page_config(
    page_title="Client Segment Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }

    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border-left: 4px solid #003168;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .kpi-label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #003168; }
    .kpi-sub   { font-size: 11px; color: #999; margin-top: 2px; }

    h2 { color: #003168 !important; }
    h3 { color: #003168 !important; }

    section[data-testid="stSidebar"] { background: #003168; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "") -> str:
    """Return an HTML string for a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def main() -> None:
    run_pipeline_if_needed()
    # loading data frames
    df, elbow = load_data()

    # sidebar filters
    with st.sidebar:
        st.markdown("## Banking Analytics")
        st.markdown("### Filters")

        # by default, initially all the options will be selected
        selected_segments = st.multiselect(
            "Segments",
            options=sorted(df["segment"].unique().tolist()),
            default=sorted(df["segment"].unique().tolist()),
        )
        selected_sectors = st.multiselect(
            "Sectors",
            options=sorted(df["sector"].unique().tolist()),
            default=sorted(df["sector"].unique().tolist()),
        )
        selected_tiers = st.multiselect(
            "Client tier",
            options=sorted(df["tier"].unique().tolist()),
            default=sorted(df["tier"].unique().tolist()),
        )

    mask = (
        df["segment"].isin(selected_segments) &
        df["sector"].isin(selected_sectors) &
        df["tier"].isin(selected_tiers)
    )
    filtered = df[mask].copy()

    # page header
    st.markdown("## Client Segment Dashboard")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(kpi_card("Total clients", f"{len(filtered):,}", "in filtered view"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Segments", str(filtered["segment"].nunique()), "distinct behaviours"), unsafe_allow_html=True)
    with c3:
        # at_risk_flag column exists if the ETL at-risk view ran correctly
        at_risk_n   = int(filtered["at_risk_flag"].sum()) if "at_risk_flag" in filtered.columns else 0
        at_risk_pct = f"{at_risk_n/len(filtered)*100:.1f}%" if len(filtered) else "0%"
        st.markdown(kpi_card("At-risk clients", f"{at_risk_n:,}", at_risk_pct + " of view"), unsafe_allow_html=True)
    with c4:
        avg_vol = filtered["avg_monthly_volume"].mean()
        st.markdown(kpi_card("Avg monthly volume", f"${avg_vol:,.0f}", "per client"), unsafe_allow_html=True)
    with c5:
        avg_flow = filtered["net_flow"].mean()
        st.markdown(kpi_card("Avg net flow", f"${avg_flow:,.0f}", "credit − debit"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PCA scatter + elbow chart
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Cluster Map (PCA)")
        st.caption(
            "Each dot = one client. Axes are the two principal components capturing "
            "the most variance. Colour = assigned segment. Hover for client details."
        )
        fig_scatter = px.scatter(
            filtered,
            x="pca_x", y="pca_y",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["client_id", "company_name", "sector", "tier",
                        "avg_monthly_volume", "net_flow"],
            labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2"},
            height=420,
            opacity=0.75,
        )
        fig_scatter.update_traces(marker=dict(size=6))
        fig_scatter.update_layout(
            legend_title_text="Segment",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("### Model Validation (Elbow)")
        st.caption(
            "Inertia drops as K grows. Silhouette score peaks at the optimal K — "
            "where clusters are most distinct from each other."
        )
        # inertia as bars (left axis), silhouette as line (right axis)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Bar(
            x=elbow["k"], y=elbow["inertia"],
            name="Inertia",
            marker_color="#CBD5E0",
            yaxis="y1",
            opacity=0.7,
        ))
        fig_elbow.add_trace(go.Scatter(
            x=elbow["k"], y=elbow["silhouette"],
            name="Silhouette",
            mode="lines+markers",
            line=dict(color="#003168", width=2),
            marker=dict(size=7),
            yaxis="y2",
        ))
        fig_elbow.update_layout(
            height=420,
            xaxis=dict(title="Number of clusters (K)", dtick=1),
            yaxis=dict(title="Inertia", showgrid=False),
            yaxis2=dict(title="Silhouette score", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Segment KPI bar charts
    st.markdown("### Segment KPI Comparison")
    st.caption("Aggregated averages per segment for the filtered view.")

    seg_stats = (
        filtered.groupby("segment")
        .agg(
            clients =("client_id", "count"),
            avg_vol =("avg_monthly_volume", "mean"),
            avg_freq=("avg_monthly_txn_count", "mean"),
            avg_net =("net_flow", "mean"),
        )
        .reset_index()
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        fig_vol = px.bar(
            seg_stats, x="segment", y="avg_vol",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Avg monthly volume ($)",
            labels={"avg_vol": "CAD", "segment": ""},
            height=300,
        )
        fig_vol.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

    with b2:
        fig_freq = px.bar(
            seg_stats, x="segment", y="avg_freq",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Avg txns per month",
            labels={"avg_freq": "Count", "segment": ""},
            height=300,
        )
        fig_freq.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_freq, use_container_width=True)

    with b3:
        fig_net = px.bar(
            seg_stats, x="segment", y="avg_net",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Avg net flow ($)",
            labels={"avg_net": "CAD", "segment": ""},
            height=300,
        )
        fig_net.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_net, use_container_width=True)

    # client lookup table
    st.markdown("### Client Lookup")
    st.caption("Click any column header to sort. Use the sidebar filters to narrow results.")

    display_cols = [
        "client_id", "company_name", "sector", "tier", "segment",
        "avg_monthly_volume", "avg_monthly_txn_count", "net_flow",
        "days_since_last_txn", "at_risk_flag",
    ]
    # keeping columns that exist in the DataFrame
    display_cols = [c for c in display_cols if c in filtered.columns]
    table_df = filtered[display_cols].copy()
    table_df.columns = [c.replace("_", " ").title() for c in table_df.columns]

    st.dataframe(table_df, use_container_width=True, height=350)

    # portfolio pie + download buttons
    p1, p2 = st.columns([2, 1])

    with p1:
        st.markdown("### Portfolio Composition")
        pie_data = filtered["segment"].value_counts().reset_index()
        pie_data.columns = ["segment", "count"]
        fig_pie = px.pie(
            pie_data, names="segment", values="count",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            height=320,
        )
        fig_pie.update_traces(textinfo="label+percent", hole=0.38)   # donut style
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with p2:
        st.markdown("### Downloads")
        st.markdown("<br>", unsafe_allow_html=True)

        # CSV download — always available, reflects current sidebar filters
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download filtered data (CSV)",
            data=csv_bytes,
            file_name="rbc_segments_filtered.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # PDF download — only shown if executive brief was generated
        if os.path.exists(PDF_PATH):
            with open(PDF_PATH, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="⬇ Download executive brief (PDF)",
                data=pdf_bytes,
                file_name="executive_brief.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Run python run_pipeline.py to generate the PDF brief.")

    # footer
    st.markdown("---")
    st.caption(
        "Portfolio project — Customer Transaction Behaviour Clustering  |  "
        "Python · SQL · Scikit-learn · Streamlit · Plotly  |  "
        "Data: synthetic (Faker)"
    )

if __name__ == "__main__":
    main()
