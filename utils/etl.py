"""
utils/etl.py
------------
ETL (Extract → Transform → Load) pipeline.

Reads from:   output_generated/transactions.db  (created by data.py)
              analytic_queries.sql               (SQL views in project root)
Writes to:    output_generated/features.csv     (ML-ready feature matrix)

Called automatically by run_pipeline.py
"""

import os
import sqlite3

import pandas as pd

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH  = os.path.join(PROJECT_ROOT, "output_generated", "transactions.db")
SQL_PATH = os.path.join(PROJECT_ROOT, "analytic_queries.sql")
OUT_PATH = os.path.join(PROJECT_ROOT, "output_generated", "features.csv")


def run_sql_views(conn: sqlite3.Connection) -> None:
    """
    Read analytic_queries.sql and register the three VIEWs in SQLite.
    Keeps SQL in a separate file edits queries without touching Python.
    """
    with open(SQL_PATH, "r") as f:
        sql_script = f.read()

    # executescript runs all semicolon-separated statements at once
    conn.executescript(sql_script)
    print("[✓] SQL views registered (v_client_summary, v_seasonality, v_at_risk_signal)")


def extract_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pull the main client summary view into a Pandas DataFrame."""
    df = pd.read_sql_query("SELECT * FROM v_client_summary;", conn)
    print(f"[✓] Extracted {len(df):,} client rows from v_client_summary")
    return df


def merge_extra_signals(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Join the seasonality and at-risk views onto the main feature DataFrame.
    """
    seasonality = pd.read_sql_query("SELECT * FROM v_seasonality;", conn)
    df = df.merge(seasonality, on="client_id", how="left")   # keeps all clients

    at_risk = pd.read_sql_query(
        "SELECT client_id, at_risk_flag FROM v_at_risk_signal;", conn
    )
    df = df.merge(at_risk, on="client_id", how="left")

    # Clients with no recent transactions ae assigned worst-case defaults
    df["at_risk_flag"]       = df["at_risk_flag"].fillna(1)
    df["q4_seasonal_index"]  = df["q4_seasonal_index"].fillna(1.0)

    print("[✓] Merged seasonality + at-risk signals")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the feature DataFrame:
      • Drop rows missing critical columns
      • Cap outliers at 1st / 99th percentile so one mega-client doesn't skew the cluster centroids
    """
    critical_cols = ["avg_monthly_volume", "avg_monthly_txn_count", "net_flow"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    if len(df) < before:
        print(f"[!] Dropped {before - len(df)} clients with missing core features")

    # clip extreme values — prevents outliers from dominating K-Means distance calculations
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].clip(
            lower=df[col].quantile(0.01),
            upper=df[col].quantile(0.99),
        )

    print(f"[✓] Cleaned: {len(df):,} clients ready for ML")
    return df


def save_features(df: pd.DataFrame) -> None:
    """Save the final feature matrix to CSV for cluster.py to consume."""
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[✓] Features saved → {OUT_PATH}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def run_etl() -> pd.DataFrame:
    """
    Orchestrator — runs all ETL steps in order and returns the final DataFrame
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        run_sql_views(conn)
        df = extract_features(conn)
        df = merge_extra_signals(df, conn)
        df = clean_features(df)
        save_features(df)
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    print("Running ETL pipeline…")
    df = run_etl()
    print("\nSample output:")
    print(df[["client_id", "sector", "avg_monthly_volume",
              "avg_monthly_txn_count", "net_flow"]].head())
