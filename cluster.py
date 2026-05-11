"""
cluster.py
----------
Machine learning clustering pipeline.

Reads from:   output_generated/features.csv        (created by utils/etl.py)
Writes to:    output_generated/clustered_clients.csv
              output_generated/elbow_scores.csv
              output_generated/transactions.db       (adds client_segments table)

Called automatically by run_pipeline.py.
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN         # clustering algorithms
from sklearn.decomposition import PCA               # reduces features to 2D for plotting
from sklearn.preprocessing import StandardScaler    # normalises features before clustering
from sklearn.metrics import silhouette_score        # measures how well-separated clusters are

warnings.filterwarnings("ignore")   # suppress noisy convergence warnings

# ── Paths — cluster.py is at project root ────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)   # project root (same folder as cluster.py)
FEATURES_CSV  = os.path.join(BASE_DIR, "output_generated", "features.csv")
OUTPUT_CSV    = os.path.join(BASE_DIR, "output_generated", "clustered_clients.csv")
DB_PATH       = os.path.join(BASE_DIR, "output_generated", "transactions.db")
ELBOW_CSV     = os.path.join(BASE_DIR, "output_generated", "elbow_scores.csv")

# ── The 8 features fed into the clustering model ─────────────────────────────
# These capture volume, frequency, health, engagement, recency, and seasonality.
# Metadata columns (client_id, company_name, etc.) are excluded — not numeric signals.
FEATURE_COLS = [
    "avg_monthly_volume",      # how much $ moves per month
    "avg_monthly_txn_count",   # how often they transact
    "net_flow",                # revenue minus expenses (positive = healthy)
    "credit_debit_ratio",      # incoming vs outgoing ratio
    "num_categories",          # breadth of banking product usage
    "days_since_last_txn",     # recency — high value = potential churn
    "volume_volatility",       # how much monthly spend swings
    "q4_seasonal_index",       # Q4 spend vs annual average
]

# ── Human-readable segment names ─────────────────────────────────────────────
# These are assigned after inspecting cluster centroids (printed during clustering).
# 0–3 map to the 4 K-Means clusters found with optimal K=4.
SEGMENT_LABELS = {
    0: "High-Growth",    # high volume, high frequency, positive net flow
    1: "At-Risk",        # declining frequency, high days_since_last_txn
    2: "Seasonal",       # high volume_volatility, high q4_seasonal_index
    3: "Stable",         # consistent mid-volume, low volatility
}


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    print(f"[✓] Loaded feature matrix: {df.shape[0]} clients × {df.shape[1]} columns")
    return df


def scale_features(df: pd.DataFrame):
    """
    Standardise all feature columns to mean=0, std=1.
    K-Means is distance-based — without scaling, the avg_monthly_volume column
    (millions of dollars) would completely dominate num_categories (0–10),
    giving meaningless clusters.
    """
    X      = df[FEATURE_COLS].fillna(0).values   # fill any remaining nulls
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)           # fit computes mean/std; transform applies it
    print("[✓] Features scaled with StandardScaler (mean=0, std=1)")
    return X_scaled, scaler


def pca_reduce(X_scaled: np.ndarray):
    """
    Reduce the 8 features down to 2 principal components for the scatter plot.
    PCA finds the two directions in the data that capture the most variance,
    so the 2D plot shows the natural groupings as clearly as possible.
    """
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[✓] PCA: 2 components explain {explained:.1f}% of total variance")
    return X_2d, pca


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 10)) -> int:
    """
    Try K-Means for each K from 2 to 9 and record:
      - Inertia (within-cluster sum of squares) — used for the elbow plot
      - Silhouette score — how distinct the clusters are (-1 to +1, higher = better)
    Saves results to elbow_scores.csv for the dashboard elbow chart.
    Auto-selects K with the highest silhouette score.
    """
    results = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil    = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  K={k}:  inertia={km.inertia_:>12,.0f}   silhouette={sil:.3f}")

    scores_df = pd.DataFrame(results)
    scores_df.to_csv(ELBOW_CSV, index=False)    # saved for dashboard elbow chart

    best_k = int(scores_df.loc[scores_df["silhouette"].idxmax(), "k"])
    print(f"[✓] Best K = {best_k}  (highest silhouette score)")
    return best_k


def fit_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """
    Fit the final K-Means model.
    n_init=10 runs 10 different random starting points and picks the best,
    avoiding bad local minima.
    """
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"[✓] K-Means fitted with K={k}")

    # Print centroid table so you can verify SEGMENT_LABELS make sense
    centroids = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLS)
    print("\nCluster centroids (scaled):")
    print(centroids.round(2).to_string())
    return labels


def run_dbscan(X_scaled: np.ndarray) -> np.ndarray:
    """
    DBSCAN labels sparse / unusual clients as -1 (noise/outlier).
    These become the "Outlier" segment — useful for compliance or RM review.
    eps=1.5 = max distance between two neighbours
    min_samples=5 = minimum points to form a dense cluster core
    """
    db        = DBSCAN(eps=1.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    print(f"[✓] DBSCAN: {(db_labels == -1).sum()} outlier clients flagged")
    return db_labels


def assign_segments(km_labels: np.ndarray, db_labels: np.ndarray) -> list:
    """
    Map K-Means cluster numbers → readable names.
    If DBSCAN also flags the client as an outlier, that takes priority.
    """
    segments = []
    for km, db in zip(km_labels, db_labels):
        if db == -1:
            segments.append("Outlier")
        else:
            segments.append(SEGMENT_LABELS.get(km, f"Segment-{km}"))
    return segments


def save_results(df: pd.DataFrame) -> None:
    """
    Save the enriched DataFrame to:
      1. clustered_clients.csv — used by dashboard.py and brief_generator.py
      2. client_segments table in SQLite — enables SQL joins with raw transactions
    """
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] Clustered clients saved → {OUTPUT_CSV}")

    conn = sqlite3.connect(DB_PATH)
    slim = df[["client_id", "segment", "km_cluster", "dbscan_label", "pca_x", "pca_y"]]
    slim.to_sql("client_segments", conn, if_exists="replace", index=False)
    conn.close()
    print("[✓] Segment labels written to SQLite → table: client_segments")


def run_clustering() -> pd.DataFrame:
    """
    Full ML pipeline orchestrator.
    Returns the enriched DataFrame with segment labels attached.
    """
    df               = load_features()
    X_scaled, scaler = scale_features(df)
    X_2d, pca        = pca_reduce(X_scaled)

    print("\nRunning elbow analysis (K = 2 to 9)…")
    best_k = find_optimal_k(X_scaled)
    K      = min(best_k, 4)          # cap at 4 so SEGMENT_LABELS always matches

    km_labels = fit_kmeans(X_scaled, K)
    db_labels = run_dbscan(X_scaled)
    segments  = assign_segments(km_labels, db_labels)

    # Attach all ML outputs back to the original DataFrame
    df["km_cluster"]   = km_labels         # numeric cluster id (0–3)
    df["dbscan_label"] = db_labels         # -1 = outlier, ≥0 = cluster
    df["segment"]      = segments          # human-readable label
    df["pca_x"]        = X_2d[:, 0]       # x-axis of the scatter plot
    df["pca_y"]        = X_2d[:, 1]       # y-axis of the scatter plot

    save_results(df)
    return df


if __name__ == "__main__":
    # Run just the ML step for testing:
    # python cluster.py     (run from the PROJECT ROOT folder)
    print("Running ML clustering pipeline…\n")
    df = run_clustering()
    print("\nSegment distribution:")
    print(df["segment"].value_counts().to_string())