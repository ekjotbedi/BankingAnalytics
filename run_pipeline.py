"""
run_pipeline.py
---------------
Master orchestration script — runs all pipeline steps in the correct order:

  Step 1: Generate synthetic transaction data → SQLite database
  Step 2: Run ETL pipeline → feature matrix CSV
  Step 3: Run ML clustering → labelled clients CSV + segment labels in SQLite
  Step 4: Generate executive brief → Markdown + PDF

After this script completes, launch the dashboard with:
  streamlit run app/dashboard.py

Expected runtime: ~30–60 seconds on a standard laptop.
"""

import sys
import os
import time

# ── Add project root to Python path so sibling imports work ───────────────────
sys.path.insert(0, os.path.dirname(__file__))


def separator(title: str) -> None:
    """Print a visible step header to the console."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    total_start = time.time()

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    separator("STEP 1 / 4 — Data Generation")
    from data import generate_clients, generate_transactions, seed_database
    clients_df = generate_clients(500)
    txns_df    = generate_transactions(clients_df, 12_000)
    seed_database(clients_df, txns_df)

    # ── Step 2: ETL Pipeline ──────────────────────────────────────────────────
    separator("STEP 2 / 4 — ETL Pipeline (SQL analytics)")
    from utils.etl import run_etl
    run_etl()

    # ── Step 3: ML Clustering ─────────────────────────────────────────────────
    separator("STEP 3 / 4 — ML Clustering (K-Means + DBSCAN)")
    from cluster import run_clustering
    clustered_df = run_clustering()
    print("\nFinal segment distribution:")
    print(clustered_df["segment"].value_counts().to_string())

    # ── Step 4: Executive Brief ───────────────────────────────────────────────
    separator("STEP 4 / 4 — Executive Brief (Markdown + PDF)")
    from utils.brief_generator import generate_brief
    generate_brief()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nOutputs generated:")
    print("  data/transactions.db          ← SQLite database")
    print("  data/features.csv             ← ML feature matrix")
    print("  data/elbow_scores.csv         ← K selection scores")
    print("  data/clustered_clients.csv    ← Labelled clients")
    print("  output/executive_brief.md     ← Markdown brief")
    print("  output/executive_brief.pdf    ← PDF brief")
    print("\nNext step:")
    print("  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()