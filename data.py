"""
data.py
-------
Generates synthetic Canadian SME bank transaction data and seeds a SQLite
database with two tables: clients and transactions.

Output:  output_generated/transactions.db

Run this first before any other script, OR just run run_pipeline.py which
calls this automatically.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

# configuration
fake = Faker("en_CA")
random.seed(42)

NUM_CLIENTS = 500
NUM_TRANSACTIONS = 12_000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# all generated files will be stored in output_generated folder
BASE_DIR = os.path.dirname(__file__)
DB_PATH  = os.path.join(BASE_DIR, "output_generated", "transactions.db")

# business sectors relevant to commercial banking
SECTORS = [
    "Retail", "Construction", "Professional Services",
    "Manufacturing", "Hospitality", "Healthcare",
    "Technology", "Real Estate", "Transportation", "Agriculture",
]

# transaction categories
CATEGORIES = [
    "Payroll", "Supplier Payment", "Client Revenue", 
    "Rent / Lease", "Utilities", "Equipment", "Tax Payment", "Loan Repayment", "Insurance", "Transfer"
]

INCOMING = {"Client Revenue", "Transfer"}   # add money; everything else removes it

def random_date(start: datetime, end: datetime) -> datetime:
    """returns a random datetime between start and end"""
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86_400)
    return start + timedelta(days=random_days, seconds=random_seconds)


def generate_clients(n: int) -> pd.DataFrame:
    """
    builds a dataFrame of n fictional clients.
    each client is assigned a tier (small / mid / large) that tells how big their transactions can be.
    """
    clients = []
    for i in range(n):
        sector = random.choice(SECTORS)
        tier = random.choices( # 60% small businesses, 30% mid-size, 10% large
            ["small", "mid", "large"],
            weights=[0.60, 0.30, 0.10])[0]

        revenue_range = {
            "small": (10_000,  100_000),
            "mid":   (100_000, 500_000),
            "large": (500_000, 5_000_000),
        }[tier]

        clients.append({
            "client_id": f"CLT{i+1:04d}",
            "company_name": fake.company(),
            "sector": sector,
            "city": fake.city(),
            "province": fake.province_abbr(),
            "tier": tier,
            "monthly_revenue_min": revenue_range[0],
            "monthly_revenue_max": revenue_range[1],
            "onboarded_date": fake.date_between( start_date="-5y", end_date="-1y").isoformat(),
        })
    return pd.DataFrame(clients)


def generate_transactions(clients_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    generates n transactions distributed across clients
    """
    tier_weight = {"small": 1, "mid": 3, "large": 8}
    weights = clients_df["tier"].map(tier_weight).tolist()

    transactions = []
    for txn_id in range(n):
        client = clients_df.sample(n=1, weights=weights).iloc[0]
        category  = random.choice(CATEGORIES)
        is_credit = category in INCOMING

        # amount scales with client size
        base_min = client["monthly_revenue_min"] * 0.01
        base_max = client["monthly_revenue_max"] * 0.15
        amount   = round(random.uniform(base_min, base_max), 2)

        # holiday/ year-end effect
        txn_date = random_date(START_DATE, END_DATE)
        if txn_date.month in (10, 11, 12):
            amount = round(amount * random.uniform(1.1, 1.3), 2)

        # ~3% of transactions are declined
        if random.random() < 0.03:
            amount = -abs(amount)

        transactions.append({
            "txn_id":      f"TXN{txn_id+1:06d}",
            "client_id":   client["client_id"],
            "date":        txn_date.strftime("%Y-%m-%d"),
            "category":    category,
            "amount":      amount if is_credit else -amount,
            "is_credit":   int(is_credit),
            "description": fake.bs().title(),
        })
    return pd.DataFrame(transactions)


def seed_database(clients_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    """
    Write both DataFrames into the SQLite database.
    SQLite creates the .db file automatically if it doesn't exist.
    Indexes on client_id and date make joins in etl.py much faster.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)   # create output_generated/ if missing
    conn = sqlite3.connect(DB_PATH)

    clients_df.to_sql("clients", conn, if_exists="replace", index=False)
    transactions_df.to_sql("transactions", conn, if_exists="replace", index=False)

    # Indexes speed up the SQL joins and WHERE clauses in analytics_queries.sql
    conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_client ON transactions(client_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_date   ON transactions(date);")
    conn.commit()
    conn.close()

    print(f"[✓] Database saved:   {DB_PATH}")
    print(f"    Clients:          {len(clients_df):,}")
    print(f"    Transactions:     {len(transactions_df):,}")


if __name__ == "__main__":
    print("Generating data…")
    clients_df      = generate_clients(NUM_CLIENTS)
    transactions_df = generate_transactions(clients_df, NUM_TRANSACTIONS)
    seed_database(clients_df, transactions_df)
    print("Done.")