-- ============================================================================
-- analytic_queries.sql
-- ---------------------
-- These SQL views transform raw transaction rows into per-client KPIs.
-- Each KPI becomes a feature column in the machine learning model.
--
-- This file is READ by etl.py using Python — you never run it manually.
-- You can open it in VS Code to read/edit the SQL logic.
-- Install the "SQLite Viewer" VS Code extension to browse transactions.db visually.
-- ============================================================================


-- ── 1. MAIN FEATURE VIEW ────────────────────────────────────────────────────
-- One row per client. Computes volume, frequency, cash flow, and recency features.
-- This is the base table the ML model trains on.

CREATE VIEW IF NOT EXISTS v_client_summary AS
WITH monthly_volumes AS (
    -- Roll up total $ activity per client per calendar month.
    -- ABS(amount) counts both inflows and outflows toward "total activity".
    SELECT
        client_id,
        STRFTIME('%Y-%m', date)       AS yr_month,
        SUM(ABS(amount))              AS monthly_volume,
        COUNT(*)                      AS monthly_txn_count
    FROM transactions
    GROUP BY client_id, yr_month
),
recency AS (
    -- Most recent transaction date per client.
    -- days_since_last_txn is a churn signal — long gaps = at-risk client.
    SELECT
        client_id,
        MAX(date)                                          AS last_txn_date,
        JULIANDAY('2024-12-31') - JULIANDAY(MAX(date))    AS days_since_last_txn
    FROM transactions
    GROUP BY client_id
),
credit_debit AS (
    -- Separate incoming (credit) from outgoing (debit) flows.
    -- net_flow = revenue - expenses; positive means more money coming in than going out.
    SELECT
        client_id,
        SUM(CASE WHEN is_credit = 1 THEN amount       ELSE 0 END) AS total_credit,
        SUM(CASE WHEN is_credit = 0 THEN ABS(amount)  ELSE 0 END) AS total_debit,
        COUNT(DISTINCT category)                                    AS distinct_categories
    FROM transactions
    GROUP BY client_id
),
category_counts AS (
    -- How many different transaction types does this client use?
    -- More categories = deeper product engagement = lower churn risk.
    SELECT client_id, COUNT(DISTINCT category) AS num_categories
    FROM transactions
    GROUP BY client_id
)
-- Join all CTEs into one wide feature row per client
SELECT
    c.client_id,
    c.company_name,
    c.sector,
    c.tier,

    -- Volume features
    AVG(mv.monthly_volume)                                          AS avg_monthly_volume,
    MAX(mv.monthly_volume)                                          AS max_monthly_volume,
    MIN(mv.monthly_volume)                                          AS min_monthly_volume,

    -- Frequency features
    AVG(mv.monthly_txn_count)                                       AS avg_monthly_txn_count,
    SUM(mv.monthly_txn_count)                                       AS total_txn_count,

    -- Cash flow features
    cd.total_credit,
    cd.total_debit,
    (cd.total_credit - cd.total_debit)                              AS net_flow,
    cd.total_credit / NULLIF(cd.total_debit, 0)                    AS credit_debit_ratio,

    -- Engagement
    cc.num_categories,

    -- Recency
    r.last_txn_date,
    r.days_since_last_txn,

    -- Volatility: how much monthly spend swings — high = seasonal or unstable
    (MAX(mv.monthly_volume) - MIN(mv.monthly_volume))
        / NULLIF(AVG(mv.monthly_volume), 0)                        AS volume_volatility

FROM clients c
JOIN monthly_volumes  mv ON c.client_id = mv.client_id
JOIN recency           r ON c.client_id = r.client_id
JOIN credit_debit     cd ON c.client_id = cd.client_id
JOIN category_counts  cc ON c.client_id = cc.client_id
GROUP BY c.client_id;


-- ── 2. SEASONALITY VIEW ──────────────────────────────────────────────────────
-- Q4 spend ratio vs annual average.
-- Ratio > 1.2 = strong seasonal spike (retail, hospitality, agriculture).

CREATE VIEW IF NOT EXISTS v_seasonality AS
SELECT
    client_id,
    AVG(CASE WHEN CAST(STRFTIME('%m', date) AS INT) BETWEEN 10 AND 12
             THEN ABS(amount) END)
    /
    NULLIF(AVG(ABS(amount)), 0)     AS q4_seasonal_index
FROM transactions
GROUP BY client_id;


-- ── 3. AT-RISK SIGNAL VIEW ───────────────────────────────────────────────────
-- Flags clients whose last-quarter activity dropped >40% below their annual average.
-- Used to validate that the "At-Risk" cluster makes business sense.

CREATE VIEW IF NOT EXISTS v_at_risk_signal AS
WITH recent_counts AS (
    SELECT client_id, COUNT(*) AS recent_txn_count
    FROM transactions
    WHERE date >= '2024-10-01'      -- last quarter of our 2-year window
    GROUP BY client_id
),
baseline_counts AS (
    SELECT client_id, COUNT(*) / 4.0 AS quarterly_avg   -- annual ÷ 4 = typical quarter
    FROM transactions
    WHERE date >= '2024-01-01'
    GROUP BY client_id
)
SELECT
    r.client_id,
    r.recent_txn_count,
    b.quarterly_avg,
    -- 1 = at risk (recent activity < 60% of normal quarter)
    CASE WHEN r.recent_txn_count < b.quarterly_avg * 0.6 THEN 1 ELSE 0 END AS at_risk_flag
FROM recent_counts r
JOIN baseline_counts b ON r.client_id = b.client_id;