"""
NYC 311 Resolution Time Predictor
Step 5: SQL Analysis & EDA
- Runs SQL queries against PostgreSQL
- Answers key business questions
- Prints findings to guide modeling decisions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", "required dependency")
    raise SystemExit(
        f"Missing Python package: {missing_module}.\n"
        "Use the project environment and rerun:\n"
        "  .\\.venv\\Scripts\\activate\n"
        "  python step5_eda.py"
    ) from exc

# ── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(".env"))

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")

required_db_keys = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
missing_db_keys = [key for key in required_db_keys if not os.getenv(key)]
if missing_db_keys:
    missing = ", ".join(missing_db_keys)
    raise SystemExit(
        f"Missing required .env values: {missing}.\n"
        "Create/update .env in the project root before running step5_eda.py"
    )


def build_db_url(database_name: str) -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT) if DB_PORT else None,
        database=database_name,
    )


engine = create_engine(build_db_url(DB_NAME), pool_pre_ping=True)

print("=" * 60)
print("STEP 5: SQL Analysis & EDA")
print("=" * 60)

# ── HELPER ────────────────────────────────────────────────────────────────────
def run_query(title, sql):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    df = pd.read_sql(sql, engine)
    print(df.to_string(index=False))
    return df

# ── Q1: OVERALL RESOLUTION TIME DISTRIBUTION ─────────────────────────────────
q1 = run_query(
    "Q1: Resolution time distribution (hours)",
    """
    SELECT
        CASE
            WHEN resolution_hours <= 1    THEN '0-1 hrs'
            WHEN resolution_hours <= 4    THEN '1-4 hrs'
            WHEN resolution_hours <= 24   THEN '4-24 hrs'
            WHEN resolution_hours <= 72   THEN '1-3 days'
            WHEN resolution_hours <= 168  THEN '3-7 days'
            WHEN resolution_hours <= 720  THEN '1-4 weeks'
            ELSE '4+ weeks'
        END AS bucket,
        COUNT(*) AS complaints,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM nyc311_clean
    GROUP BY bucket
    ORDER BY MIN(resolution_hours)
    """
)

# ── Q2: WHICH AGENCY IS SLOWEST? ─────────────────────────────────────────────
q2 = run_query(
    "Q2: Average resolution time by agency",
    """
    SELECT
        agency,
        COUNT(*)                                        AS complaints,
        ROUND(AVG(resolution_hours)::numeric, 1)        AS avg_hours,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours,
        ROUND(MAX(resolution_hours)::numeric, 1)        AS max_hours
    FROM nyc311_clean
    GROUP BY agency
    ORDER BY median_hours DESC
    """
)

# ── Q3: WHICH COMPLAINT TYPE TAKES LONGEST? ──────────────────────────────────
q3 = run_query(
    "Q3: Top 10 slowest complaint types (median hours)",
    """
    SELECT
        problem_type,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours,
        ROUND(AVG(resolution_hours)::numeric, 1)        AS avg_hours
    FROM nyc311_clean
    GROUP BY problem_type
    HAVING COUNT(*) > 100
    ORDER BY median_hours DESC
    LIMIT 10
    """
)

# ── Q4: WHICH COMPLAINT TYPE IS FASTEST? ─────────────────────────────────────
q4 = run_query(
    "Q4: Top 10 fastest complaint types (median hours)",
    """
    SELECT
        problem_type,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours
    FROM nyc311_clean
    GROUP BY problem_type
    HAVING COUNT(*) > 100
    ORDER BY median_hours ASC
    LIMIT 10
    """
)

# ── Q5: DOES BOROUGH MATTER? ─────────────────────────────────────────────────
q5 = run_query(
    "Q5: Resolution time by borough",
    """
    SELECT
        borough,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours,
        ROUND(AVG(resolution_hours)::numeric, 1)        AS avg_hours
    FROM nyc311_clean
    GROUP BY borough
    ORDER BY median_hours DESC
    """
)

# ── Q6: DOES TIME OF DAY MATTER? ─────────────────────────────────────────────
q6 = run_query(
    "Q6: Resolution time by hour of day",
    """
    SELECT
        EXTRACT(HOUR FROM created_date)::int            AS hour,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours
    FROM nyc311_clean
    GROUP BY hour
    ORDER BY hour
    """
)

# ── Q7: WEEKDAY VS WEEKEND ────────────────────────────────────────────────────
q7 = run_query(
    "Q7: Weekday vs Weekend resolution time",
    """
    SELECT
        CASE WHEN EXTRACT(DOW FROM created_date) IN (0,6)
             THEN 'Weekend' ELSE 'Weekday' END          AS day_type,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours,
        ROUND(AVG(resolution_hours)::numeric, 1)        AS avg_hours
    FROM nyc311_clean
    GROUP BY day_type
    ORDER BY median_hours DESC
    """
)

# ── Q8: CHANNEL BREAKDOWN ────────────────────────────────────────────────────
q8 = run_query(
    "Q8: Resolution time by submission channel",
    """
    SELECT
        open_data_channel_type                          AS channel,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours
    FROM nyc311_clean
    GROUP BY channel
    ORDER BY median_hours DESC
    """
)

# ── Q9: HPD vs NYPD DEEP DIVE ────────────────────────────────────────────────
q9 = run_query(
    "Q9: Top complaint types for HPD (slowest agency)",
    """
    SELECT
        problem_type,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours
    FROM nyc311_clean
    WHERE agency = 'HPD'
    GROUP BY problem_type
    ORDER BY median_hours DESC
    LIMIT 8
    """
)

# ── Q10: NIGHT VS DAY ────────────────────────────────────────────────────────
q10 = run_query(
    "Q10: Night complaints (10pm-6am) vs Day complaints",
    """
    SELECT
        CASE WHEN EXTRACT(HOUR FROM created_date) >= 22
                  OR EXTRACT(HOUR FROM created_date) < 6
             THEN 'Night (10pm-6am)'
             ELSE 'Day (6am-10pm)'
        END                                             AS time_of_day,
        COUNT(*)                                        AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1)  AS median_hours,
        ROUND(AVG(resolution_hours)::numeric, 1)        AS avg_hours
    FROM nyc311_clean
    GROUP BY time_of_day
    ORDER BY median_hours DESC
    """
)

# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
print("\n\nGenerating charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("NYC 311 — Resolution Time Analysis", fontsize=16, fontweight="bold")

# Chart 1: Distribution buckets
ax = axes[0, 0]
ax.bar(q1["bucket"], q1["complaints"], color="#4C72B0")
ax.set_title("Resolution Time Distribution")
ax.set_xlabel("Time Bucket")
ax.set_ylabel("Number of Complaints")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Chart 2: Median hours by agency
ax = axes[0, 1]
q2_sorted = q2.sort_values("median_hours", ascending=True)
ax.barh(q2_sorted["agency"], q2_sorted["median_hours"], color="#DD8452")
ax.set_title("Median Resolution by Agency")
ax.set_xlabel("Median Hours")

# Chart 3: Median hours by borough
ax = axes[0, 2]
ax.bar(q5["borough"], q5["median_hours"], color="#55A868")
ax.set_title("Median Resolution by Borough")
ax.set_xlabel("Borough")
ax.set_ylabel("Median Hours")
ax.tick_params(axis="x", rotation=30)

# Chart 4: By hour of day
ax = axes[1, 0]
ax.plot(q6["hour"], q6["median_hours"], marker="o", color="#C44E52", linewidth=2)
ax.set_title("Median Resolution by Hour of Day")
ax.set_xlabel("Hour (0=midnight)")
ax.set_ylabel("Median Hours")
ax.axvspan(22, 24, alpha=0.1, color="navy", label="Night")
ax.axvspan(0, 6, alpha=0.1, color="navy")
ax.legend()

# Chart 5: Top 10 slowest complaint types
ax = axes[1, 1]
q3_top = q3.head(10).sort_values("median_hours", ascending=True)
labels = [t[:30] for t in q3_top["problem_type"]]
ax.barh(labels, q3_top["median_hours"], color="#8172B2")
ax.set_title("Top 10 Slowest Complaint Types")
ax.set_xlabel("Median Hours")

# Chart 6: Weekday vs weekend + channel
ax = axes[1, 2]
channels = q8["channel"]
hours    = q8["median_hours"]
ax.bar(channels, hours, color="#64B5CD")
ax.set_title("Median Resolution by Channel")
ax.set_xlabel("Channel")
ax.set_ylabel("Median Hours")
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("step5_eda_charts.png", dpi=150, bbox_inches="tight")
print("      Charts saved to step5_eda_charts.png")

print("\n✓ Step 5 complete.")
engine.dispose()