"""
NYC 311 Resolution Time Predictor
Step 4: Feature Engineering
- Reads from nyc311_clean
- Creates model-ready features
- Writes to nyc311_features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

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
        "  python step4_features.py"
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
        "Create/update .env in the project root before running step4_features.py"
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
print("STEP 4: Feature Engineering")
print("=" * 60)

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────────────────────
print("\n[1/6] Loading nyc311_clean...")
df = pd.read_sql("SELECT * FROM nyc311_clean", engine)
print(f"      Loaded {len(df):,} rows")

# ── 2. TIME FEATURES ──────────────────────────────────────────────────────────
print("\n[2/6] Creating time features from created_date...")

df["created_date"] = pd.to_datetime(df["created_date"])

df["hour_of_day"]  = df["created_date"].dt.hour
df["day_of_week"]  = df["created_date"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month"]        = df["created_date"].dt.month
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["is_night"]     = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)

print(f"      hour_of_day  : 0–23")
print(f"      day_of_week  : 0 (Mon) – 6 (Sun)")
print(f"      month        : {df['month'].unique().tolist()}")
print(f"      is_weekend   : {df['is_weekend'].value_counts().to_dict()}")
print(f"      is_night     : {df['is_night'].value_counts().to_dict()}")

# ── 3. TARGET VARIABLE → LOG TRANSFORM ───────────────────────────────────────
print("\n[3/6] Log-transforming resolution_hours...")

df["resolution_hours"] = pd.to_numeric(df["resolution_hours"], errors="coerce")
df["log_resolution_hours"] = np.log1p(df["resolution_hours"])

print(f"      resolution_hours     → skewed (mean={df['resolution_hours'].mean():.1f}, "
      f"median={df['resolution_hours'].median():.1f})")
print(f"      log_resolution_hours → normalised (mean={df['log_resolution_hours'].mean():.2f}, "
      f"median={df['log_resolution_hours'].median():.2f})")

# ── 4. TARGET ENCODING: problem_type ─────────────────────────────────────────
print("\n[4/6] Target encoding problem_type (171 unique values)...")

# Target encoding = replace each category with the mean resolution hours for that category
# Using log_resolution_hours as the target to be consistent
problem_type_means = df.groupby("problem_type")["log_resolution_hours"].mean()
df["problem_type_encoded"] = df["problem_type"].map(problem_type_means)

# Also encode location_type (116 unique)
location_type_means = df.groupby("location_type")["log_resolution_hours"].mean()
df["location_type_encoded"] = df["location_type"].map(location_type_means)

print(f"      problem_type_encoded : mean per problem type (top 5 slowest):")
top5 = problem_type_means.sort_values(ascending=False).head(5)
for name, val in top5.items():
    print(f"        {name:<45} log_hours={val:.2f} ({np.expm1(val):.0f} hrs)")

# ── 5. ONE-HOT ENCODING: agency, borough, channel ────────────────────────────
print("\n[5/6] One-hot encoding agency, borough, open_data_channel_type...")

df_encoded = pd.get_dummies(
    df,
    columns=["agency", "borough", "open_data_channel_type"],
    prefix=["agency", "borough", "channel"],
    dtype=int
)

new_cols = [c for c in df_encoded.columns if c not in df.columns]
print(f"      Created {len(new_cols)} one-hot columns:")
for col in sorted(new_cols):
    print(f"        {col}")

# ── 6. SELECT FINAL FEATURE SET ───────────────────────────────────────────────
print("\n[6/6] Selecting final feature columns and saving to nyc311_features...")

# Core identifier + target
keep_cols = [
    "unique_key",
    "resolution_hours",
    "log_resolution_hours",

    # Time features
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",

    # Encoded categoricals
    "problem_type_encoded",
    "location_type_encoded",
]

# Add all one-hot columns
onehot_cols = [c for c in df_encoded.columns
               if c.startswith("agency_") or
                  c.startswith("borough_") or
                  c.startswith("channel_")]
keep_cols += onehot_cols

df_features = df_encoded[keep_cols].copy()

print(f"      Final feature set: {len(df_features.columns)} columns")
print(f"      Rows: {len(df_features):,}")

# Check for any remaining nulls
null_check = df_features.isnull().sum()
null_check = null_check[null_check > 0]
if len(null_check) > 0:
    print(f"\n      ⚠ Nulls found — filling with column median:")
    for col in null_check.index:
        median_val = df_features[col].median()
        df_features[col] = df_features[col].fillna(median_val)
        print(f"        {col}: {null_check[col]} nulls → filled with {median_val:.4f}")
else:
    print("      ✓ No nulls in feature set")

# Write to PostgreSQL
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS nyc311_features"))
    conn.commit()

df_features.to_sql(
    name      = "nyc311_features",
    con       = engine,
    if_exists = "replace",
    index     = False,
    chunksize = 2000,
)

# ── VERIFY ────────────────────────────────────────────────────────────────────
print("\n--- Verification ---")
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM nyc311_features")).scalar()
    cols  = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'nyc311_features'
        ORDER BY ordinal_position
    """)).fetchall()

print(f"Rows in nyc311_features : {count:,}")
print(f"Columns ({len(cols)}):")
for col in cols:
    print(f"  {col[0]}")

print("\n✓ Step 4 complete.")
engine.dispose()