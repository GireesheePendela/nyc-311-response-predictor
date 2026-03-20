"""
NYC 311 Resolution Time Predictor
Step 1: Data Collection & Inspection
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

# Auto-detect the first CSV in the Data folder
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Check your path.")
CSV_PATH = csv_files[0]
print(f"✓ Found dataset: {CSV_PATH.name}  ({CSV_PATH.stat().st_size / 1e6:.1f} MB)")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

df = pd.read_csv(
    CSV_PATH,
    low_memory=False,       # avoids mixed-type warnings on large files
    parse_dates=[           # parse these columns as datetime upfront
        "Created Date",
        "Closed Date",
        "Due Date",
        "Resolution Action Updated Date"
    ]
)

print(f"\n✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── 2. SHAPE & DTYPES ─────────────────────────────────────────────────────────
print("\n--- Column dtypes ---")
print(df.dtypes.to_string())

# ── 3. NULL AUDIT ─────────────────────────────────────────────────────────────
print("\n--- Null counts & % (columns with any nulls) ---")
null_summary = pd.DataFrame({
    "null_count": df.isnull().sum(),
    "null_pct":   (df.isnull().mean() * 100).round(1)
})
null_summary = null_summary[null_summary["null_count"] > 0].sort_values("null_pct", ascending=False)
print(null_summary.to_string())

# ── 4. TARGET COLUMN PREVIEW ──────────────────────────────────────────────────
# Our prediction target will be resolution_hours = Closed Date - Created Date
print("\n--- Date range sanity check ---")
print(f"Created Date range : {df['Created Date'].min()} → {df['Created Date'].max()}")
print(f"Closed Date range  : {df['Closed Date'].min()} → {df['Closed Date'].max()}")

# How many rows actually have a Closed Date? (open complaints won't have one)
closed = df["Closed Date"].notna()
print(f"\nRows WITH a Closed Date  : {closed.sum():,}  ({closed.mean()*100:.1f}%)")
print(f"Rows WITHOUT Closed Date : {(~closed).sum():,}  ({(~closed).mean()*100:.1f}%)")

# Quick preview of what resolution time looks like
df_closed = df[closed].copy()
df_closed["resolution_hours"] = (
    df_closed["Closed Date"] - df_closed["Created Date"]
).dt.total_seconds() / 3600

print("\n--- Resolution time preview (hours) ---")
print(df_closed["resolution_hours"].describe().round(2))

# Flag suspicious values (negative = Closed before Created)
neg = (df_closed["resolution_hours"] < 0).sum()
very_long = (df_closed["resolution_hours"] > 8760).sum()  # > 1 year
print(f"\n⚠ Negative resolution times  : {neg:,}")
print(f"⚠ Resolution > 1 year        : {very_long:,}")

# ── 5. KEY CATEGORICAL COLUMNS ────────────────────────────────────────────────
key_cats = [
    "Agency",
    "Problem (formerly Complaint Type)",
    "Borough",
    "Status",
    "Location Type",
    "Open Data Channel Type"
]

print("\n--- Cardinality of key categorical columns ---")
for col in key_cats:
    if col in df.columns:
        n = df[col].nunique()
        top = df[col].value_counts().head(3).to_dict()
        print(f"  {col:<45} {n:>4} unique | top: {top}")

# ── 6. STATUS BREAKDOWN ───────────────────────────────────────────────────────
print("\n--- Status distribution ---")
print(df["Status"].value_counts())

# ── 7. SAMPLE ROWS ────────────────────────────────────────────────────────────
print("\n--- Sample of 3 rows (key columns only) ---")
sample_cols = [
    "Unique Key", "Created Date", "Closed Date",
    "Agency", "Problem (formerly Complaint Type)",
    "Borough", "Status"
]
print(df[sample_cols].sample(3, random_state=42).to_string())

print("Inspection complete.")