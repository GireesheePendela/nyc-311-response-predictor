"""
NYC 311 Resolution Time Predictor
Step 9: Data for the GitHub Pages write-up
- Reads the raw CSV from data/ and applies the same filters as step 3
- Computes the medians and shares quoted on docs/index.html
- Writes docs/findings.json
"""

import json
from pathlib import Path

import pandas as pd

csv_files = sorted(Path("data").glob("*.csv"))
if not csv_files:
    raise SystemExit("No CSV found in data/. See 'Reproduce the pipeline' in README.md")
CSV_PATH = csv_files[0]

COLUMNS = {
    "Created Date": "created",
    "Closed Date": "closed",
    "Agency": "agency",
    "Problem (formerly Complaint Type)": "problem",
    "Borough": "borough",
    "Status": "status",
    "Open Data Channel Type": "channel",
}

# ── 1. LOAD & CLEAN (same rules as step3_clean.py) ────────────────────────────
print(f"[1/3] Loading {CSV_PATH.name}...")
df = pd.read_csv(CSV_PATH, usecols=list(COLUMNS), low_memory=False).rename(columns=COLUMNS)
for col in ["agency", "borough", "status", "channel"]:
    df[col] = df[col].str.strip().str.upper()
df["created"] = pd.to_datetime(df["created"], format="mixed")
df["closed"]  = pd.to_datetime(df["closed"], format="mixed")

df = df[(df["status"] == "CLOSED") & df["closed"].notna()].copy()
df["hours"] = (df["closed"] - df["created"]).dt.total_seconds() / 3600
df = df[
    (df["hours"] > 0)
    & (df["hours"] <= 8760)
    & (df["created"] >= "2026-01-01")
    & (df["created"] <= "2026-03-03")
]
print(f"      {len(df):,} closed complaints")

# ── 2. AGGREGATE ──────────────────────────────────────────────────────────────
print("[2/3] Computing medians...")


def medians(frame, col, keep=None):
    t = frame.groupby(col)["hours"].agg(["median", "count"])
    if keep is not None:
        t = t.loc[keep]
    t = t.sort_values("median")
    return [
        {"label": idx, "median_hours": round(float(r["median"]), 1), "count": int(r["count"])}
        for idx, r in t.iterrows()
    ]


boroughs = df[df["borough"] != "UNSPECIFIED"]
heat     = boroughs[boroughs["problem"] == "HEAT/HOT WATER"]
channels = df[df["channel"].isin(["MOBILE", "ONLINE", "PHONE"])]
top_problems = df["problem"].value_counts().head(15).index

findings = {
    "total": int(len(df)),
    "first_created": df["created"].min().strftime("%Y-%m-%d"),
    "last_created": df["created"].max().strftime("%Y-%m-%d"),
    "median_hours": round(float(df["hours"].median()), 1),
    "share_within_2h": round(float((df["hours"] <= 2).mean()), 3),
    "share_within_24h": round(float((df["hours"] <= 24).mean()), 3),
    "share_over_7d": round(float((df["hours"] > 168).mean()), 3),
    "by_problem": medians(df, "problem", top_problems),
    "by_borough": medians(boroughs, "borough"),
    "heat_by_borough": medians(heat, "borough"),
    "housing_share_by_borough": [
        {"label": b, "share": round(float((g["agency"] == "HPD").mean()), 3), "count": int(len(g))}
        for b, g in boroughs.groupby("borough")
    ],
    "by_channel": medians(channels, "channel"),
    "heat_by_channel": medians(channels[channels["problem"] == "HEAT/HOT WATER"], "channel"),
    "police_share_by_channel": [
        {"label": c, "share": round(float((g["agency"] == "NYPD").mean()), 3), "count": int(len(g))}
        for c, g in channels.groupby("channel")
    ],
}

# ── 3. WRITE ──────────────────────────────────────────────────────────────────
out = Path("docs/findings.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(findings, indent=2))
print(f"[3/3] Wrote {out}")
