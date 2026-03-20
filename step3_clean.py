"""
NYC 311 Resolution Time Predictor
Step 3: Data Cleaning
- Reads from nyc311_raw
- Writes cleaned closed complaints to nyc311_clean
- Writes non-closed complaints to nyc311_open
"""

import pandas as pd
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
        "  python step3_clean.py"
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
        "Create/update .env in the project root before running step3_clean.py"
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

# ── 1. LOAD FROM POSTGRES ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Data Cleaning")
print("=" * 60)

print("\n[1/7] Loading nyc311_raw from PostgreSQL...")
df = pd.read_sql("SELECT * FROM nyc311_raw", engine)
print(f"      Loaded {len(df):,} rows")

# ── 2. SPLIT CLOSED vs NON-CLOSED ────────────────────────────────────────────
print("\n[2/7] Splitting closed vs non-closed complaints...")

df_closed = df[df["status"] == "CLOSED"].copy()
df_open   = df[df["status"] != "CLOSED"].copy()

print(f"      Closed     : {len(df_closed):,} rows → will go to nyc311_clean")
print(f"      Non-closed : {len(df_open):,} rows → will go to nyc311_open")
print(f"      Non-closed status breakdown:")
for status, count in df_open["status"].value_counts().items():
    print(f"        {status:<20} {count:,}")

# ── 3. REMOVE ROWS WITH NO CLOSED DATE ───────────────────────────────────────
print("\n[3/7] Removing closed rows with no closed date...")
before = len(df_closed)
df_closed = df_closed[df_closed["closed_date"].notna()].copy()
print(f"      Removed {before - len(df_closed):,} rows → {len(df_closed):,} remaining")

# ── 4. CALCULATE RESOLUTION TIME & REMOVE BAD DATES ──────────────────────────
print("\n[4/7] Calculating resolution time and removing bad dates...")
df_closed["resolution_hours"] = (
    pd.to_datetime(df_closed["closed_date"]) - pd.to_datetime(df_closed["created_date"])
).dt.total_seconds() / 3600

before = len(df_closed)

# Remove negative resolution times (closed_date before created_date — impossible)
neg = (df_closed["resolution_hours"] <= 0).sum()
df_closed = df_closed[df_closed["resolution_hours"] > 0].copy()

# Created date must be within our 2-month dataset window
df_closed = df_closed[df_closed["created_date"] >= "2026-01-01"].copy()
df_closed = df_closed[df_closed["created_date"] <= "2026-03-03"].copy()

# Remove resolutions over 1 year (8760 hours) — likely bad timestamps
long = (df_closed["resolution_hours"] > 8760).sum()
df_closed = df_closed[df_closed["resolution_hours"] <= 8760].copy()

removed = before - len(df_closed)
print(f"      Negative resolution times removed : {neg:,}")
print(f"      Resolutions over 1 year removed   : {long:,}")
print(f"      Total removed                     : {removed:,}")
print(f"      Remaining                         : {len(df_closed):,} rows")
print(f"\n      Resolution time stats:")
print(f"        Min    : {df_closed['resolution_hours'].min():.1f} hours")
print(f"        Median : {df_closed['resolution_hours'].median():.1f} hours")
print(f"        Mean   : {df_closed['resolution_hours'].mean():.1f} hours")
print(f"        Max    : {df_closed['resolution_hours'].max():.1f} hours")

# ── 5. HANDLE NULLS (CLOSED) ─────────────────────────────────────────────────
print("\n[5/7] Handling nulls in closed complaints...")

def fill_nulls(dataframe):
    dataframe["borough"]                = dataframe["borough"].fillna("UNKNOWN")
    dataframe["problem_detail"]         = dataframe["problem_detail"].fillna("UNKNOWN")
    dataframe["location_type"]          = dataframe["location_type"].fillna("UNKNOWN")
    dataframe["city"]                   = dataframe["city"].fillna("UNKNOWN")
    dataframe["incident_zip"]           = dataframe["incident_zip"].fillna("00000")
    dataframe["open_data_channel_type"] = dataframe["open_data_channel_type"].fillna("UNKNOWN")
    return dataframe

df_closed = fill_nulls(df_closed)
df_open   = fill_nulls(df_open)

print("      Filled nulls in both closed and open complaints:")
print("        borough, problem_detail, location_type → UNKNOWN")
print("        city, open_data_channel_type           → UNKNOWN")
print("        incident_zip                           → 00000")

# ── 6. WRITE nyc311_clean ─────────────────────────────────────────────────────
print("\n[6/7] Writing nyc311_clean to PostgreSQL...")

CREATE_CLEAN_TABLE = """
CREATE TABLE IF NOT EXISTS nyc311_clean (
    unique_key                       BIGINT PRIMARY KEY,
    created_date                     TIMESTAMP,
    closed_date                      TIMESTAMP,
    resolution_hours                 NUMERIC(10, 2),
    agency                           VARCHAR(20),
    agency_name                      TEXT,
    problem_type                     TEXT,
    problem_detail                   TEXT,
    additional_details               TEXT,
    location_type                    TEXT,
    incident_zip                     VARCHAR(10),
    incident_address                 TEXT,
    street_name                      TEXT,
    cross_street_1                   TEXT,
    cross_street_2                   TEXT,
    address_type                     VARCHAR(30),
    city                             VARCHAR(100),
    status                           VARCHAR(20),
    resolution_description           TEXT,
    resolution_action_updated_date   TIMESTAMP,
    community_board                  VARCHAR(30),
    council_district                 SMALLINT,
    police_precinct                  VARCHAR(30),
    borough                          VARCHAR(20),
    x_coordinate                     NUMERIC,
    y_coordinate                     NUMERIC,
    open_data_channel_type           VARCHAR(20),
    park_facility_name               TEXT,
    park_borough                     VARCHAR(20),
    latitude                         NUMERIC(10, 7),
    longitude                        NUMERIC(10, 7)
);
"""

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS nyc311_clean"))
    conn.execute(text(CREATE_CLEAN_TABLE))
    conn.commit()

CLOSED_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "resolution_hours",
    "agency",
    "agency_name",
    "problem_type",
    "problem_detail",
    "additional_details",
    "location_type",
    "incident_zip",
    "incident_address",
    "street_name",
    "cross_street_1",
    "cross_street_2",
    "address_type",
    "city",
    "status",
    "resolution_description",
    "resolution_action_updated_date",
    "community_board",
    "council_district",
    "police_precinct",
    "borough",
    "x_coordinate",
    "y_coordinate",
    "open_data_channel_type",
    "park_facility_name",
    "park_borough",
    "latitude",
    "longitude",
]
df_closed_to_write = df_closed[CLOSED_COLUMNS].copy()

df_closed_to_write.to_sql(
    name      = "nyc311_clean",
    con       = engine,
    if_exists = "append",
    index     = False,
    chunksize = 2000,
)
print(f"      Written {len(df_closed_to_write):,} rows to nyc311_clean")

# ── 7. WRITE nyc311_open ──────────────────────────────────────────────────────
print("\n[7/7] Writing nyc311_open to PostgreSQL...")

CREATE_OPEN_TABLE = """
CREATE TABLE IF NOT EXISTS nyc311_open (
    unique_key                       BIGINT PRIMARY KEY,
    created_date                     TIMESTAMP,
    agency                           VARCHAR(20),
    agency_name                      TEXT,
    problem_type                     TEXT,
    problem_detail                   TEXT,
    additional_details               TEXT,
    location_type                    TEXT,
    incident_zip                     VARCHAR(10),
    incident_address                 TEXT,
    street_name                      TEXT,
    cross_street_1                   TEXT,
    cross_street_2                   TEXT,
    address_type                     VARCHAR(30),
    city                             VARCHAR(100),
    status                           VARCHAR(20),
    community_board                  VARCHAR(30),
    council_district                 SMALLINT,
    police_precinct                  VARCHAR(30),
    borough                          VARCHAR(20),
    x_coordinate                     NUMERIC,
    y_coordinate                     NUMERIC,
    open_data_channel_type           VARCHAR(20),
    park_facility_name               TEXT,
    park_borough                     VARCHAR(20),
    latitude                         NUMERIC(10, 7),
    longitude                        NUMERIC(10, 7)
);
"""

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS nyc311_open"))
    conn.execute(text(CREATE_OPEN_TABLE))
    conn.commit()

OPEN_COLUMNS = [
    "unique_key",
    "created_date",
    "agency",
    "agency_name",
    "problem_type",
    "problem_detail",
    "additional_details",
    "location_type",
    "incident_zip",
    "incident_address",
    "street_name",
    "cross_street_1",
    "cross_street_2",
    "address_type",
    "city",
    "status",
    "community_board",
    "council_district",
    "police_precinct",
    "borough",
    "x_coordinate",
    "y_coordinate",
    "open_data_channel_type",
    "park_facility_name",
    "park_borough",
    "latitude",
    "longitude",
]
df_open_to_write = df_open[OPEN_COLUMNS].copy()

df_open_to_write.to_sql(
    name      = "nyc311_open",
    con       = engine,
    if_exists = "append",
    index     = False,
    chunksize = 2000,
)
print(f"      Written {len(df_open_to_write):,} rows to nyc311_open")

# ── VERIFY ────────────────────────────────────────────────────────────────────
print("\n--- Verification ---")
with engine.connect() as conn:
    clean_count = conn.execute(text("SELECT COUNT(*) FROM nyc311_clean")).scalar()
    open_count  = conn.execute(text("SELECT COUNT(*) FROM nyc311_open")).scalar()
    borough_dist = conn.execute(text("""
        SELECT borough, COUNT(*) as cnt
        FROM nyc311_clean
        GROUP BY borough
        ORDER BY cnt DESC
    """)).fetchall()

print(f"nyc311_raw   : 692,052 rows  (original, untouched)")
print(f"nyc311_clean : {clean_count:,} rows  (closed, for training)")
print(f"nyc311_open  : {open_count:,} rows  (non-closed, for predictions later)")
print(f"Total        : {clean_count + open_count:,} rows")

print("\nBorough distribution in nyc311_clean:")
for row in borough_dist:
    print(f"  {row[0]:<20} {row[1]:,}")

print("\n✓ Step 3 complete.")
engine.dispose()