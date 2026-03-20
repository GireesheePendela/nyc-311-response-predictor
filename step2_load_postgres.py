"""
NYC 311 Resolution Time Predictor
Step 2: PostgreSQL Schema Design & Bulk Load
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import time

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
        "  python step2_load_postgres.py"
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
        "Create/update .env in the project root before running step2_load_postgres.py"
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

DATA_DIR = Path("data") if Path("data").exists() else Path("Data")
csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit(
        f"No CSV file found in {DATA_DIR}.\n"
        "Expected a dataset like data/Dataset-nyc-311.csv"
    )
CSV_PATH = csv_files[0]

# ── COLUMNS TO DROP (near-null, no modeling value) ────────────────────────────
COLS_TO_DROP = [
    "Vehicle Type",
    "Taxi Company Borough",
    "Taxi Pick Up Location",
    "Bridge Highway Name",
    "Bridge Highway Direction",
    "Road Ramp",
    "Bridge Highway Segment",
    "Facility Type",
    "Location",
]

# ── RENAME TO snake_case ──────────────────────────────────────────────────────
RENAME_MAP = {
    "Unique Key"                          : "unique_key",
    "Created Date"                        : "created_date",
    "Closed Date"                         : "closed_date",
    "Agency"                              : "agency",
    "Agency Name"                         : "agency_name",
    "Problem (formerly Complaint Type)"   : "problem_type",
    "Problem Detail (formerly Descriptor)": "problem_detail",
    "Additional Details"                  : "additional_details",
    "Location Type"                       : "location_type",
    "Incident Zip"                        : "incident_zip",
    "Incident Address"                    : "incident_address",
    "Street Name"                         : "street_name",
    "Cross Street 1"                      : "cross_street_1",
    "Cross Street 2"                      : "cross_street_2",
    "Intersection Street 1"               : "intersection_street_1",
    "Intersection Street 2"               : "intersection_street_2",
    "Address Type"                        : "address_type",
    "City"                                : "city",
    "Landmark"                            : "landmark",
    "Status"                              : "status",
    "Due Date"                            : "due_date",
    "Resolution Description"              : "resolution_description",
    "Resolution Action Updated Date"      : "resolution_action_updated_date",
    "Community Board"                     : "community_board",
    "Council District"                    : "council_district",
    "Police Precinct"                     : "police_precinct",
    "BBL"                                 : "bbl",
    "Borough"                             : "borough",
    "X Coordinate (State Plane)"          : "x_coordinate",
    "Y Coordinate (State Plane)"          : "y_coordinate",
    "Open Data Channel Type"              : "open_data_channel_type",
    "Park Facility Name"                  : "park_facility_name",
    "Park Borough"                        : "park_borough",
    "Latitude"                            : "latitude",
    "Longitude"                           : "longitude",
}

# ── DDL: CREATE TABLE ─────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nyc311_raw (
    unique_key                       BIGINT PRIMARY KEY,
    created_date                     TIMESTAMP,
    closed_date                      TIMESTAMP,
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
    intersection_street_1            TEXT,
    intersection_street_2            TEXT,
    address_type                     VARCHAR(30),
    city                             VARCHAR(100),
    landmark                         TEXT,
    status                           VARCHAR(20),
    due_date                         TIMESTAMP,
    resolution_description           TEXT,
    resolution_action_updated_date   TIMESTAMP,
    community_board                  VARCHAR(30),
    council_district                 SMALLINT,
    police_precinct                  VARCHAR(30),
    bbl                              BIGINT,
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

# ── 1. LOAD CSV ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: PostgreSQL Schema & Bulk Load")
print("=" * 60)

print(f"\n[1/5] Loading CSV: {CSV_PATH.name}...")
t0 = time.time()

df = pd.read_csv(
    CSV_PATH,
    low_memory=False,
    parse_dates=["Created Date", "Closed Date", "Due Date", "Resolution Action Updated Date"]
)
print(f"      Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# ── 2. DROP USELESS COLUMNS ───────────────────────────────────────────────────
print("\n[2/5] Dropping near-null columns...")
cols_present = [c for c in COLS_TO_DROP if c in df.columns]
df.drop(columns=cols_present, inplace=True)
print(f"      Dropped {len(cols_present)} columns -> {len(df.columns)} remaining")

# ── 3. RENAME + FIX DTYPES ───────────────────────────────────────────────────
print("\n[3/5] Renaming columns and fixing dtypes...")
df.rename(columns=RENAME_MAP, inplace=True)

# zip: float → string (preserve leading zeros e.g. 07001)
df["incident_zip"] = df["incident_zip"].apply(
    lambda x: str(int(x)).zfill(5) if pd.notna(x) else None
)

# x/y coordinate: object → numeric
df["x_coordinate"] = pd.to_numeric(df["x_coordinate"], errors="coerce")
df["y_coordinate"] = pd.to_numeric(df["y_coordinate"], errors="coerce")

# council_district: float → nullable int
df["council_district"] = pd.to_numeric(df["council_district"], errors="coerce") \
                           .astype("Int64")

# bbl: float → nullable int
df["bbl"] = pd.to_numeric(df["bbl"], errors="coerce").astype("Int64")

# Standardise key string columns
for col in ["agency", "borough", "status", "open_data_channel_type"]:
    df[col] = df[col].str.strip().str.upper()

print("      Dtypes fixed.")

# ── 4. CREATE DATABASE & TABLE ────────────────────────────────────────────────
print("\n[4/5] Connecting to PostgreSQL and creating table...")

engine_default = create_engine(
    build_db_url("postgres")
)
with engine_default.connect() as conn:
    conn.execute(text("COMMIT"))
    result = conn.execute(
        text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
    ).fetchone()
    if not result:
        conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
        print(f"      Created database '{DB_NAME}'")
    else:
        print(f"      Database '{DB_NAME}' already exists")
engine_default.dispose()

engine = create_engine(
    build_db_url(DB_NAME),
    pool_pre_ping=True
)

with engine.connect() as conn:
    conn.execute(text(CREATE_TABLE_SQL))
    conn.execute(text("ALTER TABLE nyc311_raw ALTER COLUMN police_precinct TYPE VARCHAR(30)"))
    conn.commit()
print("      Table 'nyc311_raw' ready.")

# ── 5. BULK INSERT ────────────────────────────────────────────────────────────
print("\n[5/5] Bulk inserting data ")
t1 = time.time()

with engine.connect() as conn:
    existing_rows = conn.execute(text("SELECT COUNT(*) FROM nyc311_raw")).scalar()
    if existing_rows:
        print(f"      Found {existing_rows:,} existing rows. Truncating for clean reload...")
        conn.execute(text("TRUNCATE TABLE nyc311_raw"))
        conn.commit()

df.to_sql(
    name      = "nyc311_raw",
    con       = engine,
    if_exists = "append",
    index     = False,
    chunksize = 5000,
    method    = "multi",
)

elapsed = time.time() - t1
print(f"      Inserted {len(df):,} rows in {elapsed:.0f}s")

# ── VERIFY ────────────────────────────────────────────────────────────────────
print("\n--- Verification ---")
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM nyc311_raw")).scalar()
    sample = conn.execute(text("""
        SELECT unique_key, created_date, closed_date, agency, problem_type, borough, status
        FROM nyc311_raw
        LIMIT 3
    """)).fetchall()

print(f"Rows in nyc311_raw: {count:,}")
print("\nSample rows:")
for row in sample:
    print(" ", row)

print("\nStep 2 complete. Data is in PostgreSQL.")
engine.dispose()