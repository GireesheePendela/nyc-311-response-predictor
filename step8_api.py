"""
NYC 311 Resolution Time Predictor
Step 8: FastAPI Deployment
- Loads best_model.pkl and feature_cols.pkl
- Exposes POST /predict endpoint
- Returns predicted resolution time in hours
"""

# pyright: reportMissingImports=false

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
model        = joblib.load("best_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
print(f"Model loaded. Features: {len(feature_cols)}")

# ── VALID VALUES (from our data) ──────────────────────────────────────────────
VALID_AGENCIES = [
    "DCWP", "DEP", "DHS", "DOB", "DOE", "DOHMH",
    "DOT", "DPR", "DSNY", "HPD", "NYPD", "OOS", "OTI", "TLC"
]

VALID_BOROUGHS = [
    "BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"
]

VALID_CHANNELS = [
    "MOBILE", "ONLINE", "OTHER", "PHONE"
]

# Target encoding means from Step 4
# These are approximate median log_resolution_hours per problem type
# In production you'd load this from the DB, but for the API we hardcode top types
PROBLEM_TYPE_ENCODINGS = {
    "HEAT/HOT WATER"           : 4.20,
    "Noise - Residential"      : 1.10,
    "Illegal Parking"          : 1.50,
    "UNSANITARY CONDITION"     : 5.35,
    "Noise - Commercial"       : 1.20,
    "PLUMBING"                 : 4.80,
    "Noise - Vehicle"          : 1.10,
    "Blocked Driveway"         : 1.40,
    "PAINT/PLASTER"            : 4.90,
    "Street Light Condition"   : 5.30,
    "DOOR/WINDOW"              : 5.30,
    "Noise - Street/Sidewalk"  : 1.15,
    "FLOORING/STAIRS"          : 5.30,
    "Dirty Conditions"         : 3.80,
    "Derelict Vehicle"         : 3.90,
    "WATER LEAK"               : 5.10,
    "Missed Collection"        : 3.75,
    "Cannabis Retailer"        : 6.70,
    "Boilers"                  : 6.70,
    "Smoking or Vaping"        : 6.26,
    "For Hire Vehicle Complaint": 6.25,
    "ELEVATOR"                 : 5.37,
    "APPLIANCE"                : 5.35,
    "Vendor Enforcement"       : 0.30,
    "Traffic"                  : 0.40,
    "Panhandling"              : 0.85,
    "Drug Activity"            : 0.95,
    "Noise - Park"             : 0.95,
    "Illegal Fireworks"        : 0.90,
}
DEFAULT_ENCODING = 3.50  # fallback for unknown problem types

LOCATION_TYPE_ENCODINGS = {
    "RESIDENTIAL BUILDING"     : 4.50,
    "Street/Sidewalk"          : 2.10,
    "Residential Building/House": 4.50,
    "Store/Commercial"         : 2.80,
    "Club/Bar/Restaurant"      : 1.50,
    "Park/Playground"          : 2.20,
    "Street"                   : 2.00,
    "Highway"                  : 2.50,
    "Bridge"                   : 3.00,
    "UNKNOWN"                  : 3.50,
}
DEFAULT_LOCATION_ENCODING = 3.50

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NYC 311 Resolution Time Predictor",
    description = "Predict how long a 311 complaint will take to resolve",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    agency            : str = Field(..., example="HPD")
    borough           : str = Field(..., example="BROOKLYN")
    problem_type      : str = Field(..., example="HEAT/HOT WATER")
    location_type     : Optional[str] = Field("UNKNOWN", example="RESIDENTIAL BUILDING")
    channel           : Optional[str] = Field("ONLINE",  example="ONLINE")
    hour_of_day       : int  = Field(..., ge=0, le=23,  example=9)
    day_of_week       : int  = Field(..., ge=0, le=6,   example=0)
    month             : int  = Field(..., ge=1, le=12,  example=1)

class PredictionResponse(BaseModel):
    predicted_hours   : float
    predicted_days    : float
    confidence_band   : str
    summary           : str
    inputs            : dict

# ── FEATURE BUILDER ───────────────────────────────────────────────────────────
def build_feature_row(req: ComplaintRequest) -> pd.DataFrame:
    row = {col: 0 for col in feature_cols}

    # Numeric features
    row["hour_of_day"]  = req.hour_of_day
    row["day_of_week"]  = req.day_of_week
    row["month"]        = req.month
    row["is_weekend"]   = 1 if req.day_of_week >= 5 else 0
    row["is_night"]     = 1 if (req.hour_of_day >= 22 or req.hour_of_day < 6) else 0

    # Target encodings
    row["problem_type_encoded"]  = PROBLEM_TYPE_ENCODINGS.get(
        req.problem_type, DEFAULT_ENCODING
    )
    row["location_type_encoded"] = LOCATION_TYPE_ENCODINGS.get(
        req.location_type or "UNKNOWN", DEFAULT_LOCATION_ENCODING
    )

    # One-hot: agency
    agency_col = f"agency_{req.agency.upper()}"
    if agency_col in row:
        row[agency_col] = 1

    # One-hot: borough
    borough_col = f"borough_{req.borough.upper()}"
    if borough_col in row:
        row[borough_col] = 1

    # One-hot: channel
    channel = (req.channel or "ONLINE").upper()
    channel_col = f"channel_{channel}"
    if channel_col in row:
        row[channel_col] = 1

    return pd.DataFrame([row])[feature_cols]

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"  : "NYC 311 Resolution Time Predictor API",
        "version"  : "1.0.0",
        "endpoints": {
            "POST /predict"   : "Predict resolution time for a complaint",
            "GET  /options"   : "List valid agencies, boroughs, channels",
            "GET  /examples"  : "See example requests",
            "GET  /docs"      : "Interactive API docs (Swagger UI)",
        }
    }

@app.get("/options")
def get_options():
    return {
        "agencies"      : VALID_AGENCIES,
        "boroughs"      : VALID_BOROUGHS,
        "channels"      : VALID_CHANNELS,
        "problem_types" : sorted(PROBLEM_TYPE_ENCODINGS.keys()),
        "location_types": sorted(LOCATION_TYPE_ENCODINGS.keys()),
        "hour_of_day"   : "0–23 (0 = midnight)",
        "day_of_week"   : "0=Monday, 1=Tuesday, ... 6=Sunday",
        "month"         : "1–12",
    }

@app.get("/examples")
def get_examples():
    return {
        "noise_complaint_night": {
            "agency"       : "NYPD",
            "borough"      : "BROOKLYN",
            "problem_type" : "Noise - Residential",
            "channel"      : "MOBILE",
            "hour_of_day"  : 23,
            "day_of_week"  : 5,
            "month"        : 1,
        },
        "heat_complaint_morning": {
            "agency"       : "HPD",
            "borough"      : "BRONX",
            "problem_type" : "HEAT/HOT WATER",
            "channel"      : "PHONE",
            "hour_of_day"  : 9,
            "day_of_week"  : 0,
            "month"        : 1,
        },
        "parking_complaint": {
            "agency"       : "NYPD",
            "borough"      : "MANHATTAN",
            "problem_type" : "Illegal Parking",
            "channel"      : "ONLINE",
            "hour_of_day"  : 14,
            "day_of_week"  : 2,
            "month"        : 2,
        },
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: ComplaintRequest):
    # Validate agency
    if req.agency.upper() not in VALID_AGENCIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid agency '{req.agency}'. Valid: {VALID_AGENCIES}"
        )

    # Validate borough
    if req.borough.upper() not in VALID_BOROUGHS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid borough '{req.borough}'. Valid: {VALID_BOROUGHS}"
        )

    # Build feature row and predict
    X = build_feature_row(req)
    log_pred   = model.predict(X)[0]
    pred_hours = float(np.expm1(log_pred))
    pred_days  = round(pred_hours / 24, 1)

    # Confidence band (±1 MAE = ±34.5 hours)
    low  = max(0, round(pred_hours - 34.5, 1))
    high = round(pred_hours + 34.5, 1)

    # Human-readable summary
    if pred_hours < 2:
        summary = f"Expected to resolve in under 2 hours."
    elif pred_hours < 24:
        summary = f"Expected to resolve within the same day (~{pred_hours:.0f} hours)."
    elif pred_hours < 72:
        summary = f"Expected to resolve in {pred_days} days (~{pred_hours:.0f} hours)."
    else:
        summary = f"Expected to take {pred_days} days (~{pred_hours:.0f} hours) to resolve."

    return PredictionResponse(
        predicted_hours = round(pred_hours, 1),
        predicted_days  = pred_days,
        confidence_band = f"{low}h – {high}h",
        summary         = summary,
        inputs          = req.dict()
    )

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("step8_api:app", host="0.0.0.0", port=7860, reload=True)