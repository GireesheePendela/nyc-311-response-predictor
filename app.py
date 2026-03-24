from importlib import import_module
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import html

gr = import_module("gradio")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model        = joblib.load("best_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# ── VALID VALUES ──────────────────────────────────────────────────────────────
AGENCIES = [
    "DCWP", "DEP", "DHS", "DOB", "DOE", "DOHMH",
    "DOT", "DPR", "DSNY", "HPD", "NYPD", "OOS", "OTI", "TLC"
]

BOROUGHS = [
    "BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"
]

CHANNELS = ["MOBILE", "ONLINE", "PHONE", "OTHER"]

PROBLEM_TYPES = sorted([
    "HEAT/HOT WATER", "Noise - Residential", "Illegal Parking",
    "UNSANITARY CONDITION", "Noise - Commercial", "PLUMBING",
    "Noise - Vehicle", "Blocked Driveway", "PAINT/PLASTER",
    "Street Light Condition", "DOOR/WINDOW", "Noise - Street/Sidewalk",
    "FLOORING/STAIRS", "Dirty Conditions", "Derelict Vehicle",
    "WATER LEAK", "Missed Collection", "Cannabis Retailer",
    "Boilers", "Smoking or Vaping", "For Hire Vehicle Complaint",
    "ELEVATOR", "APPLIANCE", "Vendor Enforcement", "Traffic",
    "Panhandling", "Drug Activity", "Noise - Park", "Illegal Fireworks",
])

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

PROBLEM_TYPE_ENCODINGS = {
    "HEAT/HOT WATER"            : 4.20,
    "Noise - Residential"       : 1.10,
    "Illegal Parking"           : 1.50,
    "UNSANITARY CONDITION"      : 5.35,
    "Noise - Commercial"        : 1.20,
    "PLUMBING"                  : 4.80,
    "Noise - Vehicle"           : 1.10,
    "Blocked Driveway"          : 1.40,
    "PAINT/PLASTER"             : 4.90,
    "Street Light Condition"    : 5.30,
    "DOOR/WINDOW"               : 5.30,
    "Noise - Street/Sidewalk"   : 1.15,
    "FLOORING/STAIRS"           : 5.30,
    "Dirty Conditions"          : 3.80,
    "Derelict Vehicle"          : 3.90,
    "WATER LEAK"                : 5.10,
    "Missed Collection"         : 3.75,
    "Cannabis Retailer"         : 6.70,
    "Boilers"                   : 6.70,
    "Smoking or Vaping"         : 6.26,
    "For Hire Vehicle Complaint": 6.25,
    "ELEVATOR"                  : 5.37,
    "APPLIANCE"                 : 5.35,
    "Vendor Enforcement"        : 0.30,
    "Traffic"                   : 0.40,
    "Panhandling"               : 0.85,
    "Drug Activity"             : 0.95,
    "Noise - Park"              : 0.95,
    "Illegal Fireworks"         : 0.90,
}

LOCATION_TYPE_ENCODINGS = {
    "RESIDENTIAL BUILDING"      : 4.50,
    "Street/Sidewalk"           : 2.10,
    "Store/Commercial"          : 2.80,
    "Club/Bar/Restaurant"       : 1.50,
    "Park/Playground"           : 2.20,
    "UNKNOWN"                   : 3.50,
}

APP_THEME = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

CUSTOM_CSS = """
body {
    background: radial-gradient(circle at top right, #18243f 0%, #0b1222 42%, #080d18 100%);
}

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 14px 18px !important;
}

.hero-card {
    background: linear-gradient(135deg, rgba(76,114,176,0.16), rgba(100,181,205,0.10));
    border: 1px solid rgba(148,163,184,0.28);
    border-radius: 16px;
    padding: 16px 20px 10px 20px;
    margin-bottom: 12px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
}

.quick-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 10px 0;
}

.quick-stat {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 12px;
    padding: 10px 12px;
}

.quick-stat-value {
    font-size: 1.08rem;
    font-weight: 700;
    color: #e8eefc;
    line-height: 1.2;
}

.quick-stat-label {
    font-size: 0.78rem;
    color: #b8c6e3;
    margin-top: 2px;
}

.subtle-note {
    color: #c3d2ef;
    font-size: 0.93rem;
    margin-top: 6px;
}

.dashboard-frame iframe {
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.24) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}

.predict-card {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 14px;
    padding: 8px 8px 2px 8px;
}

.prediction-box {
    background: rgba(10, 16, 30, 0.6);
    border: 1px solid rgba(148,163,184,0.24);
    border-radius: 12px;
    padding: 8px 10px;
}

.action-row {
    margin-top: 6px;
}

.action-row button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

@media (max-width: 900px) {
    .quick-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


def get_dashboard_embed_html():
    dashboard_path = Path("step7_dashboard.html")
    if not dashboard_path.exists():
        return """
<div style='padding:16px;border:1px solid #334155;border-radius:8px;'>
    <h3>Dashboard not found</h3>
  <p>Generate or upload <b>step7_dashboard.html</b> to view it here.</p>
</div>
"""

    dashboard_html = dashboard_path.read_text(encoding="utf-8", errors="ignore")
    escaped_html = html.escape(dashboard_html, quote=True)
    return f"""
<iframe srcdoc="{escaped_html}" style="width:100%;height:900px;border:1px solid #334155;border-radius:8px;"></iframe>
"""

# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────
def predict(agency, borough, problem_type, channel, hour, day, month):
    missing = []
    if not agency:
        missing.append("Agency")
    if not borough:
        missing.append("Borough")
    if not problem_type:
        missing.append("Problem Type")
    if not channel:
        missing.append("Channel")
    if day is None or day == "":
        missing.append("Day of Week")
    if hour is None:
        missing.append("Hour of Day")
    if month is None:
        missing.append("Month")

    if missing:
        missing_lines = "\n".join([f"- {field}" for field in missing])
        return f"""
## ⚠️ Missing Required Fields

Please fill in these fields before predicting:
{missing_lines}
"""

    # Build feature row
    row = {col: 0 for col in feature_cols}

    row["hour_of_day"]  = int(hour)
    row["day_of_week"]  = DAYS.index(day)
    row["month"]        = int(month)
    row["is_weekend"]   = 1 if DAYS.index(day) >= 5 else 0
    row["is_night"]     = 1 if (int(hour) >= 22 or int(hour) < 6) else 0

    row["problem_type_encoded"]  = PROBLEM_TYPE_ENCODINGS.get(
        problem_type, 3.50
    )
    row["location_type_encoded"] = 3.50

    agency_col = f"agency_{agency.upper()}"
    if agency_col in row:
        row[agency_col] = 1

    borough_col = f"borough_{borough.upper()}"
    if borough_col in row:
        row[borough_col] = 1

    channel_col = f"channel_{channel.upper()}"
    if channel_col in row:
        row[channel_col] = 1

    X = pd.DataFrame([row])[feature_cols]
    log_pred     = model.predict(X)[0]
    pred_hours   = float(np.expm1(log_pred))
    pred_days    = round(pred_hours / 24, 1)
    low          = max(0, round(pred_hours - 34.5, 1))
    high         = round(pred_hours + 34.5, 1)

    if pred_hours < 2:
        verdict = "🟢 Very Fast"
    elif pred_hours < 24:
        verdict = "🟡 Same Day"
    elif pred_hours < 72:
        verdict = "🟠 A Few Days"
    else:
        verdict = "🔴 Slow"

    result = f"""
## {verdict}

### Predicted Resolution Time
**{pred_hours:.1f} hours** ({pred_days} days)

### Confidence Range
{low}h – {high}h

### Summary
"""
    if pred_hours < 2:
        result += "Expected to resolve in under 2 hours."
    elif pred_hours < 24:
        result += f"Expected to resolve within the same day (~{pred_hours:.0f} hours)."
    elif pred_hours < 72:
        result += f"Expected to resolve in {pred_days} days (~{pred_hours:.0f} hours)."
    else:
        result += f"Expected to take {pred_days} days (~{pred_hours:.0f} hours) to resolve."

    return result

# ── GRADIO UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="NYC 311 Resolution Time Predictor"
) as demo:

    gr.HTML("""
    <div class="hero-card">
        <h1 style="margin:0 0 4px 0; font-size:2rem;">🗽 NYC 311 Response Intelligence App</h1>
        <p style="margin:0 0 8px 0; font-size:1rem; color:#d1def7;"><b>Workflow:</b> explore response patterns first, then generate a prediction.</p>
        <div class="quick-stats">
            <div class="quick-stat">
                <div class="quick-stat-value">589,802</div>
                <div class="quick-stat-label">Complaints in training data</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">Random Forest</div>
                <div class="quick-stat-label">Production model</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">R² = 0.78</div>
                <div class="quick-stat-label">Overall fit quality</div>
            </div>
            <div class="quick-stat">
                <div class="quick-stat-value">Jan–Mar 2026</div>
                <div class="quick-stat-label">Training time window</div>
            </div>
        </div>
        <p class="subtle-note">Use <b>Dashboard Insights</b> to understand response patterns, then use <b>Resolution Predictor</b> for new complaints.</p>
    </div>
    """)

    gr.Markdown("""
    Predictions for months outside Jan–Mar are extrapolations and may be less reliable.
    """)

    with gr.Tabs():
        with gr.Tab("Dashboard Insights"):
            gr.Markdown("""
            ### 📊 Interactive Insights Dashboard
            Review how response times vary by agency, borough, day/hour, and complaint category before running a prediction.
            """)
            gr.HTML(get_dashboard_embed_html(), elem_classes=["dashboard-frame"])

        with gr.Tab("Resolution Predictor"):
            gr.Markdown("""
            ### 🔮 Resolution Time Prediction
            Fill all required fields (*) and click **Predict Resolution Time**.
            """)

            with gr.Group(elem_classes=["predict-card"]):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🧾 Complaint Details")
                        agency       = gr.Dropdown(choices=AGENCIES,      label="Agency *",       value=None)
                        borough      = gr.Dropdown(choices=BOROUGHS,      label="Borough *",      value=None)
                        problem_type = gr.Dropdown(choices=PROBLEM_TYPES, label="Problem Type *", value=None)
                        channel      = gr.Dropdown(choices=CHANNELS,      label="Channel *",      value=None)

                    with gr.Column():
                        gr.Markdown("#### 📅 Filing Time")
                        hour  = gr.Slider(minimum=0,  maximum=23, value=9,  step=1,  label="Hour of Day * (0=midnight, 12=noon)")
                        day   = gr.Dropdown(choices=DAYS, label="Day of Week *", value=None)
                        month = gr.Slider(minimum=1,  maximum=12, value=1,  step=1,  label="Month * (1=Jan, 12=Dec)")

            output = gr.Markdown(label="Prediction", elem_classes=["prediction-box"])

            with gr.Row(elem_classes=["action-row"]):
                predict_btn = gr.Button("🔍 Predict Resolution Time", variant="primary")
                clear_btn = gr.ClearButton([agency, borough, problem_type, channel, hour, day, month, output], value="Reset")

            predict_btn.click(
                fn=predict,
                inputs=[agency, borough, problem_type, channel, hour, day, month],
                outputs=output
            )

            gr.Markdown("""
            ---
            ### Quick Examples
            - **HPD + HEAT/HOT WATER + BRONX + Monday 9am** → ~3 days
            - **NYPD + Noise - Residential + BROOKLYN + Saturday 11pm** → ~1 hour
            - **DSNY + Missed Collection + QUEENS + Wednesday 8am** → ~1-2 days
            """)

demo.launch(server_name="0.0.0.0", server_port=7860, theme=APP_THEME, css=CUSTOM_CSS)
