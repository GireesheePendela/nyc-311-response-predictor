"""
NYC 311 Resolution Time Predictor
Step 7: Interactive Plotly Dashboard (Fixed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
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
        "  python step7_visualization.py"
    ) from exc

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        "Create/update .env in the project root before running step7_visualization.py"
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
print("STEP 7: Interactive Plotly Dashboard")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1/8] Loading data...")

predictions_df = pd.read_sql("SELECT * FROM nyc311_predictions", engine)
predictions_df["actual_hours"]    = pd.to_numeric(predictions_df["actual_hours"],    errors="coerce")
predictions_df["predicted_hours"] = pd.to_numeric(predictions_df["predicted_hours"], errors="coerce")
predictions_df["error_hours"]     = pd.to_numeric(predictions_df["error_hours"],     errors="coerce")

model        = joblib.load("best_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(f"      Predictions loaded: {len(predictions_df):,} rows")

# ── 2. QUERY EDA DATA ─────────────────────────────────────────────────────────
print("\n[2/8] Querying EDA data from PostgreSQL...")

agency_df = pd.read_sql("""
    SELECT agency,
           COUNT(*) AS complaints,
           ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                 (ORDER BY resolution_hours)::numeric, 1) AS median_hours,
           ROUND(AVG(resolution_hours)::numeric, 1) AS avg_hours
    FROM nyc311_clean
    GROUP BY agency
    ORDER BY median_hours DESC
""", engine)

hour_day_df = pd.read_sql("""
    SELECT
        EXTRACT(HOUR FROM created_date)::int AS hour,
        EXTRACT(DOW  FROM created_date)::int AS dow,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1) AS median_hours
    FROM nyc311_clean
    GROUP BY
        EXTRACT(HOUR FROM created_date)::int,
        EXTRACT(DOW  FROM created_date)::int
    ORDER BY dow, hour
""", engine)

borough_agency_df = pd.read_sql("""
    SELECT
        borough,
        agency,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                            (ORDER BY resolution_hours)::numeric, 1)::float AS median_hours,
        COUNT(*) AS complaints
    FROM nyc311_clean
    WHERE borough NOT IN ('UNSPECIFIED')
      AND agency IN ('HPD','NYPD','DSNY','DOT','DEP','DOB','DPR')
    GROUP BY borough, agency
""", engine)

channel_df = pd.read_sql("""
    SELECT
        open_data_channel_type AS channel,
        COUNT(*) AS complaints,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
              (ORDER BY resolution_hours)::numeric, 1) AS median_hours
    FROM nyc311_clean
    GROUP BY channel
    ORDER BY median_hours DESC
""", engine)

slowest_df = pd.read_sql("""
    SELECT problem_type,
           ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                 (ORDER BY resolution_hours)::numeric, 1) AS median_hours,
           COUNT(*) AS complaints
    FROM nyc311_clean
    GROUP BY problem_type
    HAVING COUNT(*) > 100
    ORDER BY median_hours DESC
    LIMIT 15
""", engine)

fastest_df = pd.read_sql("""
    SELECT problem_type,
           ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                 (ORDER BY resolution_hours)::numeric, 1) AS median_hours,
           COUNT(*) AS complaints
    FROM nyc311_clean
    GROUP BY problem_type
    HAVING COUNT(*) > 100
    ORDER BY median_hours ASC
    LIMIT 15
""", engine)

daily_df = pd.read_sql("""
    SELECT
        DATE(created_date)::text AS date,
        agency,
        COUNT(*) AS complaints
    FROM nyc311_clean
    GROUP BY DATE(created_date), agency
    ORDER BY date
""", engine)

print("      All queries complete")

# ── 3. PREP DATA ──────────────────────────────────────────────────────────────
print("\n[3/8] Preparing data...")

# Scatter plot data
df_plot = predictions_df[
    (predictions_df["actual_hours"]    >= 0) &
    (predictions_df["actual_hours"]    <= 300) &
    (predictions_df["predicted_hours"] >= 0) &
    (predictions_df["predicted_hours"] <= 300)
].dropna(subset=["actual_hours", "predicted_hours"])
sample = df_plot.sample(min(8000, len(df_plot)), random_state=42)

# Heatmap 1: hour × day
hour_day_df["hour"] = hour_day_df["hour"].astype(int)
hour_day_df["dow"]  = hour_day_df["dow"].astype(int)
hour_day_df["median_hours"] = pd.to_numeric(hour_day_df["median_hours"], errors="coerce")

dow_labels = {0:"Sun", 1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat"}
hour_day_df["dow_label"] = hour_day_df["dow"].map(dow_labels)

pivot1 = hour_day_df.pivot_table(
    index="dow_label",
    columns="hour",
    values="median_hours",
    aggfunc="mean"
)
row_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
pivot1 = pivot1.reindex(index=row_order, columns=list(range(24)))
pivot1 = pivot1.astype(float)

# Heatmap 2: borough × agency
borough_agency_df["median_hours"] = pd.to_numeric(
    borough_agency_df["median_hours"], errors="coerce"
)
pivot2 = borough_agency_df.pivot_table(
    index="agency",
    columns="borough",
    values="median_hours",
    aggfunc="mean"
).fillna(0)
borough_order = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
pivot2 = pivot2.reindex(columns=[b for b in borough_order if b in pivot2.columns])
agency_order = ["NYPD", "HPD", "DSNY", "DOT", "DEP", "DOB", "DPR"]
pivot2 = pivot2.reindex([a for a in agency_order if a in pivot2.index])
pivot2 = pivot2.astype(float)

# Daily volume
daily_df["date"]       = pd.to_datetime(daily_df["date"])
daily_df["complaints"] = pd.to_numeric(daily_df["complaints"], errors="coerce")
top_agencies = ["NYPD", "HPD", "DSNY", "DOT", "DEP"]
daily_top = daily_df[daily_df["agency"].isin(top_agencies)].copy()

# ── 4. COLORS ─────────────────────────────────────────────────────────────────
COLORS = {
    "blue"   : "#4C72B0",
    "orange" : "#DD8452",
    "green"  : "#55A868",
    "red"    : "#C44E52",
    "purple" : "#8172B2",
    "teal"   : "#64B5CD",
    "bg"     : "#0f1117",
    "card"   : "#1a1d27",
    "text"   : "#e0e0e0",
    "grid"   : "#2a2d3a",
}

def dark_layout(title):
    return dict(
        title=title,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
    )

# ── 5. BUILD CHARTS ───────────────────────────────────────────────────────────
print("\n[4/8] Building charts...")

# Chart 1: Predicted vs Actual
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=sample["actual_hours"],
    y=sample["predicted_hours"],
    mode="markers",
    marker=dict(size=4, color=COLORS["blue"], opacity=0.4),
    name="Predictions",
    hovertemplate="Actual: %{x:.1f}h<br>Predicted: %{y:.1f}h<extra></extra>"
))
fig_scatter.add_trace(go.Scatter(
    x=[0, 300], y=[0, 300],
    mode="lines",
    line=dict(color=COLORS["red"], dash="dash", width=2),
    name="Perfect Prediction"
))
fig_scatter.update_layout(
    **dark_layout("Predicted vs Actual Resolution Time"),
    xaxis_title="Actual Hours",
    yaxis_title="Predicted Hours",
    legend=dict(bgcolor=COLORS["card"]),
)

# Chart 2: Error distribution
errors = predictions_df["error_hours"].dropna().clip(-200, 200)
fig_error = go.Figure()
fig_error.add_trace(go.Histogram(
    x=errors, nbinsx=80,
    marker_color=COLORS["orange"],
    opacity=0.85,
    name="Prediction Error"
))
fig_error.add_vline(x=0,             line_dash="dash", line_color=COLORS["red"],
                    annotation_text="Zero Error", annotation_position="top right")
fig_error.add_vline(x=float(errors.mean()), line_dash="dash", line_color=COLORS["green"],
                    annotation_text=f"Mean: {errors.mean():.1f}h",
                    annotation_position="top left")
fig_error.update_layout(
    **dark_layout("Prediction Error Distribution"),
    xaxis_title="Error (hours)",
    yaxis_title="Count",
)

# Chart 3: Feature importance
top15 = importances.head(15).sort_values(ascending=True)
top15_pct = (top15 * 100).round(1)
top15_labels = [f"{value:.1f}%" for value in top15_pct.values]
bar_colors = [COLORS["red"] if i == len(top15) - 1
              else COLORS["blue"] for i in range(len(top15))]
fig_importance = go.Figure(go.Bar(
    x=top15_pct.values,
    y=top15.index,
    orientation="h",
    marker_color=bar_colors,
    text=top15_labels,
    customdata=top15_labels,
    textposition="outside",
    hovertemplate="%{y}: %{customdata}<extra></extra>"
))
fig_importance.update_layout(
    **dark_layout("Top 15 Feature Importances (Random Forest)"),
    xaxis_title="Importance (%)",
)

# Chart 4: Heatmap hour × day
fig_heatmap1 = go.Figure(go.Heatmap(
    z=pivot1.values.tolist(),
    x=[str(h) for h in pivot1.columns.tolist()],
    y=pivot1.index.tolist(),
    colorscale="RdYlGn_r",
    hoverongaps=False,
    colorbar=dict(title="Median Hours"),
    zmin=float(np.nanmin(pivot1.values)),
    zmax=float(np.nanmax(pivot1.values)),
    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Median: %{z:.1f}h<extra></extra>",
))
layout1 = dark_layout("Median Resolution Time: Hour of Day × Day of Week")
layout1["xaxis"] = dict(
    title="Hour of Day",
    type="category",
    gridcolor=COLORS["grid"],
)
layout1["yaxis"] = dict(
    title="Day of Week",
    gridcolor=COLORS["grid"],
)
fig_heatmap1.update_layout(**layout1)

# Chart 5: Heatmap borough × agency
fig_heatmap2 = go.Figure(go.Heatmap(
    z=pivot2.values.tolist(),
    x=pivot2.columns.tolist(),
    y=pivot2.index.tolist(),
    colorscale="RdYlGn_r",
    hoverongaps=False,
    colorbar=dict(title="Median Hours"),
    zmin=0,
    zmax=float(np.nanmax(pivot2.values)),
    hovertemplate="Borough: %{x}<br>Agency: %{y}<br>Median: %{z:.1f}h<extra></extra>",
))
layout2 = dark_layout("Median Resolution Time: Borough × Agency Heatmap")
layout2["xaxis"] = dict(
    title="Borough",
    type="category",
    gridcolor=COLORS["grid"],
)
layout2["yaxis"] = dict(
    title="Agency",
    gridcolor=COLORS["grid"],
)
fig_heatmap2.update_layout(**layout2)

# Chart 6: Agency performance
agency_sorted = agency_df.sort_values("median_hours", ascending=True)
agency_sorted["median_hours"] = pd.to_numeric(agency_sorted["median_hours"], errors="coerce")
fig_agency = go.Figure(go.Bar(
    x=agency_sorted["median_hours"],
    y=agency_sorted["agency"],
    orientation="h",
    marker_color=[COLORS["red"] if v > 60 else COLORS["blue"]
                  for v in agency_sorted["median_hours"]],
    text=[f"{v:.1f}h" for v in agency_sorted["median_hours"]],
    textposition="outside",
    customdata=agency_sorted["complaints"],
    hovertemplate="Agency: %{y}<br>Median: %{x:.1f}h<br>Complaints: %{customdata:,}<extra></extra>"
))
fig_agency.update_layout(
    **dark_layout("Median Resolution Time by Agency"),
    xaxis_title="Median Hours",
)

# Chart 7: Daily volume animated line
fig_animated = px.line(
    daily_top,
    x="date",
    y="complaints",
    color="agency",
    title="Daily Complaint Volume by Agency (Jan 1, 2026 – Mar 3, 2026)",
    labels={"complaints": "Daily Complaints", "date": "Date"},
    color_discrete_map={
        "NYPD" : COLORS["blue"],
        "HPD"  : COLORS["red"],
        "DSNY" : COLORS["green"],
        "DOT"  : COLORS["orange"],
        "DEP"  : COLORS["purple"],
    }
)
fig_animated.update_layout(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"]),
    legend=dict(bgcolor=COLORS["card"]),
    xaxis=dict(gridcolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"]),
)

# Chart 8: Slowest complaint types
slowest_sorted = slowest_df.sort_values("median_hours", ascending=True)
slowest_sorted["median_hours"] = pd.to_numeric(slowest_sorted["median_hours"], errors="coerce")
fig_types = go.Figure(go.Bar(
    x=slowest_sorted["median_hours"],
    y=slowest_sorted["problem_type"],
    orientation="h",
    marker_color=COLORS["red"],
    hovertemplate="%{y}: %{x:.1f}h<extra></extra>"
))
fig_types.update_layout(
    **dark_layout("Top 15 Slowest Complaint Types"),
    xaxis_title="Median Hours",
    height=500
)

# Chart 9: Fastest complaint types
fastest_sorted = fastest_df.sort_values("median_hours", ascending=False)
fastest_sorted["median_hours"] = pd.to_numeric(fastest_sorted["median_hours"], errors="coerce")
fig_fastest = go.Figure(go.Bar(
    x=fastest_sorted["median_hours"],
    y=fastest_sorted["problem_type"],
    orientation="h",
    marker_color=COLORS["green"],
    hovertemplate="%{y}: %{x:.1f}h<extra></extra>"
))
fig_fastest.update_layout(
    **dark_layout("Top 15 Fastest Complaint Types"),
    xaxis_title="Median Hours",
    height=500
)

# Chart 10: Model comparison
fig_compare = make_subplots(specs=[[{"secondary_y": True}]])
models     = ["Linear Regression", "Random Forest", "XGBoost"]
r2_scores  = [0.7014, 0.7882, 0.7846]
mae_scores = [40.3,   34.5,   35.5]

fig_compare.add_trace(go.Bar(
    x=models, y=mae_scores,
    name="MAE (hrs)",
    marker_color=COLORS["orange"],
    opacity=0.85,
    text=[f"{v:.1f}h" for v in mae_scores],
    textposition="outside",
    hovertemplate="%{x}<br>MAE: %{y:.1f}h<extra></extra>",
), secondary_y=False)
fig_compare.add_trace(go.Scatter(
    x=models, y=r2_scores,
    mode="lines+markers+text",
    name="R²",
    line=dict(color=COLORS["blue"], width=3),
    marker=dict(size=10, color=COLORS["blue"]),
    text=[f"{v:.4f}" for v in r2_scores],
    textposition="top center",
    hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>",
), secondary_y=True)
fig_compare.update_layout(
    title="Model Comparison: Accuracy vs Error",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"]),
    barmode="group",
    legend=dict(
        bgcolor=COLORS["card"],
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    xaxis=dict(gridcolor=COLORS["grid"], title="Model"),
)
fig_compare.update_yaxes(
    title_text="MAE (hours) — lower is better",
    secondary_y=False,
    range=[0, max(mae_scores) + 15],
    gridcolor=COLORS["grid"]
)
fig_compare.update_yaxes(
    title_text="R² Score — higher is better",
    secondary_y=True,
    range=[0.65, 0.83],
    showgrid=False
)

# ── 6. ASSEMBLE HTML ──────────────────────────────────────────────────────────
print("\n[5/8] Assembling HTML dashboard...")

def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

def chart_card(fig, heading, context):
    return f"""
        <div class="chart-card">
            <div class="chart-title">{heading}</div>
            <div class="chart-context">{context}</div>
            {fig_to_html(fig)}
        </div>
    """

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC 311 Resolution Time Predictor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: {COLORS["bg"]};
            color: {COLORS["text"]};
            font-family: 'Segoe UI', sans-serif;
            padding: 24px;
        }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 8px;
            color: #ffffff;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 32px;
            font-size: 0.95rem;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: {COLORS["card"]};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #2a2d3a;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: {COLORS["blue"]};
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
            margin-top: 4px;
        }}
        .section-title {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #ffffff;
            margin: 32px 0 16px;
            padding-left: 12px;
            border-left: 4px solid {COLORS["blue"]};
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .grid-1 {{
            margin-bottom: 20px;
        }}
        .chart-card {{
            background: {COLORS["card"]};
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #2a2d3a;
        }}
        .chart-title {{
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .chart-context {{
            color: #b9becd;
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <h1>🗽 NYC 311 Resolution Time Predictor</h1>
    <p class="subtitle">
        Machine learning model trained on 589,802 complaints · Jan 1, 2026 to Mar 3, 2026 ·
        Random Forest · R² = 0.79
    </p>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">589,802</div>
            <div class="stat-label">Training Complaints</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">0.79</div>
            <div class="stat-label">R² Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">34.5h</div>
            <div class="stat-label">Mean Absolute Error</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">84%</div>
            <div class="stat-label">Driven by Problem Type</div>
        </div>
    </div>

    <div class="section-title">Model Performance</div>
    <div class="grid-2">
        {chart_card(
            fig_scatter,
            "Predicted vs Actual Time",
            "Each dot is one complaint. Left-to-right is the real time; bottom-to-top is the model estimate. Dots near the dashed line are accurate; dots far away are larger misses."
        )}
        {chart_card(
            fig_error,
            "Prediction Error Distribution",
            "This shows how far predictions are from reality. The center line at 0 means no error. More bars near 0 means the model is usually close; left side means overestimation, right side means underestimation."
        )}
    </div>
    <div class="grid-2">
        {chart_card(
            fig_importance,
            "What Drives Predictions",
            "Longer bars have more influence on predicted resolution time. Use this to see which complaint details matter most when estimating how long a case will take."
        )}
        {chart_card(
            fig_compare,
            "Model Comparison",
            "Orange bars show average prediction error in hours (lower is better). Blue line shows overall fit R² (higher is better). The strongest model has a lower orange bar and a higher blue point."
        )}
    </div>

    <div class="section-title">Heatmaps</div>
    <div class="grid-1">
        {chart_card(
            fig_heatmap1,
            "Resolution Time by Day and Hour",
            "Each cell shows the median time to close complaints for that day/hour combination. Green shades are faster; red shades are slower. This helps spot when service is typically quicker or slower."
        )}
    </div>
    <div class="grid-1">
        {chart_card(
            fig_heatmap2,
            "Resolution Time by Borough and Agency",
            "Rows are agencies and columns are boroughs. Each cell is the median resolution time for that pairing. Compare across a row to see location differences, or down a column to compare agencies."
        )}
    </div>

    <div class="section-title">Agency & Complaint Analysis</div>
    <div class="grid-2">
        {chart_card(
            fig_agency,
            "Agency Resolution Speed",
            "Shorter bars mean faster typical resolution times. This is a side-by-side view of agency performance based on median hours, so extreme outliers have less impact."
        )}
        {chart_card(
            fig_animated,
            "Daily Complaint Volume",
            "Each line tracks how many complaints each agency receives per day. Rising lines indicate busier periods; dips indicate quieter days. Use this to understand workload patterns over time."
        )}
    </div>
    <div class="grid-2">
        {chart_card(
            fig_types,
            "Slowest Complaint Types",
            "These complaint types usually take the longest to resolve (among categories with enough volume). Longer bars mean longer typical wait times."
        )}
        {chart_card(
            fig_fastest,
            "Fastest Complaint Types",
            "These complaint types are typically resolved quickest (among categories with enough volume). Shorter times can indicate standardized or easier workflows."
        )}
    </div>

</body>
</html>"""

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
print("\n[6/8] Saving dashboard...")
with open("step7_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)

print("      Saved: step7_dashboard.html")
print("      Open it in your browser!")
print("\n✓ Step 7 complete.")
engine.dispose()