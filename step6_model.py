"""
NYC 311 Resolution Time Predictor
Step 6: Modeling
- Loads nyc311_features from PostgreSQL
- Trains Baseline, Random Forest, XGBoost
- Evaluates with RMSE, MAE, R²
- Saves best model to disk
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
        "  python step6_model.py"
    ) from exc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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
        "Create/update .env in the project root before running step6_model.py"
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
print("STEP 6: Modeling")
print("=" * 60)

# ── 1. LOAD FEATURES ──────────────────────────────────────────────────────────
print("\n[1/6] Loading nyc311_features from PostgreSQL...")
df = pd.read_sql("SELECT * FROM nyc311_features", engine)
print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── 2. DEFINE FEATURES & TARGET ───────────────────────────────────────────────
print("\n[2/6] Defining features and target...")

# Drop non-feature columns
DROP_COLS = ["unique_key", "resolution_hours"]
TARGET    = "log_resolution_hours"

feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
X = df[feature_cols].copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
if non_numeric_cols:
    X.drop(columns=non_numeric_cols, inplace=True)
    print(f"      Dropped non-numeric features: {non_numeric_cols}")

X = X.replace([np.inf, -np.inf], np.nan)
if X.isnull().any().any():
    X = X.fillna(X.median(numeric_only=True))

valid_mask = y.notna()
if not valid_mask.all():
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    print(f"      Dropped {(~valid_mask).sum():,} rows with null target")

feature_cols = X.columns.tolist()

print(f"      Features : {len(feature_cols)}")
print(f"      Target   : {TARGET}")
print(f"      Sample features: {feature_cols[:8]}")

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
print("\n[3/6] Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train : {len(X_train):,} rows")
print(f"      Test  : {len(X_test):,} rows")

# ── HELPER: EVALUATE MODEL ────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds_log = model.predict(X_te)

    # Metrics on log scale
    rmse_log = np.sqrt(mean_squared_error(y_te, preds_log))
    mae_log  = mean_absolute_error(y_te, preds_log)
    r2       = r2_score(y_te, preds_log)

    # Convert back to hours for interpretability
    preds_hours  = np.expm1(preds_log)
    actual_hours = np.expm1(y_te)
    mae_hours    = mean_absolute_error(actual_hours, preds_hours)
    rmse_hours   = np.sqrt(mean_squared_error(actual_hours, preds_hours))

    print(f"\n  {name}")
    print(f"    R²          : {r2:.4f}  (1.0 = perfect, 0 = no better than mean)")
    print(f"    RMSE (log)  : {rmse_log:.4f}")
    print(f"    MAE  (log)  : {mae_log:.4f}")
    print(f"    MAE  (hrs)  : {mae_hours:.1f} hours  ← most interpretable")
    print(f"    RMSE (hrs)  : {rmse_hours:.1f} hours")

    return model, r2, mae_hours

# ── 4. BASELINE: LINEAR REGRESSION ───────────────────────────────────────────
print("\n[4/6] Training models...")
print("─" * 50)

lr_model, lr_r2, lr_mae = evaluate(
    "Baseline — Linear Regression",
    LinearRegression(),
    X_train, y_train, X_test, y_test
)

# ── 5. RANDOM FOREST ─────────────────────────────────────────────────────────
rf_model, rf_r2, rf_mae = evaluate(
    "Random Forest (200 trees)",
    RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 20,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = 42
    ),
    X_train, y_train, X_test, y_test
)

# ── 6. XGBOOST ────────────────────────────────────────────────────────────────
xgb_model, xgb_r2, xgb_mae = evaluate(
    "XGBoost",
    XGBRegressor(
        n_estimators      = 300,
        max_depth         = 7,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        verbosity         = 0
    ),
    X_train, y_train, X_test, y_test
)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"  {'Model':<30} {'R²':>8} {'MAE (hrs)':>12}")
print(f"  {'─'*30} {'─'*8} {'─'*12}")
print(f"  {'Linear Regression':<30} {lr_r2:>8.4f} {lr_mae:>12.1f}")
print(f"  {'Random Forest':<30} {rf_r2:>8.4f} {rf_mae:>12.1f}")
print(f"  {'XGBoost':<30} {xgb_r2:>8.4f} {xgb_mae:>12.1f}")

# ── PICK BEST MODEL ───────────────────────────────────────────────────────────
best_name, best_model = max(
    [("Linear Regression", lr_model),
     ("Random Forest",     rf_model),
     ("XGBoost",           xgb_model)],
    key=lambda x: r2_score(
        y_test,
        x[1].predict(X_test)
    )
)
print(f"\n  Best model: {best_name}")

# ── FEATURE IMPORTANCE (Random Forest) ───────────────────────────────────────
print("\n--- Feature Importance (Random Forest) ---")
importances = pd.Series(
    rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("Top 15 most important features:")
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── SAVE BEST MODEL ───────────────────────────────────────────────────────────
print(f"\n--- Saving best model and feature list ---")
joblib.dump(best_model,   "best_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print(f"  Saved: best_model.pkl")
print(f"  Saved: feature_cols.pkl")

# ── SAVE PREDICTIONS TO POSTGRES ─────────────────────────────────────────────
print("\n--- Saving predictions to PostgreSQL ---")
preds_log    = best_model.predict(X_test)
preds_hours  = np.expm1(preds_log)
actual_hours = np.expm1(y_test)

results_df = pd.DataFrame({
    "unique_key"       : df.loc[X_test.index, "unique_key"].values,
    "actual_hours"     : actual_hours.values,
    "predicted_hours"  : preds_hours,
    "error_hours"      : (preds_hours - actual_hours.values)
})

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS nyc311_predictions"))
    conn.commit()

results_df.to_sql(
    "nyc311_predictions",
    engine,
    if_exists = "replace",
    index     = False,
    chunksize = 5000,
    method    = "multi"
)
print(f"  Saved {len(results_df):,} predictions to nyc311_predictions table")

print("\n✓ Step 6 complete.")
engine.dispose()