"""
NYC 311 Resolution Time Predictor
Step 6: Modeling
- Loads nyc311_features from PostgreSQL
- Trains Baseline, Random Forest, XGBoost; picks the best on a validation split
- Refits the winner and scores it once on the held-out test set (RMSE, MAE, R²)
- Saves best model, target encodings and model_metrics.json to disk
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
import json
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

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from target_encoding import TARGET_ENCODED_COLS, fit_transform_train, apply_target_encodings

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

RAW_CAT_COLS = list(TARGET_ENCODED_COLS)

feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
X = df[feature_cols].copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

non_numeric_cols = [col for col in X.columns
                    if col not in RAW_CAT_COLS and not pd.api.types.is_numeric_dtype(X[col])]
if non_numeric_cols:
    X.drop(columns=non_numeric_cols, inplace=True)
    print(f"      Dropped non-numeric features: {non_numeric_cols}")

X[RAW_CAT_COLS] = X[RAW_CAT_COLS].fillna("UNKNOWN")
X = X.replace([np.inf, -np.inf], np.nan)

valid_mask = y.notna()
if not valid_mask.all():
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    print(f"      Dropped {(~valid_mask).sum():,} rows with null target")

print(f"      Target   : {TARGET}")

# ── 3. TRAIN / VALIDATION / TEST SPLIT ────────────────────────────────────────
print("\n[3/6] Splitting into train/validation/test sets (60/20/20)...")
# Test set is held out until the very end: models are compared on validation,
# and only the chosen model is scored on test, once.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)
print(f"      Train      : {len(X_fit):,} rows")
print(f"      Validation : {len(X_val):,} rows")
print(f"      Test       : {len(X_test):,} rows")


def encode_split(X_tr, y_tr, X_eval):
    """Target-encode and impute using training rows only (no eval target leaks in)."""
    X_tr, encodings = fit_transform_train(X_tr, y_tr)
    medians = X_tr.median(numeric_only=True)
    X_eval  = apply_target_encodings(X_eval, encodings).fillna(medians)
    return X_tr.fillna(medians), X_eval, encodings


X_fit, X_val, _ = encode_split(X_fit, y_fit, X_val)
feature_cols = X_fit.columns.tolist()
print(f"      Features : {len(feature_cols)}")
print(f"      Sample features: {feature_cols[:8]}")

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

    return {"r2": float(r2), "mae_hours": float(mae_hours), "rmse_hours": float(rmse_hours)}

# ── 4. CANDIDATE MODELS (compared on validation) ─────────────────────────────
print("\n[4/6] Training models, scoring on validation set...")
print("─" * 50)

candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 20,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = 42
    ),
    "XGBoost": XGBRegressor(
        n_estimators      = 300,
        max_depth         = 7,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        verbosity         = 0
    ),
}

val_metrics = {
    name: evaluate(f"{name} (validation)", clone(model), X_fit, y_fit, X_val, y_val)
    for name, model in candidates.items()
}

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  MODEL COMPARISON SUMMARY (validation)")
print("=" * 50)
print(f"  {'Model':<30} {'R²':>8} {'MAE (hrs)':>12}")
print(f"  {'─'*30} {'─'*8} {'─'*12}")
for name, m in val_metrics.items():
    print(f"  {name:<30} {m['r2']:>8.4f} {m['mae_hours']:>12.1f}")

best_name = max(val_metrics, key=lambda name: val_metrics[name]["r2"])
print(f"\n  Best model on validation: {best_name}")

# ── 5. REFIT BEST MODEL ON TRAIN+VALIDATION, SCORE ON TEST ONCE ──────────────
print("\n[5/6] Refitting best model on train+validation, scoring on held-out test...")
X_train_full, X_test, target_encodings = encode_split(X_train_full, y_train_full, X_test)
X_train_full = X_train_full[feature_cols]
X_test       = X_test[feature_cols]

best_model   = clone(candidates[best_name])
test_metrics = evaluate(f"{best_name} (test)", best_model, X_train_full, y_train_full, X_test, y_test)

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    print(f"\n--- Feature Importance ({best_name}) ---")
    importances = pd.Series(
        best_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("Top 15 most important features:")
    for feat, imp in importances.head(15).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

# ── 6. SAVE BEST MODEL, ENCODINGS, METRICS ────────────────────────────────────
print(f"\n[6/6] Saving best model, feature list, encodings and metrics...")
joblib.dump(best_model,   "best_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(target_encodings, "target_encodings.pkl")

# Single source of truth for reported numbers (read by step7_visualization.py)
metrics = {
    "rows"       : int(len(X)),
    "best_model" : best_name,
    "validation" : val_metrics,
    "test"       : test_metrics,
}
Path("model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"  Saved: best_model.pkl")
print(f"  Saved: feature_cols.pkl")
print(f"  Saved: target_encodings.pkl")
print(f"  Saved: model_metrics.json")

# ── SAVE PREDICTIONS TO POSTGRES ─────────────────────────────────────────────
print("\n--- Saving test-set predictions to PostgreSQL ---")
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