import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from target_encoding import TARGET_ENCODED_COLS, fit_transform_train, apply_target_encodings


load_dotenv()

engine = create_engine(
    URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )
)

print("Loading data...")
df = pd.read_sql("SELECT * FROM nyc311_features", engine)
feature_cols = joblib.load("feature_cols.pkl")
enc_to_raw = {enc: raw for raw, enc in TARGET_ENCODED_COLS.items()}
X = df[[enc_to_raw.get(c, c) for c in feature_cols]]
y = df["log_resolution_hours"]

# Same split as step6_model.py; encodings fit on training rows only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, target_encodings = fit_transform_train(X_train, y_train)
X_train = X_train[feature_cols]
X_test  = apply_target_encodings(X_test, target_encodings)[feature_cols]
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

print("Training smaller model...")
small_model = RandomForestRegressor(
    n_estimators=25,
    max_depth=10,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
)
small_model.fit(X_train, y_train)

preds_log = small_model.predict(X_test)
r2 = r2_score(y_test, preds_log)
mae_hours = mean_absolute_error(np.expm1(y_test), np.expm1(preds_log))
print(f"R²: {r2:.4f}")
print(f"MAE (hrs): {mae_hours:.1f}")

# Record the deployed model's test score next to step 6's numbers
metrics_path = Path("model_metrics.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
metrics["deployed"] = {"model": "Random Forest (25 trees, compressed)",
                       "r2": float(r2), "mae_hours": float(mae_hours)}
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

joblib.dump(small_model, "best_model.pkl", compress=9)
joblib.dump(target_encodings, "target_encodings.pkl")
size_mb = os.path.getsize("best_model.pkl") / 1024 / 1024
print(f"New size: {size_mb:.1f} MB")

engine.dispose()
