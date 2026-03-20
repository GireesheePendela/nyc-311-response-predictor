import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


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
X = df[feature_cols]
y = df["log_resolution_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training smaller model...")
small_model = RandomForestRegressor(
    n_estimators=25,
    max_depth=10,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
)
small_model.fit(X_train, y_train)

r2 = r2_score(y_test, small_model.predict(X_test))
print(f"R²: {r2:.4f}")

joblib.dump(small_model, "best_model.pkl", compress=9)
size_mb = os.path.getsize("best_model.pkl") / 1024 / 1024
print(f"New size: {size_mb:.1f} MB")

engine.dispose()
