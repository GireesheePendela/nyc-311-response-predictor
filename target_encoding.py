"""
NYC 311 Resolution Time Predictor
Target encoding helpers shared by step6_model.py and compress_model.py
- Fit on training rows only, so test rows never see their own resolution time
- Training rows themselves are encoded out-of-fold (5-fold)
- Category means are smoothed toward the global mean; unseen categories get the global mean
"""

import pandas as pd
from sklearn.model_selection import KFold

# raw categorical column → encoded feature column
TARGET_ENCODED_COLS = {
    "problem_type" : "problem_type_encoded",
    "location_type": "location_type_encoded",
}


# Smoothing strength: a category with n rows gets n / (n + SMOOTHING) of its own
# mean and the rest from the global mean, so rare categories can't memorise targets
SMOOTHING = 20


def fit_target_encodings(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    global_mean = float(y_train.mean())
    encodings = {"global_mean": global_mean}
    for raw_col in TARGET_ENCODED_COLS:
        stats = y_train.groupby(X_train[raw_col]).agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)
        encodings[raw_col] = smoothed.to_dict()
    return encodings


def fit_transform_train(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5):
    """Encode training rows out-of-fold so no row's encoding includes its own target.

    Returns the encoded training frame and the full-train encodings, which are
    what the test set and the deployed app use.
    """
    encoded = []
    for fit_idx, enc_idx in KFold(n_splits, shuffle=True, random_state=42).split(X_train):
        fold_encodings = fit_target_encodings(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        encoded.append(apply_target_encodings(X_train.iloc[enc_idx], fold_encodings))
    X_train_encoded = pd.concat(encoded).loc[X_train.index]
    return X_train_encoded, fit_target_encodings(X_train, y_train)


def apply_target_encodings(X: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    X = X.copy()
    for raw_col, enc_col in TARGET_ENCODED_COLS.items():
        X[enc_col] = (
            X[raw_col].map(encodings[raw_col])
                      .fillna(encodings["global_mean"])
                      .astype(float)
        )
    return X.drop(columns=list(TARGET_ENCODED_COLS))
