"""
predict.py

Prediction pipeline for Banca Massiccia PD model.

- Uses the SAME cleaning and feature engineering as estimation.py
- Uses the training-selected feature_names from ModelArtifacts
- Uses trained XGB model + bucket_table to output CALIBRATED PDs

Usage example (in a notebook or harness):

    import pandas as pd
    from predict import predict_pd

    df_new = pd.read_csv("new_apps.csv")
    pd_hat = predict_pd(df_new)
"""

from typing import Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from estimation import (
    ModelArtifacts,
    ARTIFACTS_PATH,
    fill_financial_identities,
    load_artifacts,
    engineer_financial_ratios,
    transform_with_frozen_imputer,
)


def _sanitize_features_for_xgb(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure X has no inf / -inf and no absurdly huge values.
    XGBoost can handle NaN but not inf or insane magnitudes.
    """
    X = X.copy()

    # Replace +/- inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def apply_calibration_to_new_data(pd_raw_new, bucket_stats):
    """
    Applies a pre-trained bucket calibration mapping to a new set of raw PDs.
    """
    df_new = pd.DataFrame({"pd_raw": pd_raw_new})

    buckets_sorted = sorted(bucket_stats.index)

    def find_bucket_pd(pd_value, buckets):
        for bucket in buckets:
            if bucket.left < pd_value <= bucket.right:
                return bucket
        if pd_value == buckets[0].left:
            return buckets[0]
        return None

    df_new["assigned_bucket"] = df_new["pd_raw"].apply(lambda x: find_bucket_pd(x, buckets_sorted))

    df_new = df_new.merge(
        bucket_stats[["pd_laplace"]],
        left_on="assigned_bucket",
        right_index=True,
        how='left'
    )

    df_new["pd_calibrated_new"] = df_new["pd_laplace"].fillna(df_new["pd_raw"])

    return df_new["pd_calibrated_new"]

def data_cleaning_inference(df: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    df1 = fill_financial_identities(df)
    df2 = transform_with_frozen_imputer(
        df1,
        artifacts.factor_imputer,
        entity_col="id",
        time_col="fs_year",
    )
    df3 = fill_financial_identities(df2)
    return df3

def add_historical_growth(df_new: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    """
    Bank-style growth:
    use last-known borrower history from TRAINING artifacts,
    not from current scoring batch.
    """
    df = df_new.copy()

    # merge borrower last-known values
    hist = artifacts.history_table.reset_index()  # bring id back as column
    df = df.merge(hist, on="id", how="left")

    def safe_growth(cur, prev):
        prev = prev.replace(0, np.nan) if hasattr(prev, "replace") else prev
        return (cur - prev) / prev

    if "rev_operating" in df.columns and "rev_operating_prev" in df.columns:
        df["sales_growth"] = safe_growth(df["rev_operating"], df["rev_operating_prev"])

    if "asst_tot" in df.columns and "asst_tot_prev" in df.columns:
        df["asset_growth"] = safe_growth(df["asst_tot"], df["asst_tot_prev"])

    if "ebitda" in df.columns and "ebitda_prev" in df.columns:
        df["ebitda_growth"] = safe_growth(df["ebitda"], df["ebitda_prev"])

    return df


def predict_pd(
    df_new: pd.DataFrame,
    artifacts: Optional[ModelArtifacts] = None,
) -> pd.Series:

    if artifacts is None:
        artifacts = load_artifacts(ARTIFACTS_PATH)

    df_new_clean = data_cleaning_inference(df_new, artifacts)
    
    new_eng = engineer_financial_ratios(df_new_clean, compute_growth=False)

    new_eng = add_historical_growth(new_eng, artifacts)

    prev_cols = [c for c in new_eng.columns if c.endswith("_prev")]
    new_eng = new_eng.drop(columns=prev_cols, errors="ignore")

    # 3) ensure all selected features exist
    missing = [f for f in artifacts.feature_names if f not in new_eng.columns]
    if missing:
        raise ValueError(f"Missing engineered features in new data: {missing}")

    X_new = new_eng[artifacts.feature_names].copy()

    # 4) sanitize for XGBoost
    X_new = _sanitize_features_for_xgb(X_new)

    # 5) raw XGB scores
    raw_scores = artifacts.model.predict_proba(X_new)[:, 1]
    raw_scores = pd.Series(raw_scores, index=df_new.index, name="score_raw")

    # 6) calibrated PD via bucket_table
    pd_hat = apply_calibration_to_new_data(raw_scores, artifacts.bucket_stats)

    return pd_hat


if __name__ == "__main__":
    # Example placeholder; you can replace with real paths if you want to test from CLI.
    # import pandas as pd
    #df_holdout = pd.read_csv("train_oot.csv")
    #pd_hat = predict_pd(df_holdout)
    #pd_hat.to_csv("predictions.csv")
    pass


