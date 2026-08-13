"""
estimation.py

Full estimation pipeline for Banca Massiccia PD model.
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

TARGET_COL = "default_next_year"
N_BUCKETS = 50
ARTIFACTS_PATH = "artifacts/model_artifacts.pkl"

# ---------------------------------------------------------
# IMPUTER CONTAINER
# ---------------------------------------------------------

@dataclass
class FactorImputerArtifacts:
    char_cols: List[str]
    means: pd.Series
    stds: pd.Series
    loadings_by_year: Dict[int, np.ndarray]  # fs_year -> Lambda_t (L x K)
    n_factors: int
    ridge_gamma: float
    standardize: bool = True


# ---------------------------------------------------------
# ARTIFACT CONTAINER
# ---------------------------------------------------------

@dataclass
class ModelArtifacts:
    feature_names: List[str]        # selected features from feature selection
    model: XGBClassifier            # trained XGB model
    bucket_stats: pd.DataFrame      # calibration table: score bucket -> PD
    laplace_alpha: float
    n_buckets: int
    min_per_bucket: int
    history_table: pd.DataFrame
    factor_imputer: FactorImputerArtifacts

# ---------------------------------------------------------
# TARGET VARIABLE ENGINEERING
# ---------------------------------------------------------
def build_lagged_pd_target(df):
    df = df.copy()
    df["stmt_date"] = pd.to_datetime(df["stmt_date"], errors="coerce")
    df["def_date"] = pd.to_datetime(df["def_date"], errors="coerce")
    # based on when they would need to have their financial statements ready. 
    # 6 months for firms required to publish a balance sheet, 10 for others based on tax filing deadlines. 
    lag_map = {
        "SPA": 6,
        "SRL": 6,
        "SRU": 6,
        "SRS": 6,
        "SAA": 10,
        "SAU": 10

    }

    df["lag_months"] = df["legal_struct"].map(lag_map).fillna(6).astype(int)

    # Compute availability and horizon 
    df["available_date"] = df["stmt_date"] + pd.to_timedelta(df["lag_months"] * 30, unit="D")
    df["horizon_end"] = df["available_date"] + pd.DateOffset(years=1)

    # default within [available_date, horizon_end) 
    df["default_next_year"] = np.where(
        (df["def_date"].notna()) &
        (df["def_date"] >= df["available_date"]) &
        (df["def_date"] < df["horizon_end"]),
        1, 0
    )
    return df



# ---------------------------------------------------------
# 3. DATA CLEANING  
# ---------------------------------------------------------

def fill_financial_identities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    op_col = None
    if "prof_operating" in df.columns:
        op_col = "prof_operating"
    elif "prof_operations" in df.columns:
        op_col = "prof_operations"

    rev_col = None
    if "rev_operations" in df.columns:
        rev_col = "rev_operations"
    elif "rev_operating" in df.columns:
        rev_col = "rev_operating"

    
    # FIXED ASSETS
    #    asst_tot = asst_intang_fixed + asst_tang_fixed + asst_fixed_fin + asst_current
    has_fixed_parts = {"asst_intang_fixed", "asst_tang_fixed", "asst_fixed_fin"}.issubset(df.columns)
    has_current = "asst_current" in df.columns
    has_asst_tot = "asst_tot" in df.columns

    fixed_assets = None
    if has_fixed_parts:
        fixed_assets = (
            df["asst_intang_fixed"] +
            df["asst_tang_fixed"] +
            df["asst_fixed_fin"]
        )

    # fill components from asst_tot when only that one is missing
    if has_asst_tot and has_fixed_parts and has_current:
        # intang
        mask = df["asst_intang_fixed"].isna() & \
               df["asst_tang_fixed"].notna() & \
               df["asst_fixed_fin"].notna() & \
               df["asst_current"].notna()
        df.loc[mask, "asst_intang_fixed"] = (
            df.loc[mask, "asst_tot"]
            - df.loc[mask, "asst_tang_fixed"]
            - df.loc[mask, "asst_fixed_fin"]
            - df.loc[mask, "asst_current"]
        )
        # tang
        mask = df["asst_tang_fixed"].isna() & \
               df["asst_intang_fixed"].notna() & \
               df["asst_fixed_fin"].notna() & \
               df["asst_current"].notna()
        df.loc[mask, "asst_tang_fixed"] = (
            df.loc[mask, "asst_tot"]
            - df.loc[mask, "asst_intang_fixed"]
            - df.loc[mask, "asst_fixed_fin"]
            - df.loc[mask, "asst_current"]
        )
        # fixed_fin
        mask = df["asst_fixed_fin"].isna() & \
               df["asst_intang_fixed"].notna() & \
               df["asst_tang_fixed"].notna() & \
               df["asst_current"].notna()
        df.loc[mask, "asst_fixed_fin"] = (
            df.loc[mask, "asst_tot"]
            - df.loc[mask, "asst_intang_fixed"]
            - df.loc[mask, "asst_tang_fixed"]
            - df.loc[mask, "asst_current"]
        )
        # current assets
        mask = df["asst_current"].isna() & \
               df["asst_intang_fixed"].notna() & \
               df["asst_tang_fixed"].notna() & \
               df["asst_fixed_fin"].notna()
        df.loc[mask, "asst_current"] = (
            df.loc[mask, "asst_tot"]
            - df.loc[mask, "asst_intang_fixed"]
            - df.loc[mask, "asst_tang_fixed"]
            - df.loc[mask, "asst_fixed_fin"]
        )

    # Recompute helper fixed_assets after possible fills
    if has_fixed_parts:
        fixed_assets = (
            df["asst_intang_fixed"] +
            df["asst_tang_fixed"] +
            df["asst_fixed_fin"]
        )


    # SHORT-TERM & LONG-TERM DEBT
    # --- SHORT-TERM DEBT: debt_st = debt_bank_st + debt_fin_st ---
    # --- SHORT-TERM DEBT: debt_st = debt_bank_st + debt_fin_st ---
    if {"debt_bank_st", "debt_fin_st", "debt_st"}.issubset(df.columns):
   
    # Fill missing bank short-term when total + fin short-term exist
        mask = df["debt_bank_st"].isna() & df["debt_st"].notna() & df["debt_fin_st"].notna()
        df.loc[mask, "debt_bank_st"] = df.loc[mask, "debt_st"] - df.loc[mask, "debt_fin_st"]

    # Fill missing fin short-term when total + bank short-term exist
        mask = df["debt_fin_st"].isna() & df["debt_st"].notna() & df["debt_bank_st"].notna()
        df.loc[mask, "debt_fin_st"] = df.loc[mask, "debt_st"] - df.loc[mask, "debt_bank_st"]

    #  Now recompute the component sum
        st_sum = df["debt_bank_st"] + df["debt_fin_st"]

    # If aggregate exists and components exist and aggregate < sum(components),
    #    treat aggregate as wrong and null it out (your original rule).
        inconsistent_st = (
           df["debt_st"].notna() &
           df["debt_bank_st"].notna() &
           df["debt_fin_st"].notna() &
           (df["debt_st"] < st_sum)
        )
        df.loc[inconsistent_st, "debt_st"] = np.nan

    #  Fill missing aggregate from the (now more complete) component sum
        mask = df["debt_st"].isna() & df["debt_bank_st"].notna() & df["debt_fin_st"].notna()
        df.loc[mask, "debt_st"] = st_sum[mask]


# --- LONG-TERM DEBT: debt_lt = debt_bank_lt + debt_fin_lt ---
    if {"debt_bank_lt", "debt_fin_lt", "debt_lt"}.issubset(df.columns):
    # Fill missing components from total where possible

    # Fill missing bank long-term when total + fin long-term exist
       mask = df["debt_bank_lt"].isna() & df["debt_lt"].notna() & df["debt_fin_lt"].notna()
       df.loc[mask, "debt_bank_lt"] = df.loc[mask, "debt_lt"] - df.loc[mask, "debt_fin_lt"]

    # Fill missing fin long-term when total + bank long-term exist
       mask = df["debt_fin_lt"].isna() & df["debt_lt"].notna() & df["debt_bank_lt"].notna()
       df.loc[mask, "debt_fin_lt"] = df.loc[mask, "debt_lt"] - df.loc[mask, "debt_bank_lt"]

    #  Recompute component sum
       lt_sum = df["debt_bank_lt"] + df["debt_fin_lt"]

    # Apply your inconsistency rule: if aggregate < sum(components),
    #    null out aggregate.
       inconsistent_lt = (
           df["debt_lt"].notna() &
           df["debt_bank_lt"].notna() &
           df["debt_fin_lt"].notna() &
           (df["debt_lt"] < lt_sum)
          )
       df.loc[inconsistent_lt, "debt_lt"] = np.nan

    # Fill missing aggregate from (now more complete) components
       mask = df["debt_lt"].isna() & df["debt_bank_lt"].notna() & df["debt_fin_lt"].notna()
       df.loc[mask, "debt_lt"] = lt_sum[mask]



    # NET PROFIT IDENTITY
    # profit = operating_profit + financial income + extraordinary income (matched for over 900k rows in train)

    if op_col is not None and \
       {"inc_financing", "inc_extraord", "profit"}.issubset(df.columns):

        #  Use components to infer profit
        implied_profit = (
            df[op_col] +
            df["inc_extraord"] +
            df["inc_financing"]
        )

        # If profit exists and all components exist and it's inconsistent, null profit
        mask_inconsistent_profit = (
            df["profit"].notna() &
            df[op_col].notna() &
            df["inc_extraord"].notna() &
            df["inc_financing"].notna() &
            ~np.isclose(df["profit"], implied_profit, rtol=1e-6, atol=1e-2)
        )
        df.loc[mask_inconsistent_profit, "profit"] = np.nan

        # Fill profit where missing and all components known
        df["profit"] = df["profit"].fillna(implied_profit)

        # Go the other way: use profit + 2 known components to fill the 3rd
        components = [op_col, "inc_financing", "inc_extraord"]

        for comp in components:
            others = [c for c in components if c != comp]

            # Need profit and both "other" components, but this one is missing
            mask = df[comp].isna() & df["profit"].notna()
            for o in others:
                mask &= df[o].notna()

            if mask.any():
                others_sum = df.loc[mask, others].sum(axis=1)
                df.loc[mask, comp] = df.loc[mask, "profit"] - others_sum

    #  ROA and ROE
    #     roa = profit / asst_tot
    #     roe = profit / eqty_tot
    # 
    if {"roa", "profit", "asst_tot"}.issubset(df.columns):
        denom = df["asst_tot"].replace(0, np.nan)
        implied_roa = df["profit"] / denom

        # If roa exists and components exist and inconsistent, null roa
        mask_inconsistent_roa = (
            df["roa"].notna() &
            df["profit"].notna() &
            denom.notna() &
            ~np.isclose(df["roa"], implied_roa, rtol=1e-6, atol=1e-6)
        )
        df.loc[mask_inconsistent_roa, "roa"] = np.nan

        # Fill roa
        df["roa"] = df["roa"].fillna(implied_roa)

        # Then fill profit where missing but roa & asst_tot exist
        mask = df["profit"].isna() & df["roa"].notna() & df["asst_tot"].notna()
        df.loc[mask, "profit"] = df.loc[mask, "roa"] * df.loc[mask, "asst_tot"]

    if {"roe", "profit", "eqty_tot"}.issubset(df.columns):
        denom = df["eqty_tot"].replace(0, np.nan)
        implied_roe = df["profit"] / denom

        mask_inconsistent_roe = (
            df["roe"].notna() &
            df["profit"].notna() &
            denom.notna() &
            ~np.isclose(df["roe"], implied_roe, rtol=1e-6, atol=1e-6)
        )
        df.loc[mask_inconsistent_roe, "roe"] = np.nan

        # Fill roe
        df["roe"] = df["roe"].fillna(implied_roe)

    

    # MARGIN_FIN (Equity - Fixed assets)
    #     margin_fin = eqty_tot - fixed_assets

    if "margin_fin" in df.columns and "eqty_tot" in df.columns and fixed_assets is not None:
        implied_margin = df["eqty_tot"] - fixed_assets

        # If margin_fin exists and eqty & fixed exist and inconsistent, null margin_fin
        mask_inconsistent_margin = (
            df["margin_fin"].notna() &
            df["eqty_tot"].notna() &
            (fixed_assets.notna()) &
            ~np.isclose(df["margin_fin"], implied_margin, rtol=1e-6, atol=1e-2)
        )
        df.loc[mask_inconsistent_margin, "margin_fin"] = np.nan

        # Fill margin_fin
        df["margin_fin"] = df["margin_fin"].fillna(implied_margin)

        # Also: fill eqty_tot if margin_fin and fixed_assets are known and eqty_tot missing
        mask = df["eqty_tot"].isna() & df["margin_fin"].notna() & fixed_assets.notna()
        df.loc[mask, "eqty_tot"] = df.loc[mask, "margin_fin"] + fixed_assets[mask]
    
    # DAYS RECEIVABLES
    #     days_rec = 365 * AR / rev
    if {"AR", "days_rec"}.issubset(df.columns) and rev_col is not None:
        rev_nonzero = df[rev_col].replace(0, np.nan)
        implied_days = 365 * df["AR"] / rev_nonzero

        # If days_rec exists and AR & rev exist and inconsistent, null days_rec
        mask_inconsistent_days = (
            df["days_rec"].notna() &
            df["AR"].notna() &
            rev_nonzero.notna() &
            ~np.isclose(df["days_rec"], implied_days, rtol=1e-4, atol=1e-1)
        )
        df.loc[mask_inconsistent_days, "days_rec"] = np.nan

        # Fill days_rec
        mask = df["days_rec"].isna() & df["AR"].notna() & df[rev_col].notna()
        df.loc[mask, "days_rec"] = 365 * df.loc[mask, "AR"] / df.loc[mask, rev_col].replace(0, np.nan)

        # Fill AR
        mask = df["AR"].isna() & df["days_rec"].notna() & df[rev_col].notna()
        df.loc[mask, "AR"] = df.loc[mask, "days_rec"] * df.loc[mask, rev_col] / 365

        # Fill revenue
        mask = df[rev_col].isna() & df["AR"].notna() & df["days_rec"].notna()
        df.loc[mask, rev_col] = 365 * df.loc[mask, "AR"] / df.loc[mask, "days_rec"].replace(0, np.nan)

    return df

# FALL BACK METHOD via Cross-Sectional Approximate Factor Models
# BASED ON “Missing Financial Data,” by Svetlana Bryzgalova, Sven Lerner, Martin Lettau, Markus Pelger.
def _pairwise_cov_matrix(C_t: np.ndarray, mask_t: np.ndarray) -> np.ndarray:
    
    N_t, L = C_t.shape
    Sigma = np.zeros((L, L), dtype=float)

    for l in range(L):
        x_l = C_t[:, l]
        m_l = mask_t[:, l]
        for p in range(l, L):
            x_p = C_t[:, p]
            m_p = mask_t[:, p]
            both = m_l & m_p
            n_both = both.sum()
            if n_both == 0:
                cov_lp = 0.0
            else:
         
                cov_lp = np.mean(x_l[both] * x_p[both])
            Sigma[l, p] = cov_lp
            Sigma[p, l] = cov_lp

    return Sigma


def _top_k_eigenvectors(Sigma: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
   
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.clip(eigvals, 0.0, None)  # no negative eigenvalues from noise
    return eigvals[:k], eigvecs[:, :k]


def _estimate_loadings(Sigma: np.ndarray, n_factors: int) -> np.ndarray:
   
    eigvals_k, eigvecs_k = _top_k_eigenvectors(Sigma, n_factors)
    sqrt_D = np.sqrt(eigvals_k)
    Lambda_t = eigvecs_k * sqrt_D[np.newaxis, :]   # (L, K)
    return Lambda_t


def _estimate_factors_ridge(
    C_t: np.ndarray,
    mask_t: np.ndarray,
    Lambda_t: np.ndarray,
    ridge_gamma: float,
) -> np.ndarray:
    
    N_t, L = C_t.shape
    _, K = Lambda_t.shape
    F_t = np.zeros((N_t, K), dtype=float)

    inv_L = 1.0 / float(L)
    I_K = np.eye(K)

    for i in range(N_t):
        obs_idx = np.where(mask_t[i])[0]
        if obs_idx.size == 0:
            # no observed chars for this firm at this time
            continue

        X_i = Lambda_t[obs_idx, :]       
        y_i = C_t[i, obs_idx]             
        XtX = X_i.T @ X_i * inv_L
        Xty = X_i.T @ y_i * inv_L

        A = XtX + ridge_gamma * I_K
        try:
            F_i = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            F_i = np.linalg.pinv(A) @ Xty

        F_t[i] = F_i

    return F_t

def factor_impute_panel(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    char_cols: List[str],
    n_factors: int = 5,
    ridge_gamma: float = 1.0,
    standardize: bool = True,
    copy: bool = True,
) -> pd.DataFrame:

    if copy:
        df = df.copy()

    char_df = df[char_cols]

    if standardize:
        means = char_df.mean(axis=0)
        stds = char_df.std(axis=0).replace(0, 1.0)
        char_std = (char_df - means) / stds
    else:
        means = pd.Series(0.0, index=char_cols)
        stds = pd.Series(1.0, index=char_cols)
        char_std = char_df

    char_imputed_std = char_std.copy()

    #  model per year
    times = np.sort(df[time_col].dropna().unique())

    for t in times:
        idx_t = (df[time_col] == t)
        if idx_t.sum() == 0:
            continue

        C_t_df = char_std.loc[idx_t, char_cols]
        C_t = C_t_df.to_numpy()          # (N_t, L)
        mask_t = ~np.isnan(C_t)

        if mask_t.sum() == 0 or C_t.shape[0] == 0:
            continue

        # For covariance estimation, fill NaNs with 0 but use mask_t to ignore them
        C_t_filled = np.where(mask_t, C_t, 0.0)

        Sigma_hat = _pairwise_cov_matrix(C_t_filled, mask_t)

    
        Lambda_t = _estimate_loadings(Sigma_hat, n_factors)

       
        F_t = _estimate_factors_ridge(C_t_filled, mask_t, Lambda_t, ridge_gamma)

        C_hat_t = F_t @ Lambda_t.T   # (N_t, L)

        # Replace missing with common component
        C_t_imputed = C_t.copy()
        missing_mask = ~mask_t
        C_t_imputed[missing_mask] = C_hat_t[missing_mask]

        # write back
        char_imputed_std.loc[idx_t, char_cols] = C_t_imputed

    # Undo standardization
    char_imputed = char_imputed_std * stds[char_cols] + means[char_cols]

    df_imputed = df.copy()
    df_imputed[char_cols] = char_imputed

    return df_imputed

def fit_factor_imputer(
    df_train: pd.DataFrame,
    entity_col: str,
    time_col: str,
    char_cols: List[str],
    n_factors: int = 5,
    ridge_gamma: float = 1.0,
    standardize: bool = True,
) -> FactorImputerArtifacts:
    """
    Fit factor imputer on TRAINING panel only.
    Stores means/stds + frozen Lambda_t per year.
    """
    df = df_train.copy()
    char_cols = [c for c in char_cols if c in df.columns]

    # compute train means/stds ON TRAINING ONLY
    char_df = df[char_cols]
    if standardize:
        means = char_df.mean(axis=0)
        stds = char_df.std(axis=0).replace(0, 1.0)
        char_std = (char_df - means) / stds
    else:
        means = pd.Series(0.0, index=char_cols)
        stds = pd.Series(1.0, index=char_cols)
        char_std = char_df

    loadings_by_year = {}

    times = np.sort(df[time_col].dropna().unique())
    for t in times:
        idx_t = (df[time_col] == t)
        if idx_t.sum() == 0:
            continue

        C_t_df = char_std.loc[idx_t, char_cols]
        C_t = C_t_df.to_numpy()  # (N_t, L)
        mask_t = ~np.isnan(C_t)

        if mask_t.sum() == 0 or C_t.shape[0] == 0:
            continue

        C_t_filled = np.where(mask_t, C_t, 0.0)

        Sigma_hat = _pairwise_cov_matrix(C_t_filled, mask_t)

        Lambda_t = _estimate_loadings(Sigma_hat, n_factors)  # (L, K)
        loadings_by_year[int(t)] = Lambda_t

    return FactorImputerArtifacts(
        char_cols=char_cols,
        means=means,
        stds=stds,
        loadings_by_year=loadings_by_year,
        n_factors=n_factors,
        ridge_gamma=ridge_gamma,
        standardize=standardize,
    )

def transform_with_frozen_imputer(
    df_new: pd.DataFrame,
    imputer: FactorImputerArtifacts,
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Apply frozen factor imputer (no refit).
    Uses train means/stds + frozen Lambda_t per year.
    """
    df = df_new.copy()
    cols = imputer.char_cols
    cols = [c for c in cols if c in df.columns]

    char_df = df[cols]
    if imputer.standardize:
        char_std = (char_df - imputer.means[cols]) / imputer.stds[cols]
    else:
        char_std = char_df.copy()

    char_imputed_std = char_std.copy()

    times = np.sort(df[time_col].dropna().unique())
    for t in times:
        idx_t = (df[time_col] == t)
        if idx_t.sum() == 0:
            continue

        # choose Lambda_t for this year, else fallback to closest / any
        t_int = int(t)
        if t_int in imputer.loadings_by_year:
            Lambda_t = imputer.loadings_by_year[t_int]
        else:
            # fallback: use latest available year’s Lambda
            Lambda_t = imputer.loadings_by_year[max(imputer.loadings_by_year.keys())]

        C_t_df = char_std.loc[idx_t, cols]
        C_t = C_t_df.to_numpy()            # (N_t, L)
        mask_t = ~np.isnan(C_t)

        if mask_t.sum() == 0 or C_t.shape[0] == 0:
            continue

        C_t_filled = np.where(mask_t, C_t, 0.0)

        # ESTIMATE FACTORS ONLY (no new Sigma / Lambda fitting)
        F_t = _estimate_factors_ridge(
            C_t=C_t_filled,
            mask_t=mask_t,
            Lambda_t=Lambda_t,
            ridge_gamma=imputer.ridge_gamma
        )

        C_hat_t = F_t @ Lambda_t.T

        C_t_imputed = C_t.copy()
        missing_mask = ~mask_t
        C_t_imputed[missing_mask] = C_hat_t[missing_mask]

        char_imputed_std.loc[idx_t, cols] = C_t_imputed

    # unstandardize using TRAIN means/stds
    if imputer.standardize:
        char_imputed = char_imputed_std * imputer.stds[cols] + imputer.means[cols]
    else:
        char_imputed = char_imputed_std

    df[cols] = char_imputed
    return df



def impute_banca_massiccia(df: pd.DataFrame,
                           n_factors: int = 6,
                           ridge_gamma: float = 0.5,
                           standardize: bool = True) -> pd.DataFrame:

    char_cols = [
        "asst_intang_fixed",
        "asst_tang_fixed",
        "asst_fixed_fin",
        "asst_current",
        "AR",
        "cash_and_equiv",
        "asst_tot",
        "eqty_tot",
        "eqty_corp_family_tot",
        "liab_lt",
        "liab_lt_emp",
        "debt_bank_st",
        "debt_bank_lt",
        "debt_fin_st",
        "debt_fin_lt",
        "AP_st",
        "AP_lt",
        "debt_st",
        "debt_lt",
        "rev_operating",
        "COGS",
        "prof_operations",
        "goodwill",
        "inc_financing",
        "exp_financing",
        "prof_financing",
        "inc_extraord",
        "taxes",
        "profit",
        "days_rec",
        "ebitda",
        "roa",
        "roe",
        "wc_net",
        "margin_fin",
        "cf_operations",
    ]

    # Keep only columns that actually exist in df (in case some are missing)
    char_cols = [c for c in char_cols if c in df.columns]

    return factor_impute_panel(
        df=df,
        entity_col="id",
        time_col="fs_year",
        char_cols=char_cols,
        n_factors=n_factors,
        ridge_gamma=ridge_gamma,
        standardize=standardize,
        copy=True,
    )

## WRAPPING ALL CLEANING 
def data_cleaning_with_frozen_imputer(df: pd.DataFrame, frozen_imputer: FactorImputerArtifacts) -> pd.DataFrame:
    df1 = fill_financial_identities(df)
    df2 = transform_with_frozen_imputer(df1, frozen_imputer, entity_col="id", time_col="fs_year")
    df3 = fill_financial_identities(df2)
    return df3

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------

def engineer_financial_ratios(df: pd.DataFrame, compute_growth=True) -> pd.DataFrame:
    df = df.copy()
    def safe_div(num, den):
        den = den.replace(0, np.nan)
        return num / den
    # asset_flag = 0
    # cf_flag = 0
    # accurals_flag = 0
    # --- Profitability ---
    if {'roa'}.issubset(df.columns):
        df['roa'] = df['roa']
    if {'roe'}.issubset(df.columns):
        df['roe'] = df['roe']
    if {'operating_roa'}.issubset(df.columns):
        df['operating_roa'] = df['operating_roa']
    if {'ebitda_to_assets'}.issubset(df.columns):
        df['ebitda_to_assets'] = df['ebitda_to_assets']
    if {'extraord_to_assets'}.issubset(df.columns):
        df['extraord_to_assets'] = df['extraord_to_assets']
    if {"prof_operations", "asst_tot"}.issubset(df.columns):
        df["operating_roa"] = safe_div(df["prof_operations"], df["asst_tot"])
    if {"ebitda", "asst_tot"}.issubset(df.columns):
        df["ebitda_to_assets"] = safe_div(df["ebitda"], df["asst_tot"])
    #if {"inc_extraord", "asst_tot"}.issubset(df.columns):
        #df["extraord_to_assets"] = safe_div(df["inc_extraord"], df["asst_tot"])

    # --- Leverage ---
    if {"debt_lt", "debt_st", "asst_tot"}.issubset(df.columns):
        df["debt_to_assets"] = safe_div(df["debt_lt"] + df["debt_st"], df["asst_tot"])
    if {"debt_bank_lt", "debt_bank_st", "asst_tot"}.issubset(df.columns):
        df["bank_debt_ratio"] = safe_div(df["debt_bank_lt"] + df["debt_bank_st"], df["asst_tot"])
    if {"debt_lt", "debt_st", "eqty_tot"}.issubset(df.columns):
        df["debt_to_equity"] = safe_div(df["debt_lt"] + df["debt_st"], df["eqty_tot"])

    # --- Liquidity ---
    if {"cash_and_equiv", "asst_tot"}.issubset(df.columns):
        df["cash_ratio_assets"] = safe_div(df["cash_and_equiv"], df["asst_tot"])
    if {"wc_net", "asst_tot"}.issubset(df.columns):
        df["wc_to_assets"] = safe_div(df["wc_net"], df["asst_tot"])
    if {"asst_current", "asst_tot"}.issubset(df.columns):
        df["current_to_asst"]= safe_div(df["asst_current"], df["asst_tot"])
    if {'cash_and_equiv', 'cf_operations'}.issubset(df.columns):
        df["cash_to_cf"]= safe_div(df["cash_and_equiv"], df["cf_operations"])
    
    # --- Size ---
    if "asst_tot" in df.columns:
        df["assets"] = np.log1p(df["asst_tot"])


    # --- Tangibility ---
    if {"asst_tang_fixed", "asst_tot"}.issubset(df.columns):
        df["tangibility"] = safe_div(df["asst_tang_fixed"], df["asst_tot"])
    if { "asst_intang_fixed", "asst_tot"}.issubset(df.columns):
        df["intang_assets_to_total"] = safe_div(
            df["asst_intang_fixed"], df["asst_tot"]
        )

    # --- Cashflow ---
    if {"cf_operations", "asst_tot"}.issubset(df.columns):
        df["cf_to_assets"] = safe_div(df["cf_operations"], df["asst_tot"])
    if {"cf_operations", "eqty_tot"}.issubset(df.columns):
        df["cf_to_equity"] = safe_div(df["cf_operations"], df["eqty_tot"])
    if {"cf_operations", "debt_lt", "debt_st"}.issubset(df.columns):
        df["cf_to_debt"] = safe_div(df["cf_operations"], df["debt_lt"] + df["debt_st"])

    # else:
    #   cf_flag += 1

    #-- Growth ratios --
    if compute_growth:
        _orig_idx = df.index
        # sort columns that are only needed for growth else incorrect calculation like 2023 to 2019 can happen
        sort_cols = ["id", "fs_year"] if "fs_year" in df.columns else ["id"]
        df_sorted = df.sort_values(sort_cols).copy()
        if {"rev_operating"}.issubset(df_sorted.columns):
            df_sorted["sales_growth"] = df_sorted.groupby("id")["rev_operating"].pct_change()
        if {"asst_tot"}.issubset(df_sorted.columns):
            df_sorted["asset_growth"] = df_sorted.groupby("id")["asst_tot"].pct_change()
        if {"ebitda"}.issubset(df_sorted.columns):
            df_sorted["ebitda_growth"] = df_sorted.groupby("id")["ebitda"].pct_change()
        
        growth_cols = ["sales_growth", "asset_growth", "ebitda_growth"]

        for col in growth_cols:
            if col in df_sorted.columns:
                df[col] = df_sorted[col].reindex(_orig_idx)

    return df
# ---------------------------------------------------------
# 5. Winsorization FOR GROWTH ONLY 
# ---------------------------------------------------------

def compute_winsor_caps(s: pd.Series, lower_q=0.01, upper_q=0.99) -> Optional[Tuple[float, float]]:
    s = s.replace([np.inf, -np.inf], np.nan)
    finite_vals = s[np.isfinite(s)]
    if finite_vals.empty:
        return None
    lower_cap = finite_vals.quantile(lower_q)
    upper_cap = finite_vals.quantile(upper_q)
    return lower_cap, upper_cap

def apply_winsor_with_caps(s: pd.Series, caps):
    if caps is None:
        return s
    lower_cap, upper_cap = caps
    s = s.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    finite_mask = np.isfinite(s)
    s.loc[finite_mask] = s.loc[finite_mask].clip(lower_cap, upper_cap)
    
    return s
# ---------------------------------------------------------
# 6. HELPER TO COMPUTE VIF
# ---------------------------------------------------------

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    #imputing ratiosonly for VIF so it runs successfully
    # not imputing engineer ratio since say total equity is 0, it means something, so just imputing so vif runs successfully
    # imputing via FALL BACK METHOD via Cross-Sectional Approximate Factor Models
    # BASED ON “Missing Financial Data,” by Svetlana Bryzgalova, Sven Lerner, Martin Lettau, Markus Pelger.
    impute_vif = impute_banca_massiccia_for_vif(df)
    if "fs_year" in impute_vif.columns:
        impute_vif = impute_vif.drop(columns=["fs_year"])
    X = impute_vif.copy()
    X = X.assign(const=1)  

    vif_data = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append((col, vif))
    return pd.DataFrame(vif_data, columns=["feature", "vif"])


def impute_banca_massiccia_for_vif(df: pd.DataFrame,
                           n_factors: int = 6,
                           ridge_gamma: float = 0.5,
                           standardize: bool = True) -> pd.DataFrame:

    char_cols = [
        "asst_intang_fixed",
        "asst_tang_fixed",
        "asst_fixed_fin",
        "asst_current",
        "AR",
        "cash_and_equiv",
        "asst_tot",
        "eqty_tot",
        "eqty_corp_family_tot",
        "liab_lt",
        "liab_lt_emp",
        "debt_bank_st",
        "debt_bank_lt",
        "debt_fin_st",
        "debt_fin_lt",
        "AP_st",
        "AP_lt",
        "debt_st",
        "debt_lt",
        "rev_operating",
        "COGS",
        "prof_operations",
        "goodwill",
        "inc_financing",
        "exp_financing",
        "prof_financing",
        "inc_extraord",
        "taxes",
        "profit",
        "days_rec",
        "ebitda",
        "roa",
        "roe",
        "wc_net",
        "margin_fin",
        "cf_operations",
        'debt_to_equity', ##added engineered columns for vif
        'wc_to_assets', 
        'assets', 
        'tangibility', 
        'cf_to_debt' , 
        "asset_growth",
        "sales_growth"
    ]
    # Keep only columns that actually exist in df (in case some are missing)
    char_cols = [c for c in char_cols if c in df.columns]

    return factor_impute_panel(
        df=df,
        entity_col="id",
        time_col="fs_year",
        char_cols=char_cols,
        n_factors=n_factors,
        ridge_gamma=ridge_gamma,
        standardize=standardize,
        copy=True,
    )


# ---------------------------------------------------------
# 6. MODEL CALIBRATION
# ---------------------------------------------------------
def bucket_calibration_laplace(
    pd_raw,
    y,
    n_buckets=20,
    min_per_bucket=300,
    laplace_alpha=0.5,   
):
   
    # Prepare Data 
    df = pd.DataFrame({"pd_raw": pd_raw, "y": y}).dropna()
    n = len(df)

    if n == 0:
        raise ValueError("No data after dropna().")

    # Adjust number of buckets based on minimum bucket size 
    max_feasible_buckets = max(1, n // min_per_bucket)
    n_buckets_eff = min(n_buckets, max_feasible_buckets)

    # If extremely small dataset
    if n_buckets_eff < 2:
        n_buckets_eff = 1

    # Create quantile buckets 
    try:
        df["bucket"] = pd.qcut(df["pd_raw"], q=n_buckets_eff, duplicates="drop")
    except ValueError:
        # Not enough variation → assign one bucket
        df["bucket"] = 0

    # Compute bucket-level stats
    bucket_stats = df.groupby("bucket").agg(
        count=("y", "size"),
        defaults=("y", "sum"),
        mean_pd_raw=("pd_raw", "mean")
    ).sort_values("mean_pd_raw")

    # --- Step 4: Laplace smoothing 
    
 
    c = bucket_stats["count"].astype(float)
    d = bucket_stats["defaults"].astype(float)
    bucket_stats["pd_laplace"] = (d + laplace_alpha) / (c + 2 * laplace_alpha)

    # --- Step 5: Map calibrated PD back to each observation ---
    df = df.merge(bucket_stats[["pd_laplace"]], left_on="bucket", right_index=True)
    df.rename(columns={"pd_laplace": "pd_calibrated"}, inplace=True)

    return df, bucket_stats

def fit_bucket_calibration_on_calib_data(
    model: XGBClassifier,
    df_calib_eng: pd.DataFrame,
    feature_names: list,
    laplace_alpha: float = 0.5,
    n_buckets: int = 20,
    min_per_bucket: int = 300,
):
   
    # X, y for calibration
    X_calib = df_calib_eng[feature_names]
    y_calib = df_calib_eng["default_next_year"].astype(int)

    # raw PDs from trained model
    pd_raw = model.predict_proba(X_calib)[:, 1]

    # fit bucket calibration
    _, bucket_stats = bucket_calibration_laplace(
        pd_raw=pd_raw,
        y=y_calib,
        n_buckets=n_buckets,
        min_per_bucket=min_per_bucket,
        laplace_alpha=laplace_alpha,
    )
    return bucket_stats


# ---------------------------------------------------------
# 7. Split data for CALIBRATION
# ---------------------------------------------------------

def split_train_oos(df_train, oos_frac=0.35, random_state=42):
  
    df = df_train.copy()

    rng = np.random.RandomState(random_state)

    # Unique firms in the training period
    unique_firms = df["id"].unique()
    rng.shuffle(unique_firms)

    # number of firms to move to OOS
    n_oos = int(len(unique_firms) * oos_frac)

    # sets
    oos_firms   = set(unique_firms[:n_oos])
    train_firms = set(unique_firms[n_oos:])

    # subsets
    df_oos = df[df["id"].isin(oos_firms)]
    df_train_final = df[df["id"].isin(train_firms)]

    return df_train_final, df_oos



# ---------------------------------------------------------
# 8. MODEL TRAINING (INCLUDING CALIBRATION)
# ---------------------------------------------------------

def train_full_pipeline(df_raw: pd.DataFrame) -> ModelArtifacts:
    """
    Master function:

    Input: raw labeled dataset with financials + default info
    Output: ModelArtifacts (selected features, trained XGB, bucket calibration)
    """

    # 1) Target engineering
    df = build_lagged_pd_target(df_raw)
    df_train,df_calib = split_train_oos(df)

    char_cols = [
        "asst_intang_fixed","asst_tang_fixed","asst_fixed_fin","asst_current","AR",
        "cash_and_equiv","asst_tot","eqty_tot","eqty_corp_family_tot","liab_lt",
        "liab_lt_emp","debt_bank_st","debt_bank_lt","debt_fin_st","debt_fin_lt",
        "AP_st","AP_lt","debt_st","debt_lt","rev_operating","COGS","prof_operations",
        "goodwill","inc_financing","exp_financing","prof_financing","inc_extraord",
        "taxes","profit","days_rec","ebitda","roa","roe","wc_net","margin_fin",
        "cf_operations"
    ]

    char_cols = [c for c in char_cols if c in df_train.columns]

    frozen_imputer = fit_factor_imputer(
        df_train=df_train,
        entity_col="id",
        time_col="fs_year",
        char_cols=char_cols,
        n_factors=6,
        ridge_gamma=0.5,
        standardize=True,
    )


    history_table = (
        df_train.sort_values(["id", "fs_year"])
        .groupby("id", as_index=False)
        .tail(1)[["id", "fs_year", "rev_operating", "asst_tot", "ebitda"]]
        .rename(columns={
            "fs_year": "fs_year_prev",
            "rev_operating": "rev_operating_prev",
            "asst_tot": "asst_tot_prev",
            "ebitda": "ebitda_prev",
        })
        .set_index("id")
    )

    # 2) Cleaning
    df_train = data_cleaning_with_frozen_imputer(df_train, frozen_imputer)

    # 4) Feature engineering
    train_eng = engineer_financial_ratios(df_train)
    
    # winsorize growth ratios
    growth_cols = [
        "sales_growth",
        "asset_growth",
        "ebitda_growth",
        ]
    for col in growth_cols:
        if col in train_eng.columns:
            caps = compute_winsor_caps(train_eng[col])
            train_eng[col] = apply_winsor_with_caps(train_eng[col], caps)
    features = ['fs_year', 'roa', 'debt_to_equity', 'wc_to_assets', 'assets', 'tangibility', 'cf_to_debt' , "sales_growth"]
    
    vif = compute_vif(train_eng[features])

    if vif[vif['vif'] > 5].shape[0] > 0:
        print('Features with VIF > 5:', vif[vif['vif'] > 5]['feature'].tolist())
        print('try another combination')

    #Selected By running VIF     
    selected_features = ['roa', 'debt_to_equity', 'wc_to_assets', 'assets', 'tangibility', 'cf_to_debt' , "sales_growth"]
    TARGET_COL= ['default_next_year']  
    X_train = train_eng[selected_features]
    y_train = train_eng[TARGET_COL].astype(int)

    # model training
    y_arr = y_train.to_numpy()
    pos = y_arr.sum()
    neg = len(y_arr) - pos
    scale_pos_weight = (neg / max(1, pos)) if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    # Calibration 
    df_calib = data_cleaning_with_frozen_imputer(df_calib, frozen_imputer)
    calib_eng = engineer_financial_ratios(df_calib)

    for col in growth_cols:
        if col in calib_eng.columns:
            caps = compute_winsor_caps(train_eng[col])
            calib_eng[col] = apply_winsor_with_caps(calib_eng[col], caps)

    #Hyperparameters for calibration
    laplace_alpha = 0.5
    n_buckets = 50
    min_per_bucket = 300

    bucket_stats = fit_bucket_calibration_on_calib_data(
        model=xgb,
        df_calib_eng=calib_eng,
        feature_names=selected_features,
        laplace_alpha=laplace_alpha,
        n_buckets=n_buckets,
        min_per_bucket=min_per_bucket,
    )

    # 8) Pack artifacts (model + calibration)
    artifacts = ModelArtifacts(
        feature_names=selected_features,
        model=xgb,
        bucket_stats=bucket_stats,
        laplace_alpha=laplace_alpha,
        n_buckets=n_buckets,
        min_per_bucket=min_per_bucket,
        history_table=history_table,
        factor_imputer=frozen_imputer
    )

    
    return artifacts


# ---------------------------------------------------------
# 9. SAVE / LOAD ARTIFACTS
# ---------------------------------------------------------

def save_artifacts(artifacts: ModelArtifacts, path: str = ARTIFACTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)


def load_artifacts(path: str = ARTIFACTS_PATH) -> ModelArtifacts:
    with open(path, "rb") as f:
        artifacts: ModelArtifacts = pickle.load(f)
    return artifacts


if __name__ == "__main__":
    # Example usage:
    #df_raw = pd.read_csv("train-2025.csv")
    #artifacts = train_full_pipeline(df_raw)
    #save_artifacts(artifacts)
    pass
