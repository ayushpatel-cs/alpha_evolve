# initial_program.py
"""
Initial program for Kaggle Playground: Predict Calorie Expenditure using OpenEvolve.

The evaluator will import this file and call fit_predict(train_df, test_df, train_idx, val_idx)
to get validation predictions (for scoring) and test predictions (for convenience).

Only the code between the EVOLVE markers will be edited by OpenEvolve.
Everything else should remain stable so the evaluator can call it reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# EVOLVE-BLOCK-START
"""
Evolvable modeling code for the Calorie Expenditure task.

Contract:
    def fit_predict(train_df, test_df, train_idx, val_idx):
        Inputs:
            - train_df: pandas.DataFrame containing the full training set with a "Calories" column
            - test_df:  pandas.DataFrame containing the Kaggle test set (no "Calories")
            - train_idx: np.ndarray or list of row indices for training subset (within train_df index order)
            - val_idx:   np.ndarray or list of row indices for validation subset (within train_df index order)

        Returns:
            dict with keys:
                - "val_pred": 1D np.ndarray of predictions for validation rows (aligned with val_idx)
                - "test_pred": 1D np.ndarray of predictions for the test set (same length as test_df)
                - "model_info": optional string for debugging

Notes:
  - Competition metric is RMSLE (RMSE on log1p target), so we train on log1p(Calories) and inverse-transform with expm1.
  - Prefer GPU training via XGBoost when available; otherwise fall back gracefully.
"""

from typing import Dict, Any, Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor


def _prepare_features(
    df: pd.DataFrame, is_train: bool = True
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Basic cleaning:
      - Leave all original columns except the target for features.
      - Don't drop columns aggressively; impute instead.
      - Return (X_df, y) with y = log1p(Calories) if training.
    """
    df = df.copy()

    y = None
    if is_train:
        if "Calories" not in df.columns:
            raise ValueError("train_df must contain 'Calories'")
        if (df["Calories"] < 0).any():
            raise ValueError("Found negative Calories; RMSLE requires non-negative targets.")
        # Target transform for RMSLE
        y = np.log1p(df["Calories"].values.astype(float))
        df = df.drop(columns=["Calories"])

    return df, y


def _split_features(X_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric/categorical columns and drop obvious ID-like columns."""
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    # Drop common identifiers
    drop_names = {"id", "Id", "ID", "Unnamed: 0"}
    numeric_features = [c for c in numeric_features if c not in drop_names]
    categorical_features = [c for c in categorical_features if c not in drop_names]

    # Drop index-like numeric columns (0..n-1 or 1..n with no repeats)
    def _is_index_like(col: pd.Series) -> bool:
        s = col.dropna().astype(float)
        if s.empty:
            return False
        # Strictly increasing integers with step 1 and either min==0 or min==1
        if not np.all(np.equal(np.mod(s, 1), 0)):
            return False
        s_sorted = np.sort(s.values)
        diffs = np.diff(s_sorted)
        if not np.all(diffs == 1):
            return False
        return s_sorted[0] in (0, 1)

    to_drop = [c for c in numeric_features if _is_index_like(X_df[c])]
    if to_drop:
        numeric_features = [c for c in numeric_features if c not in to_drop]

    return numeric_features, categorical_features


def _build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer with median impute for numeric and OHE for categoricals (dense to support HGB)."""
    num_cols, cat_cols = _split_features(X_df)

    numeric_tf = SimpleImputer(strategy="median")

    # sklearn compatibility: sparse_output (>=1.2) vs sparse (<1.2)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_tf = ohe

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # ensure dense output if any transformer yields dense
    )
    return preprocessor


def _make_xgb_regressor(use_gpu: bool = True):
    """
    Build an XGBoost regressor (returned unfit).
    - If GPU requested and supported, use CUDA.
    - If GPU requested but unsupported, caller should catch and retry with CPU.
    - If XGBoost import fails entirely, caller should fall back to HGB.
    """
    from xgboost import XGBRegressor
    import inspect

    common = dict(
        n_estimators=2000,          # combined with early stopping
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
        verbosity=0,
    )

    sig = inspect.signature(XGBRegressor.__init__)
    has_device = "device" in sig.parameters

    if use_gpu:
        if has_device:  # XGBoost >= 2.0
            return XGBRegressor(
                **common,
                tree_method="hist",
                predictor="auto",
                device="cuda",
            )
        else:  # older XGB
            return XGBRegressor(
                **common,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
            )
    else:
        if has_device:
            return XGBRegressor(
                **common,
                tree_method="hist",
                predictor="auto",
                device="cpu",
            )
        else:
            return XGBRegressor(
                **common,
                tree_method="hist",
                predictor="auto",
            )


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict[str, Any]:
    """
    Train on train_idx and predict on val_idx and the entire test_df.

    Returns:
        {
          "val_pred": np.ndarray of predicted Calories for validation rows (aligned with val_idx),
          "test_pred": np.ndarray of predicted Calories for all rows in test_df,
          "model_info": str (optional)
        }
    """
    # Split & prepare targets
    X_all, y_all = _prepare_features(train_df, is_train=True)
    X_test, _ = _prepare_features(test_df, is_train=False)

    X_tr = X_all.iloc[train_idx]
    y_tr = y_all[train_idx]
    X_va = X_all.iloc[val_idx]
    y_va = y_all[val_idx]

    # Build and fit preprocessor on TRAIN ONLY to avoid leakage
    preprocessor = _build_preprocessor(X_all)
    X_tr_t = preprocessor.fit_transform(X_tr)
    X_va_t = preprocessor.transform(X_va)
    X_te_t = preprocessor.transform(X_test)

    # Try XGBoost with GPU, then CPU, then fallback to HGB
    model_info = ""
    val_log_pred = None
    test_log_pred = None

    try:
        # GPU XGBoost
        xgb = _make_xgb_regressor(use_gpu=True)
        # early stopping via eval_set on transformed data
        xgb.fit(
            X_tr_t,
            y_tr,
            eval_set=[(X_va_t, y_va)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            verbose=False,
        )
        val_log_pred = xgb.predict(X_va_t)
        test_log_pred = xgb.predict(X_te_t)
        model_info = f"model=XGBRegressor(gpu), best_ntree_limit={getattr(xgb, 'best_ntree_limit', None)}"
    except Exception as e_gpu:
        try:
            # CPU XGBoost
            xgb = _make_xgb_regressor(use_gpu=False)
            xgb.fit(
                X_tr_t,
                y_tr,
                eval_set=[(X_va_t, y_va)],
                eval_metric="rmse",
                early_stopping_rounds=100,
                verbose=False,
            )
            val_log_pred = xgb.predict(X_va_t)
            test_log_pred = xgb.predict(X_te_t)
            model_info = f"model=XGBRegressor(cpu), best_ntree_limit={getattr(xgb, 'best_ntree_limit', None)}"
        except Exception as e_cpu:
            # Fallback: HistGradientBoostingRegressor (CPU-only)
            hgb = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.06,
                max_depth=None,
                max_iter=600,
                l2_regularization=0.0,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            )
            hgb.fit(X_tr_t, y_tr)
            val_log_pred = hgb.predict(X_va_t)
            test_log_pred = hgb.predict(X_te_t)
            model_info = "model=HistGradientBoostingRegressor"

    # Back-transform from log space and clip non-negatives
    val_pred = np.expm1(val_log_pred).astype(float)
    test_pred = np.expm1(test_log_pred).astype(float)
    val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
    test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

    # Include brief info on feature dims
    try:
        n_features = X_tr_t.shape[1]
    except Exception:
        n_features = None
    if n_features is not None:
        model_info += f", n_features={n_features}"

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "model_info": model_info,
    }
# EVOLVE-BLOCK-END


# ---- Fixed API below (do not evolve) ----

def run_calories(random_state: int = 0, val_frac: float = 0.2) -> dict:
    """
    Convenience runner to test the pipeline locally (not used by evaluator in scoring).
    Trains on a random split and prints a quick RMSLE.
    """
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=random_state, shuffle=True)

    out = fit_predict(train_df, test_df, train_idx, val_idx)

    # RMSLE (RMSE on log1p)
    y_true = np.log1p(train_df.loc[val_idx, "Calories"].values.astype(float))
    y_pred = np.log1p(out["val_pred"])
    rmsle = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    print(f"[Local run] RMSLE={rmsle:.5f} | {out.get('model_info','')}")
    return {"rmsle": rmsle, **out}


if __name__ == "__main__":
    run_calories(random_state=0, val_frac=0.2)
