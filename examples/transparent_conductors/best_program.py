# initial_program.py
"""
Initial program for Kaggle House Prices (Ames) using OpenEvolve.

The evaluator will import this file and call `fit_predict(train_df, test_df, train_idx, val_idx)`
to get validation predictions (for scoring) and test predictions (for convenience).

Only the code between the EVOLVE markers will be edited by OpenEvolve.
Everything else should remain stable so the evaluator can call it reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# EVOLVE-BLOCK-START
"""
Evolvable modeling code for the House Prices task.

Contract:
    def fit_predict(train_df, test_df, train_idx, val_idx):
        Inputs:
            - train_df: pandas.DataFrame containing the full training set with a "SalePrice" column
            - test_df:  pandas.DataFrame containing the Kaggle test set (no "SalePrice")
            - train_idx: np.ndarray or list of row indices for training subset (within train_df index order)
            - val_idx:   np.ndarray or list of row indices for validation subset (within train_df index order)

        Returns:
            dict with keys:
                - "val_pred": 1D np.ndarray of predictions for validation rows (aligned with val_idx)
                - "test_pred": 1D np.ndarray of predictions for the test set (same length as test_df)
                - "model_info": optional string for debugging

Notes:
  - Score metric is RMSE on log1p(SalePrice), so we train on log1p targets and inverse-transform with expm1.
  - Keep it reasonably fast and robust; evaluator may call many times with different splits.
"""

from typing import Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor  # Use XGBoost for potentially better performance

def _prepare_features(
    df: pd.DataFrame, is_train: bool = True
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Basic cleaning:
      - Leave all original columns except SalePrice for features.
      - Don't drop columns aggressively; impute instead.
      - Return (X_df, y) with y = log1p(SalePrice) if training.
    """
    df = df.copy()

    y = None
    if is_train:
        if "SalePrice" not in df.columns:
            raise ValueError("train_df must contain 'SalePrice'")
        # Target transform for Kaggle metric (RMSE on log targets)
        y = np.log1p(df["SalePrice"].values.astype(float))
        df = df.drop(columns=["SalePrice"])

    # Keep Id around if present; model won’t use it (we’ll drop explicitly)
    # Avoid leakage by not adding engineered features from target.

    return df, y


def _build_pipeline(X_df: pd.DataFrame) -> Pipeline:
    """
    Build a reasonably strong, fast baseline:
      - Numeric: median impute, scaling
      - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')
      - Model: XGBRegressor
    """
    # Identify columns by dtype (robust if categories come in as object)
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    # Drop purely non-informative identifiers if present
    for ident in ["Id"]:
        if ident in numeric_features:
            numeric_features = [c for c in numeric_features if c != ident]
        if ident in categorical_features:
            categorical_features = [c for c in categorical_features if c != ident]

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="passthrough",
        n_jobs=None,  # keep simple/portable
    )

    model = XGBRegressor(
        objective='reg:squarederror',  # Specify the objective function
        n_estimators=500,  # Number of boosting rounds
        learning_rate=0.05,  # Step size shrinkage
        max_depth=5,  # Maximum depth of a tree
        subsample=0.8,  # Subsample ratio of the training instance
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        random_state=42,  # Random number seed for reproducibility
        n_jobs=None,
        reg_alpha=0.01,
        reg_lambda=0.01
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe


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
          "val_pred": np.ndarray of predicted SalePrice for validation rows (aligned with val_idx),
          "test_pred": np.ndarray of predicted SalePrice for all rows in test_df,
          "model_info": str (optional)
        }
    """
    # Split & prepare
    X_all, y_all = _prepare_features(train_df, is_train=True)
    X_test, _ = _prepare_features(test_df, is_train=False)

    # Build pipeline fit on train subset
    pipe = _build_pipeline(X_all)

    X_tr = X_all.iloc[train_idx]
    y_tr = y_all[train_idx]
    X_va = X_all.iloc[val_idx]

    pipe.fit(X_tr, y_tr)

    # Predict log targets, invert with expm1, clip to non-negative
    val_log_pred = pipe.predict(X_va)
    test_log_pred = pipe.predict(X_test)

    val_pred = np.expm1(val_log_pred).astype(float)
    test_pred = np.expm1(test_log_pred).astype(float)

    # Safety: ensure no negatives due to numerical noise
    val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
    test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

    info = f"pipe={pipe.__class__.__name__}, model={pipe.named_steps['model'].__class__.__name__}"

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "model_info": info,
    }
# EVOLVE-BLOCK-END


# ---- Fixed API below (do not evolve) ----

def run_houseprice(random_state: int = 0, val_frac: float = 0.2) -> dict:
    """
    Convenience runner to test the pipeline locally (not used by evaluator in scoring).
    Trains on a random split and prints a quick score.
    """
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=random_state, shuffle=True)

    out = fit_predict(train_df, test_df, train_idx, val_idx)

    # Compute RMSE on log1p for quick feedback
    y_true = np.log1p(train_df.loc[val_idx, "SalePrice"].values.astype(float))
    y_pred = np.log1p(out["val_pred"])
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    print(f"[Local run] log-RMSE={rmse:.5f} | {out.get('model_info','')}")
    return {"rmse": rmse, **out}


if __name__ == "__main__":
    run_houseprice(random_state=0, val_frac=0.2)