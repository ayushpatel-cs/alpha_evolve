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
from sklearn.ensemble import HistGradientBoostingRegressor

def _prepare_features(
    df: pd.DataFrame, is_train: bool = True
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Basic cleaning and feature prep.
    """
    df = df.copy()

    y = None
    if is_train:
        if "SalePrice" not in df.columns:
            raise ValueError("train_df must contain 'SalePrice'")
        y = np.log1p(df["SalePrice"].values.astype(float))
        df = df.drop(columns=["SalePrice"])

    return df, y


def _build_pipeline(X_df: pd.DataFrame) -> Pipeline:
    """
    Build pipeline with preprocessing and model.
    """
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    for ident in ["Id"]:
        if ident in numeric_features:
            numeric_features = [c for c in numeric_features if c != ident]
        if ident in categorical_features:
            categorical_features = [c for c in categorical_features if c != ident]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # Add scaling
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # or 'passthrough'
        n_jobs=None,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_depth=6,
        max_iter=700,
        l2_regularization=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit and predict using the pipeline.
    """
    X_all, y_all = _prepare_features(train_df, is_train=True)
    X_test, _ = _prepare_features(test_df, is_train=False)

    pipeline = _build_pipeline(X_all)

    X_tr = X_all.iloc[train_idx]
    y_tr = y_all[train_idx]
    X_va = X_all.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)

    val_log_pred = pipeline.predict(X_va)
    test_log_pred = pipeline.predict(X_test)

    val_pred = np.expm1(val_log_pred).astype(float)
    test_pred = np.expm1(test_log_pred).astype(float)

    val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
    test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

    info = f"pipe={pipeline.__class__.__name__}, model={pipeline.named_steps['model'].__class__.__name__}"

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "model_info": info,
    }
# EVOLVE-BLOCK-END


# ---- Fixed API below (do not evolve) ----

def run_houseprice(random_state: int = 0, val_frac: float = 0.2) -> dict:
    """
    Convenience runner.
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