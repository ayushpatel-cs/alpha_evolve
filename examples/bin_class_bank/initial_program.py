# initial_program.py
"""
Initial program for Kaggle Playground: Binary Classification (Bank) using OpenEvolve.

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
Evolvable modeling code for the Bank binary classification task.

Contract:
    def fit_predict(train_df, test_df, train_idx, val_idx):
        Inputs:
            - train_df: pandas.DataFrame containing the full training set with a binary "y" column
            - test_df:  pandas.DataFrame containing the Kaggle test set (no "y")
            - train_idx: np.ndarray or list of row indices for training subset (within train_df index order)
            - val_idx:   np.ndarray or list of row indices for validation subset (within train_df index order)

        Returns:
            dict with keys:
                - "val_pred": 1D np.ndarray of P(y=1) for validation rows (aligned with val_idx)
                - "test_pred": 1D np.ndarray of P(y=1) for the test set (same length as test_df)
                - "model_info": optional string for debugging

Notes:
  - Score metric is ROC AUC on probability of the positive class (y=1).
  - Keep it fast and robust; evaluator may call many times with different splits.
"""

from typing import Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


def _prepare_features(
    df: pd.DataFrame, is_train: bool = True
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Basic cleaning:
      - Keep all original columns except target 'y' for features.
      - Do not drop columns aggressively; impute instead.
      - Return (X_df, y) with y as int/bool if training.
    """
    df = df.copy()

    y = None
    if is_train:
        if "y" not in df.columns:
            raise ValueError("train_df must contain binary target column 'y'")
        # Coerce to {0,1}
        y = df["y"].astype(int).values
        df = df.drop(columns=["y"])

    return df, y


def _build_pipeline(X_df: pd.DataFrame) -> Pipeline:
    """
    Baseline classifier:
      - Numeric: median impute
      - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')
      - Model: HistGradientBoostingClassifier (fast, strong, handles lots of features)
    """
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    # Drop purely identifying column if present
    for ident in ["id", "Id", "ID"]:
        if ident in numeric_features:
            numeric_features = [c for c in numeric_features if c != ident]
        if ident in categorical_features:
            categorical_features = [c for c in categorical_features if c != ident]

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # if your sklearn <1.2, replace sparse_output=False with sparse=False
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=None,
        max_iter=400,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
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
    Train on train_idx and predict P(y=1) on val_idx and the entire test_df.

    Returns:
        {
          "val_pred": np.ndarray of probabilities for validation rows (aligned with val_idx),
          "test_pred": np.ndarray of probabilities for all rows in test_df,
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

    # Predict probabilities for positive class
    val_pred = pipe.predict_proba(X_va)[:, 1].astype(float)
    test_pred = pipe.predict_proba(X_test)[:, 1].astype(float)

    # Safety: enforce [0,1]
    val_pred = np.clip(val_pred, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    info = f"pipe={pipe.__class__.__name__}, model={pipe.named_steps['model'].__class__.__name__}"

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "model_info": info,
    }
# EVOLVE-BLOCK-END


# ---- Fixed API below (do not evolve) ----

def run_bank_binary(random_state: int = 0, val_frac: float = 0.2) -> dict:
    """
    Convenience runner to test the pipeline locally (not used by evaluator in scoring).
    Trains on a random split and prints a quick ROC AUC.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=random_state, shuffle=True, stratify=train_df["y"])

    out = fit_predict(train_df, test_df, train_idx, val_idx)

    y_true = train_df.loc[val_idx, "y"].astype(int).values
    y_pred = out["val_pred"]
    try:
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        auc = float("nan")

    print(f"[Local run] ROC-AUC={auc:.5f} | {out.get('model_info','')}")
    return {"roc_auc": auc, **out}


if __name__ == "__main__":
    run_bank_binary(random_state=0, val_frac=0.2)
