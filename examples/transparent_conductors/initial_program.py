# initial_program.py
"""
Initial program for Kaggle Nomad2018: Predicting Transparent Conductors using OpenEvolve.

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
Evolvable modeling code for the Transparent Conductors task.

Contract:
    def fit_predict(train_df, test_df, train_idx, val_idx):
        Inputs:
            - train_df: pandas.DataFrame containing the full training set with
              targets ["formation_energy_ev_natom", "bandgap_energy_ev"]
            - test_df:  pandas.DataFrame containing the Kaggle test set (no targets)
            - train_idx: np.ndarray or list of row indices for training subset
            - val_idx:   np.ndarray or list of row indices for validation subset

        Returns:
            dict with keys:
                - "val_pred": 2D np.ndarray of shape (len(val_idx), 2)
                              columns in order: [formation_energy_ev_natom, bandgap_energy_ev]
                - "test_pred": 2D np.ndarray of shape (len(test_df), 2) with the same column order
                - "model_info": optional string for debugging

Notes:
  - Competition metric = mean column-wise RMSLE, so a common trick is to model log1p(target).
  - We'll train two independent regressors via MultiOutputRegressor to keep things sturdy.
  - Keep it fast and robust; evaluator re-splits every run.
"""

from typing import Dict, Any, Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS: List[str] = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def _safe_ohe(**kwargs):
    """Return a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        # Older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)


def _prepare_features(
    df: pd.DataFrame, is_train: bool = True
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Basic cleaning:
      - For train: separate Y (in target order) and drop them from X.
      - For both: keep all columns except targets and 'id' as features.
      - Let the pipeline impute & encode.
    """
    df = df.copy()

    y = None
    if is_train:
        for t in TARGETS:
            if t not in df.columns:
                raise ValueError(f"train_df must contain target column '{t}'")
        y = df[TARGETS].astype(float).values
        df = df.drop(columns=TARGETS)

    # Drop obvious identifiers if present
    for ident in ["id", "Id", "ID"]:
        if ident in df.columns:
            df = df.drop(columns=[ident])

    return df, y


def _build_pipeline(X_df: pd.DataFrame) -> Pipeline:
    """
    Baseline, fast, evolution-friendly:
      - Numeric: median impute
      - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')
      - Regressor: MultiOutput(HistGradientBoostingRegressor) on log1p-transformed Y
        (handled outside the pipeline)
    """
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    # Avoid non-informative ids if they slipped through
    for ident in ["id", "Id", "ID"]:
        if ident in numeric_features:
            numeric_features.remove(ident)
        if ident in categorical_features:
            categorical_features.remove(ident)

    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ohe", _safe_ohe())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ],
        remainder="drop",
    )

    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=None,
        max_iter=500,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )

    reg = MultiOutputRegressor(base)

    pipe = Pipeline(steps=[("prep", preprocessor), ("reg", reg)])
    return pipe


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict[str, Any]:
    """
    Train on train_idx, predict on val_idx and the entire test_df.

    Returns:
        {
          "val_pred": np.ndarray (n_val, 2) in TARGETS order,
          "test_pred": np.ndarray (n_test, 2) in TARGETS order,
          "model_info": str
        }
    """
    # Split & prepare
    X_all, Y_all = _prepare_features(train_df, is_train=True)
    X_test, _ = _prepare_features(test_df, is_train=False)

    # Train/val split views
    X_tr = X_all.iloc[train_idx]
    Y_tr = Y_all[train_idx]
    X_va = X_all.iloc[val_idx]

    # Train on log1p(targets) for RMSLE-friendliness; guard with clip(0)
    Y_tr_log = np.log1p(np.clip(Y_tr, a_min=0.0, a_max=None))

    pipe = _build_pipeline(X_all)
    pipe.fit(X_tr, Y_tr_log)

    # Predict log-targets, invert to original space, clip to non-negative
    val_log_pred = pipe.predict(X_va)
    test_log_pred = pipe.predict(X_test)

    val_pred = np.expm1(val_log_pred).astype(float)
    test_pred = np.expm1(test_log_pred).astype(float)

    val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
    test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

    info = (
        f"pipeline={pipe.__class__.__name__}, "
        f"reg={pipe.named_steps['reg'].__class__.__name__}("
        f"base={pipe.named_steps['reg'].estimator.__class__.__name__})"
    )

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "model_info": info,
    }
# EVOLVE-BLOCK-END


# ---- Fixed API below (do not evolve) ----

def run_nomad(random_state: int = 0, val_frac: float = 0.2) -> dict:
    """
    Convenience runner to test the pipeline locally.
    Computes mean RMSLE across both targets on a random split.
    """
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    idx = np.arange(len(train_df))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, random_state=random_state, shuffle=True
    )

    out = fit_predict(train_df, test_df, train_idx, val_idx)

    y_true = train_df.loc[val_idx, TARGETS].astype(float).values
    y_pred = np.asarray(out["val_pred"], dtype=float)

    def _rmsle_colwise(y_t, y_p):
        y_t = np.clip(y_t, 0.0, None)
        y_p = np.clip(y_p, 0.0, None)
        errs = []
        for j in range(y_t.shape[1]):
            errs.append(np.sqrt(np.mean((np.log1p(y_t[:, j]) - np.log1p(y_p[:, j])) ** 2)))
        return float(np.mean(errs))

    rmsle = _rmsle_colwise(y_true, y_pred)
    print(f"[Local run] mean RMSLE={rmsle:.6f} | {out.get('model_info','')}")
    return {"rmsle": rmsle, **out}


if __name__ == "__main__":
    run_nomad(random_state=0, val_frac=0.2)
