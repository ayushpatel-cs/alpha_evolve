# evaluator.py
"""
Evaluator for Kaggle Playground (Bank) Binary Classification using OpenEvolve.

Design:
- Every evaluation run samples a fresh random validation subset from train.csv.
- We score with ROC AUC on probability of the positive class (y=1), matching Kaggle.
- We run the candidate program in a separate Python process with a timeout.
- We do not trust the program's self-reported score; we compute the score here
  from the returned validation probabilities to keep evaluation honest.

Interface expected by OpenEvolve:
    - evaluate(program_path) -> metrics dict
    - evaluate_stage1(program_path) -> quick smoke test (smaller data fraction)
    - evaluate_stage2(program_path) -> full evaluation (delegates to evaluate)

Returns (example):
{
    "roc_auc": 0.842,
    "val_frac": 0.2,
    "n_train": 30000,
    "n_val": 7500,
    "eval_time": 3.21,
    "validity": 1.0,
    "combined_score": 0.842
}
"""

import os
import sys
import time
import pickle
import traceback
import tempfile
import subprocess
import numpy as np
import pandas as pd


class TimeoutError(Exception):
    pass


def run_with_timeout(program_path: str, timeout_seconds: int, seed: int, val_frac: float, sample_rows: int | None = None):
    """
    Execute the candidate program in a clean subprocess.
    We pass the indices split; the program returns validation probabilities aligned to val_idx.

    All loading/splitting happen inside the temp script, which calls candidate.fit_predict(...) and pickles results.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import os, sys, pickle, numpy as np, pandas as pd, traceback
sys.path.insert(0, os.path.dirname('{program_path}'))

SEED = {seed}
VAL_FRAC = {val_frac}
SAMPLE_ROWS = {repr(sample_rows)}

np.random.seed(SEED)

try:
    # Import candidate
    import importlib.util as _util
    _spec = _util.spec_from_file_location("candidate", '{program_path}')
    candidate = _util.module_from_spec(_spec)
    _spec.loader.exec_module(candidate)

    # Load data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    if SAMPLE_ROWS is not None:
        train_df = train_df.sample(n=min(SAMPLE_ROWS, len(train_df)), random_state=SEED)

    if "y" not in train_df.columns:
        raise ValueError("train.csv must contain binary target column 'y'")

    # Make stratified split when possible
    idx = np.arange(len(train_df))
    n_val = max(1, int(len(train_df) * VAL_FRAC))
    # simple stratification: shuffle separately by class if both classes exist
    if train_df["y"].nunique() == 2:
        pos_idx = np.where(train_df["y"].values == 1)[0]
        neg_idx = np.where(train_df["y"].values == 0)[0]
        rng = np.random.default_rng(SEED)
        rng.shuffle(pos_idx); rng.shuffle(neg_idx)
        n_val_pos = max(1, int(len(pos_idx) * VAL_FRAC))
        n_val_neg = max(1, int(len(neg_idx) * VAL_FRAC))
        val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
        rng.shuffle(val_idx)
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
    else:
        # fallback: unstratified
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    # Call evolvable function
    out = candidate.fit_predict(train_df, test_df, train_idx, val_idx)

    # Collect evaluation payload
    y_true_val = train_df.iloc[val_idx]["y"].astype(int).values
    y_pred_val = np.asarray(out.get("val_pred", []), dtype=float)

    results = {{
        "y_true_val": y_true_val,
        "y_pred_val": y_pred_val,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "model_info": out.get("model_info", ""),
        "ok": True,
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    with open('{temp_file.name}.results', 'wb') as f:
        import traceback as _tb
        pickle.dump({{"ok": False, "error": str(e), "trace": _tb.format_exc()}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        proc = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Subprocess timed out after {timeout_seconds}s")

        # Always show outputs (helpful when a candidate crashes)
        if stdout:
            print(stdout.decode(errors="ignore"))
        if stderr:
            print(stderr.decode(errors="ignore"))

        if proc.returncode != 0:
            raise RuntimeError(f"Candidate process exited with code {proc.returncode}")

        if not os.path.exists(results_path):
            raise RuntimeError("Results file missing")

        with open(results_path, "rb") as f:
            results = pickle.load(f)

        if not results.get("ok", False):
            err = results.get("error", "unknown")
            tr = results.get("trace", "")
            raise RuntimeError(f"Candidate error: {err}\n{tr}")

        return results

    finally:
        # Cleanup temp files
        for path in (temp_file_path, results_path):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


def _score_results(results: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(results["y_true_val"], dtype=int)
    y_pred = np.asarray(results["y_pred_val"], dtype=float)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "error": f"Shape mismatch or empty preds: y_true={y_true.shape}, y_pred={y_pred.shape}",
        }

    # If validation contains only one class, AUC is undefined.
    if len(np.unique(y_true)) < 2:
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "error": "Validation set contains a single class; cannot compute ROC AUC.",
        }

    # Clip to [0,1] and compute AUC
    y_pred = np.clip(y_pred, 0.0, 1.0)
    auc = float(roc_auc_score(y_true, y_pred))

    return {
        "validity": 1.0,
        "roc_auc": auc,
        "combined_score": auc,  # higher is better, matches Kaggle metric
    }


def evaluate(program_path: str) -> dict:
    """
    Full evaluation.
    - Uses 20% validation split sampled randomly each run (stratified if possible).
    - Timeout generous enough for modest feature processing and boosting.
    """
    start = time.time()
    seed = int.from_bytes(os.urandom(4), "little")
    val_frac = 0.2

    try:
        results = run_with_timeout(
            program_path=program_path,
            timeout_seconds=600,
            seed=seed,
            val_frac=val_frac,
            sample_rows=None,
        )

        scored = _score_results(results)
        eval_time = time.time() - start

        out = {
            "validity": float(scored["validity"]),
            "roc_auc": float(scored["roc_auc"]),
            "combined_score": float(scored["combined_score"]),
            "val_frac": float(val_frac),
            "n_train": int(results.get("n_train", 0)),
            "n_val": int(results.get("n_val", 0)),
            "eval_time": float(eval_time),
        }
        print(f"Evaluation: ROC-AUC={out['roc_auc']:.6f} | score={out['combined_score']:.6f} | n_train={out['n_train']} n_val={out['n_val']} | time={out['eval_time']:.2f}s")
        return out

    except TimeoutError as te:
        print(f"Evaluation timeout: {te}")
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "val_frac": 0.2,
            "n_train": 0,
            "n_val": 0,
            "eval_time": float(time.time() - start),
            "error": str(te),
        }
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "val_frac": 0.2,
            "n_train": 0,
            "n_val": 0,
            "eval_time": float(time.time() - start),
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> dict:
    """
    Fast smoke test for cascade evaluation.
    - Evaluate on a subsample of rows for speed.
    - Use a larger val fraction so we get a meaningful quick signal (stratified if possible).
    """
    start = time.time()
    seed = int.from_bytes(os.urandom(4), "little")
    val_frac = 0.3
    sample_rows = 600

    try:
        results = run_with_timeout(
            program_path=program_path,
            timeout_seconds=300,
            seed=seed,
            val_frac=val_frac,
            sample_rows=sample_rows,
        )

        scored = _score_results(results)
        eval_time = time.time() - start

        out = {
            "validity": float(scored["validity"]),
            "roc_auc": float(scored["roc_auc"]),
            "combined_score": float(scored["combined_score"]),
            "val_frac": float(val_frac),
            "n_train": int(results.get("n_train", 0)),
            "n_val": int(results.get("n_val", 0)),
            "eval_time": float(eval_time),
        }
        print(f"[Stage1] ROC-AUC={out['roc_auc']:.6f} | score={out['combined_score']:.6f} | n_train={out['n_train']} n_val={out['n_val']} | time={out['eval_time']:.2f}s")
        return out

    except TimeoutError as te:
        print(f"Stage 1 timeout: {te}")
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "val_frac": val_frac,
            "n_train": 0,
            "n_val": 0,
            "eval_time": float(time.time() - start),
            "error": str(te),
        }
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        print(traceback.format_exc())
        return {
            "validity": 0.0,
            "roc_auc": 0.5,
            "combined_score": 0.0,
            "val_frac": val_frac,
            "n_train": 0,
            "n_val": 0,
            "eval_time": float(time.time() - start),
            "error": str(e),
        }


def evaluate_stage2(program_path: str) -> dict:
    """Full evaluation = stage 2."""
    return evaluate(program_path)
