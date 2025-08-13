# evaluator.py
"""
Evaluator for Kaggle Nomad2018: Predicting Transparent Conductors using OpenEvolve.

Design:
- Every evaluation run samples a fresh random validation subset from train.csv.
- We score with column-wise RMSLE averaged over the two targets:
    ["formation_energy_ev_natom", "bandgap_energy_ev"].
- We run the candidate program in a separate Python process with a timeout.
- We do not trust the program's self-reported score; we compute the score here
  from the returned validation predictions to keep evaluation honest.

Interface expected by OpenEvolve:
    - evaluate(program_path) -> metrics dict
    - evaluate_stage1(program_path) -> quick smoke test (smaller data fraction)
    - evaluate_stage2(program_path) -> full evaluation (delegates to evaluate)

Returns (example):
{
    "rmsle": 0.064,
    "val_frac": 0.2,
    "n_train": 2400,
    "n_val": 600,
    "eval_time": 3.21,
    "validity": 1.0,
    "combined_score": 0.94
}
"""

import os
import sys
import time
import pickle
import random
import traceback
import tempfile
import subprocess

import numpy as np
import pandas as pd

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


class TimeoutError(Exception):
    pass


def _mean_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean column-wise RMSLE between nonnegative true/pred arrays of shape (n, 2).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        return float("inf")

    # Guard for numerical stability
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    errs = []
    for j in range(y_true.shape[1]):
        errs.append(np.sqrt(np.mean((np.log1p(y_true[:, j]) - np.log1p(y_pred[:, j])) ** 2)))
    return float(np.mean(errs))


def run_with_timeout(program_path: str, timeout_seconds: int, seed: int, val_frac: float, sample_rows: int | None = None):
    """
    Execute the candidate program in a clean subprocess.
    We pass the indices split; the program returns validation predictions aligned to val_idx.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import os, sys, pickle, numpy as np, pandas as pd, traceback, random
sys.path.insert(0, os.path.dirname('{program_path}'))

SEED = {seed}
VAL_FRAC = {val_frac}
SAMPLE_ROWS = {repr(sample_rows)}

random.seed(SEED)
np.random.seed(SEED)

try:
    # Import candidate
    import importlib.util as _util
    _spec = _util.spec_from_file_location("candidate", '{program_path}')
    candidate = _util.module_from_spec(_spec)
    _spec.loader.exec_module(candidate)

    # Load data
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")

    if SAMPLE_ROWS is not None:
        train_df = train_df.sample(n=min(SAMPLE_ROWS, len(train_df)), random_state=SEED)

    # Make split
    idx = np.arange(len(train_df))
    n_val = max(1, int(len(train_df) * VAL_FRAC))
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Call evolvable function
    out = candidate.fit_predict(train_df, test_df, train_idx, val_idx)

    # Collect evaluation payload
    y_true_val = train_df.iloc[val_idx][{TARGETS!r}].astype(float).values
    y_pred_val = np.asarray(out.get("val_pred", []), dtype=float)
    y_pred_test = np.asarray(out.get("test_pred", []), dtype=float)

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
        pickle.dump({{"ok": False, "error": str(e), "trace": traceback.format_exc()}}, f)
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
        for path in (temp_file_path, results_path):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


def _score_results(results: dict) -> dict:
    y_true = np.asarray(results["y_true_val"], dtype=float)
    y_pred = np.asarray(results["y_pred_val"], dtype=float)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        return {
            "validity": 0.0,
            "rmsle": float("inf"),
            "combined_score": 0.0,
            "error": f"Shape mismatch or empty preds: y_true={y_true.shape}, y_pred={y_pred.shape}",
        }

    rmsle = _mean_rmsle(y_true, y_pred)
    combined = 1.0 / (1.0 + rmsle)  # higher is better

    return {
        "validity": 1.0,
        "rmsle": float(rmsle),
        "combined_score": float(combined),
    }


def evaluate(program_path: str) -> dict:
    """
    Full evaluation (stage 2).
    - Uses 20% validation split sampled randomly each run.
    """
    start = time.time()
    seed = int.from_bytes(os.urandom(4), "little")
    val_frac = 0.2

    try:
        results = run_with_timeout(
            program_path=program_path,
            timeout_seconds=600,  # generous
            seed=seed,
            val_frac=val_frac,
            sample_rows=None,     # full data
        )

        scored = _score_results(results)
        eval_time = time.time() - start

        out = {
            "validity": float(scored["validity"]),
            "rmsle": float(scored["rmsle"]),
            "combined_score": float(scored["combined_score"]),
            "val_frac": float(val_frac),
            "n_train": int(results.get("n_train", 0)),
            "n_val": int(results.get("n_val", 0)),
            "eval_time": float(eval_time),
        }
        print(f"Evaluation: RMSLE={out['rmsle']:.6f} | score={out['combined_score']:.6f} | "
              f"n_train={out['n_train']} n_val={out['n_val']} | time={out['eval_time']:.2f}s")
        return out

    except TimeoutError as te:
        print(f"Evaluation timeout: {te}")
        return {
            "validity": 0.0,
            "rmsle": float("inf"),
            "combined_score": 0.0,
            "val_frac": val_frac,
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
            "rmsle": float("inf"),
            "combined_score": 0.0,
            "val_frac": val_frac,
            "n_train": 0,
            "n_val": 0,
            "eval_time": float(time.time() - start),
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> dict:
    """
    Fast smoke test for cascade evaluation.
    - Subsample rows for speed and use a larger val fraction for a quick signal.
    """
    start = time.time()
    seed = int.from_bytes(os.urandom(4), "little")
    val_frac = 0.3
    sample_rows = 900  # quickish subset (dataset is ~3k rows)

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
            "rmsle": float(scored["rmsle"]),
            "combined_score": float(scored["combined_score"]),
            "val_frac": float(val_frac),
            "n_train": int(results.get("n_train", 0)),
            "n_val": int(results.get("n_val", 0)),
            "eval_time": float(eval_time),
        }
        print(f"[Stage1] RMSLE={out['rmsle']:.6f} | score={out['combined_score']:.6f} | "
              f"n_train={out['n_train']} n_val={out['n_val']} | time={out['eval_time']:.2f}s")
        return out

    except TimeoutError as te:
        print(f"Stage 1 timeout: {te}")
        return {
            "validity": 0.0,
            "rmsle": float("inf"),
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
            "rmsle": float("inf"),
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
