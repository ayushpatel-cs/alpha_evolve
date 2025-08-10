# make_submission.py
import numpy as np
import pandas as pd
from mdoel_1 import fit_predict

def main():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    n = len(train_df)
    train_idx = np.arange(n, dtype=int)
    # Give it a non-empty validation set; one row is enough
    val_idx = train_idx[:1]

    out = fit_predict(train_df, test_df, train_idx, val_idx)

    sub = pd.DataFrame({
        "Id": test_df["Id"].values,
        "SalePrice": out["test_pred"].astype(float)
    })
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with", len(sub), "rows")

if __name__ == "__main__":
    main()
