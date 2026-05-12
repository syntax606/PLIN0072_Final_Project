# project/src/stats_stance.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ───────────────────────────────────────────────────────────────────────
# Load data
# ───────────────────────────────────────────────────────────────────────

def load_data(path="results/analysis_dataset.csv"):

    df = pd.read_csv(path)

    # Keep only valid parsed rows
    df = df.dropna(subset=["score"])

    return df


# ───────────────────────────────────────────────────────────────────────
# Prepare design matrix
# ───────────────────────────────────────────────────────────────────────

def prepare_design_matrix(df: pd.DataFrame):

    df = df.copy()

    df["order"] = df["order"].astype("category")
    df["history_mode"] = df["history_mode"].astype("category")
    df["domain"] = df["domain"].astype("category")
    df["step"] = df["step"].astype("category")

    def extract_c(step):
        if step == "R":
            return 0
        return int(step[1])

    df["C"] = df["step"].apply(extract_c).astype(int)

    # Main-effects dummies — cast to float so numpy ops work cleanly
    X = pd.get_dummies(
        df[["C", "order", "history_mode", "domain"]],
        drop_first=True
    ).astype(float)

    # Fix #4: add C×order interaction to test H3 (path dependence) in the
    # ordinal stance model, not just in the OLS drift models.
    c_values = df["C"].to_numpy(dtype=float)
    order_cols = [c for c in X.columns if c.startswith("order_")]
    for col in order_cols:
        X[f"C_x_{col}"] = c_values * X[col].to_numpy(dtype=float)

    y = df["score"].astype(int)

    return X, y, df


# ───────────────────────────────────────────────────────────────────────
# Fit ordinal model
# ───────────────────────────────────────────────────────────────────────

def fit_model(X, y):

    model = OrderedModel(
        endog=y,
        exog=X,
        distr="logit"
    )

    res = model.fit(method="bfgs", disp=False)

    return res


# ───────────────────────────────────────────────────────────────────────
# Save results
# ───────────────────────────────────────────────────────────────────────

def save_results(res, path="results/stance_ordered_logit.txt"):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(res.summary().as_text())

    print(f"Saved model summary → {path}")


# ───────────────────────────────────────────────────────────────────────
# Diagnostics
# ───────────────────────────────────────────────────────────────────────

def print_diagnostics(df):

    print("\nDataset summary:")
    print("------------------")
    print(f"Rows: {len(df)}")
    print(f"Unique conversations: {df['script_id'].nunique()}")

    print("\nScore distribution:")
    print(df["score"].value_counts().sort_index())

    print("\nBy history mode:")
    print(df.groupby("history_mode")["score"].mean())

    print("\nBy order:")
    print(df.groupby("order")["score"].mean())


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():

    print("Loading data...")
    df = load_data()

    print_diagnostics(df)

    print("\nPreparing design matrix...")
    X, y, df = prepare_design_matrix(df)

    print(f"Design matrix shape: {X.shape}")

    print("\nFitting ordered logistic regression...")
    res = fit_model(X, y)

    print("\nModel summary:")
    print(res.summary())

    save_results(res)

    print("\nDone.")


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
