# project/src/stats_drift.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

import statsmodels.formula.api as smf


# ───────────────────────────────────────────────────────────────────────
# Load data
# ───────────────────────────────────────────────────────────────────────

def load_data():

    turn_df = pd.read_csv("results/turn_metrics.csv")
    conv_df = pd.read_csv("results/conversation_metrics.csv")

    # Keep only valid rows
    turn_df = turn_df.dropna(subset=["shift", "deviation"])
    conv_df = conv_df.dropna(subset=["ndi"])

    return turn_df, conv_df


# ───────────────────────────────────────────────────────────────────────
# Add commitment level
# ───────────────────────────────────────────────────────────────────────

def add_commitment(df):

    def extract_c(step):
        if step == "R":
            return 0
        return int(step[1])

    df = df.copy()
    df["commitment"] = df["step"].apply(extract_c).astype(int)

    return df


# ───────────────────────────────────────────────────────────────────────
# Model 1: Drift vs commitment (H1)
# ───────────────────────────────────────────────────────────────────────

def model_deviation(turn_df):

    turn_df = add_commitment(turn_df)

    model = smf.ols(
        "deviation ~ commitment + C(order) + C(history_mode) + C(domain)",
        data=turn_df
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Model 2: Signed drift (directional) (H2)
# ───────────────────────────────────────────────────────────────────────

def model_shift(turn_df):

    turn_df = add_commitment(turn_df)

    model = smf.ols(
        "shift ~ commitment + C(order) + C(history_mode) + C(domain)",
        data=turn_df
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Model 3: Path dependence (H3)
# interaction between commitment and order
# ───────────────────────────────────────────────────────────────────────

def model_path_dependence(turn_df):

    turn_df = add_commitment(turn_df)

    model = smf.ols(
        "shift ~ commitment * C(order) + C(history_mode) + C(domain)",
        data=turn_df
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Model 4: Rule revision (H4) — reverse only
# ───────────────────────────────────────────────────────────────────────

def model_rule_revision(conv_df):

    rev = conv_df[conv_df["order"] == "reverse"].copy()

    # Drop NaN rule_shift
    rev = rev.dropna(subset=["rule_shift"])

    model = smf.ols(
        "rule_shift ~ C(history_mode) + C(domain)",
        data=rev
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Model 5: Dialogue vs no-history effect (H5)
# ───────────────────────────────────────────────────────────────────────

def model_history_effect(turn_df):

    turn_df = add_commitment(turn_df)

    model = smf.ols(
        "deviation ~ commitment * C(history_mode) + C(order) + C(domain)",
        data=turn_df
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Model 6: Trajectory-level instability (NDI)
# ───────────────────────────────────────────────────────────────────────

def model_ndi(conv_df):

    model = smf.ols(
        "ndi ~ C(order) + C(history_mode) + C(domain)",
        data=conv_df
    ).fit()

    return model


# ───────────────────────────────────────────────────────────────────────
# Save results
# ───────────────────────────────────────────────────────────────────────

def save_model(name, model):

    path = f"results/{name}.txt"
    Path("results").mkdir(exist_ok=True)

    with open(path, "w") as f:
        f.write(model.summary().as_text())

    print(f"Saved {name} → {path}")


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():

    print("Loading data...")
    turn_df, conv_df = load_data()

    print(f"Turn rows: {len(turn_df)}")
    print(f"Conversation rows: {len(conv_df)}")

    print("\nFitting models...")

    m1 = model_deviation(turn_df)
    save_model("drift_deviation_model", m1)

    m2 = model_shift(turn_df)
    save_model("drift_signed_model", m2)

    m3 = model_path_dependence(turn_df)
    save_model("path_dependence_model", m3)

    m4 = model_rule_revision(conv_df)
    save_model("rule_revision_model", m4)

    m5 = model_history_effect(turn_df)
    save_model("history_effect_model", m5)

    m6 = model_ndi(conv_df)
    save_model("ndi_model", m6)

    print("\nAll drift models completed.")


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
