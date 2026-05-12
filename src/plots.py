# project/src/plots.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────────────
# Load data
# ───────────────────────────────────────────────────────────────────────

def load_data():

    turn_df = pd.read_csv("results/turn_metrics.csv")
    conv_df = pd.read_csv("results/conversation_metrics.csv")

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
    df["C"] = df["step"].apply(extract_c)

    return df


# ───────────────────────────────────────────────────────────────────────
# Plot 1: Mean shift vs commitment (H2)
# ───────────────────────────────────────────────────────────────────────

def plot_mean_shift(turn_df):

    df = add_commitment(turn_df)

    grouped = df.groupby("C")["shift"].mean()

    plt.figure()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Commitment Level (C)")
    plt.ylabel("Mean Shift (S_t - Anchor)")
    plt.title("Mean Signed Drift vs Commitment")

    plt.savefig("results/figures/fig1_mean_shift_by_C.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 2: Mean deviation vs commitment (H1)
# ───────────────────────────────────────────────────────────────────────

def plot_mean_deviation(turn_df):

    df = add_commitment(turn_df)

    grouped = df.groupby("C")["deviation"].mean()

    plt.figure()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Commitment Level (C)")
    plt.ylabel("Mean Deviation")
    plt.title("Deviation from Anchor vs Commitment")

    plt.savefig("results/figures/fig2_mean_deviation_by_C.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 3: Path dependence — forward vs reverse (H3)
# ───────────────────────────────────────────────────────────────────────

def plot_path_dependence(turn_df):

    df = add_commitment(turn_df)

    grouped = df.groupby(["C", "order"])["shift"].mean().unstack()

    plt.figure()

    for col in grouped.columns:
        plt.plot(grouped.index, grouped[col], marker="o", label=col)

    plt.xlabel("Commitment Level (C)")
    plt.ylabel("Mean Shift")
    plt.title("Path Dependence: Forward vs Reverse")

    plt.legend()
    plt.savefig("results/figures/fig3_path_dependence.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 4: History vs No-History (H5)
# ───────────────────────────────────────────────────────────────────────

def plot_history_effect(turn_df):

    df = add_commitment(turn_df)

    grouped = df.groupby(["C", "history_mode"])["deviation"].mean().unstack()

    plt.figure()

    for col in grouped.columns:
        plt.plot(grouped.index, grouped[col], marker="o", label=col)

    plt.xlabel("Commitment Level (C)")
    plt.ylabel("Mean Deviation")
    plt.title("History vs No-History Effect")

    plt.legend()
    plt.savefig("results/figures/fig4_history_effect.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 5: Rule revision distribution (H4)
# ───────────────────────────────────────────────────────────────────────

def plot_rule_shift(conv_df):

    rev = conv_df[conv_df["order"] == "reverse"].copy()

    rev = rev.dropna(subset=["rule_shift"])

    plt.figure()
    plt.hist(rev["rule_shift"], bins=20)
    plt.xlabel("Rule Shift (S_R - S_C4)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rule Revision (Reverse Trajectory)")

    plt.savefig("results/figures/fig5_rule_shift_hist.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 6: NDI by trajectory order
# ───────────────────────────────────────────────────────────────────────

def plot_ndi(conv_df):

    grouped = conv_df.groupby("order")["ndi"].mean()

    plt.figure()
    plt.bar(grouped.index, grouped.values)
    plt.xlabel("Trajectory Order")
    plt.ylabel("Mean NDI")
    plt.title("Normative Drift Index by Trajectory")

    plt.savefig("results/figures/fig6_ndi_by_order.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Plot 7: Example trajectories (qualitative)
# ───────────────────────────────────────────────────────────────────────

def plot_trajectory_examples(turn_df):

    sample = turn_df.groupby("conversation_id").head(5)

    plt.figure()

    for conv_id, g in sample.groupby("conversation_id"):
        plt.plot(g["turn"], g["score"])

    plt.xlabel("Turn")
    plt.ylabel("Score")
    plt.title("Example Trajectories")

    plt.savefig("results/figures/fig7_trajectory_examples.png")
    plt.close()


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    turn_df, conv_df = load_data()

    print("Generating plots...")

    plot_mean_shift(turn_df)
    plot_mean_deviation(turn_df)
    plot_path_dependence(turn_df)
    plot_history_effect(turn_df)
    plot_rule_shift(conv_df)
    plot_ndi(conv_df)
    plot_trajectory_examples(turn_df)

    print("All plots saved in results/figures/")


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
