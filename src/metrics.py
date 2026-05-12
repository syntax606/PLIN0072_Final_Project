# project/src/metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def trajectory_consistency(shifts):
    signs = [sign(s) for s in shifts if s != 0]
    if len(signs) <= 1:
        return True
    return all(s == signs[0] for s in signs)

def main(parsed_csv="data/parsed.csv"):

    df = pd.read_csv(parsed_csv)

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    ok = df[df["parse_ok"] == True].copy()

    ok = ok.sort_values(["conversation_id", "turn"])

    turn_rows = []
    conv_rows = []

    for conv_id, g in ok.groupby("conversation_id"):

        g = g.sort_values("turn")

        scores = g["score"].values
        anchor = scores[0]

        shifts = scores - anchor
        deviations = np.abs(shifts)

        step_deltas = np.diff(scores, prepend=scores[0])

        consistent = trajectory_consistency(shifts)

        max_excursion = np.max(deviations)
        volatility = np.sum(np.abs(np.diff(scores)))
        final_shift = scores[-1] - anchor

        ndi = np.mean([max_excursion, abs(final_shift), volatility])

        # H4 rule shift (only meaningful in reverse)
        rule_shift = np.nan
        if g["order"].iloc[0] == "reverse":
            rule_shift = scores[-1] - scores[0]

        conv_rows.append({
            "conversation_id": conv_id,
            "model": g["model"].iloc[0],
            "domain": g["domain"].iloc[0],
            "order": g["order"].iloc[0],
            "history_mode": g["history_mode"].iloc[0],
            "run_id": g["run_id"].iloc[0],
            "anchor": anchor,
            "max_excursion": max_excursion,
            "volatility": volatility,
            "final_shift": final_shift,
            "ndi": ndi,
            "trajectory_consistent": consistent,
            "rule_shift": rule_shift
        })

        for i, (_, row) in enumerate(g.iterrows()):

            explanation = row.get("explanation", "")
            words = len(str(explanation).split()) if isinstance(explanation, str) else 0

            turn_rows.append({
                **row.to_dict(),
                "anchor": anchor,
                "shift": shifts[i],
                "deviation": deviations[i],
                "step_delta": step_deltas[i],
                "explanation_len_words": words
            })

    turn_df = pd.DataFrame(turn_rows)
    conv_df = pd.DataFrame(conv_rows)

    # ── Reliability metrics ─────────────────────────────────────────────
    # Fix #1: conversation_id encodes run_id, so it cannot be used as a
    # pivot index to match run 1 vs run 2.  Instead, filter by run_id
    # directly within each group.

    reliability = []

    for key, g in turn_df.groupby(["model", "script_id", "order", "history_mode", "step"]):

        run1_scores = g[g["run_id"] == 1]["score"].values
        run2_scores = g[g["run_id"] == 2]["score"].values

        if len(run1_scores) == 1 and len(run2_scores) == 1:
            s1 = run1_scores[0]
            s2 = run2_scores[0]

            reliability.append({
                "model": key[0],
                "script_id": key[1],
                "order": key[2],
                "history_mode": key[3],
                "step": key[4],
                "agreement_rate": float(s1 == s2),
                "mean_abs_diff": float(abs(s1 - s2))
            })

    reliability_df = pd.DataFrame(reliability)

    # ── Save outputs ────────────────────────────────────────────────────

    Path("results").mkdir(exist_ok=True)

    turn_df.to_csv("results/turn_metrics.csv", index=False)
    conv_df.to_csv("results/conversation_metrics.csv", index=False)
    reliability_df.to_csv("results/reliability.csv", index=False)

    analysis_cols = [
        "model", "script_id", "domain", "order", "history_mode", "run_id",
        "step", "step_type", "score", "anchor", "shift", "deviation",
        "step_delta", "confidence", "explanation_len_words"
    ]

    turn_df[analysis_cols].to_csv("results/analysis_dataset.csv", index=False)

    print("All metrics computed successfully.")

if __name__ == "__main__":
    main()
