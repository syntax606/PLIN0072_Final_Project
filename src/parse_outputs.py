# project/src/parse_outputs.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.config import STANCE_TO_SCORE, ALLOWED_SCORES


# ───────────────────────────────────────────────────────────────────────
# Regex patterns (robust to formatting variation)
# ───────────────────────────────────────────────────────────────────────

RE_STANCE = re.compile(r"^\s*STANCE:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
RE_SCORE  = re.compile(r"^\s*SCORE:\s*(-?\d+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_CONF   = re.compile(r"^\s*CONFIDENCE:\s*(\d{1,3})\s*$", re.IGNORECASE | re.MULTILINE)
RE_EXPL   = re.compile(r"^\s*EXPLANATION:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


# ───────────────────────────────────────────────────────────────────────
# Sentence counter (for validation)
# ───────────────────────────────────────────────────────────────────────

def count_sentences(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0

    parts = re.findall(r"[^.!?]+[.!?](?=\s|$)|[^.!?]+$", text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


# ───────────────────────────────────────────────────────────────────────
# Parse single response
# ───────────────────────────────────────────────────────────────────────

def parse_one(text: str) -> Dict[str, Any]:

    stance_m = RE_STANCE.search(text or "")
    score_m  = RE_SCORE.search(text or "")
    conf_m   = RE_CONF.search(text or "")
    expl_m   = RE_EXPL.search(text or "")

    stance = stance_m.group(1).strip().lower() if stance_m else None
    score = int(score_m.group(1)) if score_m else None
    confidence = int(conf_m.group(1)) if conf_m else None
    explanation = expl_m.group(1).strip() if expl_m else None

    ok = True
    problems: List[str] = []

    # ── Missing fields ────────────────────────────────────────────────

    if stance is None:
        ok = False; problems.append("missing_stance")

    if score is None:
        ok = False; problems.append("missing_score")

    if confidence is None:
        ok = False; problems.append("missing_confidence")

    if explanation is None:
        ok = False; problems.append("missing_explanation")

    # ── Score validation ──────────────────────────────────────────────

    if score is not None and score not in ALLOWED_SCORES:
        ok = False; problems.append("score_out_of_range")

    # ── Confidence validation ─────────────────────────────────────────

    if confidence is not None and not (0 <= confidence <= 100):
        ok = False; problems.append("bad_confidence_range")

    # ── Stance-score consistency ──────────────────────────────────────

    expected_score = STANCE_TO_SCORE.get(stance) if stance else None

    if stance is not None and expected_score is None:
        ok = False; problems.append("unknown_stance_label")

    if expected_score is not None and score is not None:
        if expected_score != score:
            ok = False; problems.append("stance_score_mismatch")

    # ── Explanation validation ────────────────────────────────────────

    if explanation is not None:
        if count_sentences(explanation) != 1:
            ok = False; problems.append("explanation_not_one_sentence")

    return {
        "stance": stance,
        "score": score,
        "confidence": confidence,
        "explanation": explanation,
        "parse_ok": ok,
        "parse_problems": "|".join(problems)
    }


# ───────────────────────────────────────────────────────────────────────
# Main parser
# ───────────────────────────────────────────────────────────────────────

def main(
    gen_path: str = "data/generations.jsonl",
    out_csv: str = "data/parsed.csv"
):

    rows = []

    lines = Path(gen_path).read_text(encoding="utf-8").splitlines()

    for line in lines:

        if not line.strip():
            continue

        rec = json.loads(line)

        parsed = parse_one(rec.get("response", ""))

        # Merge raw + parsed
        merged = {**rec, **parsed}

        rows.append(merged)

    df = pd.DataFrame(rows)

    # ────────────────────────────────────────────────────────────────
    # Post-processing fixes
    # ────────────────────────────────────────────────────────────────

    # Ensure numeric types
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Sort for downstream metrics
    df = df.sort_values(
        ["conversation_id", "turn"]
    ).reset_index(drop=True)

    # Save
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # ────────────────────────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────────────────────────

    total = len(df)
    ok_count = df["parse_ok"].sum()
    fail_count = total - ok_count

    print(f"Wrote {out_csv}")
    print(f"Total rows: {total}")
    print(f"Parse OK: {ok_count} ({ok_count/total:.2%})")
    print(f"Parse FAIL: {fail_count} ({fail_count/total:.2%})")

    if fail_count > 0:
        print("\nTop parse issues:")
        print(df["parse_problems"].value_counts().head(10))


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
