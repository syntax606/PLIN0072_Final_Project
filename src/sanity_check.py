# project/src/sanity_check.py
from __future__ import annotations
import json
from pathlib import Path
from src.config import *

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]

def main():

    rows = load_jsonl("data/generations.jsonl")

    scripts = sum(len(load_jsonl(f)) for f in SCRIPTS_FILES)

    expected_rows = scripts * len(MODEL_SPECS) * len(ORDERS) * len(HISTORY_MODES) * len(RUN_IDS) * len(FORWARD_PATH)

    print("Expected rows:", expected_rows)
    print("Observed rows:", len(rows))

    print("Unique conversations:", len(set(r["conversation_id"] for r in rows)))

if __name__ == "__main__":
    main()
