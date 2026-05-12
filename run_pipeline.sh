#!/usr/bin/env bash
# Run the full pipeline, committing and pushing outputs after each step.
# Usage:
#   bash run_pipeline.sh          # full experiment
#   bash run_pipeline.sh --pilot  # pilot (1 script/domain, 1 model, 1 run)
set -euo pipefail

PILOT=""
if [[ "${1:-}" == "--pilot" ]]; then
    PILOT="--pilot"
    echo "[pipeline] Running in PILOT mode"
fi

push_step() {
    local label="$1"
    git add -A
    if git diff --cached --quiet; then
        echo "[pipeline] No new files after $label — skipping commit"
    else
        git commit -m "results: $label"
        git push
        echo "[pipeline] Pushed results after $label"
    fi
}

echo "[pipeline] ── make_scripts ───────────────────────────────────────────"
python -m src.make_scripts
push_step "make_scripts"

echo "[pipeline] ── generate ─────────────────────────────────────────────"
python -m src.generate $PILOT
# generations.jsonl is gitignored; push any other data/ files that appeared
push_step "generate"

echo "[pipeline] ── sanity_check ─────────────────────────────────────────"
python -m src.sanity_check
push_step "sanity_check"

echo "[pipeline] ── parse_outputs ────────────────────────────────────────"
python -m src.parse_outputs
push_step "parse_outputs"

echo "[pipeline] ── metrics ──────────────────────────────────────────────"
python -m src.metrics
push_step "metrics"

echo "[pipeline] ── stats_stance ─────────────────────────────────────────"
python -m src.stats_stance
push_step "stats_stance"

echo "[pipeline] ── stats_drift ──────────────────────────────────────────"
python -m src.stats_drift
push_step "stats_drift"

echo "[pipeline] ── plots ────────────────────────────────────────────────"
python -m src.plots
push_step "plots"

echo ""
echo "[pipeline] ── Complete. All outputs pushed to GitHub. ───────────────"
