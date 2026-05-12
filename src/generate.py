# project/src/generate.py
from __future__ import annotations
import argparse, gc, hashlib, json, time
from pathlib import Path
from typing import List, Dict

from src.config import (
    SCRIPTS_FILES, MODEL_SPECS, MODEL_TEMPLATE_KWARGS,
    ORDERS, FORWARD_PATH, REVERSE_PATH, HISTORY_MODES, RUN_IDS,
    DOMAIN_RULE_PROMPTS, GEN_PRIMARY,
)
from src.prompt_schema import (
    SYSTEM_MSG,
    render_rule_prompt,
    render_commitment_prompt,
    render_messages_from_history,
)
from src.models_chat import HFChatModel


def run(out_path="data/generations.jsonl", pilot=False):

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    scripts = []
    for f in SCRIPTS_FILES:
        scripts += [json.loads(l) for l in Path(f).read_text().splitlines() if l]

    if pilot:
        # One script per domain, first model only, run 1 only.
        # All orders and history modes are kept so the full pipeline
        # (metrics, stats, plots) can run on the pilot output.
        seen_domains = set()
        pilot_scripts = []
        for s in scripts:
            if s["domain"] not in seen_domains:
                pilot_scripts.append(s)
                seen_domains.add(s["domain"])
        scripts     = pilot_scripts
        model_specs = MODEL_SPECS[:1]
        run_ids     = [1]
        print(f"[pilot] {len(scripts)} scripts × 1 model × {len(ORDERS)} orders "
              f"× {len(HISTORY_MODES)} history modes × 1 run "
              f"= {len(scripts) * len(ORDERS) * len(HISTORY_MODES) * len(FORWARD_PATH)} rows")
    else:
        model_specs = MODEL_SPECS
        run_ids     = RUN_IDS

    with open(out_path, "a", encoding="utf-8") as fh:

        for run_id in run_ids:
            for short, model_id in model_specs:

                model = HFChatModel(
                    model_id,
                    template_kwargs=MODEL_TEMPLATE_KWARGS.get(short, {})
                )

                try:
                    for script in scripts:
                        for order in ORDERS:
                            for history_mode in HISTORY_MODES:

                                steps = FORWARD_PATH if order == "forward" else REVERSE_PATH

                                history = []

                                conv_id = f"{short}|{script['script_id']}|{order}|{history_mode}|run{run_id}"

                                for turn, step in enumerate(steps, 1):

                                    if step == "R":
                                        prompt = render_rule_prompt(DOMAIN_RULE_PROMPTS[script["domain"]], script)
                                        step_type = "rule"
                                    else:
                                        prompt = render_commitment_prompt(script, int(step[1]))
                                        step_type = "commitment"

                                    messages = render_messages_from_history(
                                        history if history_mode == "history" else [],
                                        prompt
                                    )

                                    resp, in_len, out_len, hit_max = model.generate(messages, **GEN_PRIMARY)

                                    fh.write(json.dumps({
                                        "conversation_id": conv_id,
                                        "model": short,
                                        "script_id": script["script_id"],
                                        "domain": script["domain"],
                                        "order": order,
                                        "history_mode": history_mode,
                                        "run_id": run_id,
                                        "step": step,
                                        "step_type": step_type,
                                        "turn": turn,
                                        "prompt": prompt,
                                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                        "response": resp,
                                        "input_len": in_len,
                                        "output_len": out_len,
                                        "hit_max_tokens": hit_max,
                                        "timestamp": time.time()
                                    }) + "\n")

                                    if history_mode == "history":
                                        history += [
                                            {"role": "user", "content": prompt},
                                            {"role": "assistant", "content": resp},
                                        ]

                finally:
                    model.unload()
                    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Run a minimal slice: 1 script/domain, 1 model, 1 run")
    parser.add_argument("--out", default="data/generations.jsonl")
    args = parser.parse_args()

    run(out_path=args.out, pilot=args.pilot)
