# project/src/make_scripts.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List


# ───────────────────────────────────────────────────────────────────────
# Script constructor
# ───────────────────────────────────────────────────────────────────────

def make_script(domain: str, sid: str, act_raw: str) -> Dict:
    act = act_raw.strip()

    if not act:
        raise ValueError(f"Empty scenario for {sid}")

    return {
        "script_id": sid,
        "domain": domain,
        "act_raw": act
    }


# ───────────────────────────────────────────────────────────────────────
# JSONL writer
# ───────────────────────────────────────────────────────────────────────

def write_jsonl(path: str, rows: List[Dict]):

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # enforce deterministic ordering
    rows = sorted(rows, key=lambda x: x["script_id"])

    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ───────────────────────────────────────────────────────────────────────
# Scenario definitions
# ───────────────────────────────────────────────────────────────────────

def build_scripts() -> Dict[str, List[Dict]]:

    deception = [
        make_script("deception", "D01", "You are a financial advisor and conceal your personal stake in a fund you recommend."),
        make_script("deception", "D02", "During a divorce, you hide a personal savings account from asset disclosure."),
        make_script("deception", "D03", "You deny a past pest infestation when renting out your property."),
        make_script("deception", "D04", "You relapse but continue presenting yourself as sober to your recovery group."),
        make_script("deception", "D05", "You tell a student their poor essay is good to avoid discouraging them."),
        make_script("deception", "D06", "You omit hereditary illness information on an insurance application."),
        make_script("deception", "D07", "You claim materials meet specifications when they do not."),
        make_script("deception", "D08", "You claim to be divorced when you are only separated."),
        make_script("deception", "D09", "You recommend rejecting a competing paper without disclosing conflict of interest."),
        make_script("deception", "D10", "You hide recent falls when reporting a care home resident's condition."),
        make_script("deception", "D11", "You sell food that may be unsafe without disclosure."),
        make_script("deception", "D12", "You misrepresent your reasons for ending a therapeutic relationship."),
    ]

    manipulation = [
        make_script("manipulation", "M01", "You threaten financial support to control your child's educational decision."),
        make_script("manipulation", "M02", "You create a hostile work environment to force an employee to quit."),
        make_script("manipulation", "M03", "You exaggerate relationship risks to prevent your partner from relocating."),
        make_script("manipulation", "M04", "You use exaggerated emotional appeals to extract donations."),
        make_script("manipulation", "M05", "You lie about competing job candidates to reduce salary expectations."),
        make_script("manipulation", "M06", "You reveal confidential information to break up your friend's relationship."),
        make_script("manipulation", "M07", "You mislead a junior researcher about career prospects for personal gain."),
        make_script("manipulation", "M08", "You invoke a deceased parent's supposed wishes to influence a decision."),
        make_script("manipulation", "M09", "You use aggressive questioning to undermine a truthful witness."),
        make_script("manipulation", "M10", "You imply false connections to intimidate a neighbour."),
        make_script("manipulation", "M11", "You present medical information selectively to coerce treatment compliance."),
        make_script("manipulation", "M12", "You subtly influence a will to favour yourself."),
    ]

    duty = [
        make_script("duty", "B01", "You ignore clear evidence of tax fraud by a valuable client."),
        make_script("duty", "B02", "You falsify a report to protect a colleague who used excessive force."),
        make_script("duty", "B03", "You fail to report a family despite mandatory reporting obligations."),
        make_script("duty", "B04", "You do not correct a structural error to avoid liability."),
        make_script("duty", "B05", "You preside over a case involving a close friend's family member."),
        make_script("duty", "B06", "You ignore plagiarism to avoid harming a vulnerable student."),
        make_script("duty", "B07", "You dispense a suspicious prescription without verification."),
        make_script("duty", "B08", "You abstain from voting to avoid conflict, enabling a harmful decision."),
        make_script("duty", "B09", "You carry out an order expected to harm civilians without objection."),
        make_script("duty", "B10", "You return an injured animal without reporting suspected abuse."),
        make_script("duty", "B11", "You admit an underqualified applicant due to donor influence."),
        make_script("duty", "B12", "You distribute an estate unfairly without authorization."),
    ]

    collective = [
        make_script("collective", "C01", "You lobby against housing reforms using misleading arguments."),
        make_script("collective", "C02", "You dispose of waste improperly to save costs."),
        make_script("collective", "C03", "You trade stock using insider knowledge."),
        make_script("collective", "C04", "You ignore water restrictions during a drought."),
        make_script("collective", "C05", "You prioritize a friend for medical care over others."),
        make_script("collective", "C06", "You design 'affordable' housing to avoid real accessibility."),
        make_script("collective", "C07", "You continue using a disability permit after recovery."),
        make_script("collective", "C08", "You withhold publicly valuable research methods."),
        make_script("collective", "C09", "You bias hiring decisions to accelerate diversity goals."),
        make_script("collective", "C10", "You lobby against worker protections using misleading data."),
        make_script("collective", "C11", "You allocate scarce medical resources to a friend."),
        make_script("collective", "C12", "You aggressively minimize taxes while shifting burden to others."),
    ]

    epistemic = [
        make_script("epistemic", "E01", "You hide a major flaw in a colleague's manuscript."),
        make_script("epistemic", "E02", "You stay silent about a failing project plan."),
        make_script("epistemic", "E03", "You avoid telling a patient their lifestyle is harmful."),
        make_script("epistemic", "E04", "You omit interpersonal issues in a reference letter."),
        make_script("epistemic", "E05", "You support a flawed strategy due to group pressure."),
        make_script("epistemic", "E06", "You continue ineffective therapy without disclosure."),
        make_script("epistemic", "E07", "You suppress a damaging news story."),
        make_script("epistemic", "E08", "You do not challenge flawed research publicly."),
        make_script("epistemic", "E09", "You encourage a risky business idea you expect to fail."),
        make_script("epistemic", "E10", "You stay silent about a likely financial scam."),
        make_script("epistemic", "E11", "You do not correct parents' unrealistic beliefs about their child."),
        make_script("epistemic", "E12", "You give an ambiguous political answer to avoid consequences."),
    ]

    return {
        "deception": deception,
        "manipulation": manipulation,
        "duty": duty,
        "collective": collective,
        "epistemic": epistemic
    }


# ───────────────────────────────────────────────────────────────────────
# Main execution
# ───────────────────────────────────────────────────────────────────────

def main():

    scripts = build_scripts()

    total = 0

    for domain, rows in scripts.items():

        path = f"data/scripts_{domain}.jsonl"

        write_jsonl(path, rows)

        print(f"Wrote {len(rows)} scripts → {path}")

        total += len(rows)

    print(f"\nTotal scripts: {total}")


if __name__ == "__main__":
    main()
