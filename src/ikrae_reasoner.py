"""
ikrae_reasoner.py
-----------------
Lightweight Python semantic layer for IKRAE.

- Input:
    experiments/results/learning_objects.csv
    experiments/user_context.json

- Output:
    experiments/results/learning_objects_feasible.csv
    experiments/results/infeasible_los.json

This acts as a SWRL-like filter (mastery, language, device, bandwidth, etc.).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_infeasible(lo: Dict, ctx: Dict) -> Tuple[bool, str]:
    """
    Apply simple SWRL-like constraints:
    - mastery too low
    - language mismatch
    - low bandwidth + video
    - duration exceeds time budget
    """
    lo_id = str(lo.get("lo_id"))
    lo_type = lo.get("type", "question")
    lo_lang = lo.get("language", "en")
    duration = float(lo.get("duration_min", 5.0))
    requires_mastery = float(lo.get("requires_mastery", 0.0))

    user_lang = ctx.get("language", "en")
    bandwidth = ctx.get("bandwidth", "high")
    device = ctx.get("device", "desktop")
    mastery = float(ctx.get("mastery_level", 1.0))
    time_budget = float(ctx.get("time_budget_min", 60.0))

    # 1. Mastery requirement
    if requires_mastery > mastery:
        return True, f"requires mastery {requires_mastery:.2f} > user mastery {mastery:.2f}"

    # 2. Language mismatch
    if lo_lang and lo_lang != user_lang:
        return True, f"LO language {lo_lang} != user language {user_lang}"

    # 3. Low bandwidth + video
    if lo_type == "video" and bandwidth == "low":
        return True, "low bandwidth + video"

    # 4. Mobile + long video (example constraint)
    if lo_type == "video" and device == "mobile" and duration > 20:
        return True, f"mobile + long video ({duration:.1f} min)"

    # 5. Duration exceeds time budget
    # Very strict form: if one LO exceeds entire budget, exclude
    if duration > time_budget:
        return True, f"duration {duration:.1f} > time budget {time_budget:.1f}"

    return False, ""


def apply_semantic_filter(lo_df: pd.DataFrame, user_context: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    infeasible_records = []
    feasible_rows = []

    for _, row in lo_df.iterrows():
        lo = row.to_dict()
        infeasible, reason = is_infeasible(lo, user_context)
        if infeasible:
            infeasible_records.append({"lo_id": str(lo["lo_id"]), "reason": reason})
        else:
            feasible_rows.append(row)

    feasible_df = pd.DataFrame(feasible_rows).reset_index(drop=True)
    return feasible_df, infeasible_records


def run_reasoner(
    lo_csv: Path,
    user_json: Path,
    feasible_csv: Path,
    infeasible_json: Path,
) -> None:
    lo_df = pd.read_csv(lo_csv)
    with open(user_json, encoding="utf-8") as f:
        user_ctx = json.load(f)

    print(f"[Reasoner] Loaded {len(lo_df)} LOs from {lo_csv}")
    feasible_df, infeasible_records = apply_semantic_filter(lo_df, user_ctx)

    feasible_csv.parent.mkdir(parents=True, exist_ok=True)
    feasible_df.to_csv(feasible_csv, index=False)
    with open(infeasible_json, "w", encoding="utf-8") as f:
        json.dump(infeasible_records, f, indent=2)

    print(
        f"[Reasoner] Kept {len(feasible_df)} LOs, "
        f"filtered out {len(infeasible_records)} infeasible LOs."
    )
    print(f"[Reasoner] Saved feasible LOs to {feasible_csv}")
    print(f"[Reasoner] Saved infeasible trace to {infeasible_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IKRAE Python semantic reasoner")
    parser.add_argument(
        "--lo_csv",
        default=str(RESULTS_DIR / "learning_objects.csv"),
        help="Input learning_objects.csv",
    )
    parser.add_argument(
        "--user_json",
        default=str(ROOT / "experiments" / "user_context.json"),
        help="User context JSON",
    )
    parser.add_argument(
        "--feasible_csv",
        default=str(RESULTS_DIR / "learning_objects_feasible.csv"),
        help="Output feasible LOs CSV",
    )
    parser.add_argument(
        "--infeasible_json",
        default=str(RESULTS_DIR / "infeasible_los.json"),
        help="Output infeasible LOs trace JSON",
    )

    args = parser.parse_args()
    run_reasoner(
        lo_csv=Path(args.lo_csv),
        user_json=Path(args.user_json),
        feasible_csv=Path(args.feasible_csv),
        infeasible_json=Path(args.infeasible_json),
    )
