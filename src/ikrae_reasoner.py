import pandas as pd
import json
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# Helper: load user context file
# ----------------------------------------------------------
def load_user_context(path="experiments/user_context.json"):
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------
# Constraint checks
# ----------------------------------------------------------

def check_device_constraint(row, user):
    """Mobile users cannot view long/high-bandwidth videos."""
    if user["device"] == "mobile" and row["type"] == "lecture":
        return "device=mobile blocks lecture"
    return None


def check_bandwidth_constraint(row, user):
    """Low bandwidth cannot load video content."""
    if user["bandwidth"] == "low" and row["type"] == "lecture":
        return "low bandwidth blocks lecture"
    return None


def check_language_constraint(row, user):
    if "language" in row and row["language"] != user["language"]:
        return f"language mismatch: LO={row['language']} user={user['language']}"
    return None


def check_mastery_constraint(row, user):
    if row["requires_mastery"] > user["mastery_level"]:
        return f"requires {row['requires_mastery']:.2f} > mastery {user['mastery_level']:.2f}"
    return None


def check_time_constraint(row, user):
    """Exclude LOs individually if duration is too long."""
    if row["duration_min"] > user["time_budget_min"]:
        return "exceeds time budget"
    return None


# ----------------------------------------------------------
# Main semantic filtering function
# ----------------------------------------------------------
def semantic_filter(user_context):

    lo_path = RESULTS_DIR / "learning_objects.csv"
    lo_df = pd.read_csv(lo_path)

    feasible_rows = []
    excluded = []

    print("[Reasoner] Applying semantic constraints...")

    for _, row in lo_df.iterrows():
        reasons = []

        # Run all constraints
        for check in [
            check_device_constraint,
            check_bandwidth_constraint,
            check_language_constraint,
            check_mastery_constraint,
            check_time_constraint,
        ]:
            r = check(row, user_context)
            if r:
                reasons.append(r)

        # If no violations → feasible
        if len(reasons) == 0:
            feasible_rows.append(row)
        else:
            excluded.append({
                "lo_id": row["lo_id"],
                "reason": reasons
            })

    feasible_df = pd.DataFrame(feasible_rows)

    # Save outputs
    feasible_df.to_csv(RESULTS_DIR / "learning_objects_feasible.csv", index=False)
    with open(RESULTS_DIR / "infeasible_los.json", "w") as f:
        json.dump(excluded, f, indent=2)

    print(f"[Reasoner] Feasible LOs: {len(feasible_df)}")
    print(f"[Reasoner] Excluded LOs: {len(excluded)}")


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    user = load_user_context()
    semantic_filter(user)
