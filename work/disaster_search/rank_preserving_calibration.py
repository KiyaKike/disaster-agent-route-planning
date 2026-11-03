"""Source-only isotonic baseline repair; raw survey order is a hard constraint.

No ground truth, future quality, masks or evaluator enter these action helpers.
The unchanged conditional negative-upgrade count remains a local proxy.
"""
from copy import deepcopy
import math

from scipy.optimize import isotonic_regression

from cross_cell_count_smoke import CELLS
from run_development_budget_controls import utility_table

POLICIES = ["rank_survey", "rank_negative_unprojected", "rank_negative_isotonic"]


def project_model(model):
    """Weighted LS projection of the FOUR frozen shrunken source means.

    Weight n+8 retains the source fitter's pseudo-cell convention. This is
    a projection of its sufficient statistics, not a fit to held outcomes.
    """
    result = deepcopy(model)
    bins = model["bins"]
    if len(bins) != 4 or [b["bin"] for b in bins] != list(range(4)):
        raise ValueError("Exactly four ordered frozen bins required")
    shrinkage = model["shrinkage_pseudo_cells"]
    weights = [b["known_cells"] + shrinkage for b in bins]
    if shrinkage != 8 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Frozen positive n+8 weights required")
    detail = {"weights": weights, "modes": {}}
    for mode in ["native", "blurred"]:
        original = [b["survey_mean_count"][mode] for b in bins]
        if any(not math.isfinite(y) or y < 0 for y in original):
            raise ValueError("Finite nonnegative frozen means required")
        fitted = isotonic_regression(original, weights=weights, increasing=True)
        values = fitted.x.tolist()
        for b, value in zip(result["bins"], values):
            b["survey_mean_count"][mode] = value
        detail["modes"][mode] = {"original": original, "projected": values,
            "block_boundaries": fitted.blocks.tolist(),
            "weighted_squared_adjustment": sum(w * (a - b) ** 2 for w, a, b in zip(weights, original, values))}
    return result, detail


def raw_order(features, ranks):
    values = [features[c]["change"] for c in CELLS]
    if any(not math.isfinite(x) for x in values) or sorted(ranks) != list(range(len(CELLS))):
        raise ValueError("Finite public changes and a rank permutation required")
    return sorted(range(len(CELLS)), key=lambda i: (-values[i], ranks[i]))


def policy_table(features, original, projected, scenario, cost, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown rank-preserving policy")
    model = original if policy == "rank_negative_unprojected" else projected
    base = "count_survey" if policy == "rank_survey" else "count_negative"
    return utility_table(features, model, {}, scenario, cost, base)


def choose(state, table, remaining, cost, ranks, order):
    # Only the next unseen raw-ranked survey competes with negative upgrades.
    # This explicit constraint also preserves rank under nonmonotone values
    # (unprojected control) and isotonic plateaus (projected control).
    survey = next((i for i in order if state[i] == 0), None) if remaining >= 1 else None
    best = (survey, table[survey][0]) if survey is not None else None
    if best is not None and not math.isfinite(best[1]):
        raise ValueError("Nonfinite survey value")
    if remaining >= cost:
        for i, status in enumerate(state):
            if status != 1 or table[i][1] is None:
                continue
            value = table[i][1]
            if not math.isfinite(value):
                raise ValueError("Nonfinite negative-upgrade value")
            if best is None or value > best[1] or (value == best[1] and ranks[i] < ranks[best[0]]):
                best = (i, value)
    return best


def run_episode(environment, table, budget, cost, ranks, order):
    state = [0] * len(table)
    spent, trace = 0, []
    while spent < budget:
        selected = choose(state, table, budget - spent, cost, ranks, order)
        if selected is None:
            break
        i, value = selected
        kind = "survey" if state[i] == 0 else "upgrade"
        action_cost = 1 if kind == "survey" else cost
        result = environment.observe(CELLS[i], kind)
        if result["cost"] != action_cost:
            raise ValueError("Paid environment cost mismatch")
        spent += action_cost
        trace.append({"cell_id": CELLS[i], "kind": kind, "mode": result["mode"],
            "declared_target": result["declared_target"], "cost": action_cost,
            "cumulative_cost": spent, "utility": value})
        state[i] = 3 if result["mode"] == "native" else 2 if result["declared_target"] else 1
    return trace
