"""EVALUATOR ONLY: perfect-information budgeted coverage on frozen outcomes.

Never import into an executable search policy. True matched-instance bitsets
and future quality are deliberately available only to this offline diagnostic.
"""
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def union_bits(values):
    result = 0
    for value in values:
        result |= value
    return result


def survey_bits(cell):
    return cell[cell["survey_mode"]]


def evaluate(cells, levels, budget, cost, allow_upgrade):
    if len(levels) != len(cells):
        raise ValueError("Wrong oracle decision shape")
    found = spent = 0
    for cell, level in zip(cells, levels):
        if level not in [0, 1, 2]:
            raise ValueError("Oracle decisions must be none/survey/survey+upgrade")
        if level:
            spent += 1
            found |= survey_bits(cell)
        if level == 2:
            if not allow_upgrade or cell["survey_mode"] != "blurred":
                raise ValueError("Oracle bypassed observation eligibility")
            spent += cost
            found |= cell["native"]
    if spent > budget:
        raise ValueError("Oracle exceeded budget")
    return {"found": found.bit_count(), "spent": spent, "bits": found}


def greedy_feasible(cells, budget, cost, allow_upgrade):
    levels = [0] * len(cells)
    remaining = budget
    found = 0
    while remaining:
        best = None
        for i, cell in enumerate(cells):
            if levels[i] == 0:
                options = [(1, 1, survey_bits(cell))]
                if allow_upgrade and cell["survey_mode"] == "blurred":
                    options.append((2, 1 + cost, cell["blurred"] | cell["native"]))
            elif levels[i] == 1 and allow_upgrade and cell["survey_mode"] == "blurred":
                options = [(2, cost, cell["native"])]
            else:
                continue
            for level, price, bits in options:
                gain = (bits & ~found).bit_count()
                if price <= remaining and gain and (best is None or gain / price > best[0]):
                    best = (gain / price, i, level, price, bits)
        if best is None:
            break
        _, i, level, price, bits = best
        levels[i] = level
        found |= bits
        remaining -= price
    return levels


def bit_positions(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def solve(cells, budget, cost, allow_upgrade=True, incumbents=(), time_limit=2.):
    """Return a replayable feasible lower bound and a certified integer upper.

    UB uses the HiGHS minimization dual bound (sign reversed and floored), or
    the trivial accessible-instance union if no finite solver bound exists.
    Feasible incumbents never become claimed upper bounds.
    """
    if budget < 0 or cost < 1 or int(budget) != budget or int(cost) != cost or time_limit <= 0:
        raise ValueError("Invalid fixed oracle budget/cost/time limit")
    if any(c["survey_mode"] not in ["native", "blurred"] or min(c["native"], c["blurred"]) < 0 for c in cells):
        raise ValueError("Invalid cached oracle cell")
    started = time.perf_counter()
    accessible = union_bits(survey_bits(c) | (c["native"] if allow_upgrade and c["survey_mode"] == "blurred" else 0) for c in cells)
    candidates = [("empty", [0] * len(cells)), ("greedy_feasible", greedy_feasible(cells, budget, cost, allow_upgrade))]
    candidates.extend(("supplied_feasible_incumbent", list(v)) for v in incumbents)
    best_origin, best = candidates[0]
    best_score = evaluate(cells, best, budget, cost, allow_upgrade)
    for origin, levels in candidates[1:]:
        value = evaluate(cells, levels, budget, cost, allow_upgrade)
        if (value["found"], -value["spent"]) > (best_score["found"], -best_score["spent"]):
            best_origin, best, best_score = origin, levels, value
    if best_score["found"] == accessible.bit_count() or budget == 0:
        upper = best_score["found"]
        return {"levels": best, "lower": upper, "upper": upper, "spent": best_score["spent"],
            "accessible_union": accessible.bit_count(), "certified_optimal": True,
            "certificate": "accessible_union_closed" if budget else "zero_budget", "solver_called": False,
            "solver_status": None, "solver_message": None, "raw_maximization_dual_bound": None,
            "incumbent_origin": best_origin, "wall_s": time.perf_counter() - started}
    # Remove only actions provably irrelevant to this perfect-information
    # objective, including upgrades whose native matches are a subset of B.
    active = [i for i, c in enumerate(cells) if survey_bits(c) or
              (allow_upgrade and c["survey_mode"] == "blurred" and c["native"] & ~c["blurred"])]
    upgrades = [i for i in active if allow_upgrade and cells[i]["survey_mode"] == "blurred"
                and cells[i]["native"] & ~cells[i]["blurred"]]
    targets = list(bit_positions(accessible))
    s_index = {i: j for j, i in enumerate(active)}
    u_index = {i: len(active) + j for j, i in enumerate(upgrades)}
    z_index = {t: len(active) + len(upgrades) + j for j, t in enumerate(targets)}
    n_variables = len(active) + len(upgrades) + len(targets)
    rr, cc, vv = [], [], []
    upper_constraints = [float(budget)] + [0.] * (len(upgrades) + len(targets))
    for i, j in s_index.items():
        rr.append(0); cc.append(j); vv.append(1.)
    for i, j in u_index.items():
        rr.append(0); cc.append(j); vv.append(float(cost))
    for k, i in enumerate(upgrades, start=1):
        rr.extend([k, k]); cc.extend([u_index[i], s_index[i]]); vv.extend([1., -1.])
    target_row = {t: 1 + len(upgrades) + j for j, t in enumerate(targets)}
    for t in targets:
        rr.append(target_row[t]); cc.append(z_index[t]); vv.append(1.)
    for i, j in s_index.items():
        for t in bit_positions(survey_bits(cells[i])):
            rr.append(target_row[t]); cc.append(j); vv.append(-1.)
    for i, j in u_index.items():
        for t in bit_positions(cells[i]["native"]):
            rr.append(target_row[t]); cc.append(j); vv.append(-1.)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(upper_constraints), n_variables)).tocsc()
    objective = np.zeros(n_variables)
    for j in z_index.values():
        objective[j] = -1.
    result = milp(objective, integrality=np.ones(n_variables), bounds=Bounds(0., 1.),
        constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper_constraints)),
        options={"time_limit": time_limit, "mip_rel_gap": 0., "presolve": True})
    if result.status not in [0, 1]:
        raise ValueError(f"Unexpected oracle solver state {result.status}: {result.message}")
    if result.x is not None:
        if not np.isfinite(result.x).all() or np.max(np.abs(result.x - np.rint(result.x))) > 1e-5:
            raise ValueError("Oracle incumbent was not integer feasible")
        levels = [0] * len(cells)
        for i, j in s_index.items():
            if result.x[j] > .5:
                levels[i] = 1
        for i, j in u_index.items():
            if result.x[j] > .5:
                if levels[i] != 1:
                    raise ValueError("MILP incumbent upgraded before survey")
                levels[i] = 2
        value = evaluate(cells, levels, budget, cost, allow_upgrade)
        if (value["found"], -value["spent"]) > (best_score["found"], -best_score["spent"]):
            best_origin, best, best_score = "milp_incumbent", levels, value
    dual = getattr(result, "mip_dual_bound", None)
    raw_upper = -float(dual) if dual is not None and math.isfinite(dual) else None
    upper = accessible.bit_count()
    if raw_upper is not None:
        if raw_upper < best_score["found"] - 1e-6:
            raise ValueError("Dual bound below replayed feasible discovery count")
        upper = min(upper, max(best_score["found"], math.floor(raw_upper + 1e-6)))
    if result.status == 0 and upper != best_score["found"]:
        raise ValueError("Solver claimed optimal but bounds did not close")
    return {"levels": best, "lower": best_score["found"], "upper": upper,
        "spent": best_score["spent"], "accessible_union": accessible.bit_count(),
        "certified_optimal": upper == best_score["found"],
        "certificate": "integer_dual_bound_closed" if upper == best_score["found"] else "unclosed_dual_or_union_bound",
        "solver_called": True, "solver_status": int(result.status), "solver_message": result.message,
        "raw_maximization_dual_bound": raw_upper, "incumbent_origin": best_origin,
        "wall_s": time.perf_counter() - started}
