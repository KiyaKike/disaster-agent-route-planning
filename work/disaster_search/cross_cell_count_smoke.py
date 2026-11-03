"""Six-scene engineering diagnostic; never a validation or new-method result.

Offline labels fit an aggregate boundary-conditioned remaining-reward ratio.
The online planner receives only public coarse cues and paid mask edge flags.
True building identities are used only by the fitter and post-hoc evaluator.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from build_xbd_mini import atomic_json, sha
from paired_count_baseline import action_score
from plan_search_v2 import freeze


POLICIES = ["random_survey", "coverage_survey", "prior_survey", "random_upgrade",
            "prior_gate", "local_count", "boundary_count"]
SCENARIOS = ["native_survey", "mixed_survey", "blurred_survey"]
CELLS = [f"r{r:02d}c{c:02d}" for r in range(8) for c in range(8)]
QUALITY_SEEDS = list(range(20261025, 20261030))
ORDER_SEEDS = list(range(20261030, 20261040))
FIT_ORDER_SEEDS = list(range(20261101, 20261106))
OPPOSITE = {"north": "south", "south": "north", "west": "east", "east": "west"}


def facing_neighbors(cell):
    row, col = int(cell[1:3]), int(cell[4:6])
    return [(f"r{r:02d}c{c:02d}", OPPOSITE[edge])
            for r, c, edge in [(row-1, col, "north"), (row+1, col, "south"),
                               (row, col-1, "west"), (row, col+1, "east")]
            if 0 <= r < 8 and 0 <= c < 8]


def mask_edges(mask):
    if mask.shape != (128, 128) or mask.dtype != np.bool_:
        raise ValueError("Expected a frozen 128x128 boolean severe mask")
    return {name: bool(strip.sum() >= 4) for name, strip in
            [("north", mask[:4]), ("south", mask[-4:]),
             ("west", mask[:, :4]), ("east", mask[:, -4:])]}


def quality(scenario, seed, sid, cid):
    if scenario != "mixed_survey":
        return {"native_survey": "native", "blurred_survey": "blurred"}[scenario]
    # Same quality assignment as the legacy cache environment, hidden online.
    value = int(hashlib.sha256(f"{seed}|{sid}|{cid}|survey_quality".encode()).hexdigest(), 16)
    return "native" if value % 2 == 0 else "blurred"


class PaidEnvironment:
    """No annotations. A mask is read only when its corresponding action is paid."""
    _verified_public_masks = {}

    def __init__(self, observations, sid, scenario, quality_seed, upgrade_cost):
        self._rows = {(r["cell_id"], r["mode"]): r for r in observations if r["sample_id"] == sid}
        self._sid, self._scenario, self._seed = sid, scenario, quality_seed
        self._upgrade_cost = upgrade_cost
        self._latest = {}

    def observe(self, cell, kind):
        previous = self._latest.get(cell)
        if kind == "survey" and previous is None:
            mode, cost = quality(self._scenario, self._seed, self._sid, cell), 1
        elif kind == "upgrade" and previous == "blurred":
            mode, cost = "native", self._upgrade_cost
        else:
            raise ValueError("Repeated or ineligible action")
        row = self._rows[cell, mode]
        key = (row["cache_file"], row["cache_sha256"])
        if key not in self._verified_public_masks:
            path = Path(row["cache_file"])
            if sha(path.read_bytes()) != row["cache_sha256"]:
                raise ValueError("Changed frozen mask")
            with np.load(path, allow_pickle=False) as cache:
                mask = cache["severe_mask"]
            if bool(mask.sum() >= 4) != row["declared_target"]:
                raise ValueError("Changed declaration")
            self._verified_public_masks[key] = {"cell_id": cell, "mode": mode,
                "declared_target": row["declared_target"], "edges": mask_edges(mask)}
        self._latest[cell] = mode
        # Explicit whitelist: no cache pathname, future quality or evaluator data.
        return {**self._verified_public_masks[key], "edges": dict(self._verified_public_masks[key]["edges"]), "cost": cost}


class PublicState:
    def __init__(self):
        self.latest = {}

    def observe(self, result):
        cell, mode = result["cell_id"], result["mode"]
        if cell not in CELLS or mode not in {"native", "blurred"}:
            raise ValueError("Invalid paid observation")
        old = self.latest.get(cell)
        if old and (old["mode"] != "blurred" or mode != "native"):
            raise ValueError("Observation is repeated or degrades native evidence")
        if type(result["declared_target"]) is not bool or set(result["edges"]) != set(OPPOSITE):
            raise ValueError("Invalid public prediction")
        if any(type(v) is not bool for v in result["edges"].values()):
            raise ValueError("Invalid edge flags")
        self.latest[cell] = {"mode": mode, "declared_target": result["declared_target"],
                             "edges": dict(result["edges"])}

    def candidates(self, remaining, upgrade_cost):
        actions = []
        for cell in CELLS:
            old = self.latest.get(cell)
            if old is None and remaining >= 1:
                actions.append((cell, "survey", 1))
            elif old and old["mode"] == "blurred" and remaining >= upgrade_cost:
                actions.append((cell, "upgrade", upgrade_cost))
        return actions

    def boundary_contact(self, cell):
        # No candidate mask or truth footprint is available here.
        return any(neighbor in self.latest and self.latest[neighbor]["edges"][edge]
                   for neighbor, edge in facing_neighbors(cell))


def stratum(kind, blurred_positive):
    return "survey" if kind == "survey" else f"upgrade_B{int(blurred_positive)}"


def fit_discount(records, pseudo_count=1.):
    """Records are offline reward labels, averaged per cell/mode/context first."""
    if pseudo_count <= 0:
        raise ValueError("Positive reward pseudocount required")
    per_cell = defaultdict(list)
    for r in records:
        local, new = r["local_new"], r["global_new"]
        if not 0 <= new <= local:
            raise ValueError("Global new reward must be a subset of local new reward")
        key = (r["sample_id"], r["cell_id"], r["mode"], r["stratum"], r["boundary_contact"])
        per_cell[key].append((local, new))
    groups = defaultdict(list)
    for key, values in per_cell.items():
        groups[key[-2:]].append(tuple(sum(x[i] for x in values) / len(values) for i in [0, 1]))
    params = {}
    for name in ["survey", "upgrade_B0", "upgrade_B1"]:
        params[name] = {}
        for contact in [False, True]:
            values = groups[name, contact]
            local, new = [sum(x[i] for x in values) for i in [0, 1]]
            params[name][str(int(contact))] = {
                "cell_mode_contexts": len(values), "mean_local_reward_sum": local,
                "mean_global_reward_sum": new, "reward_support": bool(local > 0),
                "remaining_fraction": (new+pseudo_count)/(local+pseudo_count) if contact and local > 0 else 1.}
    return {"stage": "SIX_SCENE_IN_SAMPLE_ENGINEERING_ONLY", "reward_pseudocount_toward_one": pseudo_count,
            "parameters": params, "zero_support_fallback": 1.,
            "no_contact_rule": "No discount. Absence of a predicted edge does not establish absence of duplicates.",
            "offline_action_records": len(records), "cell_mode_contexts": len(per_cell),
            "true_ids_retained_in_parameters": False}


def discount_factor(model, kind, positive, contact):
    return model["parameters"][stratum(kind, positive)][str(int(contact))]["remaining_fraction"]


def choose(state, cues, model, discount, policy, scenario, upgrade_cost, remaining, rng):
    if policy not in POLICIES:
        raise ValueError("Unknown fixed policy")
    ranked = []
    for cell, kind, cost in state.candidates(remaining, upgrade_cost):
        if policy in {"random_survey", "coverage_survey", "prior_survey"} and kind != "survey":
            continue
        old = state.latest.get(cell)
        positive = old["declared_target"] if old else None
        contact = state.boundary_contact(cell)
        factor = 1.
        if policy.startswith("random_"):
            base = rng.random()
        elif policy == "coverage_survey":
            base = 64 - CELLS.index(cell)
        elif policy in {"prior_survey", "prior_gate"}:
            # Gate is deliberately simple: revisit only a blurred negative.
            if kind == "upgrade" and positive:
                continue
            base = cues[cell] / cost
        else:
            base = action_score(model, cues[cell], kind, scenario=scenario,
                                blurred_positive=positive, cost=cost)
            if base is None:
                continue
            if policy == "boundary_count":
                factor = discount_factor(discount, kind, positive, contact)
        score = base * factor
        if not math.isfinite(score):
            raise ValueError("Non-finite action utility")
        ranked.append((-score, cell, kind, {"cell_id": cell, "kind": kind, "cost": cost,
            "base_utility": base, "discount": factor, "utility": score, "boundary_contact": contact}))
    return min(ranked, key=lambda x: x[:3])[3] if ranked else None


def run_episode(observations, sid, cues, model, discount, policy, scenario,
                quality_seed, order_seed, upgrade_cost, budget):
    environment = PaidEnvironment(observations, sid, scenario, quality_seed, upgrade_cost)
    state, rng, spent, trace = PublicState(), random.Random(order_seed), 0, []
    planning_s = 0.
    while spent < budget:
        start = time.perf_counter()
        action = choose(state, cues, model, discount, policy, scenario, upgrade_cost, budget-spent, rng)
        planning_s += time.perf_counter() - start
        if action is None:
            break
        result = environment.observe(action["cell_id"], action["kind"])
        if result["cost"] != action["cost"]:
            raise ValueError("Planner and environment costs differ")
        spent += action["cost"]
        trace.append({**action, "mode": result["mode"], "declared_target": result["declared_target"],
                      "edges": result["edges"], "cumulative_cost": spent})
        state.observe(result)
    return trace, planning_s


def matched(row, threshold=.1):
    return {r["uid"] for r in row["overlap_records"]
            if r["visible_pixels"] >= 4 and r["severe_overlap_pixels"] >= 4
            and r["visible_footprint_coverage"] >= threshold}


def training_records(trajectories, lookup):
    records = []
    for sid, trace in trajectories:
        found, per_cell, state = set(), defaultdict(set), PublicState()
        for action in trace:
            cell = action["cell_id"]
            row = lookup[sid, cell, action["mode"]]
            ids = matched(row)
            old = state.latest.get(cell)
            if row["target"] is not None:
                records.append({"sample_id": sid, "cell_id": cell, "mode": action["mode"],
                    "stratum": stratum(action["kind"], old["declared_target"] if old else None),
                    "boundary_contact": state.boundary_contact(cell),
                    "local_new": len(ids-per_cell[cell]), "global_new": len(ids-found)})
            found |= ids
            per_cell[cell] |= ids
            state.observe(action)
    return records


def evaluate(trace, lookup, sid, budget, threshold):
    found, per_cell = set(), defaultdict(set)
    false_cells, unknown_cells, unmatched_positive = set(), set(), set()
    duplicate, local_total, first, upgrades, contacts = 0, 0, None, 0, 0
    targets = set().union(*(set(r["visible_target_uids"]) for (s, _, _), r in lookup.items() if s == sid))
    for a in trace:
        if a["cumulative_cost"] > budget:
            break
        row = lookup[sid, a["cell_id"], a["mode"]]
        ids = matched(row, threshold)
        local = ids-per_cell[a["cell_id"]]
        new = ids-found
        local_total += len(local)
        duplicate += len(local & found)
        if new and first is None:
            first = a["cumulative_cost"]
        if row["predicted"]:
            if row["target"] is False: false_cells.add(a["cell_id"])
            if row["target"] is None: unknown_cells.add(a["cell_id"])
            if not ids: unmatched_positive.add(a["cell_id"])
        upgrades += a["kind"] == "upgrade"
        contacts += a["boundary_contact"]
        found |= ids
        per_cell[a["cell_id"]] |= ids
    return {"distinct_found": len(found), "visible_targets": len(targets),
            "recall": len(found)/len(targets) if targets else None,
            "local_new_match_events": local_total, "cross_cell_duplicate_match_events": duplicate,
            "false_positive_cells": len(false_cells), "unknown_positive_cells": len(unknown_cells),
            "unmatched_positive_cells": len(unmatched_positive), "upgrades": upgrades,
            "boundary_contact_actions": contacts, "first_discovery_cost": first,
            "first_discovery_censored": first is None}


def make_spatial_groups(selection, spatial):
    ids = {r["sample_id"] for r in selection}
    if ids != {r["sample_id"] for r in spatial["per_sample"]}:
        raise ValueError("Spatial audit and frozen cohort differ")
    components = {}
    for group in spatial["candidate_envelope_components"]:
        group_id = "bbox-" + sha("|".join(sorted(group)).encode())[:16]
        for sid in group:
            if sid in components: raise ValueError("Repeated spatial component member")
            components[sid] = group_id
    if set(components) != ids:
        raise ValueError("Incomplete spatial components")
    by_id = {r["sample_id"]: r for r in spatial["per_sample"]}
    result = []
    for sample in selection:
        sid = sample["sample_id"]
        row = by_id[sid]
        available = row["annotation_envelope_wgs84"] is not None
        result.append({**sample,
            "annotation_envelope_component": components[sid] if available else None,
            "analysis_group": components[sid] if available else "unverified-"+sample["event"],
            "geometry_status": "ANNOTATION_ENVELOPE_ONLY" if available else "MISSING_GEOMETRY",
            "training_envelope_overlap": bool(row["training_annotation_envelope_overlap_ids"]),
            "full_image_independence_verified": False,
            "membership_unchanged": True})
    return result


def macro_summary(metrics):
    # Average order/quality repeats within scenes, then scenes within events.
    scene_groups = defaultdict(list)
    for r in metrics:
        key = (r["policy"], r["scenario"], r["upgrade_cost"], r["budget"], r["threshold"], r["event"], r["sample_id"])
        scene_groups[key].append(r)
    fields = ["distinct_found", "false_positive_cells", "upgrades", "cross_cell_duplicate_match_events"]
    event_groups = defaultdict(list)
    scene_rows = []
    for key, rows in scene_groups.items():
        values = {k: sum(r[k] for r in rows)/len(rows) for k in fields}
        record = dict(zip(["policy", "scenario", "upgrade_cost", "budget", "threshold", "event", "sample_id"], key))
        scene_rows.append({**record, **values, "repeats": len(rows)})
        event_groups[key[:-1]].append(values)
    event_rows, macro = [], defaultdict(list)
    for key, rows in event_groups.items():
        values = {k: sum(r[k] for r in rows)/len(rows) for k in fields}
        event_rows.append({**dict(zip(["policy", "scenario", "upgrade_cost", "budget", "threshold", "event"], key)), **values})
        macro[key[:-1]].append(values)
    macro_rows = [{**dict(zip(["policy", "scenario", "upgrade_cost", "budget", "threshold"], key)),
                   **{k: sum(r[k] for r in rows)/len(rows) for k in fields}} for key, rows in macro.items()]
    return {"scene_means": scene_rows, "event_means": event_rows, "event_macro_means": macro_rows}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["perception", "components", "design", "spatial", "out"]:
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "observations": args.perception/"observations.json", "association": args.perception/"association_report.json",
        "cues": args.perception/"public_change_scores.json", "count_parameters": args.components/"component_parameters.json",
        "selection": args.design/"cohort_selection.json", "contract": args.design/"experiment_contract_v2_1.json",
        "probes": args.design/"next_development_probes.json", "spatial": args.spatial/"report.json"}
    sources = {k: sha(p.read_bytes()) for k, p in input_paths.items()}
    protocol = {"stage": "SIX_DEVELOPMENT_IN_SAMPLE_ENGINEERING_DIAGNOSTIC", "sources": sources,
        "runner_sha256": sha(Path(__file__).read_bytes()), "policies": POLICIES,
        "quality_scenarios": SCENARIOS, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS,
        "fit_history_order_seeds": FIT_ORDER_SEEDS, "fit_history_policy": "uniform eligible random action",
        "fit_history_budget": 64, "fit_history_upgrade_cost": 2,
        "fit_weighting": "Within cell/mode/stratum/boundary flag average rewards over histories; then pool ratios.",
        "discount": "If any paid neighboring 4px edge strip has >=4 severe pixels: (new_global+1)/(new_local+1), by survey/upgrade_B0/upgrade_B1. No support or no edge contact: factor 1.",
        "mask_access": "Only chosen/paid observation; expose four boolean edge flags, not ground truth or future masks.",
        "evaluation_budgets": [8,16,32], "separate_budget_trajectories": True,
        "upgrade_costs": [1,2,4], "measurement_thresholds": [.05,.1,.25],
        "spatial": "Preserve all fixed IDs. Connected annotation-envelope groups; missing geometry pooled by event as unverified; no full-image independence claim. Training overlaps are a separate stratum, never outcome-based exclusion.",
        "new_method_selected": False, "validation_or_confirmation_access": False,
        "limitations": ["Previously inspected six scenes, fitted and scored in sample; no effect-size go/no-go test.",
                        "Boundary contact is an imperfect proxy, not a predicted or true building identity.",
                        "Full-image geographic independence and complete 72-scene development fit remain pending."]}
    freeze(args.out/"protocol.json", protocol)
    data = {k: json.loads(p.read_text()) for k,p in input_paths.items()}
    expected = {r["sample_id"] for r in data["probes"] if r["split"] == "development"}
    if len(expected) != 6 or {r["sample_id"] for r in data["observations"]} != expected or set(data["cues"]) != expected:
        raise ValueError("Only the six frozen development probes are allowed")
    if {r["sample_id"] for r in data["association"]["rows"]} != expected:
        raise ValueError("Evaluator has scenes outside allowed probes")
    ledger = make_spatial_groups(data["selection"], data["spatial"])
    atomic_json(args.out/"spatial_groups.json", ledger)
    observations, cues, model = data["observations"], data["cues"], data["count_parameters"]
    lookup = {(r["sample_id"],r["cell_id"],r["mode"]): r for r in data["association"]["rows"]}
    fit_traces = []
    for sid in sorted(expected):
        for scenario in SCENARIOS:
            for quality_seed in QUALITY_SEEDS:
                for order_seed in FIT_ORDER_SEEDS:
                    trace,_ = run_episode(observations,sid,cues[sid],model,None,"random_upgrade",scenario,quality_seed,order_seed,2,64)
                    fit_traces.append((sid,trace))
    discount = fit_discount(training_records(fit_traces,lookup))
    atomic_json(args.out/"discount_parameters.json", discount)
    print(json.dumps({"stage":"OFFLINE_DISCOUNT_FIT_COMPLETE", "parameters":discount["parameters"]}),flush=True)
    metrics, comparisons, trajectories, planning_s = [], [], 0, 0.
    events = {r["sample_id"]:r["event"] for r in data["probes"]}
    with (args.out/"trajectories.jsonl").open("w") as handle:
        for sid in sorted(expected):
            for scenario in SCENARIOS:
                for upgrade_cost in [1,2,4]:
                    for quality_seed in QUALITY_SEEDS:
                        for budget in [8,16,32]:
                            paired = {}
                            for policy in POLICIES:
                                seeds = ORDER_SEEDS if policy.startswith("random_") else [ORDER_SEEDS[0]]
                                for order_seed in seeds:
                                    trace,elapsed = run_episode(observations,sid,cues[sid],model,discount,policy,scenario,quality_seed,order_seed,upgrade_cost,budget)
                                    config = {"sample_id":sid,"event":events[sid],"scenario":scenario,"upgrade_cost":upgrade_cost,
                                              "quality_seed":quality_seed,"order_seed":order_seed,"policy":policy,"budget":budget}
                                    handle.write(json.dumps({**config,"trace":trace},separators=(",",":"))+"\n")
                                    trajectories += 1
                                    planning_s += elapsed
                                    for threshold in [.05,.1,.25]:
                                        metrics.append({**config,"threshold":threshold,**evaluate(trace,lookup,sid,budget,threshold),
                                                        "planning_s":elapsed,"spent":trace[-1]["cumulative_cost"] if trace else 0})
                                    if policy in {"local_count","boundary_count"}: paired[policy] = trace
                            a,b = paired["local_count"],paired["boundary_count"]
                            comparisons.append({"sample_id":sid,"scenario":scenario,"upgrade_cost":upgrade_cost,
                                "quality_seed":quality_seed,"budget":budget,
                                "changed_action_order": [(x["cell_id"],x["kind"]) for x in a] != [(x["cell_id"],x["kind"]) for x in b],
                                "discounted_actions":sum(x["discount"] < 1 for x in b)})
            print(json.dumps({"stage":"SCENE_COMPLETE","sample_id":sid,"trajectories":trajectories}),flush=True)
    with (args.out/"metrics.jsonl").open("w") as handle:
        for r in metrics: handle.write(json.dumps(r,separators=(",",":"))+"\n")
    summary = macro_summary(metrics)
    atomic_json(args.out/"summary.json",summary)
    atomic_json(args.out/"paired_action_checks.json",comparisons)
    primary = [r for r in summary["event_macro_means"] if r["scenario"] == "mixed_survey" and r["upgrade_cost"] == 2 and r["budget"] == 32 and r["threshold"] == .1]
    spatial_summary = {"fixed_scenes":len(ledger),"annotation_envelope_groups":len({r["annotation_envelope_component"] for r in ledger if r["annotation_envelope_component"]}),
        "missing_geometry":sum(r["geometry_status"]=="MISSING_GEOMETRY" for r in ledger),
        "training_envelope_overlap_scenes":sum(r["training_envelope_overlap"] for r in ledger),
        "full_image_independence_verified":False,"membership_changed":False}
    preserved = {k:sha(p.read_bytes()) == sources[k] for k,p in input_paths.items()}
    report = {"stage":"SIX_SCENE_BUDGETED_SEARCH_ENGINEERING_COMPLETE", "fit_scenes":6,"scored_scenes":6,
        "fit_histories":len(fit_traces),"trajectories":trajectories,"trajectory_threshold_scores":len(metrics),
        "primary_setting_diagnostic":primary,"paired_configs":len(comparisons),
        "paired_configs_with_changed_action_order":sum(r["changed_action_order"] for r in comparisons),
        "discounted_actions_across_paired_configs":sum(r["discounted_actions"] for r in comparisons),
        "planning_wall_s":planning_s,"spatial":spatial_summary,"inputs_preserved":preserved,
        "trajectory_sha256":sha((args.out/"trajectories.jsonl").read_bytes()),
        "metrics_sha256":sha((args.out/"metrics.jsonl").read_bytes()),
        "independent_policy_improvement_demonstrated":False,"new_method_selected":False,
        "full_72_development_completed":False,"gpu_rented":False}
    if not all(preserved.values()): raise ValueError("Source artifacts changed")
    atomic_json(args.out/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
