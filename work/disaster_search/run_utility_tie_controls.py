"""Prospective tie-only controls for the frozen development72 count proxy.

No new models or observations. Public-parameter symmetry classes are expanded
back to all eligible cells; their representative's identical Q is reused.
"""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import time

from build_xbd_mini import atomic_json, sha
from cross_cell_count_smoke import CELLS
from plan_search_v2 import freeze
from run_budget_planning import EPS, POLICIES, action_values, planning_inputs
from run_development_budget_controls import (PaidBinaryEnvironment, QUALITY_SEEDS,
    ORDER_SEEDS, SCENARIOS, THRESHOLDS, aggregate_scene_metrics, private_evaluator,
    score_trace, summarize)

TIES = ["row_major", "raw_change", "fixed_random"]
CONFIG_FIELDS = ["sample_id", "fit_scope", "scenario", "upgrade_cost", "budget",
                 "quality_seed", "order_seed", "policy", "tie_rule"]
SOURCE_NAMES = ["run_utility_tie_controls.py", "run_budget_planning.py",
    "run_development_budget_controls.py", "paired_count_baseline.py",
    "cross_cell_count_smoke.py", "build_xbd_mini.py", "plan_search_v2.py"]


def symmetry_key(i, state, inputs):
    status = state[i]
    return (status, tuple(inputs["reward"][i]),
            inputs["blur_positive"][i] if status == 0 else None)


def expanded_values(state, inputs, remaining, cost, policy):
    grouped = action_values(state, inputs, remaining, cost, policy)
    if policy == "proxy_greedy":
        return grouped
    values = {symmetry_key(i, state, inputs): (q, greedy) for i, q, greedy in grouped}
    result = []
    for i, status in enumerate(state):
        if status == 3 or (1 if status == 0 else cost) > remaining:
            continue
        if inputs["reward"][i][status] is None:
            continue
        q, greedy = values[symmetry_key(i, state, inputs)]
        result.append((i, q, greedy))
    return result


def tie_ranks(features, rule, seed):
    """One fixed permutation of cells per rule/seed; no future or labels."""
    if rule not in TIES:
        raise ValueError("Unknown tie rule")
    order = list(range(len(CELLS)))
    if rule == "raw_change":
        changes = [features[c]["change"] for c in CELLS]
        if any(not math.isfinite(v) for v in changes):
            raise ValueError("Nonfinite public change")
        order.sort(key=lambda i: (-changes[i], i))
    elif rule == "fixed_random":
        random.Random(seed).shuffle(order)
    ranks = [0] * len(order)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def choose(state, inputs, remaining, cost, policy, ranks):
    # Iterate in the original row-major order. Original EPS and the secondary
    # greedy objective are unchanged. Only the final exact tie is altered.
    best = None
    for i, q, greedy in expanded_values(state, inputs, remaining, cost, policy):
        if not math.isfinite(q):
            raise ValueError("Nonfinite planning value")
        if best is None:
            best = (i, q, greedy)
            continue
        if policy == "proxy_greedy":
            better = q > best[1] or (q == best[1] and ranks[i] < ranks[best[0]])
        else:
            better = q > best[1] + EPS or (abs(q - best[1]) <= EPS and
                (greedy > best[2] or (greedy == best[2] and ranks[i] < ranks[best[0]])))
        if better:
            best = (i, q, greedy)
    return best


def run_episode(environment, inputs, policy, budget, cost, ranks):
    state = [0] * len(inputs["reward"])
    spent = 0
    trace = []
    while spent < budget:
        selected = choose(state, inputs, budget - spent, cost, policy, ranks)
        if selected is None:
            break
        i, q, greedy = selected
        kind = "survey" if state[i] == 0 else "upgrade"
        action_cost = 1 if state[i] == 0 else cost
        result = environment.observe(CELLS[i], kind)
        if result["cost"] != action_cost:
            raise ValueError("Cost differs from frozen environment")
        spent += action_cost
        trace.append({"cell_id": CELLS[i], "kind": kind, "mode": result["mode"],
            "declared_target": result["declared_target"], "cost": action_cost,
            "cumulative_cost": spent, "utility": q, "greedy_utility": greedy})
        state[i] = 3 if result["mode"] == "native" else 2 if result["declared_target"] else 1
    return trace


def verify_receipt(directory):
    receipt = json.loads((directory / "completion_receipt.json").read_text())
    for name, digest in receipt["files"].items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError(f"Frozen previous artifact changed: {directory / name}")


def aggregate_ties(local):
    result = []
    for rule in TIES:
        result.extend({**r, "tie_rule": rule} for r in
                      aggregate_scene_metrics([r for r in local if r["tie_rule"] == rule]))
    return result


def summarize_ties(scene_means, spatial):
    result = {}
    for rule in TIES:
        part = summarize([r for r in scene_means if r["tie_rule"] == rule], spatial)
        for key, rows in part.items():
            result.setdefault(key, []).extend({**r, "tie_rule": rule} for r in rows)
    return result


def is_primary(row):
    return (row["scenario"], row["upgrade_cost"], row["budget"], row["threshold"]) == ("mixed_survey", 2, 32, .1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["planning", "previous", "perception", "design", "spatial", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing to overwrite a completed run")
    paths = {"models": args.previous / "models.json", "features": args.previous / "public_features.json",
        "previous_receipt": args.previous / "completion_receipt.json",
        "planning_receipt": args.planning / "completion_receipt.json",
        "planning_decision": args.planning / "next_step_decision.json",
        "observations": args.perception / "observations.json",
        "association": args.perception / "association_report.json",
        "selection": args.design / "next_development_batch.json",
        "contract": args.design / "experiment_contract_v2_2.json", "spatial": args.spatial}
    sources = {key: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for key, p in paths.items()}
    code = {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES}
    # Frozen before reading outcome cache or generating trajectories.
    freeze(args.out / "protocol.json", {"stage": "DEVELOPMENT72_TIE_ONLY_CONTROLS",
        "sources": sources, "source_code_sha256": code, "policies": POLICIES,
        "tie_rules": TIES, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS,
        "scenarios": SCENARIOS, "upgrade_costs": [1, 2, 4], "budgets": [8, 16, 32],
        "thresholds": THRESHOLDS, "primary": {"scenario": "mixed_survey", "upgrade_cost": 2, "budget": 32, "threshold": .1},
        "fit_scopes": {"development_fit": "All 72 fit, entire grid; in-sample.",
            "event_holdout": "Frozen other-two-development-events models; primary configuration only. Previously inspected data; not independent validation."},
        "expected_trajectories": 362880, "expected_threshold_scores": 1088640,
        "tie_only": "Same four-bin count means, paired likelihoods, actions, costs, eligibility, cached outcomes and quality maps. No fit or utility change.",
        "candidate_symmetry": "Expand each exact state/reward/branch-probability class to all eligible cells with the same representative Q. Same candidate universe under all rules; no shortlist.",
        "tie_comparison": "Original row-major candidate iteration. Greedy strict score with exact equality ties. Planning original EPS=1e-12, then higher original greedy utility, then new cell rank. No EPS in greedy or secondary utility.",
        "ranks": "Row-major; descending initial public raw change with row-major fallback; Python random.Random(order_seed).shuffle(range(64)). One permutation fixed throughout each trajectory, identical across scenes/policies/quality seeds/settings for a given order_seed. Random rules averaged across all ten established seeds, never select a seed.",
        "original_controls": "Reuse frozen prior_survey, count_survey and count_negative scene means as references; every row-major trajectory must reproduce the preceding planning run.",
        "statistics": "Average quality/order repeats within scene, then scenes within event and events equally. Random order range is descriptive algorithm sensitivity, not a confidence interval. All thresholds and grid reported, no outcome-dependent rule/setting selection.",
        "limits": ["Same-cell count proxy can duplicate global buildings.",
            "Batch tail omits future unlocks by other surveys; not full MF-ENS.",
            "Mexico has zero cached detectable targets; retains denominator.",
            "Full-image spatial independence unverified; fitted detector has seen all three events."],
        "no_tuning_after_run": True, "new_method_selected": False,
        "validation_or_confirmation_access": False, "new_inference_calls": 0, "paid_gpu": False})
    verify_receipt(args.planning)
    verify_receipt(args.previous)
    selected = json.loads(paths["selection"].read_text())
    ids = {r["sample_id"] for r in selected}
    if len(ids) != 72 or any(r["split"] != "development" for r in selected):
        raise ValueError("Exact 72 development scenes required")
    models = json.loads(paths["models"].read_text())
    features = json.loads(paths["features"].read_text())
    observations = json.loads(paths["observations"].read_text())
    association = json.loads(paths["association"].read_text())
    if len(observations) != 9216 or {r["sample_id"] for r in observations} != ids or set(features) != ids:
        raise ValueError("Foreign or incomplete development cache")
    if association["stage"] != "DEVELOPMENT72_SENSOR_QA" or len(association["rows"]) != 9216 or {r["sample_id"] for r in association["rows"]} != ids:
        raise ValueError("Foreign or incomplete private evaluator")
    by_obs = defaultdict(list)
    by_rows = defaultdict(list)
    for r in observations:
        by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:
        by_rows[r["sample_id"]].append(r)
    spatial = {r["sample_id"]: r for r in json.loads(args.spatial.read_text())}
    scene_means = []
    trajectories = scores = 0
    planning_s = 0.
    started = time.perf_counter()
    with gzip.open(args.out / "trajectories.jsonl.gz", "wt", compresslevel=3) as traces, gzip.open(args.out / "metrics.jsonl.gz", "wt", compresslevel=3) as metrics:
        for sample in selected:
            sid, event = sample["sample_id"], sample["event"]
            evaluator, targets = private_evaluator(by_rows[sid])
            local = []
            ranks = {(rule, seed): tie_ranks(features[sid], rule, seed) for rule in TIES
                     for seed in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1])}
            for scope, params, configs in [
                ("development_fit", models["development_fit"], [(s, c, b) for s in SCENARIOS for c in [1, 2, 4] for b in [8, 16, 32]]),
                ("event_holdout", models["held_" + event], [("mixed_survey", 2, 32)])]:
                for scenario, cost, budget in configs:
                    inputs = planning_inputs(features[sid], params, scenario, cost)
                    for quality_seed in QUALITY_SEEDS:
                        for policy in POLICIES:
                            for (rule, order_seed), rank in ranks.items():
                                env = PaidBinaryEnvironment(by_obs[sid], sid, scenario, quality_seed, cost)
                                t = time.perf_counter()
                                trace = run_episode(env, inputs, policy, budget, cost, rank)
                                elapsed = time.perf_counter() - t
                                planning_s += elapsed
                                config = {"sample_id": sid, "event": event, "fit_scope": scope, "policy": policy,
                                    "scenario": scenario, "upgrade_cost": cost, "budget": budget,
                                    "quality_seed": quality_seed, "order_seed": order_seed, "tie_rule": rule}
                                traces.write(json.dumps({**config, "trace": trace}, separators=(",", ":")) + "\n")
                                trajectories += 1
                                for result in score_trace(trace, evaluator, targets):
                                    record = {**config, **result, "planning_s": elapsed}
                                    local.append(record)
                                    metrics.write(json.dumps(record, separators=(",", ":")) + "\n")
                                    scores += 1
            scene_means.extend({"sample_id": sid, "event": event, **r} for r in aggregate_ties(local))
            print(json.dumps({"stage": "SCENE_COMPLETE", "sample_id": sid, "trajectories": trajectories}), flush=True)
    atomic_json(args.out / "scene_means.json", scene_means)
    summary = summarize_ties(scene_means, spatial)
    atomic_json(args.out / "summary.json", summary)
    preserved = {k: sha(p.read_bytes()) == sources[k]["sha256"] for k, p in paths.items()}
    if not all(preserved.values()):
        raise ValueError("Source artifacts changed")
    if trajectories != 362880 or scores != 1088640:
        raise ValueError("Incomplete fixed grid")
    report = {"stage": "DEVELOPMENT72_TIE_CONTROLS_RUN_COMPLETE", "scenes": 72,
        "trajectories": trajectories, "threshold_scores": scores,
        "primary": [r for r in summary["event_macro_means"] if is_primary(r)],
        "planning_wall_s": planning_s, "total_wall_s": time.perf_counter() - started,
        "sources_preserved": preserved, "new_method_selected": False,
        "validation_or_confirmation_scored": False, "independent_efficacy_claim": False,
        "new_inference_calls": 0, "gpu_rented": False,
        "trajectory_sha256": sha((args.out / "trajectories.jsonl.gz").read_bytes()),
        "metrics_sha256": sha((args.out / "metrics.jsonl.gz").read_bytes())}
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "primary"}), flush=True)


if __name__ == "__main__":
    main()
