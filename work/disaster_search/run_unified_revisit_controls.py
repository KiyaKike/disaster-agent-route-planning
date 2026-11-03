"""Matched-tie revisit ablations on frozen development72 caches and models."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time

from build_xbd_mini import atomic_json, sha
from cross_cell_count_smoke import CELLS
from plan_search_v2 import freeze
from run_development_budget_controls import (PaidBinaryEnvironment, QUALITY_SEEDS,
    ORDER_SEEDS, SCENARIOS, THRESHOLDS, utility_table, private_evaluator, score_trace)
from run_utility_tie_controls import (TIES, CONFIG_FIELDS, aggregate_ties,
    summarize_ties, tie_ranks, is_primary, verify_receipt)

POLICIES = ["prior_survey", "prior_gate", "count_survey", "count_negative", "count_all", "presence"]
SOURCE_NAMES = ["run_unified_revisit_controls.py", "run_utility_tie_controls.py",
    "run_budget_planning.py", "run_development_budget_controls.py", "paired_count_baseline.py",
    "cross_cell_count_smoke.py", "build_xbd_mini.py", "plan_search_v2.py"]


def choose(state, table, remaining, cost, ranks):
    best = None
    for i, status in enumerate(state):
        if status == 3 or (1 if status == 0 else cost) > remaining:
            continue
        value = table[i][status]
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError("Nonfinite utility")
        if best is None or value > best[1] or (value == best[1] and ranks[i] < ranks[best[0]]):
            best = (i, value)
    return best


def run_episode(environment, table, budget, cost, ranks):
    state = [0] * len(table)
    spent = 0
    trace = []
    while spent < budget:
        selected = choose(state, table, budget - spent, cost, ranks)
        if selected is None:
            break
        i, value = selected
        kind = "survey" if state[i] == 0 else "upgrade"
        action_cost = 1 if state[i] == 0 else cost
        result = environment.observe(CELLS[i], kind)
        if result["cost"] != action_cost:
            raise ValueError("Environment cost mismatch")
        spent += action_cost
        trace.append({"cell_id": CELLS[i], "kind": kind, "mode": result["mode"],
            "declared_target": result["declared_target"], "cost": action_cost,
            "cumulative_cost": spent, "utility": value})
        state[i] = 3 if result["mode"] == "native" else 2 if result["declared_target"] else 1
    return trace


def previous_count_rows(path):
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            if row["policy"] == "proxy_greedy":
                yield row


def import_count_trace(source, config):
    row = next(source)
    for key in CONFIG_FIELDS:
        if key != "policy" and row[key] != config[key]:
            raise ValueError("Reused count trajectory configuration differs")
    if row["policy"] != "proxy_greedy" or config["policy"] != "count_all":
        raise ValueError("Wrong source policy")
    trace = []
    for action in row["trace"]:
        if action["utility"] != action["greedy_utility"]:
            raise ValueError("Source was not original count greedy")
        trace.append({k: v for k, v in action.items() if k != "greedy_utility"})
    return trace


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["ties", "previous", "perception", "design", "spatial", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing to overwrite a completed run")
    paths = {"models": args.previous / "models.json", "features": args.previous / "public_features.json",
        "previous_receipt": args.previous / "completion_receipt.json", "ties_receipt": args.ties / "completion_receipt.json",
        "decision": args.ties / "next_step_decision.json", "reused_trajectories": args.ties / "trajectories.jsonl.gz",
        "observations": args.perception / "observations.json", "association": args.perception / "association_report.json",
        "selection": args.design / "next_development_batch.json", "contract": args.design / "experiment_contract_v2_2.json",
        "spatial": args.spatial}
    sources = {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k, p in paths.items()}
    freeze(args.out / "protocol.json", {"stage": "DEVELOPMENT72_UNIFIED_TIE_REVISIT_CONTROLS",
        "sources": sources, "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES},
        "policies": POLICIES, "tie_rules": TIES, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS,
        "scenarios": SCENARIOS, "upgrade_costs": [1, 2, 4], "budgets": [8, 16, 32], "thresholds": THRESHOLDS,
        "primary": {"scenario": "mixed_survey", "budget": 32, "upgrade_cost": 2, "threshold": .1},
        "fit_scopes": {"development_fit": "Same frozen all72 models; entire grid; in-sample.",
            "event_holdout": "Same other-two-development-events models; primary scenario/cost/budget only. Previously inspected data, not independent validation."},
        "expected_trajectories": 725760, "computed_trajectories": 604800, "reused_count_all_trajectories": 120960,
        "expected_threshold_scores": 2177280,
        "reuse": "Import only original proxy_greedy from preceding tie run with every config key matched; strip redundant greedy_utility. No inference or recomputation of its decisions. Every imported trace re-scored. planning_s=0 for reused traces means no decision computation this run, not intrinsic algorithm runtime.",
        "action_rules": "Original utility_table values unchanged: survey cost1; one native upgrade after blurred; no native repeat. count_survey/prior_survey disable upgrades; count_negative/prior_gate disable B+ upgrades. presence keeps original zero B+ utility rather than adding a new stopping rule.",
        "ties": "Strict original utility comparison, exact equality only. Same fixed row/raw-change/random cell ranks as preceding tie experiment. All10 random orders and all5 quality seeds averaged within scene; no best-seed selection.",
        "causal_comparisons": ["prior_gate_vs_prior_survey", "count_negative_vs_count_survey", "count_all_vs_count_negative", "count_all_vs_count_survey"],
        "other_comparisons": ["presence_vs_count_all", "presence_vs_count_negative", "count_negative_vs_prior_gate", "count_all_vs_prior_gate", "presence_vs_prior_gate"],
        "diagnostic_stop": "Apply preceding frozen decision: if count positive revisits do not have consistent net primary gain over survey-only/negative-only under matched ties in both development fit scopes, stop promoting this positive-revisit policy; do not tune more heuristics on these72. Cost1-only gains do not replace primary cost2. This is not the independent-validation go criterion.",
        "statistics": "Average repetitions within scene, then scenes within event, then events equally. Report all settings, all thresholds, event/spatial/train-envelope strata; retain Mexico and zero-target scenes. Seeds are not independent disasters.",
        "no_fit_or_tuning": True, "new_method_selected": False, "validation_or_confirmation_access": False,
        "new_inference_calls": 0, "paid_gpu": False,
        "limits": ["Same-cell count reward is not exact global new-building utility.", "Only two development events have cached detections; Mexico remains zero.",
            "Detector has training-event exposure; full-image geographic independence unverified.", "No flight-distance, energy, return path, or new-viewpoint model."]})
    verify_receipt(args.ties)
    verify_receipt(args.previous)
    selected = json.loads(paths["selection"].read_text())
    ids = {r["sample_id"] for r in selected}
    if len(ids) != 72 or any(r["split"] != "development" for r in selected):
        raise ValueError("Exact72 development scenes required")
    features = json.loads(paths["features"].read_text())
    models = json.loads(paths["models"].read_text())
    observations = json.loads(paths["observations"].read_text())
    association = json.loads(paths["association"].read_text())
    if set(features) != ids or len(observations) != 9216 or {r["sample_id"] for r in observations} != ids:
        raise ValueError("Incomplete or foreign features/cache")
    if association["stage"] != "DEVELOPMENT72_SENSOR_QA" or len(association["rows"]) != 9216 or {r["sample_id"] for r in association["rows"]} != ids:
        raise ValueError("Incomplete or foreign evaluator")
    by_obs = defaultdict(list)
    by_rows = defaultdict(list)
    for r in observations:
        by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:
        by_rows[r["sample_id"]].append(r)
    spatial = {r["sample_id"]: r for r in json.loads(args.spatial.read_text())}
    imported = previous_count_rows(paths["reused_trajectories"])
    scene_means = []
    trajectories = scores = reused = 0
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
                    tables = {policy: utility_table(features[sid], params["count"], params["presence"], scenario, cost, policy) for policy in POLICIES}
                    for quality_seed in QUALITY_SEEDS:
                        for policy in POLICIES:
                            for (rule, order_seed), rank in ranks.items():
                                config = {"sample_id": sid, "event": event, "fit_scope": scope, "policy": policy,
                                    "scenario": scenario, "upgrade_cost": cost, "budget": budget,
                                    "quality_seed": quality_seed, "order_seed": order_seed, "tie_rule": rule}
                                if policy == "count_all":
                                    trace = import_count_trace(imported, config)
                                    elapsed = 0.
                                    reused += 1
                                    origin = "reused_count_greedy"
                                else:
                                    env = PaidBinaryEnvironment(by_obs[sid], sid, scenario, quality_seed, cost)
                                    t = time.perf_counter()
                                    trace = run_episode(env, tables[policy], budget, cost, rank)
                                    elapsed = time.perf_counter() - t
                                    origin = "computed_this_run"
                                planning_s += elapsed
                                traces.write(json.dumps({**config, "execution_origin": origin, "trace": trace}, separators=(",", ":")) + "\n")
                                trajectories += 1
                                for result in score_trace(trace, evaluator, targets):
                                    record = {**config, **result, "planning_s": elapsed}
                                    metrics.write(json.dumps(record, separators=(",", ":")) + "\n")
                                    local.append(record)
                                    scores += 1
            scene_means.extend({"sample_id": sid, "event": event, **r} for r in aggregate_ties(local))
            print(json.dumps({"stage": "SCENE_COMPLETE", "sample_id": sid, "trajectories": trajectories, "reused": reused}), flush=True)
    if next(imported, None) is not None:
        raise ValueError("Unconsumed source count trajectories")
    if (trajectories, reused, scores) != (725760, 120960, 2177280):
        raise ValueError("Incomplete fixed grid")
    atomic_json(args.out / "scene_means.json", scene_means)
    summary = summarize_ties(scene_means, spatial)
    atomic_json(args.out / "summary.json", summary)
    preserved = {k: sha(p.read_bytes()) == sources[k]["sha256"] for k, p in paths.items()}
    if not all(preserved.values()):
        raise ValueError("Source artifacts changed")
    report = {"stage": "DEVELOPMENT72_UNIFIED_REVISIT_RUN_COMPLETE", "scenes": 72,
        "trajectories": trajectories, "computed_trajectories": trajectories - reused,
        "reused_count_all_trajectories": reused, "threshold_scores": scores,
        "primary": [r for r in summary["event_macro_means"] if is_primary(r)],
        "decision_compute_s": planning_s, "total_wall_s": time.perf_counter() - started,
        "timing_note": "Imported count_all has planning_s=0; timing not a fresh runtime comparison.",
        "sources_preserved": preserved, "new_method_selected": False,
        "independent_efficacy_claim": False, "validation_or_confirmation_scored": False,
        "new_inference_calls": 0, "gpu_rented": False,
        "trajectory_sha256": sha((args.out / "trajectories.jsonl.gz").read_bytes()),
        "metrics_sha256": sha((args.out / "metrics.jsonl.gz").read_bytes())}
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "primary"}), flush=True)


if __name__ == "__main__":
    main()
