"""Posthoc simple coverage controls over unchanged calibration/validation caches.

No new detector inference, training, validation refit, or test-set access.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_cached_search import (Action, BUDGETS, SCENARIOS, CacheEnvironment, Evaluator,
                               Planner, PublicCell, digest, survey_mode)
from run_changeos_pilot import freeze_plan


CONTROL_POLICIES = ("prior_survey", "uniform_survey", "uniform_gate", "spatial_survey",
                    "spatial_gate", "random_survey", "random_gate")
ORDER_SEEDS = tuple(range(20261030, 20261040))
PROTOCOL = {
    "version": "coverage_controls_v1", "splits": ["calibration", "validation"],
    "timing": "Posthoc supplement after inspecting calibration and validation policy results. Definitions frozen before these controls run; not a new blind validation.",
    "policies": list(CONTROL_POLICIES), "random_order_seeds": list(ORDER_SEEDS),
    "budgets": list(BUDGETS), "quality_scenarios_and_seeds": SCENARIOS,
    "uniform_prior": "Original calibration global smoothed prevalence, identical for all cells and events; not the validation prevalence.",
    "uniform_order": "Euclidean nearest tie-break, exactly the original quality-gate selector after flattening its prior.",
    "spatial_order": "Fixed maximin ordering of public grid coordinates, starting at lexicographically first cell; iteratively farthest from already ordered cells, lexicographic ties. Spreads early observations over the map. No motion cost is charged in this E0 proxy.",
    "random_order": "Sort SHA256(search_seed|sample_id|cell_id|coverage_order), independent of labels and quality seed. Same cell order paired between random_survey and random_gate. Report all 10 orders, their mean and range; never select best seed.",
    "survey_variants": "No upgrades, static priority with the indicated prior/order. Standard cached survey quality unchanged.",
    "gate_variants": "Same action inventory and quality-gate posterior as existing controls: survey cost1, one native upgrade cost2 after negative blurred survey. Common expected-presence-per-cost criterion. Ordered coverage changes tie-breaking only; it does not override upgrade scores.",
    "prior_survey": "Original four-bin fitted probabilities and original nearest tie-break, no upgrades. Reproduces the old prior baseline to isolate observation-upgrade effects.",
    "random_exact": "For random survey only, compute exact expected unique discoveries at budget k: sum_buildings [1-C(64-m,k)/C(64,k)], m=number of survey cells whose fixed outcome matches this building. Evaluator-only expectation is never passed to planner. Also exact known-negative positive-cell expectation.",
    "metrics": "Unique target buildings, false-positive cells, upgrades/surveys, discovery-scene count, conditional first-discovery cost AND budget+1-censored first-discovery cost for scenes with known targets. No-target recall stays null.",
    "aggregation": "Within each scene/quality map, average random-order seeds first; average quality maps second, then sum discoveries across 24 scenes. Seeds are Monte Carlo allocations, not independent events. Retain per-event and scene-paired comparisons.",
    "inference": "Fixed cached outcomes only. Calibration parameters and all earlier reports/traces unchanged. No test inference or evaluation.",
    "limitations": ["Posthoc controls cannot turn previously inspected validation results into an untouched test.",
        "Validation targets occur only in Michael; the sampled Mexico scenes contain no severe-damage targets.",
        "Artificial blur restoration with declared costs is not physical drone revisiting or measured compute cost.",
        "The sensor checkpoint is expected to have training exposure; target-building coverage matching is an evaluator proxy, not instance AP.",
        "Presence-oriented acquisition and terminal positive cells may miss additional targets in a dense cell; travel costs are excluded."]}


def spatial_order(cells):
    remaining = {c.cell_id: c for c in cells}
    if not remaining:
        return []
    ordered = [min(remaining)]
    selected = [remaining.pop(ordered[0])]
    while remaining:
        cid = min(remaining, key=lambda cid: (
            -min((remaining[cid].x - s.x) ** 2 + (remaining[cid].y - s.y) ** 2 for s in selected), cid))
        ordered.append(cid)
        selected.append(remaining.pop(cid))
    return ordered


def random_order(cells, sid, seed):
    return sorted((c.cell_id for c in cells), key=lambda cid: (
        hashlib.sha256(f"{seed}|{sid}|{cid}|coverage_order".encode()).hexdigest(), cid))


def exact_random_discoveries(total_cells, budget, matched_cell_counts):
    if not 0 <= budget <= total_cells or any(not 0 <= m <= total_cells for m in matched_cell_counts):
        raise ValueError("Invalid sampling population")
    denominator = math.comb(total_cells, budget)
    return sum(1 - (math.comb(total_cells - m, budget) / denominator
                    if total_cells - m >= budget else 0.) for m in matched_cell_counts)


class CoveragePlanner(Planner):
    def __init__(self, cells, policy, rates, scenario, sid, seed=-1):
        if policy not in CONTROL_POLICIES:
            raise ValueError("Unknown control policy")
        self.control_policy = policy
        self.order = (spatial_order(cells) if policy.startswith("spatial") else
                      random_order(cells, sid, seed) if policy.startswith("random") else None)
        self.rank = {cid: i for i, cid in enumerate(self.order or [])}
        super().__init__(cells, "quality_gate" if policy.endswith("gate") else "prior", rates, scenario)

    def choose(self, remaining):
        if self.order is None:
            return super().choose(remaining)
        candidates = []
        for cid in self.cells:
            previous = self.latest.get(cid)
            if previous is None:
                kind, score = "survey", self.current(cid) * self.survey_rates["tpr"]
            elif (self.policy != "prior" and previous["mode"] == "blurred" and
                  not previous["declared_target"] and (cid, "upgrade") not in self.used):
                kind, score = "upgrade", self.current(cid) * self.rates["native"]["tpr"] / 2
            else:
                continue
            action = Action(cid, kind)
            if action.cost <= remaining:
                candidates.append((-score, self.rank[cid], cid, kind))
        if not candidates:
            return None
        _, _, cid, kind = min(candidates)
        return Action(cid, kind)


def run_control(cells, policy, rates, scenario, budget, sid, seed, provider, evaluator):
    planner = CoveragePlanner(cells, policy, rates, scenario, sid, seed)
    cost, trace = 0, []
    while (action := planner.choose(budget - cost)) is not None:
        observation = provider(action)
        audit = planner.observe(action, observation)
        cost += action.cost
        evaluator.observe(observation, cost)
        trace.append({"action": asdict(action), "cost_total": cost, "observation": observation,
                      **{k: audit[k] for k in ("belief_before", "original_prior", "belief_after", "representative_replaced")}})
    result = {**evaluator.result(), "cost": cost, "actions": len(trace),
              "surveys": sum(t["action"]["kind"] == "survey" for t in trace),
              "upgrades": sum(t["action"]["kind"] == "upgrade" for t in trace),
              "evidence_groups": len(planner.registry.representatives())}
    return result, trace


def summarize_scene(rows):
    return {"sample_id": rows[0]["sample_id"], "event": rows[0]["event"], "scene_type": rows[0]["scene_type"],
        "target_count": rows[0]["target_count"], "runs_averaged": len(rows),
        "buildings_found": statistics.mean(r["unique_buildings_found"] for r in rows),
        "false_positive_cells": statistics.mean(r["false_positive_cells"] for r in rows),
        "upgrades": statistics.mean(r["upgrades"] for r in rows),
        "surveys": statistics.mean(r["surveys"] for r in rows),
        "discovery_probability": statistics.mean(not r["no_true_discovery"] for r in rows),
        "first_cost_censored": statistics.mean(r["first_discovery_cost"] if r["first_discovery_cost"] is not None else r["budget"] + 1 for r in rows) if rows[0]["target_count"] else None}


def aggregate(results, reference_rows, exact_rows):
    full = results + reference_rows
    grouped = defaultdict(list)
    for r in full:
        grouped[(r["split"], r["scenario"], r["budget"], r["policy"], r["sample_id"])].append(r)
    scene_summaries = [{"split": key[0], "scenario": key[1], "budget": key[2], "policy": key[3], **summarize_scene(rows)}
                       for key, rows in sorted(grouped.items())]
    summaries = []
    for split, scenario, budget, policy in sorted({k[:4] for k in grouped}):
        selected = [r for r in scene_summaries if (r["split"], r["scenario"], r["budget"], r["policy"]) == (split, scenario, budget, policy)]
        target_scenes = [r for r in selected if r["target_count"]]
        rows = [r for r in full if (r["split"], r["scenario"], r["budget"], r["policy"]) == (split, scenario, budget, policy)]
        first_costs = [r["first_discovery_cost"] for r in rows if r["first_discovery_cost"] is not None]
        summary = {"split": split, "scenario": scenario, "budget": budget, "policy": policy,
            "scenes": len(selected), "target_scenes": len(target_scenes),
            "sum_buildings_found": sum(r["buildings_found"] for r in selected),
            "sum_false_positive_cells": sum(r["false_positive_cells"] for r in selected),
            "mean_upgrades": statistics.mean(r["upgrades"] for r in selected),
            "mean_surveys": statistics.mean(r["surveys"] for r in selected),
            "mean_target_scenes_with_discovery": sum(r["discovery_probability"] for r in target_scenes),
            "mean_first_cost_censored_target_scenes": statistics.mean(r["first_cost_censored"] for r in target_scenes) if target_scenes else None,
            "first_cost_conditional_on_discovery": statistics.mean(first_costs) if first_costs else None,
            "by_event": {event: {"sum_buildings_found": sum(r["buildings_found"] for r in selected if r["event"] == event),
                                "sum_false_positive_cells": sum(r["false_positive_cells"] for r in selected if r["event"] == event)}
                         for event in sorted({r["event"] for r in selected})}}
        if policy.startswith("random"):
            per_order = []
            for seed in ORDER_SEEDS:
                chosen = [r for r in rows if r["search_seed"] == seed]
                per_order.append({"search_seed": seed, "sum_buildings_found": sum(r["unique_buildings_found"] for r in chosen) / len(SCENARIOS[scenario])})
            summary["all_order_results"] = per_order
            summary["random_order_minimum_found"] = min(r["sum_buildings_found"] for r in per_order)
            summary["random_order_maximum_found"] = max(r["sum_buildings_found"] for r in per_order)
        summaries.append(summary)
    paired = []
    for split in ("calibration", "validation"):
        for scenario in SCENARIOS:
            for budget in BUDGETS:
                for a, b in (("quality_gate", "uniform_gate"), ("quality_gate", "spatial_gate"), ("quality_gate", "random_gate"),
                             ("local_residual", "random_gate"), ("local_residual", "spatial_gate"),
                             ("quality_gate", "prior_survey"), ("uniform_gate", "uniform_survey"),
                             ("spatial_gate", "spatial_survey"), ("random_gate", "random_survey")):
                    left = {r["sample_id"]: r for r in scene_summaries if (r["split"], r["scenario"], r["budget"], r["policy"]) == (split, scenario, budget, a)}
                    right = {r["sample_id"]: r for r in scene_summaries if (r["split"], r["scenario"], r["budget"], r["policy"]) == (split, scenario, budget, b)}
                    differences = [{"sample_id": sid, "event": left[sid]["event"],
                                    "building_delta": left[sid]["buildings_found"] - right[sid]["buildings_found"],
                                    "false_positive_delta": left[sid]["false_positive_cells"] - right[sid]["false_positive_cells"]} for sid in sorted(left)]
                    paired.append({"split": split, "scenario": scenario, "budget": budget, "policy": a, "baseline": b,
                        "sum_building_delta": sum(r["building_delta"] for r in differences),
                        "wins": sum(r["building_delta"] > 1e-9 for r in differences),
                        "ties": sum(abs(r["building_delta"]) <= 1e-9 for r in differences),
                        "losses": sum(r["building_delta"] < -1e-9 for r in differences), "scene_differences": differences})
    exact_summary = []
    for split in ("calibration", "validation"):
        for scenario in SCENARIOS:
            for budget in BUDGETS:
                rows = [r for r in exact_rows if (r["split"], r["scenario"], r["budget"]) == (split, scenario, budget)]
                exact_summary.append({"split": split, "scenario": scenario, "budget": budget,
                    "sum_expected_buildings_found": sum(r["expected_buildings_found"] for r in rows) / len(SCENARIOS[scenario]),
                    "sum_expected_false_positive_cells": sum(r["expected_false_positive_cells"] for r in rows) / len(SCENARIOS[scenario])})
    return summaries, scene_summaries, paired, exact_summary


def sources(output):
    return {"parameters": output / "disaster_search_calibration_v2/calibration_parameters.json",
        **{f"{split}_{name}": path for split, inference, evaluation, old in
            (("calibration", "disaster_search_changeos_calibration", "disaster_search_calibration_v2", "disaster_search_local_feedback_v1"),
             ("validation", "disaster_search_changeos_validation", "disaster_search_validation_v1", "disaster_search_validation_v1"))
           for name, path in (("observations", output / inference / "observations.json"),
                              ("association", output / evaluation / "association_report.json"),
                              ("old_runs", output / old / "runs.json"), ("old_report", output / old / "report.json"),
                              ("old_traces", output / old / "traces.jsonl"), ("old_checks", output / old / "integrity_checks.json"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = sources(args.output_root)
    protocol = {**PROTOCOL, "sources": {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in inputs.items()},
        "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("run_coverage_controls.py", "run_cached_search.py", "real_evidence.py")}}
    freeze_plan(args.out / "control_protocol.json", protocol)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "control_protocol.json"), "new_episodes": 25200}))
        return
    params = json.loads(inputs["parameters"].read_text())
    prior = params["initial_presence_prior"]
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    results, references, exact_rows, cue_hashes = [], [], [], {}
    with gzip.open(args.out / "traces.jsonl.gz", "wt", compresslevel=1) as trace_file:
        for split in ("calibration", "validation"):
            observations = json.loads(inputs[split + "_observations"].read_text())
            association = json.loads(inputs[split + "_association"].read_text())
            old_report = json.loads(inputs[split + "_old_report"].read_text())
            checks = json.loads(inputs[split + "_old_checks"].read_text())
            if not checks["passed"] or digest(inputs[split + "_old_traces"]) != checks["traces_sha256"]:
                raise ValueError("Old traces must be audited and unchanged")
            if association["observations_sha256"] != digest(inputs[split + "_observations"]) or len(observations) != 3072 or {o["split"] for o in observations} != {split}:
                raise ValueError("Incomplete or mismatched split")
            old_rows = json.loads(inputs[split + "_old_runs"].read_text())
            for row in old_rows:
                references.append({**row, "split": split, "search_seed": -1})
            expected_cues = params["public_cue_hashes"] if split == "calibration" else old_report["public_cue_hashes"]
            for sid, expected in sorted(expected_cues.items()):
                cue_path = args.prepared_data / sid / "public_cues.json"
                if digest(cue_path) != expected:
                    raise ValueError("Changed public cue")
                cues = json.loads(cue_path.read_text())
                cue_hashes[sid] = expected
                if cues["split"] != split:
                    raise ValueError("Mixed split cues")
                fitted, uniform = [], []
                for c in cues["cells"]:
                    group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                    fitted.append(PublicCell(c["cell_id"], c["col"], c["row"], group, prior["bins"][group]["fitted_presence_probability"]))
                    uniform.append(PublicCell(c["cell_id"], c["col"], c["row"], group, prior["global_smoothed_prevalence"]))
                rows = [r for r in association["rows"] if r["sample_id"] == sid]
                truth = {(r["cell_id"], r["mode"]): r for r in rows}
                for scenario, quality_seeds in SCENARIOS.items():
                    for quality_seed in quality_seeds:
                        matching_cells = defaultdict(set)
                        false_cells = set()
                        for c in uniform:
                            mode = survey_mode(scenario, quality_seed, sid, c.cell_id)
                            r = truth[(c.cell_id, mode)]
                            for uid in r["matched_target_uids"]:
                                matching_cells[uid].add(c.cell_id)
                            if r["predicted"] and r["target"] is False:
                                false_cells.add(c.cell_id)
                        for budget in BUDGETS:
                            exact_rows.append({"split": split, "sample_id": sid, "event": cues["event"], "scenario": scenario,
                                "quality_seed": quality_seed, "budget": budget,
                                "expected_buildings_found": exact_random_discoveries(64, budget, [len(cells) for cells in matching_cells.values()]),
                                "expected_false_positive_cells": len(false_cells) * budget / 64})
                            for policy in CONTROL_POLICIES:
                                for search_seed in ORDER_SEEDS if policy.startswith("random") else (-1,):
                                    cells = fitted if policy == "prior_survey" else uniform
                                    env = CacheEnvironment(sid, observations, scenario, quality_seed)
                                    row, trace = run_control(cells, policy, rates, scenario, budget, sid, search_seed, env.observe, Evaluator(rows))
                                    if row["cost"] > budget or len(env.requests) != len(set(env.requests)):
                                        raise AssertionError("Invalid budget or repeated observation")
                                    row.update(split=split, sample_id=sid, event=cues["event"], scenario=scenario,
                                               quality_seed=quality_seed, search_seed=search_seed, budget=budget, policy=policy)
                                    results.append(row)
                                    trace_file.write(json.dumps({"run": row, "steps": trace}, ensure_ascii=False) + "\n")
                print(json.dumps({"split": split, "scene": sid, "episodes_done": len(results)}), flush=True)
    if len(results) != 25200:
        raise AssertionError("Incomplete coverage experiment")
    with gzip.open(args.out / "runs.json.gz", "wt") as f:
        json.dump(results, f, ensure_ascii=False)
    summary, scene_summary, paired, exact_summary = aggregate(results, references, exact_rows)
    report = {"stage": "POSTHOC_COVERAGE_CONTROLS", "new_episodes": len(results), "reused_reference_episodes": len(references),
              "protocol_sha256": digest(args.out / "control_protocol.json"), "runner_sha256": digest(Path(__file__)),
              "public_cue_hashes": cue_hashes, "summary": summary, "paired_comparisons": paired,
              "exact_random_survey_expectation": exact_summary, "test_used": False, "parameters_refitted": False,
              "new_inference_run": False, "paid_gpu_used": False, "limitations": PROTOCOL["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / "scene_summaries.json").write_text(json.dumps(scene_summary, ensure_ascii=False, indent=2))
    print(json.dumps({"new_episodes": len(results), "validation_mixed_budget32": [r for r in summary if r["split"] == "validation" and r["scenario"] == "mixed_survey" and r["budget"] == 32]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
