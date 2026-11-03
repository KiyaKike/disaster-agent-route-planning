"""Fixed calibration follow-up: bounded local feedback and a simple comparator.

Reuses v1 source caches and action semantics. No labels enter LocalPlanner.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_cached_search import (BUDGETS, SCENARIOS, CacheEnvironment, Evaluator,
                               Planner, PublicCell, PROTOCOL as BASE_PROTOCOL, digest)
from run_changeos_pilot import freeze_plan


LOCAL_POLICIES = ("local_residual", "local_positive", "neighbor_positive")
PROTOCOL = {
    "version": "local_feedback_calibration_v1", "split": "calibration",
    "timing": "Exploratory follow-up designed after scope diagnostics; not preregistered before seeing calibration labels. No validation/test or hyperparameter search.",
    "policies": ["quality_gate", "quality", "adaptive_prior", *LOCAL_POLICIES],
    "scenarios_and_seeds": SCENARIOS, "budgets": list(BUDGETS),
    "scope": "Online relations use only immediate 8-connected neighbors in the same original RGB-change quartile. Never use shared-building UID exclusions online; these were evaluator diagnostic controls only.",
    "local_residual": "For each unvisited candidate, sum latest per-neighbor (TPR-FPR)^2 * (source posterior-original source prior); logit shift clip(4*sum/(8+sum reliability),-1,1). Both signs. Original v1 constants unchanged.",
    "local_positive": "Same local residual formula but include only latest detector-positive source neighbors; negative sources contribute neither residual nor denominator. Observation quality represented by same fixed reliability.",
    "neighbor_positive": "Simple baseline: candidate logit +0.25 if any latest detector-positive eligible neighbor exists, otherwise zero; independent of number of positive neighbors or likelihood values. Fixed modest bias, not tuned.",
    "cell_posteriors": "All three local policies use the unchanged quality posterior for observed cells, anchored to their original fitted prior. Only unvisited scores receive neighbor corrections.",
    "correlation": BASE_PROTOCOL["correlated_evidence"], "actions": BASE_PROTOCOL["actions"],
    "selection": BASE_PROTOCOL["selection"], "cost_scope": BASE_PROTOCOL["cost_scope"],
    "paired_design": "Reuse all v1 control trajectories unchanged after hash checks. Run three new local policies on exactly the same fixed observation map and budgets; report all outcomes.",
    "progression_rule": "Do not present a contribution or spend test events if local residual policies cannot consistently beat the simple neighbor-positive and quality-gate controls across budgets and events. Gains concentrated in a few scenes remain exploratory.",
    "limitations": BASE_PROTOCOL["limitations"] + ["Fixed-radius same-bin relation motivated by this calibration data, with no held-out confirmation.",
        "Local detector-positive neighbors may refer to the same crossing building; unique discovery evaluator removes repeat counts but the planner does not know true object IDs.",
        "The 0.25 simple-bias constant is an engineering comparator choice; no claim it is the strongest possible local heuristic."]}


class LocalPlanner(Planner):
    def __init__(self, cells, policy, rates, scenario):
        if policy not in LOCAL_POLICIES:
            raise ValueError("Unknown local policy")
        self.local_policy = policy
        self.cell_shifts = {c.cell_id: 0. for c in cells}
        super().__init__(cells, "quality", rates, scenario)

    def neighbors(self, cid):
        candidate = self.cells[cid]
        return [source for source in self.latest if source != cid and
                self.cells[source].group == candidate.group and
                max(abs(self.cells[source].x - candidate.x), abs(self.cells[source].y - candidate.y)) <= 1]

    def current(self, cid):
        if cid in self.latest:
            return self.values[cid]
        prior = self.cells[cid].prior
        return 1 / (1 + math.exp(-(math.log(prior / (1 - prior)) + self.cell_shifts[cid])))

    def observe(self, action, observation):
        before_order = [cid for cid in self.unvisited_order() if cid != action.cell_id]
        before = self.current(action.cell_id)
        old_position = self.position
        audit = super().observe(action, observation)
        # Rebuild from one latest representative per neighbor, never append duplicate residuals.
        for cid in self.cells:
            if cid in self.latest:
                self.cell_shifts[cid] = 0.
                continue
            peers = self.neighbors(cid)
            if self.local_policy == "neighbor_positive":
                self.cell_shifts[cid] = .25 if any(self.latest[p]["declared_target"] for p in peers) else 0.
                continue
            residual, weight = 0., 0.
            for peer in peers:
                if self.local_policy == "local_positive" and not self.latest[peer]["declared_target"]:
                    continue
                rates = self.rates[self.latest[peer]["mode"]]
                reliability = (rates["tpr"] - rates["fpr"]) ** 2
                residual += reliability * (self.values[peer] - self.cells[peer].prior)
                weight += reliability
            self.cell_shifts[cid] = max(-1., min(1., 4 * residual / (8 + weight)))
        # Use the same old position when testing whether feedback itself changed ranking.
        self.position = old_position
        after_order = self.unvisited_order()
        self.position = (self.cells[action.cell_id].x, self.cells[action.cell_id].y)
        audit.update(belief_before=before, feedback_reordered_unvisited=before_order != after_order,
                     cell_logit_shifts={cid: shift for cid, shift in self.cell_shifts.items() if shift},
                     next_unvisited_top3=after_order[:3])
        audit["unvisited_rank_sha256"] = hashlib.sha256(json.dumps(after_order).encode()).hexdigest()
        return audit


def run_local_episode(cells, policy, rates, scenario, budget, provider, evaluator):
    planner = LocalPlanner(cells, policy, rates, scenario)
    cost, trace = 0, []
    while (action := planner.choose(budget - cost)) is not None:
        observation = provider(action)
        audit = planner.observe(action, observation)
        cost += action.cost
        evaluator.observe(observation, cost)
        trace.append({"step": len(trace) + 1, "action": asdict(action), "cost_total": cost,
                      "observation": observation, **audit})
    return {**evaluator.result(), "cost": cost, "actions": len(trace),
            "surveys": sum(s["action"]["kind"] == "survey" for s in trace),
            "upgrades": sum(s["action"]["kind"] == "upgrade" for s in trace),
            "evidence_groups": len(planner.registry.representatives()),
            "feedback_rank_changes": sum(s["feedback_reordered_unvisited"] for s in trace)}, trace


def comparisons(results, policies):
    index = {(r["sample_id"], r["scenario"], r["quality_seed"], r["budget"], r["policy"]): r for r in results}
    scene_ids = sorted({r["sample_id"] for r in results})
    events = {r["sample_id"]: r["event"] for r in results}
    report = []
    for scenario, seeds in SCENARIOS.items():
        for budget in BUDGETS:
            for policy in policies:
                for baseline in ("quality_gate", "neighbor_positive"):
                    if policy == baseline:
                        continue
                    scene_deltas = []
                    for sid in scene_ids:
                        deltas, false_deltas = [], []
                        for seed in seeds:
                            a = index[(sid, scenario, seed, budget, policy)]
                            b = index[(sid, scenario, seed, budget, baseline)]
                            deltas.append(a["unique_buildings_found"] - b["unique_buildings_found"])
                            false_deltas.append(a["false_positive_cells"] - b["false_positive_cells"])
                        scene_deltas.append({"sample_id": sid, "event": events[sid], "building_delta": statistics.mean(deltas),
                                             "false_positive_delta": statistics.mean(false_deltas)})
                    report.append({"scenario": scenario, "budget": budget, "policy": policy, "baseline": baseline,
                        "sum_building_delta": sum(s["building_delta"] for s in scene_deltas),
                        "sum_false_positive_delta": sum(s["false_positive_delta"] for s in scene_deltas),
                        "wins": sum(s["building_delta"] > 0 for s in scene_deltas),
                        "ties": sum(s["building_delta"] == 0 for s in scene_deltas),
                        "losses": sum(s["building_delta"] < 0 for s in scene_deltas),
                        "by_event": {e: sum(s["building_delta"] for s in scene_deltas if s["event"] == e) for e in sorted(set(events.values()))},
                        "scene_deltas": scene_deltas})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "association", "parameters", "prepared-data", "old-run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "source_hashes": {k: digest(getattr(args, k)) for k in ("observations", "association", "parameters")},
                "old_control_traces_sha256": digest(args.old_run / "traces.jsonl"),
                "old_control_report_sha256": digest(args.old_run / "report.json")}
    freeze_plan(args.out / "search_protocol.json", protocol)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "search_protocol.json"), "new_episodes": 24 * 7 * 3 * 3}))
        return
    params = json.loads(args.parameters.read_text())
    observations = json.loads(args.observations.read_text())
    association = json.loads(args.association.read_text())
    old_checks = json.loads((args.old_run / "integrity_checks.json").read_text())
    if not old_checks["passed"] or old_checks["traces_sha256"] != protocol["old_control_traces_sha256"]:
        raise ValueError("Old control traces must be audited and unchanged")
    if len(observations) != 3072 or any(o["split"] != "calibration" for o in observations) or association["stage"] != "FULL_CALIBRATION_BOUNDARY_DIAGNOSTICS":
        raise ValueError("Calibration inputs only")
    if params["source_observations_sha256"] != digest(args.observations) or params["source_association_sha256"] != digest(args.association):
        raise ValueError("Source mismatch")
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    prior = params["initial_presence_prior"]
    results = []
    with (args.old_run / "traces.jsonl").open() as f:
        for line in f:
            episode = json.loads(line)
            if episode["run"]["policy"] in ("quality_gate", "quality", "adaptive_prior"):
                results.append(episode["run"])
    with (args.out / "traces.jsonl").open("w") as trace_file:
        for sid, expected in sorted(params["public_cue_hashes"].items()):
            cue_path = args.prepared_data / sid / "public_cues.json"
            if digest(cue_path) != expected:
                raise ValueError("Changed public cue")
            cues = json.loads(cue_path.read_text())
            cells = []
            for c in cues["cells"]:
                group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                cells.append(PublicCell(c["cell_id"], c["col"], c["row"], group, prior["bins"][group]["fitted_presence_probability"]))
            rows = [r for r in association["rows"] if r["sample_id"] == sid]
            for scenario, seeds in SCENARIOS.items():
                for seed in seeds:
                    for budget in BUDGETS:
                        for policy in LOCAL_POLICIES:
                            env = CacheEnvironment(sid, observations, scenario, seed)
                            row, trace = run_local_episode(cells, policy, rates, scenario, budget, env.observe, Evaluator(rows))
                            if row["cost"] > budget or len(env.requests) != len(set(env.requests)):
                                raise AssertionError("Budget/evidence failure")
                            row.update(sample_id=sid, event=cues["event"], scenario=scenario, quality_seed=seed, budget=budget, policy=policy)
                            results.append(row)
                            trace_file.write(json.dumps({"run": row, "steps": trace}, ensure_ascii=False) + "\n")
    if len(results) != 3024:
        raise AssertionError("Incomplete follow-up")
    (args.out / "runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    summaries = []
    for scenario, seeds in SCENARIOS.items():
        for budget in BUDGETS:
            for policy in PROTOCOL["policies"]:
                rows = [r for r in results if (r["scenario"], r["budget"], r["policy"]) == (scenario, budget, policy)]
                summaries.append({"scenario": scenario, "budget": budget, "policy": policy,
                    "sum_buildings_found": sum(r["unique_buildings_found"] for r in rows) / len(seeds),
                    "sum_false_positive_cells": sum(r["false_positive_cells"] for r in rows) / len(seeds),
                    "mean_upgrades": statistics.mean(r["upgrades"] for r in rows),
                    "mean_feedback_rank_changes": statistics.mean(r["feedback_rank_changes"] for r in rows)})
    report = {"stage": "EXPLORATORY_CALIBRATION_LOCAL_FEEDBACK", "new_episodes": 1512, "reused_controls": 1512,
              "protocol_sha256": digest(args.out / "search_protocol.json"), "runner_sha256": digest(Path(__file__)),
              "summary": summaries, "paired_comparisons": comparisons(results, ("adaptive_prior", *LOCAL_POLICIES)),
              "validation_used": False, "test_used": False, "new_inference": False, "paid_gpu_used": False,
              "limitations": PROTOCOL["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"new_episodes": 1512, "mixed_survey_results": [r for r in summaries if r["scenario"] == "mixed_survey"]}))


if __name__ == "__main__":
    main()
