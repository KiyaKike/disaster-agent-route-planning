"""Calibration-only sequential search over fixed real-image detector caches.

Planner receives public image cues and requested observations only. All object
IDs, labels, and future outcomes stay in the environment/evaluator.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from real_evidence import ImageEvidenceRegistry
from run_changeos_pilot import freeze_plan


POLICIES = ("prior", "pooled", "quality_gate", "quality", "adaptive_prior")
SCENARIOS = {"native_survey": [20261024], "mixed_survey": list(range(20261024, 20261029)),
             "blurred_survey": [20261024]}
BUDGETS = (8, 16, 32)
PROTOCOL = {
    "version": "cached_search_calibration_v1", "split": "calibration", "policies": list(POLICIES),
    "scenarios_and_quality_map_seeds": SCENARIOS, "budgets": list(BUDGETS),
    "input": "Frozen coarse RGB-change bins/prior, aggregate calibration parameters; detector outcomes supplied only after action.",
    "schedule": "Native-survey, blurred-survey, or mixed 50% per-cell SHA256(seed|sample|cell|survey_quality), independent of labels and policy. Future survey mode hidden from planner; only scenario mixture is known.",
    "actions": "Survey costs 1; a negative blurred survey permits one native quality upgrade costing 2. Positive regions terminal for all policies; native survey has no further upgrade. No direct native request on an unvisited region in degraded-survey conditions.",
    "cost_scope": "Controlled quality-restoration proxy with declared 1:2 action costs; not measured compute, flight time, UAV viewpoints, or a claim that reprocessing recovers genuinely lost information.",
    "selection": "Expected target-presence detections per cost: unvisited probability * scenario-average sensitivity; upgrade latest cell probability * native sensitivity / 2. Common forecast rates across feedback methods; Euclidean distance breaks ties, no travel charge.",
    "prior_baseline": "Static initial-probability survey ranking, no upgrades. Other four policies share the same upgrade inventory and acquisition rule.",
    "pooled": "Ignore observed mode for posterior likelihood, use scenario-mixture rates for both survey and upgrade; common action forecast unchanged. This is an explicit mode-agnostic ablation, not an optimally fitted pooled baseline.",
    "quality_gate": "Quality likelihoods, except negative blurred observations leave the initial cell probability unchanged.",
    "correlated_evidence": "One registry representative per sample/cell. Upgrade replaces blurred result: recompute from frozen original cell prior using native likelihood, not previous posterior. Current peer residual replaced too.",
    "adaptive_prior": {"group": "public raw-change quartile bin", "residual": "sum reliability * (latest cell posterior - original cell prior)",
        "reliability": "(sensitivity - false_positive_rate)^2", "logit_shift": "clip(4 * residual / (8 + sum reliability), -1, 1)",
        "scope": "Unvisited cells only; observed-cell posterior always anchored to original fitted prior, avoiding recursive peer amplification. Constants inherited from synthetic prototype, not tuned here."},
    "metrics": "Union of evaluator (sample_id, building_uid) matches, full-scene labeled target denominator, known-negative positive cells, unmatched positive observations, unknown outcomes, first matched-building cost; no-target recall null.",
    "audit": "Trace actions, available observations, representative replacements, rank signatures and group shifts; compare paired adaptive vs quality trajectories, including unvisited-ranking changes.",
    "statistics": "Same 24 scenes for every method. Five mixed seeds are controlled quality allocations, not new independent scenes. Report paired scene differences averaged across quality seeds and separate events, no significance claim.",
    "limitations": ["Calibration data reused to fit sensor/prior and run these diagnostics: no held-out policy result.",
        "Checkpoint training exposure expected; two events, no real UAV measurements.",
        "Presence-oriented acquisition and terminal positive cells may miss additional buildings in dense positive cells.",
        "Coverage-based building matching is an evaluator proxy, not instance AP; spatial peers are not guaranteed exchangeable."]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PublicCell:
    cell_id: str
    x: int
    y: int
    group: int
    prior: float


@dataclass(frozen=True)
class Action:
    cell_id: str
    kind: str

    @property
    def cost(self):
        return {"survey": 1, "upgrade": 2}[self.kind]


def survey_mode(scenario, seed, sid, cid):
    if scenario == "native_survey":
        return "native"
    if scenario == "blurred_survey":
        return "blurred"
    if scenario != "mixed_survey":
        raise ValueError("Unknown quality scenario")
    key = f"{seed}|{sid}|{cid}|survey_quality"
    return "native" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "blurred"


def mixture_rates(rates, scenario):
    w = {"native_survey": 1., "mixed_survey": .5, "blurred_survey": 0.}[scenario]
    return {key: w * rates["native"][key] + (1 - w) * rates["blurred"][key] for key in ("tpr", "fpr")}


def update_probability(prior, positive, rates):
    if not 0 < prior < 1 or not 0 < rates["fpr"] < rates["tpr"] < 1:
        raise ValueError("Expected strictly bounded prior and informative smoothed likelihoods")
    a, b = (rates["tpr"], rates["fpr"]) if positive else (1 - rates["tpr"], 1 - rates["fpr"])
    return prior * a / (prior * a + (1 - prior) * b)


class Planner:
    def __init__(self, cells, policy, rates, scenario):
        if policy not in POLICIES or len({c.cell_id for c in cells}) != len(cells):
            raise ValueError("Invalid public cells or policy")
        self.cells = {c.cell_id: c for c in cells}
        self.policy, self.rates = policy, rates
        self.survey_rates = mixture_rates(rates, scenario)
        self.registry = ImageEvidenceRegistry()
        self.used = set()
        self.latest = {}
        self.values = {c.cell_id: c.prior for c in cells}
        self.shifts = {c.group: 0. for c in cells}
        self.position = (0, 0)

    def current(self, cid):
        cell = self.cells[cid]
        if self.policy != "adaptive_prior" or cid in self.latest:
            return self.values[cid]
        return 1 / (1 + math.exp(-(math.log(cell.prior / (1 - cell.prior)) + self.shifts[cell.group])))

    def unvisited_order(self):
        return sorted((cid for cid in self.cells if cid not in self.latest),
                      key=lambda cid: (-self.current(cid), math.dist(self.position, (self.cells[cid].x, self.cells[cid].y)), cid))

    def choose(self, remaining):
        candidates = []
        for cid, cell in self.cells.items():
            previous = self.latest.get(cid)
            if previous is None:
                action = Action(cid, "survey")
                score = self.current(cid) * self.survey_rates["tpr"]
            elif (self.policy != "prior" and previous["mode"] == "blurred" and
                  not previous["declared_target"] and (cid, "upgrade") not in self.used):
                action = Action(cid, "upgrade")
                score = self.current(cid) * self.rates["native"]["tpr"] / 2
            else:
                continue
            if action.cost <= remaining:
                candidates.append((-score, math.dist(self.position, (cell.x, cell.y)), cid, action.kind))
        if not candidates:
            return None
        _, _, cid, kind = min(candidates)
        return Action(cid, kind)

    def observe(self, action, observation):
        cid = action.cell_id
        if cid not in self.cells or (cid, action.kind) in self.used:
            raise ValueError("Unknown cell or duplicate action")
        previous = self.latest.get(cid)
        if action.kind == "survey" and previous is not None:
            raise ValueError("Survey already performed")
        if action.kind == "upgrade" and (self.policy == "prior" or previous is None or previous["mode"] != "blurred" or
                                         previous["declared_target"] or observation["mode"] != "native"):
            raise ValueError("Upgrade requires a negative blurred survey")
        if action.kind not in {"survey", "upgrade"} or observation["cell_id"] != cid:
            raise ValueError("Observation/action mismatch")
        # Snapshot ranking at the same position before and after feedback, excluding the selected cell.
        before_order = [c for c in self.unvisited_order() if c != cid]
        before = self.current(cid)
        replaced = self.registry.add(observation)
        self.used.add((cid, action.kind))
        self.latest[cid] = {k: observation[k] for k in ("mode", "declared_target", "observation_id")}
        rates = self.survey_rates if self.policy == "pooled" else self.rates[observation["mode"]]
        p = self.cells[cid].prior
        if self.policy == "prior" or (self.policy == "quality_gate" and observation["mode"] == "blurred" and not observation["declared_target"]):
            self.values[cid] = p
        else:
            # Replace, never multiply: both modes share one underlying image.
            self.values[cid] = update_probability(p, observation["declared_target"], rates)
        if self.policy == "adaptive_prior":
            sums = defaultdict(lambda: [0., 0.])
            for peer, latest in self.latest.items():
                likelihood = self.rates[latest["mode"]]
                reliability = (likelihood["tpr"] - likelihood["fpr"]) ** 2
                group = self.cells[peer].group
                sums[group][0] += reliability * (self.values[peer] - self.cells[peer].prior)
                sums[group][1] += reliability
            self.shifts = {g: max(-1., min(1., 4 * sums[g][0] / (8 + sums[g][1]))) for g in self.shifts}
        after_order = self.unvisited_order()
        feedback_reordered = before_order != after_order
        self.position = (self.cells[cid].x, self.cells[cid].y)
        return {"belief_before": before, "original_prior": p, "belief_after": self.values[cid],
                "representative_replaced": previous is not None and replaced,
                "likelihood_used": rates if self.policy != "prior" else None,
                "peer_count": len(self.latest), "group_logit_shifts": dict(self.shifts),
                "feedback_reordered_unvisited": feedback_reordered,
                "unvisited_rank_sha256": hashlib.sha256(json.dumps(after_order).encode()).hexdigest(),
                "next_unvisited_top3": after_order[:3]}


class CacheEnvironment:
    """Environment-only lookup. Return only a requested observation's safe fields."""
    FIELDS = ("sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target", "uncalibrated_severe_score")

    def __init__(self, sid, observations, scenario, seed):
        self.sid, self.scenario, self.seed = sid, scenario, seed
        self._observations = {(o["cell_id"], o["mode"]): o for o in observations if o["sample_id"] == sid}
        self.requests = []

    def observe(self, action):
        mode = survey_mode(self.scenario, self.seed, self.sid, action.cell_id) if action.kind == "survey" else "native"
        source = self._observations[(action.cell_id, mode)]
        self.requests.append(source["observation_id"])
        return {key: source[key] for key in self.FIELDS}


class Evaluator:
    def __init__(self, rows):
        self.rows = {(r["cell_id"], r["mode"]): r for r in rows}
        self.targets = {uid for row in rows for uid in row["visible_target_uids"]}
        self.unknown_scene = any(row["target"] is None for row in rows)
        self.found, self.false_cells = set(), set()
        self.unknown_positives = set()
        self.unmatched_positives = 0
        self.first_cost = None

    def observe(self, observation, cost):
        row = self.rows[(observation["cell_id"], observation["mode"])]
        matched = set(row["matched_target_uids"])
        self.found |= matched
        if matched and self.first_cost is None:
            self.first_cost = cost
        if observation["declared_target"]:
            if row["target"] is None:
                self.unknown_positives.add(observation["cell_id"])
            else:
                if not matched:
                    self.unmatched_positives += 1
                if row["target"] is False:
                    self.false_cells.add(observation["cell_id"])

    def result(self):
        return {"target_count": len(self.targets), "unique_buildings_found": len(self.found),
                "recall": len(self.found) / len(self.targets) if self.targets else None,
                "false_positive_cells": len(self.false_cells), "unmatched_positive_observations": self.unmatched_positives,
                "unknown_positive_cells": len(self.unknown_positives),
                "scene_type": "target_present" if self.targets else ("indeterminate" if self.unknown_scene else "known_no_target"),
                "first_discovery_cost": self.first_cost, "no_true_discovery": not self.found}


def run_episode(cells, policy, rates, scenario, budget, provider, evaluator):
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("Budget must be a positive integer")
    planner = Planner(cells, policy, rates, scenario)
    cost, trace = 0, []
    while (action := planner.choose(budget - cost)) is not None:
        observation = provider(action)
        audit = planner.observe(action, observation)
        cost += action.cost
        evaluator.observe(observation, cost)  # Evaluator output never fed to planner.
        trace.append({"step": len(trace) + 1, "action": asdict(action), "cost_total": cost,
                      "observation": observation, **audit})
    return {**evaluator.result(), "cost": cost, "actions": len(trace),
            "surveys": sum(t["action"]["kind"] == "survey" for t in trace),
            "upgrades": sum(t["action"]["kind"] == "upgrade" for t in trace),
            "evidence_groups": len(planner.registry.representatives()),
            "feedback_rank_changes": sum(t["feedback_reordered_unvisited"] for t in trace)}, trace


def summarize(results):
    summary = []
    for scenario, seeds in SCENARIOS.items():
        for budget in BUDGETS:
            for policy in POLICIES:
                per_seed = []
                for seed in seeds:
                    rows = [r for r in results if (r["scenario"], r["budget"], r["policy"], r["quality_seed"]) == (scenario, budget, policy, seed)]
                    target_count = sum(r["target_count"] for r in rows)
                    found = sum(r["unique_buildings_found"] for r in rows)
                    per_seed.append({"sum_buildings_found": found, "micro_recall": found / target_count if target_count else None,
                        "macro_recall_target_scenes": statistics.mean(r["recall"] for r in rows if r["recall"] is not None),
                        "sum_false_positive_cells": sum(r["false_positive_cells"] for r in rows),
                        "sum_unmatched_positive_observations": sum(r["unmatched_positive_observations"] for r in rows),
                        "mean_upgrades": statistics.mean(r["upgrades"] for r in rows),
                        "scenes_with_discovery": sum(not r["no_true_discovery"] for r in rows),
                        "mean_feedback_rank_changes": statistics.mean(r["feedback_rank_changes"] for r in rows)})
                summary.append({"scenario": scenario, "budget": budget, "policy": policy, "scenes": 24,
                    "quality_maps": len(seeds), **{k: statistics.mean(r[k] for r in per_seed) for k in per_seed[0]}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "source_hashes": {key: digest(getattr(args, key)) for key in ("observations", "association", "parameters")}}
    freeze_plan(args.out / "search_protocol.json", protocol)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "search_protocol.json"), "episodes": 24 * 7 * 3 * 5}))
        return
    params = json.loads(args.parameters.read_text())
    observations = json.loads(args.observations.read_text())
    association = json.loads(args.association.read_text())
    if params["source_observations_sha256"] != digest(args.observations) or params["source_association_sha256"] != digest(args.association):
        raise ValueError("Parameters and source caches differ")
    if association["stage"] != "FULL_CALIBRATION_BOUNDARY_DIAGNOSTICS" or len(observations) != 3072 or any(o["split"] != "calibration" for o in observations):
        raise ValueError("Only complete calibration input is authorized for this run")
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    prior = params["initial_presence_prior"]
    results, pair_traces = [], defaultdict(dict)
    traces_file = args.out / "traces.jsonl"
    with traces_file.open("w") as traces:
        for sid in sorted(params["public_cue_hashes"]):
            path = args.prepared_data / sid / "public_cues.json"
            if digest(path) != params["public_cue_hashes"][sid]:
                raise ValueError("Changed public cues")
            cues = json.loads(path.read_text())
            if cues["split"] != "calibration":
                raise ValueError("Non-calibration public scene")
            cells = []
            for cell in cues["cells"]:
                group = bisect_right(prior["bin_edges"], cell["raw_change_score"])
                cells.append(PublicCell(cell["cell_id"], cell["col"], cell["row"], group, prior["bins"][group]["fitted_presence_probability"]))
            scene_rows = [r for r in association["rows"] if r["sample_id"] == sid]
            for scenario, seeds in SCENARIOS.items():
                for seed in seeds:
                    for budget in BUDGETS:
                        for policy in POLICIES:
                            env = CacheEnvironment(sid, observations, scenario, seed)
                            row, trace = run_episode(cells, policy, rates, scenario, budget, env.observe, Evaluator(scene_rows))
                            row.update(sample_id=sid, event=cues["event"], scenario=scenario, quality_seed=seed, budget=budget, policy=policy)
                            if row["cost"] > budget or len(env.requests) != len(set(env.requests)):
                                raise AssertionError("Budget or evidence identity failed")
                            results.append(row)
                            traces.write(json.dumps({"run": row, "steps": trace}, ensure_ascii=False) + "\n")
                            if policy in {"quality", "adaptive_prior"}:
                                pair_traces[(sid, scenario, seed, budget)][policy] = [t["action"] for t in trace]
            print(json.dumps({"scene": sid, "episodes_done": len(results)}), flush=True)
    if len(results) != 2520:
        raise AssertionError("Incomplete paired run")
    with (args.out / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = summarize(results)
    comparisons = []
    for scenario, seeds in SCENARIOS.items():
        for budget in BUDGETS:
            scenes = []
            for sid in sorted(params["public_cue_hashes"]):
                differences = []
                for seed in seeds:
                    match = {r["policy"]: r for r in results if (r["sample_id"], r["scenario"], r["quality_seed"], r["budget"]) == (sid, scenario, seed, budget)}
                    differences.append(match["adaptive_prior"]["unique_buildings_found"] - match["quality"]["unique_buildings_found"])
                scenes.append({"sample_id": sid, "mean_building_delta": statistics.mean(differences)})
            comparisons.append({"scenario": scenario, "budget": budget, "scene_mean_differences": scenes,
                "sum_mean_building_delta": sum(r["mean_building_delta"] for r in scenes),
                "scene_wins": sum(r["mean_building_delta"] > 0 for r in scenes),
                "scene_ties": sum(r["mean_building_delta"] == 0 for r in scenes),
                "scene_losses": sum(r["mean_building_delta"] < 0 for r in scenes),
                "paired_trajectories_different": sum(pair_traces[(sid, scenario, seed, budget)]["quality"] != pair_traces[(sid, scenario, seed, budget)]["adaptive_prior"]
                                                     for sid in params["public_cue_hashes"] for seed in seeds),
                "paired_trajectories_total": 24 * len(seeds)})
    event_summary = []
    for event in sorted({r["event"] for r in results}):
        for row in summary:
            subset = [r for r in results if (r["event"], r["scenario"], r["budget"], r["policy"]) == (event, row["scenario"], row["budget"], row["policy"])]
            event_summary.append({"event": event, "scenario": row["scenario"], "budget": row["budget"], "policy": row["policy"],
                "mean_sum_buildings_found": sum(r["unique_buildings_found"] for r in subset) / row["quality_maps"],
                "mean_sum_false_positive_cells": sum(r["false_positive_cells"] for r in subset) / row["quality_maps"]})
    report = {"stage": "CALIBRATION_SEARCH_ENGINEERING_DIAGNOSTICS", "episodes": len(results), "scenes": 24, "events": 2,
              "protocol_sha256": digest(args.out / "search_protocol.json"), "runner_sha256": digest(Path(__file__)),
              "validation_used": False, "test_used": False, "paid_gpu_used": False, "new_inference_run": False,
              "summary": summary, "adaptive_vs_quality": comparisons, "event_summary": event_summary,
              "limitations": PROTOCOL["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"stage": report["stage"], "episodes": len(results), "adaptive_vs_quality": [{k: r[k] for k in ("scenario", "budget", "sum_mean_building_delta", "scene_wins", "scene_losses", "paired_trajectories_different")} for r in comparisons]}), flush=True)


if __name__ == "__main__":
    main()
