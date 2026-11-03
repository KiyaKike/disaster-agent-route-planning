"""CPU-only sequential-search harness. Synthetic results are engineering checks.

python3 work/disaster_search/experiment.py --out output/disaster_search_smoke
No image model, real UAV observation, or measured disaster-search efficacy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


POLICIES = ("nearest", "prior", "mixed", "pooled", "quality_gate", "quality", "adaptive_prior")
SCENARIOS = ("clean", "structured_prior_error", "random_prior_error", "poor_visibility", "persistent_occlusion")
# Known likelihoods only for synthetic checks; empirical calibration is still required.
SENSOR = {"clear": {"tpr": .90, "fpr": .04}, "poor": {"tpr": .25, "fpr": .12}}
# Marginal rates under the normal synthetic quality mix, by known action mode.
# This is a declared simulation calibration, not an empirical real-image estimate.
POOLED = {"survey": {"tpr": .705, "fpr": .064},
          "verify": {"tpr": .85125, "fpr": .046}}


def rng_for(seed, *parts):
    key = json.dumps([seed, *parts], separators=(",", ":"))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


@dataclass(frozen=True)
class Cell:
    cell_id: str
    x: int
    y: int
    group: str  # Observable cue category, never a truth-defined grouping.
    prior: float
    prior_unknown: bool


@dataclass(frozen=True)
class Action:
    cell_id: str
    slot: str  # Separate observation opportunity, not rereading the same evidence.

    @property
    def cost(self):
        return 1 if self.slot == "survey" else 2


@dataclass(frozen=True)
class Observation:
    observation_id: str
    positive: bool
    quality: str


@dataclass
class Scene:
    cells: list[Cell]
    truth: dict[str, bool]
    observations: dict[tuple[str, str], Observation]
    scenario: str
    seed: int


def synthetic_scene(seed=0, scenario="clean", side=8):
    if scenario not in SCENARIOS or side < 2:
        raise ValueError("Invalid synthetic scene configuration")
    cells, truth, observations = [], {}, {}
    for y in range(side):
        for x in range(side):
            cid = f"r{y:02d}c{x:02d}"
            group = "high_cue" if x < side // 2 else "low_cue"
            r = rng_for(seed, cid, "map")
            actual = .55 if group == "high_cue" else .15
            prior = actual
            if scenario == "structured_prior_error":
                prior = .15 if group == "high_cue" else .55
            if scenario == "random_prior_error" and r.random() < .4:
                prior = 1 - actual
            unknown = r.random() < .15
            if unknown:
                prior = .35  # Synthetic fallback; not a PS change-score conversion.
            cells.append(Cell(cid, x, y, group, prior, unknown))
            truth[cid] = rng_for(seed, cid, "truth").random() < actual
            permanently_poor = (scenario == "persistent_occlusion" and
                                rng_for(seed, cid, "occlusion").random() < .6)
            shared_bad = rng_for(seed, cid, "persistent_likelihood").random()
            for slot in ("survey", "verify"):
                r = rng_for(seed, cid, slot, "sensor")
                poor_chance = .75 if scenario == "poor_visibility" else .3
                if slot == "verify":
                    poor_chance *= .25
                quality = "poor" if permanently_poor or r.random() < poor_chance else "clear"
                rates = SENSOR[quality]
                threshold = rates["tpr"] if truth[cid] else rates["fpr"]
                # Persistent obscuration produces correlated failures across opportunities.
                draw = shared_bad if permanently_poor else r.random()
                observations[(cid, slot)] = Observation(f"{seed}:{cid}:{slot}", draw < threshold, quality)
    return Scene(cells, truth, observations, scenario, seed)


def posterior(prior, positive, rates):
    if not 0 < prior < 1:
        raise ValueError("Prior must be strictly between zero and one")
    tp, fp = rates["tpr"], rates["fpr"]
    if not 0 <= fp <= tp <= 1:
        raise ValueError("Invalid likelihoods")
    a, b = (tp, fp) if positive else (1 - tp, 1 - fp)
    denominator = prior * a + (1 - prior) * b
    if denominator <= 0:
        raise ValueError("Impossible observation")
    return min(.999, max(.001, prior * a / denominator))


class Belief:
    """Planner-visible state. Neither labels nor future observation outcomes enter it."""

    def __init__(self, cells, policy):
        if policy not in POLICIES:
            raise ValueError("Unknown policy")
        self.cells = {c.cell_id: c for c in cells}
        self.policy = policy
        self.values = {c.cell_id: c.prior for c in cells}
        self.used = set()
        self.seen_ids = set()
        self.positive_cells = set()
        self.latest_reliability = {}
        self.position = (0, 0)
        self.steps = 0

    def current(self, cid):
        cell = self.cells[cid]
        value = self.values[cid]
        if self.policy != "adaptive_prior" or (cid, "survey") in self.used:
            return value
        peers = [c for c in self.cells.values() if c.group == cell.group and
                 (c.cell_id, "survey") in self.used]
        # Exploratory shrinkage heuristic, not a novel or calibrated estimator.
        residual = sum(self.latest_reliability.get(c.cell_id, 0) *
                       (self.values[c.cell_id] - c.prior) for c in peers)
        weight = sum(self.latest_reliability.get(c.cell_id, 0) for c in peers)
        correction = max(-1., min(1., 4 * residual / (8 + weight)))
        logit = math.log(value / (1 - value)) + correction
        return 1 / (1 + math.exp(-logit))

    def choose(self, remaining):
        candidates = []
        dynamic = self.policy in {"pooled", "quality", "quality_gate", "adaptive_prior"}
        for cid, cell in self.cells.items():
            if cid in self.positive_cells:
                continue
            slot = "verify" if (cid, "survey") in self.used else "survey"
            if slot == "verify" and not dynamic:
                continue
            if (cid, slot) in self.used:
                continue
            action = Action(cid, slot)
            if action.cost > remaining:
                continue
            distance = math.dist(self.position, (cell.x, cell.y))
            # Distance only breaks ties in E0; travel is NOT part of its budget.
            score = self.current(cid) / action.cost
            if self.policy == "nearest":
                score = -distance
            if self.policy == "mixed" and (self.steps + 1) % 4 == 0:
                score = 2. if cell.prior_unknown else score
            candidates.append((-score, distance, cid, slot, action))
        return min(candidates)[-1] if candidates else None

    def observe(self, action, observation):
        if action.cell_id not in self.cells or action.slot not in {"survey", "verify"}:
            raise ValueError("Invalid action")
        if (action.cell_id, action.slot) in self.used or observation.observation_id in self.seen_ids:
            raise ValueError("Duplicate evidence cannot be accumulated")
        if action.slot == "verify" and (action.cell_id, "survey") not in self.used:
            raise ValueError("Verify requires an initial survey")
        before = self.current(action.cell_id)
        self.used.add((action.cell_id, action.slot))
        self.seen_ids.add(observation.observation_id)
        self.steps += 1
        cell = self.cells[action.cell_id]
        self.position = (cell.x, cell.y)
        if self.policy in {"pooled", "quality", "quality_gate", "adaptive_prior"}:
            rates = POOLED[action.slot] if self.policy == "pooled" else SENSOR[observation.quality]
            if not (self.policy == "quality_gate" and observation.quality == "poor" and not observation.positive):
                self.values[action.cell_id] = posterior(before, observation.positive, rates)
            self.latest_reliability[action.cell_id] = (rates["tpr"] - rates["fpr"]) ** 2
        if observation.positive:
            self.positive_cells.add(action.cell_id)
        return before, self.current(action.cell_id)


def evaluate(scene, state, cost, first_cost):
    targets = {cid for cid, present in scene.truth.items() if present}
    predicted = state.positive_cells
    found = predicted & targets
    return {"target_count": len(targets), "true_discoveries": len(found),
            "false_positives": len(predicted - targets),
            "recall": len(found) / len(targets) if targets else None,
            "precision": len(found) / len(predicted) if predicted else None,
            "first_true_discovery_cost": first_cost, "no_true_discovery": not found,
            "cost": cost, "actions": state.steps,
            "revisits": sum(slot == "verify" for _, slot in state.used),
            "brier": statistics.mean((state.current(c.cell_id) - scene.truth[c.cell_id]) ** 2 for c in scene.cells)}


def run_episode(scene, policy, budget):
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("Budget must be a positive integer")
    state = Belief(scene.cells, policy)
    cost, first_cost, log = 0, None, []
    while True:
        action = state.choose(budget - cost)
        if action is None:
            break
        observation = scene.observations[(action.cell_id, action.slot)]
        before, after = state.observe(action, observation)
        cost += action.cost
        # Ground truth used only by the evaluator, after the planner acted.
        if first_cost is None and observation.positive and scene.truth[action.cell_id]:
            first_cost = cost
        log.append({"step": state.steps, "action": asdict(action), "cost_total": cost,
                    "observation": asdict(observation), "belief_before": before, "belief_after": after})
    result = {"scenario": scene.scenario, "seed": scene.seed, "policy": policy, "budget": budget,
              **evaluate(scene, state, cost, first_cost)}
    return result, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("seeds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results, example = [], None
    for scenario in SCENARIOS:
        for seed in range(args.seeds):
            scene = synthetic_scene(seed, scenario)
            for budget in (10, 20, 40):
                for policy in POLICIES:
                    row, log = run_episode(scene, policy, budget)
                    results.append(row)
                    if scenario == "structured_prior_error" and seed == 0 and budget == 20 and policy == "adaptive_prior":
                        example = {"result": row, "steps": log}
    with (args.out / "synthetic_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for scenario in SCENARIOS:
        for budget in (10, 20, 40):
            for policy in POLICIES:
                rows = [r for r in results if (r["scenario"], r["budget"], r["policy"]) == (scenario, budget, policy)]
                summary.append({"scenario": scenario, "budget": budget, "policy": policy,
                                "episodes": len(rows),
                                "mean_recall": statistics.mean(r["recall"] for r in rows if r["recall"] is not None),
                                "mean_false_positives": statistics.mean(r["false_positives"] for r in rows),
                                "mean_revisits": statistics.mean(r["revisits"] for r in rows),
                                "no_discovery_rate": statistics.mean(r["no_true_discovery"] for r in rows)})
    report = {"stage": "SYNTHETIC_ENGINEERING_CHECK_ONLY", "real_search_efficacy_validated": False,
              "configuration": {"seeds": args.seeds, "side": 8, "budgets": [10, 20, 40], "sensor": SENSOR,
                                "pooled": POOLED, "policies": POLICIES, "scenarios": SCENARIOS},
              "limitations": ["Synthetic priors are probabilities; existing PS change scores are not.",
                              "Likelihoods are known in this simulation; not empirically estimated.",
                              "E0 cost is observation effort only: survey=1, verify=2; no travel cost.",
                              "Two pre-generated observation opportunities per cell; no repeated use of an observation ID.",
                              "Most observations are conditionally independent; persistent_occlusion explicitly correlates them.",
                              "Positive means provisional declaration, not independently verified damage confirmation.",
                              "Adaptive prior correction is an exploratory heuristic, not a demonstrated research contribution.",
                              "Synthetic seeds are not independent real disaster events; no generalization claim."],
              "total_runs": len(results), "summary": summary}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / "example_trace.json").write_text(json.dumps(example, ensure_ascii=False, indent=2))
    print(json.dumps({"stage": report["stage"], "total_runs": len(results), "output": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
