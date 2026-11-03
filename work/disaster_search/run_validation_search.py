"""Apply all frozen follow-up policies to the untouched validation events.

Calibration-fitted rates and prior are immutable. Validation labels score
predictions and search only; they never refit or enter the online planner.
"""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import statistics

from evaluate_changeos_pilot import counts
from run_cached_search import (BUDGETS, SCENARIOS, CacheEnvironment, Evaluator, PublicCell,
                               run_episode, digest)
from run_changeos_pilot import freeze_plan
from run_local_feedback import (LOCAL_POLICIES, PROTOCOL as LOCAL_PROTOCOL,
                                 comparisons, run_local_episode)


def make_protocol(args):
    return {"version": "frozen_validation_search_v1", "split": "validation", "scenes": 24, "events": ["hurricane-michael", "mexico-earthquake"],
            "timing": "Frozen after calibration scope diagnosis and local follow-up, before validation model inference/search.",
            "policy_definitions": LOCAL_PROTOCOL, "policies": LOCAL_PROTOCOL["policies"],
            "source_parameters_sha256": digest(args.parameters),
            "inference_plan_sha256": digest(args.inference_plan), "association_plan_sha256": digest(args.out / "association_protocol.json"),
            "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                ("run_cached_search.py", "run_local_feedback.py", "real_evidence.py", "associate_buildings.py", "run_changeos_calibration.py", "run_validation_search.py")},
            "fit": "Apply original calibration bin edges, bin probabilities and mode likelihoods unchanged. No event-specific fit, threshold tuning or score-dependent filtering. Report every frozen policy/condition/budget.",
            "diagnostics": "Report validation cell detection rates, prior Brier versus original-calibration global probability, event-level policy differences and scene pairs. No confidence based on treating grid cells or quality seeds as independent events.",
            "decision": "Local residual remains a candidate only if it has a consistent incremental advantage over quality gate and simple neighbor-positive; preserve regressions and event concentration. No test run in this request.",
            "interpretation": "These two events are held out from prior/sensor/policy fitting, but checkpoint training exposure is expected. Not unseen-event perception evaluation or a real UAV deployment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "association", "parameters", "inference-plan", "prepared-data", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args)
    freeze_plan(args.out / "search_protocol.json", protocol)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "search_protocol.json"), "episodes": 3024}))
        return
    params = json.loads(args.parameters.read_text())
    observations = json.loads(args.observations.read_text())
    association = json.loads(args.association.read_text())
    plan = json.loads(args.inference_plan.read_text())
    selected = {s["sample_id"] for s in plan["samples"]}
    expected_ids = {f"{sid}:r{r:02d}c{c:02d}:{mode}" for sid in selected
                    for r in range(8) for c in range(8) for mode in ("native", "blurred")}
    if (len(observations) != 3072 or {o["observation_id"] for o in observations} != expected_ids
            or any(o["split"] != "validation" for o in observations)
            or association["stage"] != "FULL_VALIDATION_BOUNDARY_DIAGNOSTICS"):
        raise ValueError("Only complete frozen validation grid accepted")
    if selected & set(params["public_cue_hashes"]) or len(selected) != 24:
        raise ValueError("Validation/calibration scenes overlap or scope incomplete")
    if association["observations_sha256"] != digest(args.observations):
        raise ValueError("Observation/evaluator mismatch")
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    prior = params["initial_presence_prior"]
    results, diagnostics = [], []
    cue_hashes = {}
    policies = LOCAL_PROTOCOL["policies"]
    with (args.out / "traces.jsonl").open("w") as f:
        for sid in sorted(selected):
            cue_path = args.prepared_data / sid / "public_cues.json"
            cues = json.loads(cue_path.read_text())
            cue_hashes[sid] = digest(cue_path)
            if cues["split"] != "validation":
                raise ValueError("Non-validation cues")
            cells = []
            for c in cues["cells"]:
                group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                cells.append(PublicCell(c["cell_id"], c["col"], c["row"], group, prior["bins"][group]["fitted_presence_probability"]))
            rows = [r for r in association["rows"] if r["sample_id"] == sid]
            p_by_cell = {c.cell_id: c.prior for c in cells}
            known = [r for r in rows if r["mode"] == "native" and r["target"] is not None]
            diagnostics.append({"sample_id": sid, "event": cues["event"], "known_cells": len(known),
                "positive_cells": sum(r["target"] for r in known),
                "prior_brier": statistics.mean((p_by_cell[r["cell_id"]] - r["target"]) ** 2 for r in known),
                "calibration_global_probability_brier": statistics.mean((prior["global_smoothed_prevalence"] - r["target"]) ** 2 for r in known),
                "sensor_counts": {mode: counts([r for r in rows if r["mode"] == mode and r["target"] is not None]) for mode in ("native", "blurred")}})
            for scenario, seeds in SCENARIOS.items():
                for seed in seeds:
                    for budget in BUDGETS:
                        for policy in policies:
                            env = CacheEnvironment(sid, observations, scenario, seed)
                            runner = run_local_episode if policy in LOCAL_POLICIES else run_episode
                            row, trace = runner(cells, policy, rates, scenario, budget, env.observe, Evaluator(rows))
                            if row["cost"] > budget or len(env.requests) != len(set(env.requests)):
                                raise AssertionError("Budget/evidence failure")
                            row.update(sample_id=sid, event=cues["event"], scenario=scenario, quality_seed=seed, budget=budget, policy=policy)
                            results.append(row)
                            f.write(json.dumps({"run": row, "steps": trace}, ensure_ascii=False) + "\n")
            print(json.dumps({"validation_scene": sid, "episodes_done": len(results)}), flush=True)
    if len(results) != 3024:
        raise AssertionError("Incomplete validation run")
    summaries = []
    for scenario, seeds in SCENARIOS.items():
        for budget in BUDGETS:
            for policy in policies:
                rows = [r for r in results if (r["scenario"], r["budget"], r["policy"]) == (scenario, budget, policy)]
                summaries.append({"scenario": scenario, "budget": budget, "policy": policy,
                    "sum_buildings_found": sum(r["unique_buildings_found"] for r in rows) / len(seeds),
                    "sum_false_positive_cells": sum(r["false_positive_cells"] for r in rows) / len(seeds),
                    "mean_upgrades": statistics.mean(r["upgrades"] for r in rows),
                    "mean_feedback_rank_changes": statistics.mean(r["feedback_rank_changes"] for r in rows)})
    diagnostics_report = {}
    for event in sorted({d["event"] for d in diagnostics}):
        rows = [d for d in diagnostics if d["event"] == event]
        diagnostics_report[event] = {"scenes": len(rows), "known_cells": sum(r["known_cells"] for r in rows),
            "positive_cells": sum(r["positive_cells"] for r in rows),
            "equal_scene_prior_brier": statistics.mean(r["prior_brier"] for r in rows),
            "equal_scene_calibration_global_probability_brier": statistics.mean(r["calibration_global_probability_brier"] for r in rows),
            "sensor_totals": {mode: {key: sum(r["sensor_counts"][mode][key] for r in rows) for key in ("tp", "fn", "fp", "tn")}
                              for mode in ("native", "blurred")}}
    report = {"stage": "FROZEN_VALIDATION_SEARCH", "episodes": len(results), "validation_used": True, "test_used": False,
              "parameters_refitted": False, "protocol_sha256": digest(args.out / "search_protocol.json"),
              "observations_sha256": digest(args.observations), "association_sha256": digest(args.association),
              "source_parameters_sha256": digest(args.parameters), "public_cue_hashes": cue_hashes,
              "summary": summaries, "paired_comparisons": comparisons(results, ("adaptive_prior", *LOCAL_POLICIES)),
              "transfer_diagnostics_by_event": diagnostics_report, "scene_transfer_diagnostics": diagnostics,
              "interpretation": protocol["interpretation"], "limitations": LOCAL_PROTOCOL["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / "runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"episodes": len(results), "mixed_survey": [r for r in summaries if r["scenario"] == "mixed_survey"], "transfer": diagnostics_report}))


if __name__ == "__main__":
    main()
