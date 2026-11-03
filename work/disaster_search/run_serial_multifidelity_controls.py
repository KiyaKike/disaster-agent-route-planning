"""Primary-only matched two-stage serial multifidelity baseline controls."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from run_rank_calibration_control import assert_fit_ids
from run_budget_planning import planning_inputs
from run_development_budget_controls import (PaidBinaryEnvironment, QUALITY_SEEDS,
    ORDER_SEEDS, THRESHOLDS, private_evaluator, score_trace)
from run_utility_tie_controls import (TIES, aggregate_ties, summarize_ties, tie_ranks, verify_receipt)
from serial_multifidelity_baseline import POLICIES, compress_inputs, run_episode

SOURCE_NAMES = ["serial_multifidelity_baseline.py", "run_serial_multifidelity_controls.py",
    "audit_serial_multifidelity_controls.py", "test_serial_multifidelity_baseline.py",
    "run_budget_planning.py", "run_utility_tie_controls.py", "run_development_budget_controls.py",
    "run_rank_calibration_control.py", "paired_count_baseline.py", "cross_cell_count_smoke.py",
    "audit_development_budget_controls.py", "build_xbd_mini.py", "plan_search_v2.py"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Fresh output directory required")
    base = Path.cwd()
    models_dir = base / "output/disaster_search_development72_budget_controls_v1"
    perception = base / "output/disaster_search_hold_development72_perception_v1"
    design = base / "output/disaster_search_v2_design"
    paths = {"models": models_dir / "models.json", "features": models_dir / "public_features.json",
        "model_receipt": models_dir / "completion_receipt.json",
        "previous_receipt": base / "output/disaster_search_metadata_admission_v1/completion_receipt.json",
        "ties_receipt": base / "output/disaster_search_utility_ties_v1/completion_receipt.json",
        "old_batch_trajectories": base / "output/disaster_search_utility_ties_v1/trajectories.jsonl.gz",
        "old_batch_scenes": base / "output/disaster_search_utility_ties_v1/scene_means.json",
        "simple_receipt": base / "output/disaster_search_unified_revisit_v1/completion_receipt.json",
        "simple_scenes": base / "output/disaster_search_unified_revisit_v1/scene_means.json",
        "observations": perception / "observations.json", "association": perception / "association_report.json",
        "selection": design / "next_development_batch.json", "contract": design / "experiment_contract_v2_2.json",
        "spatial": base / "output/disaster_search_cross_cell_smoke_v1/spatial_groups.json",
        "adaptation": Path(__file__).with_name("serial_multifidelity_adaptation.md").resolve(),
        "paper": base / "output/disaster_search_budget_planning_v1/sources/nguyen21f.pdf",
        "paper_text": base / "output/disaster_search_budget_planning_v1/sources/nguyen21f.txt",
        "paper_receipt": base / "output/disaster_search_budget_planning_v1/sources/source_receipt.json",
        "preflight": Path(__file__).with_name("serial_multifidelity_public_preflight.json").resolve()}
    sources = {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k,p in paths.items()}
    freeze(args.out / "protocol.json", {"stage": "DEVELOPMENT72_SERIAL_TWO_STAGE_MULTIFIDELITY_PRIMARY_CONTROL",
        "sources": sources, "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES},
        "policies": POLICIES, "scenarios": ["mixed_survey"], "upgrade_costs": [2], "budgets": [32],
        "thresholds": THRESHOLDS, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS,
        "tie_rules": TIES, "primary_tie_rule": "fixed_random", "primary_threshold": .1,
        "fit_scopes": {"development_fit": "Frozen all72 models; in-sample.",
            "event_holdout": "Frozen other-two-development-events models; already inspected development, not independent validation."},
        "expected_trajectories": 17280, "expected_threshold_scores": 51840,
        "exploration": "At most min(C,R-C) unseen cells outside current cost-optimal final batch; exchange-value ranking; all prefixes including0; pay every survey and possible upgrade. Exact native/B-/B+ enumeration then outcome-adapted costed tail. No new unlocks inside final tail batch.",
        "source_model": "Exactly frozen local count means and paired blur declaration probabilities; no refit, spatial propagation, global building identity or future outcomes.",
        "numeric_anchor": "serial_one_stage directly uses frozen batch_tail values. two_stage adds nonnegative extra value to the same values; compressed h0 must match original Q within1e-9. All original online EPS/greedy/cell ties unchanged.",
        "control_required": "Every one-stage trajectory must exactly reproduce the previously saved matched batch_tail trace after removing greedy_utility; all policies fully decision/paid-outcome/score replayed.",
        "statistics": "All repeats within scene, scenes within event, then events equally; all3 tie rules and all3 thresholds, retain Mexico. No efficacy claim from surrogate Q or number of trajectories.",
        "timing": "Bounded public-parameter caches reused within scene/fit scope across repetitions, then cleared; saved planning_s is execution accounting, not a cold-cache runtime benchmark.",
        "interpretation_stop": "If two-stage improves only surrogate values, or fails to beat raw ranking/gate or its one-stage control, record that result and do not deepen this planner or tune k on these72. Any positive development result remains provisional; no method selected or reserved scoring automatically enabled.",
        "not_faithful_original_mfens": True, "new_method_selected": False,
        "old_stopped_variants_remain_stopped": True, "validation_or_confirmation_access": False,
        "new_model_fit": False, "new_inference_calls": 0, "paid_gpu": False,
        "limits": ["Serial shared costs and noisy native feedback differ from original exact parallel high fidelity.",
            "Expected same-cell count proxy omits cross-cell duplicate discoveries.",
            "Only2 development events have cached detections, geographic independence unverified, all candidate events exposed to detector training.",
            "Only primary B32,C2,mixed tested in this bounded stage; no full cost/quality/budget robustness claim."]})
    for key in ["model_receipt", "previous_receipt", "ties_receipt", "simple_receipt"]:
        verify_receipt(paths[key].parent)
    original_paper = json.loads(paths["paper_receipt"].read_text())["sources"]["mfens"]
    if sha(paths["paper"].read_bytes()) != original_paper["sha256"]:
        raise ValueError("Primary paper changed")
    (args.out / "adaptation.md").write_bytes(paths["adaptation"].read_bytes())
    (args.out / "public_model_preflight.json").write_bytes(paths["preflight"].read_bytes())
    selected = json.loads(paths["selection"].read_text())
    models, features = json.loads(paths["models"].read_text()), json.loads(paths["features"].read_text())
    assert_fit_ids(models, selected)
    ids = {r["sample_id"] for r in selected}
    observations = json.loads(paths["observations"].read_text())
    association = json.loads(paths["association"].read_text())
    if set(features) != ids or len(observations) != 9216 or {r["sample_id"] for r in observations} != ids:
        raise ValueError("Foreign/incomplete public or paid cache")
    if association["stage"] != "DEVELOPMENT72_SENSOR_QA" or len(association["rows"]) != 9216 or {r["sample_id"] for r in association["rows"]} != ids:
        raise ValueError("Foreign/incomplete evaluator")
    by_obs, by_rows = defaultdict(list), defaultdict(list)
    for r in observations:
        by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:
        by_rows[r["sample_id"]].append(r)
    spatial = {r["sample_id"]: r for r in json.loads(paths["spatial"].read_text())}
    trajectories, scores, scene_means, diagnostics = 0, 0, [], []
    started = time.perf_counter()
    with gzip.open(args.out / "trajectories.jsonl.gz", "wt", compresslevel=3) as traces, gzip.open(args.out / "metrics.jsonl.gz", "wt", compresslevel=3) as metrics:
        for sample in selected:
            sid, event = sample["sample_id"], sample["event"]
            evaluator, targets = private_evaluator(by_rows[sid])
            ranks = {(rule, seed): tie_ranks(features[sid], rule, seed) for rule in TIES
                for seed in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1])}
            local = []
            for scope, key in [("development_fit", "development_fit"), ("event_holdout", "held_" + event)]:
                inputs = planning_inputs(features[sid], models[key], "mixed_survey", 2)
                model, membership = compress_inputs(inputs, 2)
                for quality_seed in QUALITY_SEEDS:
                    for policy in POLICIES:
                        for (rule, order_seed), rank in ranks.items():
                            config = {"sample_id": sid, "event": event, "fit_scope": scope, "scenario": "mixed_survey",
                                "upgrade_cost": 2, "budget": 32, "quality_seed": quality_seed,
                                "order_seed": order_seed, "policy": policy, "tie_rule": rule}
                            env = PaidBinaryEnvironment(by_obs[sid], sid, "mixed_survey", quality_seed, 2)
                            t = time.perf_counter()
                            trace, stats = run_episode(env, inputs, 32, 2, rank, policy, model, membership)
                            elapsed = time.perf_counter() - t
                            traces.write(json.dumps({**config, "trace": trace, "planning_diagnostics": stats}, separators=(",", ":")) + "\n")
                            trajectories += 1
                            diagnostics.append({**config, **stats})
                            for result in score_trace(trace, evaluator, targets):
                                row = {**config, **result, "planning_s": elapsed}
                                metrics.write(json.dumps(row, separators=(",", ":")) + "\n")
                                local.append(row)
                                scores += 1
                for cached in [model.tail, model.after_surveys, model.future, model.q]:
                    cached.cache_clear()
            scene_means.extend({"sample_id": sid, "event": event, **r} for r in aggregate_ties(local))
            print(json.dumps({"stage": "SCENE_COMPLETE", "sample_id": sid, "trajectories": trajectories}), flush=True)
    if (trajectories, scores) != (17280, 51840):
        raise ValueError("Incomplete primary grid")
    atomic_json(args.out / "scene_means.json", scene_means)
    summary = summarize_ties(scene_means, spatial)
    atomic_json(args.out / "summary.json", summary)
    atomic_json(args.out / "planning_diagnostics.json", diagnostics)
    report = {"stage": "SERIAL_TWO_STAGE_MULTIFIDELITY_PRIMARY_RUN_COMPLETE", "scenes":72,
        "trajectories": trajectories, "threshold_scores": scores, "wall_s": time.perf_counter()-started,
        "primary": [r for r in summary["event_macro_means"] if r["threshold"] == .1],
        "source_artifacts_preserved": {k: sha(p.read_bytes()) == sources[k]["sha256"] for k,p in paths.items()},
        "new_fit": False, "new_inference_calls": 0, "gpu_rented": False,
        "new_method_selected": False, "validation_or_confirmation_scored": False,
        "faithful_original_mfens_reproduction": False,
        "trajectory_sha256": sha((args.out / "trajectories.jsonl.gz").read_bytes()),
        "metrics_sha256": sha((args.out / "metrics.jsonl.gz").read_bytes())}
    if not all(report["source_artifacts_preserved"].values()):
        raise ValueError("Previous artifact changed")
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!='primary'}), flush=True)


if __name__ == "__main__":
    main()
