"""One prospective, source-only rank-preserving calibration control on dev72."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from rank_preserving_calibration import POLICIES, project_model, raw_order, policy_table, run_episode
from run_development_budget_controls import (PaidBinaryEnvironment, QUALITY_SEEDS,
    ORDER_SEEDS, SCENARIOS, THRESHOLDS, private_evaluator, score_trace)
from run_utility_tie_controls import (TIES, aggregate_ties, summarize_ties,
    tie_ranks, is_primary, verify_receipt)

SOURCE_NAMES = ["rank_preserving_calibration.py", "run_rank_calibration_control.py",
    "audit_rank_calibration_control.py", "test_rank_preserving_calibration.py",
    "run_unified_revisit_controls.py", "run_utility_tie_controls.py", "run_budget_planning.py",
    "run_development_budget_controls.py", "paired_count_baseline.py", "cross_cell_count_smoke.py",
    "build_xbd_mini.py", "plan_search_v2.py", "audit_development_budget_controls.py"]


def assert_fit_ids(models, selected):
    event_by_id = {r["sample_id"]: r["event"] for r in selected}
    if len(event_by_id) != 72 or any(r["split"] != "development" for r in selected):
        raise ValueError("Exact development72 selection required")
    expected_keys = {"development_fit"} | {"held_" + e for e in event_by_id.values()}
    if set(models) != expected_keys:
        raise ValueError("Unexpected fit scope")
    for event in set(event_by_id.values()):
        expected = {s for s, e in event_by_id.items() if e != event}
        if set(models["held_" + event]["fit_ids"]) != expected:
            raise ValueError("Held-event source model contaminated")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["models", "unified", "diagnostic", "perception", "design", "spatial", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("A fresh output directory is required")
    paths = {"models": args.models / "models.json", "features": args.models / "public_features.json",
        "model_receipt": args.models / "completion_receipt.json", "unified_receipt": args.unified / "completion_receipt.json",
        "references": args.unified / "scene_means.json", "reference_trajectories": args.unified / "trajectories.jsonl.gz",
        "diagnostic_receipt": args.diagnostic / "completion_receipt.json", "previous_decision": args.diagnostic / "next_step_decision.json",
        "observations": args.perception / "observations.json", "association": args.perception / "association_report.json",
        "selection": args.design / "next_development_batch.json", "contract": args.design / "experiment_contract_v2_2.json",
        "spatial": args.spatial}
    sources = {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k, p in paths.items()}
    # Freeze before projecting source means, opening evaluator outcomes or scoring.
    freeze(args.out / "protocol.json", {"stage": "DEVELOPMENT72_RANK_PRESERVING_CALIBRATION_CONTROL",
        "sources": sources, "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES},
        "policies": POLICIES, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS,
        "tie_rules": TIES, "scenarios": SCENARIOS, "upgrade_costs": [1, 2, 4], "budgets": [8, 16, 32], "thresholds": THRESHOLDS,
        "primary": {"scenario": "mixed_survey", "upgrade_cost": 2, "budget": 32, "threshold": .1, "tie_rule": "fixed_random"},
        "fit_scopes": {"development_fit": "Project frozen all72 aggregate means; entire grid; in-sample.",
            "event_holdout": "Project ONLY other-two-development-events frozen aggregate means; mixed,cost2,B32. Already inspected development data, not independent validation."},
        "projection": "Separately by native/blurred: weighted least squares nondecreasing isotonic projection of four frozen shrunken survey means, weights known_cells+8; frozen quantile edges. No outcome-based hyperparameter selection.",
        "upgrade_model": "Unchanged source-fitted conditional B-negative native-minus-blurred same-cell count divided by upgrade cost. This is NOT a new global opportunity-cost estimator. B-positive upgrades remain disabled.",
        "survey_rank": "Hard constraint: every survey subsequence is a prefix of descending public raw change, with the established tie permutation for EXACT raw ties. Always offer only next unseen raw-ranked survey, including in the nonmonotone unprojected control.",
        "cross_action_ties": "Strict per-cost score; only exact equality uses the fixed cell tie rank. No EPS, no zero-utility stop, no new native shortcut.",
        "policy_definitions": {"rank_survey": "Isotonic values, no upgrades; must reproduce original raw-survey actions/outcomes exactly, ignoring changed utility field.",
            "rank_negative_unprojected": "Restore raw survey order; original nonmonotone count survey prices and original negative-upgrade prices.",
            "rank_negative_isotonic": "Same raw survey order; projected monotone survey prices and unchanged negative-upgrade prices. Difference from unprojected isolates price projection."},
        "references": ["prior_survey", "prior_gate", "count_survey", "count_negative"],
        "expected_trajectories": 362880, "expected_threshold_scores": 1088640,
        "statistics": "Average all quality/order repetitions within scene, scenes within event, then events equally. All thresholds and settings reported. No seeds treated as independent disasters.",
        "interpretation_and_stop": "Rank-only recovery is baseline repair. If isotonic pricing has no positive held-development primary gain over both unprojected rank control and raw negative gate, stop this calibration variant; do not tune another mapping on these72. Any positive development gain remains provisional and requires event/tie sensitivity, strong related baselines and independent-data admission; it does not select a new method or reopen validation automatically.",
        "authorized_new_fit": "Only aggregate source-model isotonic projection. Prior decision no-fit flags described its completed stage; this follow-up was authorized by the user's continue instruction. No new detector/controller training.",
        "positive_revisit_policy_still_stopped": True, "new_method_selected": False,
        "validation_or_confirmation_access": False, "new_inference_calls": 0, "paid_gpu": False,
        "limits": ["Only two development events yield cached detections; keep Mexico and all empty scenes.",
            "Raw-change monotonicity is a control constraint, not an established property of held truth.",
            "Full-image independence unverified, detector training-event exposure, synthetic quality and observation-only costs remain."]})
    for directory in [args.models, args.unified, args.diagnostic]:
        verify_receipt(directory)
    selected = json.loads(paths["selection"].read_text())
    models = json.loads(paths["models"].read_text())
    assert_fit_ids(models, selected)
    projected, details = {}, {}
    for scope, params in models.items():
        projected[scope], details[scope] = project_model(params["count"])
        details[scope]["fit_ids"] = params.get("fit_ids", [r["sample_id"] for r in selected])
    atomic_json(args.out / "projected_models.json", projected)
    atomic_json(args.out / "projection_details.json", details)
    features = json.loads(paths["features"].read_text())
    observations = json.loads(paths["observations"].read_text())
    association = json.loads(paths["association"].read_text())
    ids = {r["sample_id"] for r in selected}
    if set(features) != ids or len(observations) != 9216 or {r["sample_id"] for r in observations} != ids:
        raise ValueError("Foreign or incomplete public/paid cache")
    if association["stage"] != "DEVELOPMENT72_SENSOR_QA" or len(association["rows"]) != 9216 or {r["sample_id"] for r in association["rows"]} != ids:
        raise ValueError("Foreign or incomplete evaluator")
    by_obs, by_rows = defaultdict(list), defaultdict(list)
    for r in observations:
        by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:
        by_rows[r["sample_id"]].append(r)
    spatial = {r["sample_id"]: r for r in json.loads(paths["spatial"].read_text())}
    scene_means, trajectories, scores = [], 0, 0
    started = time.perf_counter()
    with gzip.open(args.out / "trajectories.jsonl.gz", "wt", compresslevel=3) as traces, gzip.open(args.out / "metrics.jsonl.gz", "wt", compresslevel=3) as metrics:
        for sample in selected:
            sid, event = sample["sample_id"], sample["event"]
            evaluator, targets = private_evaluator(by_rows[sid])
            local = []
            ranks = {(rule, seed): tie_ranks(features[sid], rule, seed) for rule in TIES
                for seed in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1])}
            orders = {key: raw_order(features[sid], rank) for key, rank in ranks.items()}
            for scope, model_key, configs in [
                ("development_fit", "development_fit", [(s, c, b) for s in SCENARIOS for c in [1, 2, 4] for b in [8, 16, 32]]),
                ("event_holdout", "held_" + event, [("mixed_survey", 2, 32)])]:
                for scenario, cost, budget in configs:
                    tables = {p: policy_table(features[sid], models[model_key]["count"], projected[model_key], scenario, cost, p) for p in POLICIES}
                    for quality_seed in QUALITY_SEEDS:
                        for policy in POLICIES:
                            for (rule, order_seed), rank in ranks.items():
                                config = {"sample_id": sid, "event": event, "fit_scope": scope, "policy": policy,
                                    "scenario": scenario, "upgrade_cost": cost, "budget": budget,
                                    "quality_seed": quality_seed, "order_seed": order_seed, "tie_rule": rule}
                                env = PaidBinaryEnvironment(by_obs[sid], sid, scenario, quality_seed, cost)
                                t = time.perf_counter()
                                trace = run_episode(env, tables[policy], budget, cost, rank, orders[rule, order_seed])
                                elapsed = time.perf_counter() - t
                                traces.write(json.dumps({**config, "trace": trace}, separators=(",", ":")) + "\n")
                                trajectories += 1
                                for result in score_trace(trace, evaluator, targets):
                                    record = {**config, **result, "planning_s": elapsed}
                                    metrics.write(json.dumps(record, separators=(",", ":")) + "\n")
                                    local.append(record)
                                    scores += 1
            scene_means.extend({"sample_id": sid, "event": event, **r} for r in aggregate_ties(local))
            print(json.dumps({"stage": "SCENE_COMPLETE", "sample_id": sid, "trajectories": trajectories}), flush=True)
    if (trajectories, scores) != (362880, 1088640):
        raise ValueError("Incomplete fixed grid")
    atomic_json(args.out / "scene_means.json", scene_means)
    summary = summarize_ties(scene_means, spatial)
    atomic_json(args.out / "summary.json", summary)
    preserved = {k: sha(p.read_bytes()) == sources[k]["sha256"] for k, p in paths.items()}
    if not all(preserved.values()):
        raise ValueError("Frozen source artifact changed")
    report = {"stage": "DEVELOPMENT72_RANK_CALIBRATION_RUN_COMPLETE", "scenes": 72,
        "trajectories": trajectories, "threshold_scores": scores, "total_wall_s": time.perf_counter() - started,
        "primary": [r for r in summary["event_macro_means"] if is_primary(r)],
        "sources_preserved": preserved, "source_aggregate_projections": 4,
        "new_method_selected": False, "independent_efficacy_claim": False,
        "validation_or_confirmation_scored": False, "new_inference_calls": 0, "gpu_rented": False,
        "trajectory_sha256": sha((args.out / "trajectories.jsonl.gz").read_bytes()),
        "metrics_sha256": sha((args.out / "metrics.jsonl.gz").read_bytes())}
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "primary"}), flush=True)


if __name__ == "__main__":
    main()
