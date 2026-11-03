"""Full action/score replay and matched-family revisit net effects."""
from collections import Counter, defaultdict
import gzip
from itertools import groupby
import json
from pathlib import Path

from audit_development_budget_controls import ACTION_FIELDS, audit_actions
from build_xbd_mini import atomic_json, sha
from run_development_budget_controls import (QUALITY_SEEDS, ORDER_SEEDS, SCENARIOS,
    PaidBinaryEnvironment, utility_table, private_evaluator, score_trace)
from run_unified_revisit_controls import POLICIES, run_episode, import_count_trace, previous_count_rows
from run_utility_tie_controls import (CONFIG_FIELDS, TIES, aggregate_ties, summarize_ties,
    is_primary, tie_ranks, verify_receipt)

COMPARISONS = [("prior_gate", "prior_survey"), ("count_negative", "count_survey"),
    ("count_all", "count_negative"), ("count_all", "count_survey"),
    ("presence", "count_all"), ("presence", "count_negative"),
    ("count_negative", "prior_gate"), ("count_all", "prior_gate"), ("presence", "prior_gate")]
KEYS = ["fit_scope", "scenario", "upgrade_cost", "budget", "threshold", "tie_rule"]
EFFECT_FIELDS = ["candidate_found", "baseline_found", "delta_found", "candidate_fp", "baseline_fp", "delta_fp",
    "candidate_upgrades", "baseline_upgrades", "delta_upgrades", "candidate_new_from_positive_upgrades"]


def paired_effects(scene_means):
    groups = defaultdict(dict)
    for r in scene_means:
        key = tuple(r[k] for k in KEYS + ["sample_id", "event"])
        groups[key][r["policy"]] = r
    scenes = []
    event_groups = defaultdict(list)
    for key, policies in groups.items():
        for a, b in COMPARISONS:
            if a not in policies or b not in policies:
                continue
            ar, br = policies[a], policies[b]
            row = {**dict(zip(KEYS + ["sample_id", "event"], key)), "comparison": a + "_vs_" + b,
                "candidate_found": ar["distinct_found"], "baseline_found": br["distinct_found"],
                "delta_found": ar["distinct_found"] - br["distinct_found"],
                "candidate_fp": ar["false_positive_cells"], "baseline_fp": br["false_positive_cells"],
                "delta_fp": ar["false_positive_cells"] - br["false_positive_cells"],
                "candidate_upgrades": ar["upgrades"], "baseline_upgrades": br["upgrades"],
                "delta_upgrades": ar["upgrades"] - br["upgrades"],
                "candidate_new_from_positive_upgrades": ar["new_buildings_from_positive_upgrades"]}
            scenes.append(row)
            event_groups[key[:6] + (row["comparison"], key[-1])].append(row)
    events = []
    macro = defaultdict(list)
    for key, rows in sorted(event_groups.items()):
        row = {**dict(zip(KEYS + ["comparison", "event"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS}, "scenes": len(rows),
            "scene_positive": sum(r["delta_found"] > 1e-12 for r in rows),
            "scene_negative": sum(r["delta_found"] < -1e-12 for r in rows),
            "scene_zero": sum(abs(r["delta_found"]) <= 1e-12 for r in rows)}
        events.append(row)
        macro[key[:-1]].append(row)
    macros = []
    for key, rows in sorted(macro.items()):
        row = {**dict(zip(KEYS + ["comparison"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS},
            "events": len(rows), "positive_events": sum(r["delta_found"] > 1e-12 for r in rows),
            "negative_events": sum(r["delta_found"] < -1e-12 for r in rows)}
        row["relative_delta"] = row["delta_found"] / row["baseline_found"] if row["baseline_found"] else None
        macros.append(row)
    return {"scene_effects": scenes, "event_effects": events, "event_macro_effects": macros}


def old_key(r):
    return tuple(r[k] for k in CONFIG_FIELDS if k != "tie_rule")


def old_scenes(path):
    def rows():
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row["policy"] in POLICIES:
                    yield row
    for sid, group in groupby(rows(), key=lambda r: r["sample_id"]):
        yield sid, {old_key(r): r["trace"] for r in group}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "previous", "ties", "perception"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    for name, digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Frozen source code changed")
    for record in protocol["sources"].values():
        if sha(Path(record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError("Frozen input changed")
    for name, key in [("trajectories.jsonl.gz", "trajectory_sha256"), ("metrics.jsonl.gz", "metrics_sha256")]:
        if sha((args.run / name).read_bytes()) != report[key]:
            raise ValueError("Run output changed")
    verify_receipt(args.previous)
    verify_receipt(args.ties)
    selected = json.loads(Path(protocol["sources"]["selection"]["path"]).read_text())
    event_by_id = {r["sample_id"]: r["event"] for r in selected}
    ids = set(event_by_id)
    features = json.loads(Path(protocol["sources"]["features"]["path"]).read_text())
    models = json.loads(Path(protocol["sources"]["models"]["path"]).read_text())
    for event in set(event_by_id.values()):
        if set(models["held_" + event]["fit_ids"]) != {s for s in ids if event_by_id[s] != event}:
            raise ValueError("Held-event model fit contaminated")
    observations = json.loads((args.perception / "observations.json").read_text())
    by_obs = defaultdict(list)
    obs = defaultdict(dict)
    by_rows = defaultdict(list)
    for r in observations:
        if sha(Path(r["cache_file"]).read_bytes()) != r["cache_sha256"]:
            raise ValueError("Fixed prediction mask changed")
        by_obs[r["sample_id"]].append(r)
        obs[r["sample_id"]][r["cell_id"], r["mode"]] = r["declared_target"]
    for r in json.loads((args.perception / "association_report.json").read_text())["rows"]:
        by_rows[r["sample_id"]].append(r)
    evaluators = {sid: private_evaluator(rows) for sid, rows in by_rows.items()}
    old = old_scenes(args.previous / "trajectories.jsonl.gz")
    imported = previous_count_rows(args.ties / "trajectories.jsonl.gz")
    counts = Counter()
    scene_counts = Counter()
    seen = set()
    previous_sid = None
    rebuilt = []
    local = []
    native_reference = {}
    parameter_free = {}
    original = {}
    with gzip.open(args.run / "trajectories.jsonl.gz", "rt") as traces, gzip.open(args.run / "metrics.jsonl.gz", "rt") as metrics:
        for line in traces:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in ids or r["event"] != event_by_id[sid]:
                raise ValueError("Foreign scene/event")
            if sid != previous_sid:
                if previous_sid is not None:
                    if original:
                        raise ValueError("Original row controls unconsumed")
                    rebuilt.extend({"sample_id": previous_sid, "event": event_by_id[previous_sid], **v} for v in aggregate_ties(local))
                    print(json.dumps({"stage": "SCENE_REPLAY_COMPLETE", "sample_id": previous_sid, "trajectories": counts["trajectories"]}), flush=True)
                old_sid, original = next(old)
                if old_sid != sid:
                    raise ValueError("Original scene order changed")
                previous_sid = sid
                native_reference = {}
                parameter_free = {}
                local = []
            rule, policy = r["tie_rule"], r["policy"]
            key = tuple(r[k] for k in CONFIG_FIELDS)
            if key in seen:
                raise ValueError("Duplicate configuration")
            seen.add(key)
            if policy not in POLICIES or rule not in TIES or r["quality_seed"] not in QUALITY_SEEDS:
                raise ValueError("Unplanned policy/rule/quality seed")
            if r["order_seed"] not in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1]):
                raise ValueError("Unplanned order seed")
            if r["fit_scope"] == "development_fit":
                if r["scenario"] not in SCENARIOS or r["upgrade_cost"] not in [1, 2, 4] or r["budget"] not in [8, 16, 32]:
                    raise ValueError("Unplanned development setting")
            elif r["fit_scope"] != "event_holdout" or (r["scenario"], r["upgrade_cost"], r["budget"]) != ("mixed_survey", 2, 32):
                raise ValueError("Unplanned held-event setting")
            if any(set(a) != ACTION_FIELDS for a in r["trace"]):
                raise ValueError("Private or missing saved action field")
            counts.update(audit_actions(r, obs[sid]))
            counts["trajectories"] += 1
            scene_counts[sid] += 1
            if policy == "count_all":
                if r["execution_origin"] != "reused_count_greedy" or import_count_trace(imported, r) != r["trace"]:
                    raise ValueError("Imported count control differs")
                counts["reused_count_all_trajectories_verified"] += 1
            elif r["execution_origin"] != "computed_this_run":
                raise ValueError("Wrong execution origin")
            if rule == "row_major":
                source_trace = original.pop(old_key(r), None)
                if source_trace is None:
                    if (policy, r["fit_scope"]) != ("prior_gate", "event_holdout"):
                        raise ValueError("Missing original row control")
                    counts["new_parameter_free_held_prior_gate_rows"] += 1
                elif source_trace != r["trace"]:
                    raise ValueError("Original row trajectory changed")
                else:
                    counts["original_row_trajectories_reproduced"] += 1
            nk = tuple(r[k] for k in ["fit_scope", "scenario", "upgrade_cost", "budget", "quality_seed", "order_seed", "tie_rule"])
            if r["scenario"] == "native_survey":
                if policy in ["prior_survey", "count_survey"]:
                    native_reference[(policy, nk)] = r["trace"]
                elif policy in ["prior_gate", "count_negative", "count_all"]:
                    base = "prior_survey" if policy == "prior_gate" else "count_survey"
                    if r["trace"] != native_reference[(base, nk)]:
                        raise ValueError("Native-only family invariant failed")
                    counts["native_only_same_family_trajectory_invariants"] += 1
            primary_config = (r["scenario"], r["upgrade_cost"], r["budget"]) == ("mixed_survey", 2, 32)
            if primary_config and policy in ["prior_survey", "prior_gate"]:
                pk = (policy, r["quality_seed"], r["order_seed"], rule)
                if r["fit_scope"] == "development_fit":
                    parameter_free[pk] = r["trace"]
                elif r["trace"] != parameter_free[pk]:
                    raise ValueError("Parameter-free control changed by fit scope")
                else:
                    counts["parameter_free_held_scope_invariants"] += 1
            if primary_config and r["quality_seed"] == QUALITY_SEEDS[0]:
                params = models["development_fit" if r["fit_scope"] == "development_fit" else "held_" + r["event"]]
                table = utility_table(features[sid], params["count"], params["presence"], r["scenario"], r["upgrade_cost"], policy)
                env = PaidBinaryEnvironment(by_obs[sid], sid, r["scenario"], r["quality_seed"], r["upgrade_cost"])
                actual = run_episode(env, table, r["budget"], r["upgrade_cost"], tie_ranks(features[sid], rule, r["order_seed"]))
                if actual != r["trace"]:
                    raise ValueError("Full decision replay differs")
                counts["primary_first_quality_complete_decision_replays"] += 1
            for expected in score_trace(r["trace"], *evaluators[sid]):
                saved = json.loads(next(metrics))
                if tuple(saved[k] for k in CONFIG_FIELDS) != key or any(saved[k] != v for k, v in expected.items()):
                    raise ValueError("Saved metric differs from full score replay")
                if policy == "count_all" and saved["planning_s"] != 0.:
                    raise ValueError("Imported timing was reported as fresh runtime")
                counts["threshold_scores"] += 1
                local.append(saved)
        if metrics.readline() or next(imported, None) is not None or next(old, None) is not None or original:
            raise ValueError("Extra scores or unconsumed controls")
    rebuilt.extend({"sample_id": previous_sid, "event": event_by_id[previous_sid], **v} for v in aggregate_ties(local))
    if counts["trajectories"] != 725760 or counts["threshold_scores"] != 2177280 or set(scene_counts) != ids or set(scene_counts.values()) != {10080}:
        raise ValueError("Incomplete fixed grid")
    for key, expected in [("reused_count_all_trajectories_verified", 120960), ("original_row_trajectories_reproduced", 60120),
        ("new_parameter_free_held_prior_gate_rows", 360), ("primary_first_quality_complete_decision_replays", 10368),
        ("native_only_same_family_trajectory_invariants", 116640), ("parameter_free_held_scope_invariants", 8640)]:
        if counts[key] != expected:
            raise ValueError(f"Incomplete invariant: {key} = {counts[key]}, expected {expected}")
    if rebuilt != json.loads((args.run / "scene_means.json").read_text()):
        raise ValueError("Scene/seed aggregation differs")
    spatial = {r["sample_id"]: r for r in json.loads(Path(protocol["sources"]["spatial"]["path"]).read_text())}
    if summarize_ties(rebuilt, spatial) != json.loads((args.run / "summary.json").read_text()):
        raise ValueError("Event/spatial/stratum aggregation differs")
    effects = paired_effects(rebuilt)
    atomic_json(args.run / "paired_effects.json", effects)
    audit = {"stage": "DEVELOPMENT72_UNIFIED_REVISIT_FULL_REPLAY_PASSED", "counts": dict(counts),
        "configs_per_scene": 10080, "prediction_masks_verified": len(observations),
        "primary_effects": [r for r in effects["event_macro_effects"] if is_primary(r)],
        "all_budgets_eligibility_paid_outcomes_and_quality_maps_checked": True,
        "all_scores_and_aggregates_replayed": True, "held_event_fit_ids_checked": True,
        "previous_outputs_preserved": True, "new_method_selected": False,
        "validation_or_confirmation_scored": False, "independent_efficacy_claim": False,
        "new_inference_calls": 0, "gpu_rented": False, "source_sha256": sha(Path(__file__).read_bytes())}
    atomic_json(args.run / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "primary_effects"}), flush=True)


if __name__ == "__main__":
    main()
