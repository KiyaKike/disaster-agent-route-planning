"""Replay fixed-budget tie controls; descriptive paired effects only."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from audit_budget_planning import action_signature, trajectory_key
from audit_development_budget_controls import ACTION_FIELDS, audit_actions
from build_xbd_mini import atomic_json, sha
from run_budget_planning import POLICIES, planning_inputs
from run_development_budget_controls import (ORDER_SEEDS, QUALITY_SEEDS, SCENARIOS,
    SUMMARY_FIELDS, THRESHOLDS, PaidBinaryEnvironment, private_evaluator, score_trace)
from run_utility_tie_controls import (CONFIG_FIELDS, TIES, aggregate_ties, is_primary,
    run_episode, summarize_ties, tie_ranks, verify_receipt)

EFFECT_KEYS = ["fit_scope", "scenario", "upgrade_cost", "budget", "threshold"]
EFFECT_FIELDS = ["candidate_found", "baseline_found", "delta_found", "delta_fp",
                 "delta_upgrades", "delta_positive_upgrades"]


def paired_effects(scene_means, old_means):
    groups = defaultdict(dict)
    for r in scene_means:
        key = tuple(r[k] for k in EFFECT_KEYS + ["sample_id", "event"])
        groups[key][r["policy"] + "/" + r["tie_rule"]] = r
    for r in old_means:
        key = tuple(r[k] for k in EFFECT_KEYS + ["sample_id", "event"])
        groups[key]["old/" + r["policy"]] = r
    comparisons = [(p + "/" + rule, p + "/row_major") for p in POLICIES
                   for rule in ["raw_change", "fixed_random"]]
    comparisons += [(p + "/raw_change", p + "/fixed_random") for p in POLICIES]
    comparisons += [(p + "/" + rule, "old/" + b) for p in POLICIES for rule in TIES
                    for b in ["prior_survey", "count_survey", "count_negative"]]
    comparisons += [(p + "/" + rule, "proxy_greedy/" + rule)
                    for p in ["two_step", "batch_tail"] for rule in TIES]
    scene_rows = []
    events = defaultdict(list)
    for key, rows in groups.items():
        for a, b in comparisons:
            if a not in rows or b not in rows:
                continue
            ar, br = rows[a], rows[b]
            row = {**dict(zip(EFFECT_KEYS + ["sample_id", "event"], key)),
                "comparison": a + "_vs_" + b, "candidate_found": ar["distinct_found"],
                "baseline_found": br["distinct_found"], "delta_found": ar["distinct_found"] - br["distinct_found"],
                "delta_fp": ar["false_positive_cells"] - br["false_positive_cells"],
                "delta_upgrades": ar["upgrades"] - br["upgrades"],
                "delta_positive_upgrades": ar["positive_upgrades"] - br["positive_upgrades"]}
            scene_rows.append(row)
            events[key[:5] + (row["comparison"], key[-1])].append(row)
    event_rows = []
    macro = defaultdict(list)
    for key, rows in sorted(events.items()):
        row = {**dict(zip(EFFECT_KEYS + ["comparison", "event"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS},
            "scenes": len(rows), "scene_positive": sum(r["delta_found"] > 1e-12 for r in rows),
            "scene_negative": sum(r["delta_found"] < -1e-12 for r in rows),
            "scene_zero": sum(abs(r["delta_found"]) <= 1e-12 for r in rows)}
        event_rows.append(row)
        macro[key[:-1]].append(row)
    macro_rows = []
    for key, rows in sorted(macro.items()):
        row = {**dict(zip(EFFECT_KEYS + ["comparison"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS},
            "positive_events": sum(r["delta_found"] > 1e-12 for r in rows), "events": len(rows)}
        row["relative_delta"] = row["delta_found"] / row["baseline_found"] if row["baseline_found"] else None
        macro_rows.append(row)
    return {"scene_effects": scene_rows, "event_effects": event_rows, "event_macro_effects": macro_rows}


def random_order_summary(order_scene_values):
    """Event-equal macro per fixed order seed; not a sampling-error interval."""
    groups = defaultdict(list)
    for key, values in order_scene_values.items():
        scope, policy, seed, event, sid = key
        if len(values) != len(QUALITY_SEEDS):
            raise ValueError("Random order missing quality seed")
        groups[(scope, policy, seed, event)].append(sum(values) / len(values))
    events = []
    macro = defaultdict(list)
    for (scope, policy, seed, event), values in sorted(groups.items()):
        if len(values) != 24:
            raise ValueError("Random order event missing scene")
        value = sum(values) / len(values)
        events.append({"fit_scope": scope, "policy": policy, "order_seed": seed,
                       "event": event, "distinct_found": value, "scenes": len(values)})
        macro[(scope, policy, seed)].append(value)
    macro_rows = [{"fit_scope": scope, "policy": policy, "order_seed": seed,
        "distinct_found": sum(v) / len(v), "events": len(v)} for (scope, policy, seed), v in sorted(macro.items())]
    ranges = defaultdict(list)
    for r in macro_rows:
        ranges[(r["fit_scope"], r["policy"])].append(r["distinct_found"])
    range_rows = [{"fit_scope": scope, "policy": policy, "minimum": min(v), "maximum": max(v),
        "mean": sum(v) / len(v), "orders": len(v), "confidence_interval": False} for (scope, policy), v in sorted(ranges.items())]
    return {"primary_per_order_event_means": events, "primary_per_order_event_macro_means": macro_rows,
            "primary_order_sensitivity": range_rows}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "planning", "previous", "perception"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    for name, digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Frozen source code changed")
    for record in protocol["sources"].values():
        if sha(Path(record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError("Frozen source input changed")
    for name, key in [("trajectories.jsonl.gz", "trajectory_sha256"), ("metrics.jsonl.gz", "metrics_sha256")]:
        if sha((args.run / name).read_bytes()) != report[key]:
            raise ValueError("Output changed")
    verify_receipt(args.planning)
    verify_receipt(args.previous)
    selected = json.loads(Path(protocol["sources"]["selection"]["path"]).read_text())
    ids = {r["sample_id"] for r in selected}
    event_by_id = {r["sample_id"]: r["event"] for r in selected}
    features = json.loads(Path(protocol["sources"]["features"]["path"]).read_text())
    models = json.loads(Path(protocol["sources"]["models"]["path"]).read_text())
    for event in set(event_by_id.values()):
        fitted_ids = set(models["held_" + event]["fit_ids"])
        if fitted_ids != {sid for sid in ids if event_by_id[sid] != event}:
            raise ValueError("Held-event fit contamination")
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
    counts = Counter()
    changed = Counter()
    config_counts = Counter()
    scene_counts = Counter()
    seen = set()
    native_reference = {}
    previous_scene = None
    local = []
    rebuilt = []
    order_scene_values = defaultdict(list)
    with gzip.open(args.run / "trajectories.jsonl.gz", "rt") as traces, gzip.open(args.run / "metrics.jsonl.gz", "rt") as metrics, gzip.open(args.planning / "trajectories.jsonl.gz", "rt") as old:
        for line in traces:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in ids or r["event"] != event_by_id[sid]:
                raise ValueError("Foreign scene/event")
            if sid != previous_scene:
                if previous_scene is not None:
                    rebuilt.extend({"sample_id": previous_scene, "event": event_by_id[previous_scene], **a} for a in aggregate_ties(local))
                previous_scene = sid
                local = []
                native_reference = {}
            rule = r["tie_rule"]
            key = tuple(r[k] for k in CONFIG_FIELDS)
            if key in seen:
                raise ValueError("Duplicate configuration")
            seen.add(key)
            if rule not in TIES or r["policy"] not in POLICIES or r["quality_seed"] not in QUALITY_SEEDS:
                raise ValueError("Unplanned rule/policy/quality seed")
            if r["order_seed"] not in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1]):
                raise ValueError("Unplanned order seed")
            if r["fit_scope"] == "development_fit":
                if r["scenario"] not in SCENARIOS or r["upgrade_cost"] not in [1, 2, 4] or r["budget"] not in [8, 16, 32]:
                    raise ValueError("Unplanned development setting")
            elif r["fit_scope"] != "event_holdout" or (r["scenario"], r["upgrade_cost"], r["budget"]) != ("mixed_survey", 2, 32):
                raise ValueError("Unplanned held-event setting")
            config_counts[(r["fit_scope"], rule)] += 1
            scene_counts[sid] += 1
            if any(set(a) != ACTION_FIELDS | {"greedy_utility"} for a in r["trace"]):
                raise ValueError("Private or missing action fields")
            stripped = {**r, "trace": [{k: v for k, v in a.items() if k in ACTION_FIELDS} for a in r["trace"]]}
            counts.update(audit_actions(stripped, obs[sid]))
            counts["trajectories"] += 1
            if rule == "row_major":
                old_r = json.loads(next(old))
                if trajectory_key(r) != trajectory_key(old_r) or r["policy"] != old_r["policy"] or r["trace"] != old_r["trace"]:
                    raise ValueError("Row-major controls changed from frozen planning")
                counts["previous_planning_trajectories_reproduced"] += 1
            if r["scenario"] == "native_survey":
                nk = tuple(r[k] for k in ["fit_scope", "scenario", "upgrade_cost", "budget", "quality_seed", "order_seed", "tie_rule"])
                signature = action_signature(r["trace"])
                if r["policy"] == "proxy_greedy":
                    native_reference[nk] = signature
                elif signature != native_reference[nk]:
                    raise ValueError("Native-only same-order policy invariant failed")
                else:
                    counts["native_only_policy_invariants_checked"] += 1
            if (r["scenario"], r["upgrade_cost"], r["budget"]) == ("mixed_survey", 2, 32) and r["quality_seed"] == QUALITY_SEEDS[0]:
                params = models["development_fit" if r["fit_scope"] == "development_fit" else "held_" + r["event"]]
                model = planning_inputs(features[sid], params, r["scenario"], r["upgrade_cost"])
                env = PaidBinaryEnvironment(by_obs[sid], sid, r["scenario"], r["quality_seed"], r["upgrade_cost"])
                replayed = run_episode(env, model, r["policy"], r["budget"], r["upgrade_cost"], tie_ranks(features[sid], rule, r["order_seed"]))
                if replayed != r["trace"]:
                    raise ValueError("Tie/decision replay differs")
                counts["primary_first_quality_full_decision_replays"] += 1
            for expected in score_trace(r["trace"], *evaluators[sid]):
                saved = json.loads(next(metrics))
                if tuple(saved[k] for k in CONFIG_FIELDS) != key or any(saved[k] != v for k, v in expected.items()):
                    raise ValueError("Budget score replay differs")
                local.append(saved)
                counts["threshold_scores"] += 1
                if rule == "fixed_random" and is_primary(saved):
                    order_scene_values[(r["fit_scope"], r["policy"], r["order_seed"], r["event"], sid)].append(saved["distinct_found"])
            if rule != "row_major":
                # The immediately preceding row-major trace for this policy is
                # still old_r while processing its raw/random variants.
                changed[(r["fit_scope"], r["policy"], rule)] += action_signature(r["trace"]) != action_signature(old_r["trace"])
        if metrics.readline() or old.readline():
            raise ValueError("Extra metric or missing original controls")
    rebuilt.extend({"sample_id": previous_scene, "event": event_by_id[previous_scene], **a} for a in aggregate_ties(local))
    if counts["trajectories"] != 362880 or counts["threshold_scores"] != 1088640 or set(scene_counts) != ids or set(scene_counts.values()) != {5040}:
        raise ValueError("Incomplete fixed grid")
    if counts["previous_planning_trajectories_reproduced"] != 30240 or counts["primary_first_quality_full_decision_replays"] != 5184:
        raise ValueError("Missing baseline or full decision replay")
    saved_means = json.loads((args.run / "scene_means.json").read_text())
    if rebuilt != saved_means:
        raise ValueError("Scene/seed aggregation differs")
    spatial = {r["sample_id"]: r for r in json.loads(Path(protocol["sources"]["spatial"]["path"]).read_text())}
    if summarize_ties(rebuilt, spatial) != json.loads((args.run / "summary.json").read_text()):
        raise ValueError("Event/spatial aggregation differs")
    effects = paired_effects(saved_means, json.loads((args.previous / "scene_means.json").read_text()))
    atomic_json(args.run / "paired_effects.json", effects)
    random_summary = random_order_summary(order_scene_values)
    atomic_json(args.run / "random_order_sensitivity.json", random_summary)
    audit = {"stage": "DEVELOPMENT72_TIE_BUDGET_AND_SCORE_REPLAY_PASSED", "counts": dict(counts),
        "configs_per_scene": 5040, "changed_vs_row_major": [{"fit_scope": s, "policy": p, "tie_rule": t, "trajectories_changed": v}
            for (s, p, t), v in sorted(changed.items())],
        "primary_effects": [r for r in effects["event_macro_effects"] if is_primary(r)],
        "fixed_masks_verified": len(observations), "all_scores_and_aggregates_replayed": True,
        "all_budgets_eligibility_paid_outcomes_and_quality_maps_checked": True,
        "original_planning_controls_reproduced": True, "previous_outputs_preserved": True,
        "held_event_fit_ids_checked": True, "new_method_selected": False,
        "independent_efficacy_claim": False, "validation_or_confirmation_scored": False,
        "new_inference_calls": 0, "gpu_rented": False, "audit_source_sha256": sha(Path(__file__).read_bytes())}
    atomic_json(args.run / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "primary_effects"}), flush=True)


if __name__ == "__main__":
    main()
