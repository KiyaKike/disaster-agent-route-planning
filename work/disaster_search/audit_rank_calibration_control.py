"""Replay every paid decision/score and isolate rank repair from price changes."""
from collections import Counter, defaultdict
import gzip
import itertools
import json
from pathlib import Path

from audit_development_budget_controls import audit_actions
from build_xbd_mini import atomic_json, sha
from cross_cell_count_smoke import CELLS
from rank_preserving_calibration import POLICIES, project_model, policy_table, raw_order, run_episode
from run_rank_calibration_control import assert_fit_ids
from run_development_budget_controls import (QUALITY_SEEDS, ORDER_SEEDS, SCENARIOS,
    PaidBinaryEnvironment, utility_table, private_evaluator, score_trace, SUMMARY_FIELDS)
from run_unified_revisit_controls import run_episode as original_episode
from run_utility_tie_controls import (CONFIG_FIELDS, TIES, aggregate_ties, summarize_ties,
    is_primary, tie_ranks, verify_receipt)

KEYS = ["fit_scope", "scenario", "upgrade_cost", "budget", "threshold", "tie_rule"]
COMPARISONS = [("rank_survey", "prior_survey"),
    ("rank_survey", "count_survey"),
    ("rank_negative_unprojected", "count_negative"),
    ("rank_negative_unprojected", "rank_survey"),
    ("rank_negative_isotonic", "rank_negative_unprojected"),
    ("rank_negative_isotonic", "rank_survey"),
    ("rank_negative_isotonic", "prior_gate"),
    ("rank_negative_isotonic", "count_negative")]
EFFECT_FIELDS = ["candidate_found", "baseline_found", "delta_found", "delta_fp", "delta_upgrades"]


def independent_projection(values, weights):
    """Enumerate all contiguous partitions of FOUR bins; no SciPy PAVA call."""
    best = None
    for cuts in itertools.product([False, True], repeat=3):
        edges = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [4]
        means = [sum(values[i] * weights[i] for i in range(a, b)) / sum(weights[a:b])
                 for a, b in zip(edges, edges[1:])]
        if any(a > b + 1e-14 for a, b in zip(means, means[1:])):
            continue
        fitted = [mean for mean, a, b in zip(means, edges, edges[1:]) for _ in range(a, b)]
        loss = sum(w * (y - x) ** 2 for w, y, x in zip(weights, values, fitted))
        if best is None or loss < best[0]:
            best = loss, fitted
    return best[1]


def paired_effects(scene_means):
    groups = defaultdict(dict)
    for row in scene_means:
        key = tuple(row[k] for k in KEYS + ["sample_id", "event"])
        if row["policy"] in groups[key]:
            raise ValueError("Duplicated scene-policy mean")
        groups[key][row["policy"]] = row
    scenes, events = [], defaultdict(list)
    for key, policies in groups.items():
        for a, b in COMPARISONS:
            if a not in policies or b not in policies:
                continue
            ar, br = policies[a], policies[b]
            row = {**dict(zip(KEYS + ["sample_id", "event"], key)), "comparison": a + "_vs_" + b,
                "candidate_found": ar["distinct_found"], "baseline_found": br["distinct_found"],
                "delta_found": ar["distinct_found"] - br["distinct_found"],
                "delta_fp": ar["false_positive_cells"] - br["false_positive_cells"],
                "delta_upgrades": ar["upgrades"] - br["upgrades"]}
            scenes.append(row)
            events[key[:6] + (row["comparison"], key[-1])].append(row)
    event_rows, macro = [], defaultdict(list)
    for key, rows in sorted(events.items()):
        row = {**dict(zip(KEYS + ["comparison", "event"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS}, "scenes": len(rows),
            "scene_positive": sum(r["delta_found"] > 1e-12 for r in rows),
            "scene_negative": sum(r["delta_found"] < -1e-12 for r in rows)}
        event_rows.append(row)
        macro[key[:-1]].append(row)
    macros = []
    for key, rows in sorted(macro.items()):
        row = {**dict(zip(KEYS + ["comparison"], key)),
            **{f: sum(r[f] for r in rows) / len(rows) for f in EFFECT_FIELDS}, "events": len(rows),
            "positive_events": sum(r["delta_found"] > 1e-12 for r in rows),
            "negative_events": sum(r["delta_found"] < -1e-12 for r in rows)}
        row["relative_delta"] = row["delta_found"] / row["baseline_found"] if row["baseline_found"] else None
        macros.append(row)
    return {"scene_effects": scenes, "event_effects": event_rows, "event_macro_effects": macros}


def without_utility(trace):
    return [{k: v for k, v in action.items() if k != "utility"} for action in trace]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    paths = {k: Path(r["path"]) for k, r in protocol["sources"].items()}
    for k, path in paths.items():
        if sha(path.read_bytes()) != protocol["sources"][k]["sha256"]:
            raise ValueError("Frozen input changed: " + k)
    for name, digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Frozen source code changed: " + name)
    for key in ["model_receipt", "unified_receipt", "diagnostic_receipt"]:
        verify_receipt(paths[key].parent)
    for name, key in [("trajectories.jsonl.gz", "trajectory_sha256"), ("metrics.jsonl.gz", "metrics_sha256")]:
        if sha((args.run / name).read_bytes()) != report[key]:
            raise ValueError("Output hash changed")
    selected = json.loads(paths["selection"].read_text())
    event_by_id = {r["sample_id"]: r["event"] for r in selected}
    models = json.loads(paths["models"].read_text())
    assert_fit_ids(models, selected)
    projected = json.loads((args.run / "projected_models.json").read_text())
    details = json.loads((args.run / "projection_details.json").read_text())
    for key, params in models.items():
        expected, detail = project_model(params["count"])
        detail["fit_ids"] = params.get("fit_ids", list(event_by_id))
        if expected != projected[key] or detail != details[key]:
            raise ValueError("Saved source projection differs")
        weights = detail["weights"]
        for mode in ["native", "blurred"]:
            m = detail["modes"][mode]
            independent = independent_projection(m["original"], weights)
            if any(abs(a - b) > 1e-12 for a, b in zip(independent, m["projected"])):
                raise ValueError("Independent projection enumeration differs")
        for original_bin, new_bin in zip(params["count"]["bins"], projected[key]["bins"]):
            if original_bin["upgrade"] != new_bin["upgrade"]:
                raise ValueError("Upgrade prices changed")
    features = json.loads(paths["features"].read_text())
    observations = json.loads(paths["observations"].read_text())
    by_obs, obs, by_rows = defaultdict(list), defaultdict(dict), defaultdict(list)
    for r in observations:
        if sha(Path(r["cache_file"]).read_bytes()) != r["cache_sha256"]:
            raise ValueError("Fixed prediction changed")
        by_obs[r["sample_id"]].append(r)
        obs[r["sample_id"]][r["cell_id"], r["mode"]] = r["declared_target"]
    for r in json.loads(paths["association"].read_text())["rows"]:
        by_rows[r["sample_id"]].append(r)
    evaluators = {s: private_evaluator(rows) for s, rows in by_rows.items()}
    counts, seen, scene_counts = Counter(), set(), Counter()
    rebuilt, local, previous_sid = [], [], None
    ranks_cache, orders_cache, tables_cache = {}, {}, {}
    with gzip.open(args.run / "trajectories.jsonl.gz", "rt") as traces, gzip.open(args.run / "metrics.jsonl.gz", "rt") as metrics:
        for line in traces:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in event_by_id or row["event"] != event_by_id[sid]:
                raise ValueError("Foreign scene/event")
            if previous_sid != sid:
                if previous_sid is not None:
                    rebuilt.extend({"sample_id": previous_sid, "event": event_by_id[previous_sid], **r} for r in aggregate_ties(local))
                    print(json.dumps({"stage": "SCENE_REPLAY_COMPLETE", "sample_id": previous_sid, "trajectories": counts["trajectories"]}), flush=True)
                previous_sid, local = sid, []
                ranks_cache = {(rule, seed): tie_ranks(features[sid], rule, seed) for rule in TIES
                    for seed in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1])}
                orders_cache = {k: raw_order(features[sid], v) for k, v in ranks_cache.items()}
                tables_cache = {}
            key = tuple(row[k] for k in CONFIG_FIELDS)
            if key in seen:
                raise ValueError("Duplicate configuration")
            seen.add(key)
            policy, rule = row["policy"], row["tie_rule"]
            if policy not in POLICIES or rule not in TIES or row["quality_seed"] not in QUALITY_SEEDS:
                raise ValueError("Unplanned policy/rule/quality seed")
            if row["order_seed"] not in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1]):
                raise ValueError("Unplanned order seed")
            scope, scenario, cost, budget = (row[k] for k in ["fit_scope", "scenario", "upgrade_cost", "budget"])
            if scope == "development_fit":
                if scenario not in SCENARIOS or cost not in [1, 2, 4] or budget not in [8, 16, 32]:
                    raise ValueError("Unplanned full-fit setting")
                model_key = "development_fit"
            elif scope == "event_holdout" and (scenario, cost, budget) == ("mixed_survey", 2, 32):
                model_key = "held_" + row["event"]
            else:
                raise ValueError("Unplanned held-event setting")
            counts.update(audit_actions(row, obs[sid]))
            prior = {}
            survey_cells = []
            for action in row["trace"]:
                cid = action["cell_id"]
                if action["kind"] == "upgrade":
                    if policy == "rank_survey" or prior[cid] is not False:
                        raise ValueError("Survey-only or B-positive upgrade forbidden")
                else:
                    survey_cells.append(cid)
                prior[cid] = action["declared_target"]
            rk = rule, row["order_seed"]
            expected_prefix = [CELLS[i] for i in orders_cache[rk]][:len(survey_cells)]
            if survey_cells != expected_prefix:
                raise ValueError("Raw survey ranking not preserved")
            counts["raw_survey_prefixes_verified"] += 1
            tk = model_key, scenario, cost, policy
            if tk not in tables_cache:
                tables_cache[tk] = policy_table(features[sid], models[model_key]["count"], projected[model_key], scenario, cost, policy)
            env = PaidBinaryEnvironment(by_obs[sid], sid, scenario, row["quality_seed"], cost)
            actual = run_episode(env, tables_cache[tk], budget, cost, ranks_cache[rk], orders_cache[rk])
            if actual != row["trace"]:
                raise ValueError("Decision replay differs")
            counts["complete_decision_replays"] += 1
            if policy == "rank_survey":
                original = original_episode(PaidBinaryEnvironment(by_obs[sid], sid, scenario, row["quality_seed"], cost),
                    utility_table(features[sid], {}, {}, scenario, cost, "prior_survey"), budget, cost, ranks_cache[rk])
                if without_utility(original) != without_utility(actual):
                    raise ValueError("Rank-only control differs from original raw survey")
                counts["raw_survey_exact_action_equivalences"] += 1
            counts["trajectories"] += 1
            scene_counts[sid] += 1
            for expected in score_trace(row["trace"], *evaluators[sid]):
                saved = json.loads(next(metrics))
                if tuple(saved[k] for k in CONFIG_FIELDS) != key or any(saved[k] != v for k, v in expected.items()):
                    raise ValueError("Score replay differs")
                counts["threshold_scores"] += 1
                local.append(saved)
        if metrics.readline():
            raise ValueError("Extra saved metrics")
    rebuilt.extend({"sample_id": previous_sid, "event": event_by_id[previous_sid], **r} for r in aggregate_ties(local))
    if counts["trajectories"] != 362880 or counts["threshold_scores"] != 1088640 or counts["raw_survey_exact_action_equivalences"] != 120960 or counts["positive_upgrades"] != 0:
        raise ValueError("Incomplete or forbidden trajectories")
    if set(scene_counts) != set(event_by_id) or set(scene_counts.values()) != {5040}:
        raise ValueError("Incomplete per-scene grid")
    if rebuilt != json.loads((args.run / "scene_means.json").read_text()):
        raise ValueError("Scene aggregation differs")
    spatial = {r["sample_id"]: r for r in json.loads(paths["spatial"].read_text())}
    if summarize_ties(rebuilt, spatial) != json.loads((args.run / "summary.json").read_text()):
        raise ValueError("Event/spatial aggregation differs")
    references = [r for r in json.loads(paths["references"].read_text()) if r["policy"] in protocol["references"]]
    old = {tuple(r[k] for k in KEYS + ["sample_id"]): r for r in references if r["policy"] == "prior_survey"}
    for row in rebuilt:
        if row["policy"] != "rank_survey":
            continue
        ref = old[tuple(row[k] for k in KEYS + ["sample_id"])]
        if any(row[f] != ref[f] for f in SUMMARY_FIELDS if f != "planning_s"):
            raise ValueError("Rank-only score differs from frozen raw reference")
        counts["frozen_raw_scene_mean_equivalences"] += 1
    effects = paired_effects(rebuilt + references)
    atomic_json(args.run / "paired_effects.json", effects)
    audit = {"stage": "DEVELOPMENT72_RANK_CALIBRATION_FULL_REPLAY_PASSED", "counts": dict(counts),
        "prediction_masks_verified": len(observations), "independently_enumerated_source_projections": 8,
        "held_event_fit_ids_checked": True, "all_decisions_scores_and_aggregations_replayed": True,
        "all_raw_survey_prefixes_preserved": True, "negative_upgrade_model_unchanged": True,
        "primary_effects": [r for r in effects["event_macro_effects"] if is_primary(r)],
        "old_positive_revisit_policy_still_stopped": True, "new_method_selected": False,
        "independent_efficacy_claim": False, "validation_or_confirmation_scored": False,
        "new_inference_calls": 0, "gpu_rented": False}
    atomic_json(args.run / "audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "primary_effects"}), flush=True)


if __name__ == "__main__":
    main()
