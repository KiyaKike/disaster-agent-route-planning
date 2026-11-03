"""Replay diagnostics and oracle certificates with independent UID sets."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

from audit_development_budget_controls import audit_actions
from build_xbd_mini import atomic_json, sha
from cross_cell_count_smoke import CELLS, quality
from run_development_budget_controls import (QUALITY_SEEDS, ORDER_SEEDS,
    utility_table, private_evaluator, score_trace)
from run_utility_failure_diagnostic import (BUDGETS, COST, SCENARIO, THRESHOLDS,
    aggregate, component_calibration, levels_from_trace, selected_episode)
from run_utility_tie_controls import TIES, aggregate_ties, summarize_ties, tie_ranks, verify_receipt

CONFIG = ["sample_id", "fit_scope", "policy", "tie_rule", "scenario", "upgrade_cost", "budget", "quality_seed", "order_seed"]


def uid_sets(rows):
    return {(r["cell_id"], r["mode"], threshold): {
        v["uid"] for v in r["overlap_records"] if v["visible_pixels"] >= 4
        and v["severe_overlap_pixels"] >= 4 and v["visible_footprint_coverage"] >= threshold}
        for r in rows for threshold in THRESHOLDS}


def found_uids(trace, sets, threshold=.1):
    result = set()
    for action in trace:
        result.update(sets[action["cell_id"], action["mode"], threshold])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    paths = {k: Path(v["path"]) for k, v in protocol["sources"].items()}
    for name, digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Frozen diagnostic source changed")
    for name, path in paths.items():
        if sha(path.read_bytes()) != protocol["sources"][name]["sha256"]:
            raise ValueError("Frozen input changed")
    verify_receipt(paths["unified_receipt"].parent)
    verify_receipt(paths["review_receipt"].parent)
    selected = json.loads(paths["selection"].read_text())
    events = {r["sample_id"]: r["event"] for r in selected}
    features = json.loads(paths["features"].read_text())
    models = json.loads(paths["models"].read_text())
    by_rows = defaultdict(list)
    by_obs = defaultdict(list)
    declarations = defaultdict(dict)
    for r in json.loads(paths["association"].read_text())["rows"]:
        by_rows[r["sample_id"]].append(r)
    observations = json.loads(paths["observations"].read_text())
    for r in observations:
        if sha(Path(r["cache_file"]).read_bytes()) != r["cache_sha256"]:
            raise ValueError("Cached mask changed")
        by_obs[r["sample_id"]].append(r)
        declarations[r["sample_id"]][r["cell_id"], r["mode"]] = r["declared_target"]
    evaluator = {sid: private_evaluator(rows) for sid, rows in by_rows.items()}
    sets = {sid: uid_sets(rows) for sid, rows in by_rows.items()}
    pairs = defaultdict(dict)
    for sid, rows in by_rows.items():
        for r in rows:
            pairs[sid, r["cell_id"]][r["mode"]] = {**r, "bits": evaluator[sid][0][r["cell_id"], r["mode"]]["matches"][1]}
    if component_calibration(models, features, pairs, selected) != json.loads((args.run / "component_calibration.json").read_text())["rows"]:
        raise ValueError("Frozen-bin calibration replay differs")
    counts = Counter()
    ranking_seen = set()
    rank_lists = set()
    ranking_means = []
    local = []
    previous_sid = None
    previous_key = None
    expected_scores = None
    with gzip.open(args.run / "ranking_metrics.jsonl.gz", "rt") as source:
        for line in source:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in events or r["event"] != events[sid] or r["policy"] not in ["count_survey", "prior_survey"]:
                raise ValueError("Foreign ranking configuration")
            if r["fit_scope"] not in ["development_fit", "event_holdout"] or r["tie_rule"] not in TIES or r["quality_seed"] not in QUALITY_SEEDS or r["budget"] not in BUDGETS or r["scenario"] != SCENARIO or r["upgrade_cost"] != COST:
                raise ValueError("Unplanned ranking settings")
            if r["order_seed"] not in (ORDER_SEEDS if r["tie_rule"] == "fixed_random" else ORDER_SEEDS[:1]) or r["threshold"] not in THRESHOLDS:
                raise ValueError("Unplanned ranking seed/threshold")
            key = tuple(r[k] for k in CONFIG)
            sk = key + (r["threshold"],)
            if sk in ranking_seen:
                raise ValueError("Duplicate ranking score")
            ranking_seen.add(sk)
            rank_lists.add(tuple(r[k] for k in CONFIG if k != "budget"))
            if sid != previous_sid:
                if previous_sid is not None:
                    ranking_means.extend({"sample_id": previous_sid, "event": events[previous_sid], **v} for v in aggregate_ties(local))
                previous_sid = sid
                local = []
            if key != previous_key:
                params = models["development_fit" if r["fit_scope"] == "development_fit" else "held_" + events[sid]]
                table = utility_table(features[sid], params["count"], params["presence"], SCENARIO, COST, r["policy"])
                rank = tie_ranks(features[sid], r["tie_rule"], r["order_seed"])
                order = sorted(range(64), key=lambda i: (-table[i][0], rank[i]))[:r["budget"]]
                trace = [{"cell_id": CELLS[i], "kind": "survey", "mode": quality(SCENARIO, r["quality_seed"], sid, CELLS[i]),
                    "declared_target": declarations[sid][CELLS[i], quality(SCENARIO, r["quality_seed"], sid, CELLS[i])],
                    "cost": 1, "cumulative_cost": j + 1, "utility": table[i][0]} for j, i in enumerate(order)]
                expected_scores = {v["threshold"]: v for v in score_trace(trace, *evaluator[sid])}
                previous_key = key
            expected = expected_scores[r["threshold"]]
            if any(r[k] != v for k, v in expected.items()) or r["planning_s"] != 0.:
                raise ValueError("Ranking score replay differs")
            if len(found_uids(trace, sets[sid], r["threshold"])) != r["distinct_found"]:
                raise ValueError("Independent ranking UID-union replay differs")
            local.append(r)
            counts["ranking_threshold_scores"] += 1
    ranking_means.extend({"sample_id": previous_sid, "event": events[previous_sid], **v} for v in aggregate_ties(local))
    if ranking_means != json.loads((args.run / "ranking_scene_means.json").read_text()):
        raise ValueError("Ranking scene/seed aggregation differs")
    spatial = {r["sample_id"]: r for r in json.loads(paths["spatial"].read_text())}
    if summarize_ties(ranking_means, spatial) != json.loads((args.run / "ranking_summary.json").read_text()):
        raise ValueError("Ranking event/spatial aggregation differs")
    if len(rank_lists) != 17280 or len(ranking_seen) != 155520:
        raise ValueError("Incomplete ranking grid")
    max_source = defaultdict(int)
    episode_rows = []
    with gzip.open(paths["unified_traces"], "rt") as source, gzip.open(args.run / "selected_episode_diagnostics.jsonl.gz", "rt") as saved, gzip.open(args.run / "upgrade_counterfactuals.jsonl.gz", "rt") as visits:
        for line in source:
            row = json.loads(line)
            if (row["scenario"], row["upgrade_cost"]) != (SCENARIO, COST) or row["budget"] not in BUDGETS:
                continue
            sid = row["sample_id"]
            discovered = len(found_uids(row["trace"], sets[sid]))
            key = sid, row["quality_seed"], row["budget"]
            max_source[key + ("survey_and_upgrade",)] = max(max_source[key + ("survey_and_upgrade",)], discovered)
            if all(a["kind"] == "survey" for a in row["trace"]):
                max_source[key + ("survey_only",)] = max(max_source[key + ("survey_only",)], discovered)
            if row["budget"] != 32:
                continue
            replay_counts = audit_actions(row, declarations[sid])
            counts["factual_actions_replayed"] += replay_counts["actions"]
            params = models["development_fit" if row["fit_scope"] == "development_fit" else "held_" + events[sid]]
            table = utility_table(features[sid], params["count"], params["presence"], SCENARIO, COST, row["policy"])
            ranks = tie_ranks(features[sid], row["tie_rule"], row["order_seed"])
            values, actions = selected_episode(row, table, ranks, evaluator[sid][0], declarations[sid])
            config = {k: v for k, v in row.items() if k not in ["trace", "execution_origin"]}
            stored = json.loads(next(saved))
            if stored != {**config, **values}:
                raise ValueError("Selected-episode diagnostic replay differs")
            episode_rows.append(stored)
            fresh_found = set()
            own = defaultdict(set)
            action_by_step = {a["step"]: a for a in actions}
            for step, a in enumerate(row["trace"]):
                cid = a["cell_id"]
                current = sets[sid][cid, a["mode"], .1]
                if a["kind"] == "upgrade":
                    record = json.loads(next(visits))
                    expected = action_by_step[step]
                    if record != {**config, **expected}:
                        raise ValueError("Upgrade counterfactual replay differs")
                    alternate = set()
                    for alternate_cell in record["publicly_ranked_equal_cost_alternative_cells"]:
                        if alternate_cell in own:
                            raise ValueError("Counterfactual surveyed a visited cell")
                        alternate.update(sets[sid][alternate_cell, quality(SCENARIO, row["quality_seed"], sid, alternate_cell), .1])
                    if record["actual_same_cell_new"] != len(current - own[cid]) or record["actual_global_new"] != len(current - fresh_found):
                        raise ValueError("Independent local/global UID-set count differs")
                    if len(record["publicly_ranked_equal_cost_alternative_cells"]) == COST and record["alternative_actual_global_new"] != len(alternate - fresh_found):
                        raise ValueError("Independent equal-cost alternate UID-set count differs")
                    counts["upgrade_counterfactuals_replayed"] += 1
                fresh_found.update(current)
                own[cid].update(current)
            if stored["actual_global"] != len(fresh_found):
                raise ValueError("Independent factual final count differs")
            counts["selected_episodes_replayed"] += 1
        if saved.readline() or visits.readline():
            raise ValueError("Extra selected-action records")
    fields = list(values)
    scenes = aggregate(episode_rows, ["sample_id", "event", "fit_scope", "policy", "tie_rule"], fields)
    event_rows = aggregate(scenes, ["event", "fit_scope", "policy", "tie_rule"], fields)
    macros = aggregate(event_rows, ["fit_scope", "policy", "tie_rule"], fields)
    action_summary = json.loads((args.run / "selected_action_summary.json").read_text())
    if (scenes, event_rows, macros) != (action_summary["scene_means"], action_summary["event_means"], action_summary["event_macro_means"]):
        raise ValueError("Selected-action aggregation differs")
    oracle_seen = set()
    oracle_rows = []
    oracle_index = {}
    with gzip.open(args.run / "oracle_solutions.jsonl.gz", "rt") as source:
        for line in source:
            r = json.loads(line)
            sid, seed, budget, kind = r["sample_id"], r["quality_seed"], r["budget"], r["oracle_kind"]
            key = sid, seed, budget, kind
            if key in oracle_seen or sid not in events or r["event"] != events[sid] or seed not in QUALITY_SEEDS or budget not in BUDGETS or kind not in ["survey_only", "survey_and_upgrade"]:
                raise ValueError("Duplicate or unplanned oracle configuration")
            oracle_seen.add(key)
            counts["oracle_actions_replayed"] += audit_actions({**r, "policy": "offline_oracle"}, declarations[sid])["actions"]
            if r["levels"] != levels_from_trace(r["trace"]):
                raise ValueError("Oracle decision vector and trace differ")
            if kind == "survey_only" and any(a["kind"] == "upgrade" for a in r["trace"]):
                raise ValueError("Survey-only oracle upgraded")
            cache_union = set()
            for cid in CELLS:
                mode = quality(SCENARIO, seed, sid, cid)
                cache_union.update(sets[sid][cid, mode, .1])
                if kind == "survey_and_upgrade" and mode == "blurred":
                    cache_union.update(sets[sid][cid, "native", .1])
            scored = score_trace(r["trace"], *evaluator[sid])
            if scored != r["threshold_replays"]:
                raise ValueError("Oracle sensitivity score replay differs")
            for threshold in THRESHOLDS:
                expected = next(v for v in scored if v["threshold"] == threshold)
                if len(found_uids(r["trace"], sets[sid], threshold)) != expected["distinct_found"]:
                    raise ValueError("Independent oracle UID-union score differs")
                counts["oracle_threshold_replays"] += 1
            if r["lower"] != scored[1]["distinct_found"] or r["spent"] != scored[1]["spent"] or r["accessible_union"] != len(cache_union):
                raise ValueError("Oracle feasible or union bound differs")
            if not r["lower"] <= r["upper"] <= len(cache_union) or r["lower"] < max_source[key]:
                raise ValueError("Oracle worse than supplied feasible source or bound invalid")
            if r["certificate"] == "accessible_union_closed":
                if r["lower"] != r["upper"] or r["upper"] != len(cache_union):
                    raise ValueError("Invalid union-closed certificate")
            else:
                raw = r["raw_maximization_dual_bound"]
                expected_upper = len(cache_union) if raw is None else min(len(cache_union), max(r["lower"], math.floor(raw + 1e-6)))
                if r["upper"] != expected_upper:
                    raise ValueError("Integer dual bound certificate differs")
            if r["certified_optimal"] != (r["lower"] == r["upper"]):
                raise ValueError("Unclosed incumbent was mislabeled optimal")
            if r["solver_called"]:
                counts["solver_certificates_checked"] += 1
                if r["solver_status"] not in [0, 1]:
                    raise ValueError("Invalid oracle solver state")
            if r["certified_optimal"]:
                counts["certified_optimal_oracles"] += 1
            oracle_rows.append(r)
            oracle_index[key] = r
    for (sid, seed, budget, kind), r in oracle_index.items():
        if kind == "survey_and_upgrade" and r["lower"] < oracle_index[sid, seed, budget, "survey_only"]["lower"]:
            raise ValueError("Full-oracle feasible lower bound lost survey-only incumbent")
    fields = ["lower", "upper", "spent", "accessible_union", "visible_targets", "wall_s"]
    oscene = aggregate(oracle_rows, ["sample_id", "event", "budget", "oracle_kind"], fields)
    oevents = aggregate(oscene, ["event", "budget", "oracle_kind"], fields)
    omacro = aggregate(oevents, ["budget", "oracle_kind"], fields)
    summary = json.loads((args.run / "oracle_summary.json").read_text())
    if (oscene, oevents, omacro) != (summary["scene_means"], summary["event_means"], summary["event_macro_means"]):
        raise ValueError("Oracle seed/scene/event aggregation differs")
    if len(oracle_rows) != 2160 or counts["selected_episodes_replayed"] != 51840 or counts["upgrade_counterfactuals_replayed"] != report["upgrade_counterfactuals"]:
        raise ValueError("Incomplete diagnostic replay")
    if counts["certified_optimal_oracles"] != report["certified_optimal_oracles"]:
        raise ValueError("Reported oracle optimal count differs")
    result = {"stage": "UTILITY_FAILURE_AND_ORACLE_REPLAY_PASSED", "counts": dict(counts),
        "ranking_orders_checked": len(rank_lists), "oracle_cases_checked": len(oracle_rows),
        "independent_uid_set_replay": True, "all_oracle_budgets_quality_maps_and_prerequisites_checked": True,
        "all_saved_bound_certificates_checked": True, "all_aggregations_replayed": True,
        "fixed_prediction_masks_verified": len(observations), "previous_outputs_preserved": True,
        "new_policy_selected": False, "new_model_fits": 0, "new_inference_calls": 0,
        "validation_or_confirmation_scored": False, "paid_gpu": False, "source_sha256": sha(Path(__file__).read_bytes())}
    atomic_json(args.run / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
