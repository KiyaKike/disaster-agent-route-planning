"""Frozen evaluator-only utility failure diagnosis and cache budget ceilings.

No policy is fitted or proposed. Private future outcomes enter only component
diagnostics, one-step counterfactuals and perfect-information oracle bounds.
"""
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time

import scipy

from build_xbd_mini import atomic_json, sha
from cache_budget_oracle import evaluate, solve, survey_bits, union_bits
from cross_cell_count_smoke import CELLS, quality
from plan_search_v2 import freeze
from run_development_budget_controls import (QUALITY_SEEDS, ORDER_SEEDS, SUMMARY_FIELDS,
    utility_table, PaidBinaryEnvironment, private_evaluator, score_trace)
from run_utility_tie_controls import (TIES, tie_ranks, aggregate_ties, summarize_ties, verify_receipt)

BUDGETS = [8, 16, 32]
COST = 2
SCENARIO = "mixed_survey"
THRESHOLDS = [.05, .1, .25]
POLICIES = ["prior_survey", "prior_gate", "count_survey", "count_negative", "count_all", "presence"]
SOURCE_NAMES = ["run_utility_failure_diagnostic.py", "cache_budget_oracle.py", "run_unified_revisit_controls.py",
    "run_utility_tie_controls.py", "run_budget_planning.py", "run_development_budget_controls.py",
    "paired_count_baseline.py", "cross_cell_count_smoke.py", "build_xbd_mini.py", "plan_search_v2.py"]


def mean(values):
    return sum(values) / len(values) if values else None


def aggregate(rows, keys, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, key)), **{f: mean([r[f] for r in values if r[f] is not None]) for f in fields},
             "averaged_units": len(values)} for key, values in sorted(groups.items())]


def fold_models(models, event):
    return [("development_fit", models["development_fit"]), ("event_holdout", models["held_" + event])]


def component_calibration(models, features, pairs, selected):
    """Known-cell local count means; global dedup is intentionally separate."""
    records = []
    for event in sorted({r["event"] for r in selected}):
        held_ids = {r["sample_id"] for r in selected if r["event"] == event}
        for scope, parameters in fold_models(models, event):
            model = parameters["count"]
            source_ids = set(features) if scope == "development_fit" else set(parameters["fit_ids"])
            if scope == "event_holdout" and source_ids & held_ids:
                raise ValueError("Held event leaked into component fit")
            source = defaultdict(list)
            held = defaultdict(list)
            unknown = defaultdict(int)
            for (sid, cid), pair in pairs.items():
                b, n = pair["blurred"], pair["native"]
                bin_number = bisect_right(model["public_quantile_edges"], features[sid][cid]["change"])
                if n["target"] is None:
                    if sid in held_ids:
                        unknown[bin_number] += 1
                    continue
                nb, bb = n["bits"], b["bits"]
                row = {"native": nb.bit_count(), "blurred": bb.bit_count(),
                    "upgrade": (nb & ~bb).bit_count(), "positive": b["predicted"]}
                if sid in source_ids:
                    source[bin_number].append(row)
                if sid in held_ids:
                    held[bin_number].append(row)
            for number in range(4):
                fitted = model["bins"][number]
                if len(source[number]) != fitted["known_cells"]:
                    raise ValueError("Frozen source-bin support differs")
                source_native = mean([r["native"] for r in source[number]])
                source_blurred = mean([r["blurred"] for r in source[number]])
                for mode in ["native", "blurred"]:
                    total = sum(r[mode] for r in source[number])
                    expected = (total + model["shrinkage_pseudo_cells"] * model["global_survey_mean_count"][mode]) / (len(source[number]) + model["shrinkage_pseudo_cells"])
                    if abs(expected - fitted["survey_mean_count"][mode]) > 1e-12:
                        raise ValueError("Frozen component mean no longer reproducible")
                hn = mean([r["native"] for r in held[number]])
                hb = mean([r["blurred"] for r in held[number]])
                upgrades = []
                for positive in [False, True]:
                    src = [r for r in source[number] if r["positive"] == positive]
                    target = [r for r in held[number] if r["positive"] == positive]
                    upgrades.append({"blurred_positive": positive, "source_cells": len(src), "held_cells": len(target),
                        "source_observed_increment": mean([r["upgrade"] for r in src]),
                        "frozen_predicted_increment": fitted["upgrade"][str(int(positive))]["mean_new_within_cell"],
                        "held_observed_increment": mean([r["upgrade"] for r in target])})
                records.append({"event": event, "fit_scope": scope, "raw_change_bin": number,
                    "source_known_cells": len(source[number]), "held_known_cells": len(held[number]),
                    "held_unknown_pairs_excluded": unknown[number],
                    "source_observed_native": source_native, "source_observed_blurred": source_blurred,
                    "predicted_native": fitted["survey_mean_count"]["native"],
                    "predicted_blurred": fitted["survey_mean_count"]["blurred"],
                    "held_observed_native": hn, "held_observed_blurred": hb,
                    "predicted_mixed": (fitted["survey_mean_count"]["native"] + fitted["survey_mean_count"]["blurred"]) / 2,
                    "held_observed_mixed": (hn + hb) / 2 if hn is not None else None,
                    "upgrades": upgrades})
    return records


def levels_from_trace(trace):
    levels = [0] * 64
    indices = {c: i for i, c in enumerate(CELLS)}
    for action in trace:
        i = indices[action["cell_id"]]
        if action["kind"] == "survey" and levels[i] == 0:
            levels[i] = 1
        elif action["kind"] == "upgrade" and levels[i] == 1:
            levels[i] = 2
        else:
            raise ValueError("Invalid factual or oracle action sequence")
    return levels


def oracle_cells(sid, seed, evaluator):
    return [{"cell_id": cid, "survey_mode": quality(SCENARIO, seed, sid, cid),
             "native": evaluator[cid, "native"]["matches"][1],
             "blurred": evaluator[cid, "blurred"]["matches"][1]} for cid in CELLS]


def trace_from_levels(levels, observations, sid, seed):
    env = PaidBinaryEnvironment(observations, sid, SCENARIO, seed, COST)
    trace = []
    spent = 0
    for kind in ["survey", "upgrade"]:
        for i, level in enumerate(levels):
            if (kind == "survey" and level >= 1) or (kind == "upgrade" and level == 2):
                result = env.observe(CELLS[i], kind)
                spent += result["cost"]
                trace.append({"cell_id": CELLS[i], "kind": kind, "mode": result["mode"],
                    "declared_target": result["declared_target"], "cost": result["cost"],
                    "cumulative_cost": spent, "utility": 0.})
    return trace


def update_incumbent(store, key, levels, cells, budget, allow_upgrade):
    value = evaluate(cells, levels, budget, COST, allow_upgrade)
    if key not in store or (value["found"], -value["spent"]) > (store[key]["found"], -store[key]["spent"]):
        store[key] = {**value, "levels": list(levels)}


def selected_episode(row, table, ranks, evaluator, declarations):
    """Private diagnostic at factual states; no alternate policy is executed."""
    is_count = row["policy"].startswith("count_")
    fields = {"actions": 0, "upgrades": 0, "unknown_actions": 0,
        "actual_global": 0, "actual_local": 0, "cross_cell_duplicates": 0,
        "known_actions": 0, "actual_known_local": 0, "actual_known_global": 0,
        "expected_known_local": 0. if is_count else None,
        "survey_actual_global": 0, "upgrade_actual_global": 0,
        "positive_upgrade_actual_global": 0, "negative_upgrade_actual_global": 0,
        "upgrade_known_cases": 0, "upgrade_known_local": 0, "upgrade_known_global": 0,
        "upgrade_expected_known_local": 0. if is_count else None,
        "opportunity_cases": 0, "opportunity_upgrade_global": 0, "opportunity_alternative_global": 0,
        "opportunity_actual_difference": 0}
    state = [0] * 64
    found = 0
    own = defaultdict(int)
    indices = {c: i for i, c in enumerate(CELLS)}
    visits = []
    for step, action in enumerate(row["trace"]):
        cid = action["cell_id"]
        i = indices[cid]
        status = state[i]
        reward = table[i][status]
        if reward is None or reward != action["utility"]:
            raise ValueError("Factual action utility differs from frozen table")
        if action["declared_target"] is not declarations[cid, action["mode"]]:
            raise ValueError("Factual paid declaration changed")
        observed = evaluator[cid, action["mode"]]
        bits = observed["matches"][1]
        local = (bits & ~own[cid]).bit_count()
        global_new = (bits & ~found).bit_count()
        known = observed["truth"] is not None
        expected = reward * action["cost"] if is_count else None
        fields["actions"] += 1
        fields["actual_local"] += local
        fields["actual_global"] += global_new
        fields["cross_cell_duplicates"] += local - global_new
        fields["known_actions"] += known
        fields["unknown_actions"] += not known
        if known:
            fields["actual_known_local"] += local
            fields["actual_known_global"] += global_new
            if is_count:
                fields["expected_known_local"] += expected
        if action["kind"] == "upgrade":
            if status not in [1, 2] or action["mode"] != "native" or action["cost"] != COST:
                raise ValueError("Factual upgrade eligibility/cost differs")
            fields["upgrades"] += 1
            fields["upgrade_actual_global"] += global_new
            fields["positive_upgrade_actual_global" if status == 2 else "negative_upgrade_actual_global"] += global_new
            if known:
                fields["upgrade_known_cases"] += 1
                fields["upgrade_known_local"] += local
                fields["upgrade_known_global"] += global_new
                if is_count:
                    fields["upgrade_expected_known_local"] += expected
            unseen = [j for j, s in enumerate(state) if s == 0 and table[j][0] is not None]
            unseen.sort(key=lambda j: (-table[j][0], ranks[j]))
            alternatives = unseen[:COST]
            alternative_count = None
            if len(alternatives) == COST:
                alternative_bits = union_bits(evaluator[CELLS[j], quality(SCENARIO, row["quality_seed"], row["sample_id"], CELLS[j])]["matches"][1] for j in alternatives)
                alternative_count = (alternative_bits & ~found).bit_count()
                fields["opportunity_cases"] += 1
                fields["opportunity_upgrade_global"] += global_new
                fields["opportunity_alternative_global"] += alternative_count
                fields["opportunity_actual_difference"] += global_new - alternative_count
            visits.append({"step": step, "cell_id": cid, "blurred_positive": status == 2,
                "truth_known_for_calibration": known, "expected_same_cell_new_count": expected,
                "actual_same_cell_new": local, "actual_global_new": global_new,
                "same_cell_minus_global_duplicate": local - global_new,
                "publicly_ranked_equal_cost_alternative_cells": [CELLS[j] for j in alternatives],
                "alternative_actual_global_new": alternative_count,
                "one_step_actual_difference": global_new - alternative_count if alternative_count is not None else None})
        else:
            if status != 0 or action["cost"] != 1:
                raise ValueError("Factual survey eligibility/cost differs")
            fields["survey_actual_global"] += global_new
        found |= bits
        own[cid] |= bits
        state[i] = 3 if action["mode"] == "native" else 2 if action["declared_target"] else 1
    if is_count:
        fields["local_estimation_excess"] = fields["expected_known_local"] - fields["actual_known_local"]
        fields["known_duplicate_excess"] = fields["actual_known_local"] - fields["actual_known_global"]
        fields["total_estimate_minus_global"] = fields["expected_known_local"] - fields["actual_known_global"]
        if abs(fields["local_estimation_excess"] + fields["known_duplicate_excess"] - fields["total_estimate_minus_global"]) > 1e-9:
            raise ValueError("Selected-action error identity failed")
    else:
        fields.update(local_estimation_excess=None, known_duplicate_excess=None, total_estimate_minus_global=None)
    return fields, visits


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["unified", "previous", "perception", "design", "review", "spatial", "out"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing to overwrite completed diagnostic")
    paths = {"unified_receipt": args.unified / "completion_receipt.json", "unified_protocol": args.unified / "protocol.json",
        "unified_traces": args.unified / "trajectories.jsonl.gz", "unified_means": args.unified / "scene_means.json",
        "review_receipt": args.review / "completion_receipt.json", "review": args.review / "review.json",
        "models": args.previous / "models.json", "features": args.previous / "public_features.json",
        "observations": args.perception / "observations.json", "association": args.perception / "association_report.json",
        "selection": args.design / "next_development_batch.json", "contract": args.design / "experiment_contract_v2_2.json",
        "spatial": args.spatial}
    sources = {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k, p in paths.items()}
    freeze(args.out / "protocol.json", {"stage": "DEVELOPMENT72_UTILITY_FAILURE_AND_PERFECT_INFORMATION_DIAGNOSTIC",
        "sources": sources, "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES},
        "scipy_version": scipy.__version__, "primary": {"scenario": SCENARIO, "upgrade_cost": COST, "budget": 32, "threshold": .1},
        "budgets": BUDGETS, "quality_seeds": QUALITY_SEEDS, "random_order_seeds": ORDER_SEEDS, "tie_rules": TIES,
        "component_calibration": "Frozen full/other-two-event bins. Known truth pairs only as in original fit. Source counts and shrunk means reconstructed; held native/blurred/same-cell upgrade means reported per bin/B result. Unknown excluded and counted. No refit or monotonic correction.",
        "ranking": "Only unchanged prior_survey and count_survey; all3 ties/5 quality/10 random orders at B8/16/32 in both development fit scopes. Same cost2/mixed scenario. All3 measurement thresholds. Reproduce all available old scene means exactly, excluding runtime.",
        "selected_actions": "Replay all6 policies at primary B32/mixed/cost2. Compare count-policy expected local increments with actual local then global unique increments; noncount utilities never treated as expected building counts. Known/unknown calibration separated.",
        "counterfactual": "At each factual upgrade, privately evaluate surveying the next2 unvisited cells selected by that policy's own public initial ranking and same fixed tie order. Compare equal cost from the factual prefix. No true-count selection of alternatives. One-step diagnostic only: alternatives can overlap across distinct hypothetical prefixes; summed effects are NOT a complete alternate-policy gain.",
        "oracle": {"cases": 2160, "kinds": ["survey_only", "survey_and_upgrade"],
            "objective_threshold": .1, "solver": "scipy.optimize.milp/HiGHS", "time_limit_s_per_call": 2., "mip_rel_gap": 0.,
            "formulation": "Binary survey s, blurred-only upgrade u<=s, unique-instance coverage z. Cost sum(s)+2sum(u)<=B. Survey mode fixed by original hidden quality map. Blurred and native target sets are unioned, never replace previous discoveries.",
            "bounds": "Replay feasible solver/greedy/previous-policy incumbents for LB. UB is floor(sign-reversed finite MILP dual bound +1e-6), capped by accessible union; union UB if no finite dual. Optimal only if integer bounds close. No time-limit result called exact without certificate.",
            "incumbents": "Best valid prior factual or newly recomputed unchanged survey traces may seed feasible lower bound, evaluator-only. Full oracle also admits survey-only oracle incumbent.",
            "sensitivity": "Re-score threshold10 oracle traces at5/25; these are feasible traces, NOT reoptimized ceilings or certified bounds at5/25.",
            "interpretation": "Perfect-information offline diagnostic sees private instances/future quality. Not an executable method, not fair efficacy comparison, and not evidence that its gap is learnable from allowed inputs."},
        "aggregation": "Seeds/orders within scene, scenes within event, events equally; retain zero targets and zero-detection Mexico. No seed-level statistical inference.",
        "preflight": "Implementation schema probe inspected first already-used development cached row; only keys and that row printed. No new aggregate diagnostic before freeze; development data previously inspected in earlier experiments.",
        "no_new_policy_or_fit": True, "validation_or_confirmation_access": False, "new_inference_calls": 0, "paid_gpu": False})
    verify_receipt(args.unified)
    verify_receipt(args.review)
    old_protocol = json.loads(paths["unified_protocol"].read_text())
    for record in old_protocol["sources"].values():
        if sha(Path(record["path"]).read_bytes()) != record["sha256"]:
            raise ValueError("Previous diagnostic input changed")
    selected = json.loads(paths["selection"].read_text())
    event_by_id = {r["sample_id"]: r["event"] for r in selected}
    if len(event_by_id) != 72 or any(r["split"] != "development" for r in selected):
        raise ValueError("Exact72 development identities required")
    features = json.loads(paths["features"].read_text())
    models = json.loads(paths["models"].read_text())
    observations = json.loads(paths["observations"].read_text())
    association = json.loads(paths["association"].read_text())
    if len(observations) != 9216 or len(association["rows"]) != 9216 or association["stage"] != "DEVELOPMENT72_SENSOR_QA" or set(features) != set(event_by_id):
        raise ValueError("Incomplete frozen development inputs")
    by_obs = defaultdict(list)
    by_rows = defaultdict(list)
    for r in observations:
        if r["sample_id"] not in event_by_id:
            raise ValueError("Foreign observation")
        by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:
        if r["sample_id"] not in event_by_id:
            raise ValueError("Foreign evaluator row")
        by_rows[r["sample_id"]].append(r)
    evaluators = {sid: private_evaluator(rows) for sid, rows in by_rows.items()}
    pairs = defaultdict(dict)
    for sid, rows in by_rows.items():
        for r in rows:
            pairs[sid, r["cell_id"]][r["mode"]] = {**r, "bits": evaluators[sid][0][r["cell_id"], r["mode"]]["matches"][1]}
    if len(pairs) != 4608 or any(set(v) != {"native", "blurred"} for v in pairs.values()):
        raise ValueError("Incomplete paired cells")
    started = time.perf_counter()
    atomic_json(args.out / "component_calibration.json", {"stage": "EVALUATOR_ONLY_FROZEN_COMPONENT_CALIBRATION",
        "rows": component_calibration(models, features, pairs, selected), "unit": "Known-cell local matched counts; not globally distinct buildings.",
        "new_fit": False, "threshold": .1})
    spatial = {r["sample_id"]: r for r in json.loads(args.spatial.read_text())}
    ranking_means = []
    survey_incumbents = {}
    full_incumbents = {}
    cells_by_seed = {(sid, seed): oracle_cells(sid, seed, evaluators[sid][0]) for sid in event_by_id for seed in QUALITY_SEEDS}
    rank_episodes = rank_scores = 0
    with gzip.open(args.out / "ranking_metrics.jsonl.gz", "wt", compresslevel=3) as saved:
        for sample in selected:
            sid, event = sample["sample_id"], sample["event"]
            local = []
            for scope, params in fold_models(models, event):
                for policy in ["prior_survey", "count_survey"]:
                    table = utility_table(features[sid], params["count"], params["presence"], SCENARIO, COST, policy)
                    for rule in TIES:
                        for order_seed in (ORDER_SEEDS if rule == "fixed_random" else ORDER_SEEDS[:1]):
                            ranks = tie_ranks(features[sid], rule, order_seed)
                            order = sorted(range(64), key=lambda i: (-table[i][0], ranks[i]))
                            for seed in QUALITY_SEEDS:
                                env = PaidBinaryEnvironment(by_obs[sid], sid, SCENARIO, seed, COST)
                                trace = []
                                for i in order[:32]:
                                    result = env.observe(CELLS[i], "survey")
                                    trace.append({"cell_id": CELLS[i], "kind": "survey", "mode": result["mode"],
                                        "declared_target": result["declared_target"], "cost": 1,
                                        "cumulative_cost": len(trace) + 1, "utility": table[i][0]})
                                rank_episodes += 1
                                for budget in BUDGETS:
                                    prefix = trace[:budget]
                                    key = sid, seed, budget
                                    levels = levels_from_trace(prefix)
                                    update_incumbent(survey_incumbents, key, levels, cells_by_seed[sid, seed], budget, False)
                                    update_incumbent(full_incumbents, key, levels, cells_by_seed[sid, seed], budget, True)
                                    config = {"sample_id": sid, "event": event, "fit_scope": scope, "policy": policy, "tie_rule": rule,
                                        "scenario": SCENARIO, "upgrade_cost": COST, "budget": budget, "quality_seed": seed, "order_seed": order_seed}
                                    for scored in score_trace(prefix, *evaluators[sid]):
                                        record = {**config, **scored, "planning_s": 0.}
                                        local.append(record)
                                        saved.write(json.dumps(record, separators=(",", ":")) + "\n")
                                        rank_scores += 1
            ranking_means.extend({"sample_id": sid, "event": event, **r} for r in aggregate_ties(local))
    atomic_json(args.out / "ranking_scene_means.json", ranking_means)
    atomic_json(args.out / "ranking_summary.json", summarize_ties(ranking_means, spatial))
    old_means = json.loads(paths["unified_means"].read_text())
    old_index = {tuple(r[k] for k in ["sample_id", "fit_scope", "policy", "tie_rule", "scenario", "upgrade_cost", "budget", "threshold"]): r for r in old_means}
    reproduced = 0
    for r in ranking_means:
        key = tuple(r[k] for k in ["sample_id", "fit_scope", "policy", "tie_rule", "scenario", "upgrade_cost", "budget", "threshold"])
        if key in old_index:
            for field in SUMMARY_FIELDS:
                if field != "planning_s" and r[field] != old_index[key][field]:
                    raise ValueError("Unchanged survey ranking failed prior-result replay")
            reproduced += 1
    if (rank_episodes, rank_scores, reproduced) != (17280, 155520, 5184):
        raise ValueError("Incomplete ranking diagnostics")
    print(json.dumps({"stage": "RANKING_AND_COMPONENT_DIAGNOSTIC_COMPLETE", "ranking_scores": rank_scores, "old_scene_means_reproduced": reproduced}), flush=True)
    episode_rows = []
    upgrade_rows = source_rows = 0
    cached_tables = {}
    with gzip.open(paths["unified_traces"], "rt") as source, gzip.open(args.out / "selected_episode_diagnostics.jsonl.gz", "wt", compresslevel=3) as episodes, gzip.open(args.out / "upgrade_counterfactuals.jsonl.gz", "wt", compresslevel=3) as visits:
        for line in source:
            row = json.loads(line)
            source_rows += 1
            if row["scenario"] != SCENARIO or row["upgrade_cost"] != COST or row["budget"] not in BUDGETS:
                continue
            sid, seed, budget = row["sample_id"], row["quality_seed"], row["budget"]
            levels = levels_from_trace(row["trace"])
            update_incumbent(full_incumbents, (sid, seed, budget), levels, cells_by_seed[sid, seed], budget, True)
            if 2 not in levels:
                update_incumbent(survey_incumbents, (sid, seed, budget), levels, cells_by_seed[sid, seed], budget, False)
            if budget != 32:
                continue
            table_key = sid, row["fit_scope"], row["policy"]
            if table_key not in cached_tables:
                params = models["development_fit" if row["fit_scope"] == "development_fit" else "held_" + row["event"]]
                cached_tables[table_key] = utility_table(features[sid], params["count"], params["presence"], SCENARIO, COST, row["policy"])
            ranks = tie_ranks(features[sid], row["tie_rule"], row["order_seed"])
            declarations = {(r["cell_id"], r["mode"]): r["declared_target"] for r in by_obs[sid]}
            values, actions = selected_episode(row, cached_tables[table_key], ranks, evaluators[sid][0], declarations)
            reference = score_trace(row["trace"], *evaluators[sid])[1]
            if (values["actual_global"], values["cross_cell_duplicates"], values["upgrades"]) != (reference["distinct_found"], reference["cross_cell_duplicate_match_events"], reference["upgrades"]):
                raise ValueError("Selected-action diagnosis changed scores")
            config = {k: v for k, v in row.items() if k not in ["trace", "execution_origin"]}
            record = {**config, **values}
            episode_rows.append(record)
            episodes.write(json.dumps(record, separators=(",", ":")) + "\n")
            for action in actions:
                visits.write(json.dumps({**config, **action}, separators=(",", ":")) + "\n")
                upgrade_rows += 1
    if source_rows != 725760 or len(episode_rows) != 51840:
        raise ValueError("Incomplete source-action diagnostics")
    action_fields = list(values)
    action_scene = aggregate(episode_rows, ["sample_id", "event", "fit_scope", "policy", "tie_rule"], action_fields)
    action_events = aggregate(action_scene, ["event", "fit_scope", "policy", "tie_rule"], action_fields)
    action_macro = aggregate(action_events, ["fit_scope", "policy", "tie_rule"], action_fields)
    atomic_json(args.out / "selected_action_summary.json", {"scene_means": action_scene, "event_means": action_events,
        "event_macro_means": action_macro, "counterfactual_scope": "One-step comparisons at factual prefixes, not full alternate-policy gains; private future outcomes only in evaluator."})
    print(json.dumps({"stage": "SELECTED_ACTION_DIAGNOSTIC_COMPLETE", "episodes": len(episode_rows), "upgrades": upgrade_rows}), flush=True)
    oracle_rows = []
    with gzip.open(args.out / "oracle_solutions.jsonl.gz", "wt", compresslevel=3) as saved:
        for sample in selected:
            sid, event = sample["sample_id"], sample["event"]
            evaluator, targets = evaluators[sid]
            for seed in QUALITY_SEEDS:
                cells = cells_by_seed[sid, seed]
                for budget in BUDGETS:
                    key = sid, seed, budget
                    survey_result = solve(cells, budget, COST, False, [survey_incumbents[key]["levels"]])
                    full_result = solve(cells, budget, COST, True, [full_incumbents[key]["levels"], survey_result["levels"]])
                    for kind, result in [("survey_only", survey_result), ("survey_and_upgrade", full_result)]:
                        trace = trace_from_levels(result["levels"], by_obs[sid], sid, seed)
                        scores = score_trace(trace, evaluator, targets)
                        if scores[1]["distinct_found"] != result["lower"] or scores[1]["spent"] != result["spent"]:
                            raise ValueError("Oracle feasible trace does not replay")
                        if result["upper"] < result["lower"] or (result["certified_optimal"] and result["upper"] != result["lower"]):
                            raise ValueError("Invalid oracle certificate")
                        record = {"sample_id": sid, "event": event, "quality_seed": seed, "scenario": SCENARIO,
                            "upgrade_cost": COST, "budget": budget, "oracle_kind": kind, "visible_targets": targets,
                            **result, "trace": trace, "threshold_replays": scores,
                            "analysis_group": spatial[sid]["analysis_group"], "training_envelope_overlap": spatial[sid]["training_envelope_overlap"]}
                        oracle_rows.append(record)
                        saved.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps({"stage": "SCENE_ORACLE_COMPLETE", "sample_id": sid, "oracle_cases": len(oracle_rows)}), flush=True)
    if len(oracle_rows) != 2160:
        raise ValueError("Incomplete oracle cases")
    fields = ["lower", "upper", "spent", "accessible_union", "visible_targets", "wall_s"]
    oracle_scenes = aggregate(oracle_rows, ["sample_id", "event", "budget", "oracle_kind"], fields)
    oracle_events = aggregate(oracle_scenes, ["event", "budget", "oracle_kind"], fields)
    oracle_macro = aggregate(oracle_events, ["budget", "oracle_kind"], fields)
    atomic_json(args.out / "oracle_summary.json", {"scene_means": oracle_scenes, "event_means": oracle_events, "event_macro_means": oracle_macro,
        "cases": len(oracle_rows), "certified_optimal_cases": sum(r["certified_optimal"] for r in oracle_rows),
        "solver_called_cases": sum(r["solver_called"] for r in oracle_rows),
        "unclosed_cases": [{k: r[k] for k in ["sample_id", "quality_seed", "budget", "oracle_kind", "lower", "upper", "solver_status"]} for r in oracle_rows if not r["certified_optimal"]],
        "scope": "Perfect-information cached oracle, not executable method. Only10% objective optimized;5/25% are fixed-trace sensitivity replays."})
    preserved = {k: sha(p.read_bytes()) == sources[k]["sha256"] for k, p in paths.items()}
    if not all(preserved.values()):
        raise ValueError("Input changed during diagnosis")
    report = {"stage": "DEVELOPMENT72_UTILITY_FAILURE_DIAGNOSTIC_RUN_COMPLETE", "scenes": 72,
        "component_bin_rows": 24, "ranking_episodes": rank_episodes, "ranking_threshold_scores": rank_scores,
        "previous_ranking_scene_means_reproduced": reproduced, "source_trajectories_scanned": source_rows,
        "selected_episodes_diagnosed": len(episode_rows), "upgrade_counterfactuals": upgrade_rows,
        "oracle_cases": len(oracle_rows), "certified_optimal_oracles": sum(r["certified_optimal"] for r in oracle_rows),
        "oracle_threshold_replays": len(oracle_rows) * 3, "total_wall_s": time.perf_counter() - started,
        "sources_preserved": preserved, "new_policy_selected": False, "new_model_fits": 0,
        "new_inference_calls": 0, "validation_or_confirmation_scored": False, "paid_gpu": False}
    atomic_json(args.out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
