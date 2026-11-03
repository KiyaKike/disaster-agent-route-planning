"""Exploratory identifiability check: can a short probe diagnose prior utility?

Signals use public prior and observed model outputs only. Evaluator-only
counterfactuals compare identical probe prefixes and equal total survey costs.
No classifier, switch threshold, refitting, extra inference, or test access.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_cached_search import PublicCell, SCENARIOS, digest, survey_mode
from run_changeos_pilot import freeze_plan
from run_coverage_controls import ORDER_SEEDS, exact_random_discoveries, random_order, sources


PREFIX_LENGTHS = (4, 8)
TOTAL_BUDGETS = (16, 32)
SIGNALS = ("positive_count", "response_surplus", "binary_alignment", "soft_alignment", "public_prior_spread")
PROTOCOL = {
    "version": "early_evidence_identifiability_v1", "prefix_lengths": list(PREFIX_LENGTHS),
    "total_survey_budgets": list(TOTAL_BUDGETS), "probe_order_seeds": list(ORDER_SEEDS), "quality_maps": SCENARIOS,
    "timing": "Exploratory protocol frozen after all earlier calibration/validation results were inspected. No untouched-test or novelty claim.",
    "probe": "First 4/8 distinct cells in the fixed random coverage order, costs included in total budget, no upgrades. Future quality hidden; probe features use observed quality only.",
    "continuations": "After identical paid probe, either static fitted-prior survey ordering with original nearest tie starting at last probe position, or uniform random survey over remaining cells. No upgrades in either continuation: isolate ranking utility from quality restoration.",
    "random_evaluator": "Exact expected new unique buildings over a uniform subset of remaining cells; remove prefix-discovered IDs first. Same frozen quality map, remaining budget and building-matching rule for both continuations.",
    "features": {
        "positive_count": "Count detector positives in the probe; density control, not itself a ranking-failure diagnosis.",
        "response_surplus": "sum(z-q)/sqrt(sum q*(1-q)), q=p*TPR(mode)+(1-p)*FPR(mode), all fixed calibration parameters. A mismatch need not imply incorrect spatial ranking.",
        "binary_alignment": "Mean sign agreement between fitted prior and binary response over probe pairs with same observed mode, unequal priors and unequal responses; null if no comparable pairs.",
        "soft_alignment": "Same pair agreement, using frozen uncalibrated model score instead of binary response; null if no comparable pairs. No comparison across native/blurred modes.",
        "public_prior_spread": "Whole-map variance of fitted prior, available before probing; negative-control baseline for incremental observation value."},
    "labels": "Evaluator gain=prior continuation unique discoveries minus random continuation expectation. Normalize by distinct yet-undiscovered targets reachable by any remaining survey result; null when no remaining opportunity, not an invalid-prior label.",
    "summaries": "Primary: evaluate scene-level Spearman/AUC separately for each fixed quality-map/probe-order replicate, using ONE 4/8-cell probe per scene; average replicate diagnostics, no seed/cell independence p-values. AUC compares normalized gain >=+0.10 vs <=-0.10; also report all 0.00/0.10/0.20 margins without selecting best. No-opportunity scenes excluded. Secondary scene-averaged associations are descriptive only, cannot be read as a single short-probe classifier.",
    "missingness": "Report all-negative probes, comparable-pair availability and scene counts for each feature. Scores computed only on available scenes cannot be treated as full-coverage diagnosis.",
    "oracle": "Evaluator chooses the larger of prior total and random expected total after each probe. An upper bound for these two continuations in expected-utility terms, not an attainable trained selector and not oracle hindsight over random realizations. Compare with original no-probe prior baseline including the probe cost.",
    "progression": "Do not implement a learned switcher just because response counts detect sparse scenes. Seek directional consistency within target-opportunity scenes, sufficient signal availability, and room for gains after probe cost; preserve failures or event reversals.",
    "limitations": ["Already-inspected validation, effectively one event with validation targets; weights expected to have training exposure.",
        "Target-count and ranking utility are different; no-opportunity/low-density cases must not masquerade as evidence of prior failure.",
        "Same-image artificial degradation and fixed observation costs are engineering proxies; no true UAV viewpoints or flight costs.",
        "No signal tested here exhausts possible visual features or proves information-theoretic impossibility."]}


def pair_alignment(priors, observations, score_key):
    signs = []
    for i in range(len(priors)):
        for j in range(i):
            if observations[i]["mode"] != observations[j]["mode"]:
                continue
            a = priors[i] - priors[j]
            b = observations[i][score_key] - observations[j][score_key]
            if a and b:
                signs.append(1. if a * b > 0 else -1.)
    return statistics.mean(signs) if signs else None, len(signs)


def probe_features(cells, observations, rates):
    """No label table or unrequested outcomes are accepted by this function."""
    ids = [o["cell_id"] for o in observations]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Probe must contain distinct observations")
    priors = [cells[cid].prior for cid in ids]
    q = [p * rates[o["mode"]]["tpr"] + (1 - p) * rates[o["mode"]]["fpr"] for p, o in zip(priors, observations)]
    variance = sum(v * (1 - v) for v in q)
    binary, binary_pairs = pair_alignment(priors, observations, "declared_target")
    soft, soft_pairs = pair_alignment(priors, observations, "uncalibrated_severe_score")
    return {"positive_count": sum(o["declared_target"] for o in observations),
        "response_surplus": sum(o["declared_target"] - expected for o, expected in zip(observations, q)) / math.sqrt(variance) if variance else None,
        "binary_alignment": binary, "soft_alignment": soft,
        "public_prior_spread": statistics.pvariance(c.prior for c in cells.values()),
        "binary_comparable_pairs": binary_pairs, "soft_comparable_pairs": soft_pairs,
        "native_count": sum(o["mode"] == "native" for o in observations)}


def prior_order(cells, prefix, count):
    remaining = {cid: c for cid, c in cells.items() if cid not in prefix}
    position = (cells[prefix[-1]].x, cells[prefix[-1]].y) if prefix else (0, 0)
    result = []
    for _ in range(count):
        cid = min(remaining, key=lambda cid: (-remaining[cid].prior,
                  math.dist(position, (remaining[cid].x, remaining[cid].y)), cid))
        cell = remaining.pop(cid)
        result.append(cid)
        position = cell.x, cell.y
    return result


def counterfactual(cells, prefix, total_budget, matched_cells):
    if len(prefix) != len(set(prefix)) or not 0 <= len(prefix) < total_budget <= len(cells):
        raise ValueError("Invalid paid prefix/budget")
    found = set().union(*(matched_cells[cid] for cid in prefix)) if prefix else set()
    remaining = [cid for cid in cells if cid not in prefix]
    remaining_steps = total_budget - len(prefix)
    reachable = defaultdict(int)
    for cid in remaining:
        for uid in matched_cells[cid] - found:
            reachable[uid] += 1
    continuation = prior_order(cells, prefix, remaining_steps)
    prior_found = found | set().union(*(matched_cells[cid] for cid in continuation))
    random_expected = len(found) + exact_random_discoveries(len(remaining), remaining_steps, list(reachable.values()))
    gain = len(prior_found) - random_expected
    return {"prefix_discoveries": len(found), "remaining_reachable_targets": len(reachable),
            "prior_continuation_cells": continuation, "prior_total": len(prior_found), "random_expected_total": random_expected,
            "gain": gain, "normalized_gain": gain / len(reachable) if reachable else None,
            "oracle_expected_total": max(len(prior_found), random_expected)}


def rank(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for i in ordered[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def correlation(x, y):
    if len(x) < 3:
        return None
    a, b = statistics.mean(x), statistics.mean(y)
    denominator = math.sqrt(sum((v - a) ** 2 for v in x) * sum((v - b) ** 2 for v in y))
    return sum((v - a) * (w - b) for v, w in zip(x, y)) / denominator if denominator else None


def oriented_auc(values, gains, margin):
    high = [value for value, gain in zip(values, gains) if gain > 0 and gain >= margin]
    low = [value for value, gain in zip(values, gains) if gain <= 0 and gain <= -margin]
    if not high or not low:
        return {"auc": None, "prior_advantage_scenes": len(high), "coverage_advantage_scenes": len(low)}
    score = sum(1. if a > b else .5 if a == b else 0. for a in high for b in low) / (len(high) * len(low))
    return {"auc": score, "prior_advantage_scenes": len(high), "coverage_advantage_scenes": len(low)}


def scene_summary(rows):
    # Association uses only rows with future opportunity; no-opportunity trials are
    # still retained for probe availability and complete cost/utility accounting.
    opportunity = [r for r in rows if r["outcome"]["normalized_gain"] is not None]
    features = {}
    for signal in SIGNALS:
        available = [r["features"][signal] for r in opportunity if r["features"][signal] is not None]
        features[signal] = statistics.mean(available) if available else None
    return {"sample_id": rows[0]["sample_id"], "event": rows[0]["event"], "trials": len(rows),
        "opportunity_trials": len(opportunity), "features_on_opportunity_trials": features,
        "feature_availability_on_opportunity_trials": {signal: sum(r["features"][signal] is not None for r in opportunity) / len(opportunity) if opportunity else None for signal in SIGNALS},
        "mean_normalized_gain": statistics.mean(r["outcome"]["normalized_gain"] for r in opportunity) if opportunity else None,
        "zero_positive_probe_fraction": statistics.mean(r["features"]["positive_count"] == 0 for r in rows),
        "mean_prior_total": statistics.mean(r["outcome"]["prior_total"] for r in rows),
        "mean_random_total": statistics.mean(r["outcome"]["random_expected_total"] for r in rows),
        "mean_oracle_total": statistics.mean(r["outcome"]["oracle_expected_total"] for r in rows),
        "mean_no_probe_prior_total": statistics.mean(r["no_probe_prior_total"] for r in rows)}


def associations(scenes):
    opportunity = [s for s in scenes if s["mean_normalized_gain"] is not None]
    result = {}
    for signal in SIGNALS:
        available = [s for s in opportunity if s["features_on_opportunity_trials"][signal] is not None]
        values = [s["features_on_opportunity_trials"][signal] for s in available]
        gains = [s["mean_normalized_gain"] for s in available]
        result[signal] = {"opportunity_scenes": len(opportunity), "available_scenes": len(available),
            "mean_trial_availability": statistics.mean(s["feature_availability_on_opportunity_trials"][signal] for s in opportunity) if opportunity else None,
            "spearman": correlation(rank(values), rank(gains)),
            "oriented_auc_by_margin": {str(margin): oriented_auc(values, gains, margin) for margin in (0., .1, .2)}}
    return result


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def single_probe_performance(rows):
    replicates = defaultdict(list)
    for row in rows:
        replicates[(row["quality_seed"], row["probe_seed"])].append(row)
    performance = []
    for (quality_seed, probe_seed), values in sorted(replicates.items()):
        scenes = [{"sample_id": r["sample_id"], "mean_normalized_gain": r["outcome"]["normalized_gain"],
                   "features_on_opportunity_trials": r["features"],
                   "feature_availability_on_opportunity_trials": {s: int(r["features"][s] is not None) for s in SIGNALS}}
                  for r in values]
        performance.append({"quality_seed": quality_seed, "probe_seed": probe_seed,
                            "association": associations(scenes)})
    aggregate = {}
    for signal in SIGNALS:
        values = [p["association"][signal] for p in performance]
        aggregate[signal] = {"replicates": len(values),
            "mean_opportunity_scenes": mean_or_none(v["opportunity_scenes"] for v in values),
            "mean_available_scenes": mean_or_none(v["available_scenes"] for v in values),
            "mean_availability": mean_or_none(v["mean_trial_availability"] for v in values),
            "mean_spearman": mean_or_none(v["spearman"] for v in values),
            "replicates_with_spearman": sum(v["spearman"] is not None for v in values),
            "oriented_auc_by_margin": {str(m): {
                "mean_auc": mean_or_none(v["oriented_auc_by_margin"][str(m)]["auc"] for v in values),
                "replicates_with_two_classes": sum(v["oriented_auc_by_margin"][str(m)]["auc"] is not None for v in values),
                "mean_prior_advantage_scenes": mean_or_none(v["oriented_auc_by_margin"][str(m)]["prior_advantage_scenes"] for v in values),
                "mean_coverage_advantage_scenes": mean_or_none(v["oriented_auc_by_margin"][str(m)]["coverage_advantage_scenes"] for v in values)} for m in (0., .1, .2)}}
    return aggregate, performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = sources(args.output_root)
    control = args.output_root / "disaster_search_coverage_controls_v1"
    inputs.update(coverage_report=control / "report.json", coverage_checks=control / "integrity_checks.json")
    protocol = {**PROTOCOL, "source_files": {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in inputs.items()},
                "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                    ("diagnose_early_evidence.py", "run_coverage_controls.py", "run_cached_search.py")}}
    freeze_plan(args.out / "diagnostic_protocol.json", protocol)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "diagnostic_protocol.json"), "probe_budget_cases": 13440}))
        return
    params = json.loads(inputs["parameters"].read_text())
    old_report = json.loads(inputs["coverage_report"].read_text())
    checks = json.loads(inputs["coverage_checks"].read_text())
    if not checks["passed"] or digest(inputs["coverage_report"]) != checks["report_sha256"]:
        raise ValueError("Coverage reference must be unchanged and audited")
    prior = params["initial_presence_prior"]
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    rows = []
    case_file = args.out / "cases.jsonl.gz"
    with gzip.open(case_file, "wt", compresslevel=1) as f:
        for split in ("calibration", "validation"):
            observations = json.loads(inputs[split + "_observations"].read_text())
            association = json.loads(inputs[split + "_association"].read_text())
            if len(observations) != 3072 or {o["split"] for o in observations} != {split} or digest(inputs[split + "_observations"]) != association["observations_sha256"]:
                raise ValueError("Incomplete or mismatched source")
            observation_index = {(o["sample_id"], o["cell_id"], o["mode"]): o for o in observations}
            truth_index = {(r["sample_id"], r["cell_id"], r["mode"]): r for r in association["rows"]}
            scene_ids = sorted({o["sample_id"] for o in observations})
            for sid in scene_ids:
                cue_path = args.prepared_data / sid / "public_cues.json"
                if digest(cue_path) != old_report["public_cue_hashes"][sid]:
                    raise ValueError("Changed public cue")
                cues = json.loads(cue_path.read_text())
                cells = {}
                for c in cues["cells"]:
                    group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                    cells[c["cell_id"]] = PublicCell(c["cell_id"], c["col"], c["row"], group, prior["bins"][group]["fitted_presence_probability"])
                for scenario, quality_seeds in SCENARIOS.items():
                    for quality_seed in quality_seeds:
                        modes = {cid: survey_mode(scenario, quality_seed, sid, cid) for cid in cells}
                        # This entire mapping belongs to the evaluator, never probe_features.
                        matched = {cid: set(truth_index[(sid, cid, modes[cid])]["matched_target_uids"]) for cid in cells}
                        no_probe = {b: len(set().union(*(matched[cid] for cid in prior_order(cells, [], b)))) for b in TOTAL_BUDGETS}
                        for seed in ORDER_SEEDS:
                            order = random_order(list(cells.values()), sid, seed)
                            for length in PREFIX_LENGTHS:
                                prefix = order[:length]
                                # Explicit field whitelist; no evaluator target IDs included.
                                probe = [{key: observation_index[(sid, cid, modes[cid])][key] for key in
                                          ("sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target", "uncalibrated_severe_score")}
                                         for cid in prefix]
                                features = probe_features(cells, probe, rates)
                                for budget in TOTAL_BUDGETS:
                                    result = counterfactual(cells, prefix, budget, matched)
                                    row = {"split": split, "sample_id": sid, "event": cues["event"], "scenario": scenario,
                                        "quality_seed": quality_seed, "probe_seed": seed, "prefix_length": length, "budget": budget,
                                        "probe": probe, "features": features, "outcome": result,
                                        "no_probe_prior_total": no_probe[budget]}
                                    rows.append(row)
                                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({"split": split, "scene": sid, "cases_done": len(rows)}), flush=True)
    if len(rows) != 13440:
        raise AssertionError("Incomplete diagnostic matrix")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["split"], r["scenario"], r["prefix_length"], r["budget"])].append(r)
    summaries, scene_rows, single_probe_details = [], [], []
    for key, values in sorted(grouped.items()):
        per_scene = defaultdict(list)
        for r in values:
            per_scene[r["sample_id"]].append(r)
        scenes = [scene_summary(v) for _, v in sorted(per_scene.items())]
        single, detail = single_probe_performance(values)
        common = dict(zip(("split", "scenario", "prefix_length", "budget"), key))
        sum_oracle = sum(s["mean_oracle_total"] for s in scenes)
        sum_prior_no_probe = sum(s["mean_no_probe_prior_total"] for s in scenes)
        summaries.append({**common, "scenes": len(scenes), "cases": len(values),
            "opportunity_scenes": sum(s["mean_normalized_gain"] is not None for s in scenes),
            "all_negative_probe_fraction": statistics.mean(r["features"]["positive_count"] == 0 for r in values),
            "all_negative_probe_fraction_with_opportunity": mean_or_none(int(r["features"]["positive_count"] == 0) for r in values if r["outcome"]["normalized_gain"] is not None),
            "sum_prior_after_probe": sum(s["mean_prior_total"] for s in scenes),
            "sum_random_after_probe": sum(s["mean_random_total"] for s in scenes),
            "sum_oracle_after_probe": sum_oracle, "sum_no_probe_prior": sum_prior_no_probe,
            "oracle_minus_no_probe_prior": sum_oracle - sum_prior_no_probe,
            "single_probe_performance": single,
            "scene_averaged_associations_descriptive_only": associations(scenes),
            "single_probe_by_event": {e: single_probe_performance([r for r in values if r["event"] == e])[0] for e in sorted({r["event"] for r in values})}})
        scene_rows.extend([{**common, **s} for s in scenes])
        single_probe_details.extend([{**common, **r} for r in detail])
    examples = []
    for sign in ("prior_advantage", "coverage_advantage"):
        candidates = [r for r in rows if r["features"]["positive_count"] == 0 and r["outcome"]["normalized_gain"] is not None
                      and (r["outcome"]["normalized_gain"] >= .1 if sign == "prior_advantage" else r["outcome"]["normalized_gain"] <= -.1)]
        chosen = sorted(candidates, key=lambda r: (r["split"], r["scenario"], r["prefix_length"], r["budget"], r["sample_id"], r["quality_seed"], r["probe_seed"]))[:1]
        examples.extend({"evaluator_class": sign, "selection": "First lexicographic matching all-negative case, posthoc illustration only; no representativeness claim.", **r} for r in chosen)
    report = {"stage": "EXPLORATORY_EARLY_EVIDENCE_IDENTIFIABILITY", "cases": len(rows), "scenes": 48, "events": 4,
              "protocol_sha256": digest(args.out / "diagnostic_protocol.json"), "runner_sha256": digest(Path(__file__)),
              "summary": summaries, "parameters_refitted": False, "switcher_trained": False,
              "new_inference_run": False, "paid_gpu_used": False, "test_used": False, "limitations": PROTOCOL["limitations"]}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / "scene_summaries.json").write_text(json.dumps(scene_rows, ensure_ascii=False, indent=2))
    with gzip.open(args.out / "single_probe_details.json.gz", "wt") as f:
        json.dump(single_probe_details, f, ensure_ascii=False)
    (args.out / "negative_probe_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(rows), "mixed_budget32": [{k: r[k] for k in
        ("split", "prefix_length", "opportunity_scenes", "all_negative_probe_fraction", "sum_prior_after_probe", "sum_oracle_after_probe", "sum_no_probe_prior")}
        for r in summaries if r["scenario"] == "mixed_survey" and r["budget"] == 32]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
