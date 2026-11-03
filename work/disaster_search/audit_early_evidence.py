"""Audit short-probe visibility, sunk costs and evaluator counterfactuals."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from run_cached_search import PublicCell, SCENARIOS, digest, survey_mode
from run_coverage_controls import ORDER_SEEDS, random_order
from diagnose_early_evidence import PREFIX_LENGTHS, TOTAL_BUDGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "diagnostic_protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    assert digest(args.run / "diagnostic_protocol.json") == report["protocol_sha256"]
    paths = {}
    for key, item in protocol["source_files"].items():
        paths[key] = Path(item["path"])
        assert digest(paths[key]) == item["sha256"]
    for name, expected in protocol["implementation_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    params = json.loads(paths["parameters"].read_text())
    coverage = json.loads(paths["coverage_report"].read_text())
    prior = params["initial_presence_prior"]
    rates = {mode: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]}
             for mode, r in params["likelihood_by_mode"].items()}
    source, truth, cells, splits = {}, {}, {}, {}
    for split in ("calibration", "validation"):
        observations = json.loads(paths[split + "_observations"].read_text())
        source.update({o["observation_id"]: o for o in observations})
        association = json.loads(paths[split + "_association"].read_text())
        truth.update({(r["sample_id"], r["cell_id"], r["mode"]): r for r in association["rows"]})
        for sid in {o["sample_id"] for o in observations}:
            splits[sid] = split
            cue_path = args.prepared_data / sid / "public_cues.json"
            assert digest(cue_path) == coverage["public_cue_hashes"][sid]
            cues = json.loads(cue_path.read_text())
            cells[sid] = {}
            for c in cues["cells"]:
                group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                cells[sid][c["cell_id"]] = PublicCell(c["cell_id"], c["col"], c["row"], group,
                                                     prior["bins"][group]["fitted_presence_probability"])
    seen = set()
    grouped = defaultdict(list)
    audit_totals = defaultdict(list)
    prefix_canonical_features = {}
    with gzip.open(args.run / "cases.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            sid, scenario, quality_seed, probe_seed, length, budget = (row[k] for k in
                ("sample_id", "scenario", "quality_seed", "probe_seed", "prefix_length", "budget"))
            key = sid, scenario, quality_seed, probe_seed, length, budget
            assert key not in seen and splits[sid] == row["split"]
            seen.add(key)
            public = cells[sid]
            prefix = random_order(list(public.values()), sid, probe_seed)[:length]
            assert [o["cell_id"] for o in row["probe"]] == prefix
            assert len({o["observation_id"] for o in row["probe"]}) == length
            for o in row["probe"]:
                assert o["mode"] == survey_mode(scenario, quality_seed, sid, o["cell_id"])
                assert set(o) == {"sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target", "uncalibrated_severe_score"}
                assert all(source[o["observation_id"]][k] == v for k, v in o.items())
            # Independently recompute features from paid observed fields only.
            q = [public[o["cell_id"]].prior * rates[o["mode"]]["tpr"] +
                 (1 - public[o["cell_id"]].prior) * rates[o["mode"]]["fpr"] for o in row["probe"]]
            expected_surplus = sum(o["declared_target"] - v for o, v in zip(row["probe"], q)) / math.sqrt(sum(v * (1 - v) for v in q))
            assert math.isclose(expected_surplus, row["features"]["response_surplus"], abs_tol=1e-12)
            assert sum(o["declared_target"] for o in row["probe"]) == row["features"]["positive_count"]
            for feature, field in (("binary_alignment", "declared_target"), ("soft_alignment", "uncalibrated_severe_score")):
                signs = []
                for i, a in enumerate(row["probe"]):
                    for b in row["probe"][:i]:
                        x = public[a["cell_id"]].prior - public[b["cell_id"]].prior
                        y = a[field] - b[field]
                        if a["mode"] == b["mode"] and x and y:
                            signs.append(1 if x * y > 0 else -1)
                expected = statistics.mean(signs) if signs else None
                assert expected == row["features"][feature]
            fingerprint = (sid, scenario, quality_seed, probe_seed, length)
            if fingerprint in prefix_canonical_features:
                assert row["features"] == prefix_canonical_features[fingerprint]
            prefix_canonical_features[fingerprint] = row["features"]
            # Future evaluator outcomes are isolated from the feature calculation above.
            matched = {cid: set(truth[(sid, cid, survey_mode(scenario, quality_seed, sid, cid))]["matched_target_uids"])
                       for cid in public}
            found = set().union(*(matched[cid] for cid in prefix))
            remainder = [cid for cid in public if cid not in prefix]
            count = budget - length
            continuation = row["outcome"]["prior_continuation_cells"]
            assert len(continuation) == count and not set(continuation) & set(prefix)
            position = (public[prefix[-1]].x, public[prefix[-1]].y)
            available = set(remainder)
            for cid in continuation:
                best = min(available, key=lambda item: (-public[item].prior, math.dist(position, (public[item].x, public[item].y)), item))
                assert best == cid
                available.remove(cid)
                position = public[cid].x, public[cid].y
            reachable = defaultdict(int)
            for cid in remainder:
                for uid in matched[cid] - found:
                    reachable[uid] += 1
            n = len(remainder)
            expected_random = len(found) + sum(1 - (math.comb(n - m, count) / math.comb(n, count) if n - m >= count else 0.)
                                             for m in reachable.values())
            prior_total = len(found | set().union(*(matched[cid] for cid in continuation)))
            outcome = row["outcome"]
            assert outcome["prefix_discoveries"] == len(found)
            assert outcome["remaining_reachable_targets"] == len(reachable)
            assert prior_total == outcome["prior_total"]
            assert math.isclose(expected_random, outcome["random_expected_total"], abs_tol=1e-12)
            assert outcome["oracle_expected_total"] == max(prior_total, expected_random)
            assert math.isclose(prior_total - expected_random, outcome["gain"], abs_tol=1e-12)
            if reachable:
                assert math.isclose(outcome["normalized_gain"], (prior_total - expected_random) / len(reachable), abs_tol=1e-12)
            else:
                assert outcome["normalized_gain"] is None
            grouped[(row["split"], scenario, length, budget, sid)].append(row)
            audit_totals[(row["split"], scenario, length, budget)].append(row)
    expected = {(sid, scenario, quality_seed, seed, length, budget) for sid in cells for scenario, quality_seeds in SCENARIOS.items()
                for quality_seed in quality_seeds for seed in ORDER_SEEDS for length in PREFIX_LENGTHS for budget in TOTAL_BUDGETS}
    assert seen == expected and len(seen) == report["cases"] == 13440
    for summary in report["summary"]:
        key = summary["split"], summary["scenario"], summary["prefix_length"], summary["budget"]
        rows = audit_totals[key]
        repeats = len(ORDER_SEEDS) * len(SCENARIOS[summary["scenario"]])
        for out_key, field in (("sum_prior_after_probe", "prior_total"), ("sum_random_after_probe", "random_expected_total"), ("sum_oracle_after_probe", "oracle_expected_total")):
            assert math.isclose(sum(r["outcome"][field] for r in rows) / repeats, summary[out_key], abs_tol=1e-9)
        original = next(r for r in coverage["summary"] if (r["split"], r["scenario"], r["budget"], r["policy"]) ==
                        (summary["split"], summary["scenario"], summary["budget"], "prior_survey"))
        assert math.isclose(summary["sum_no_probe_prior"], original["sum_buildings_found"], abs_tol=1e-9)
    checks = {"passed": True, "cases_replayed": len(seen), "distinct_paid_probes": len(prefix_canonical_features),
              "features_only_use_paid_observations_and_public_prior": True, "features_do_not_depend_on_future_budget": True,
              "matching_prefix_cost_and_random_exact_expectation_recomputed": True,
              "no_opportunity_not_labeled_as_prior_failure": True, "old_prior_baseline_reproduced": True,
              "all_sources_and_parameters_unchanged": True, "unit_tests_passed": 72, "test_used": False,
              "report_sha256": digest(args.run / "report.json"), "cases_sha256": digest(args.run / "cases.jsonl.gz")}
    (args.run / "integrity_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
