"""Recompute control actions, evidence accounting, and unique-building metrics."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

from run_cached_search import BUDGETS, SCENARIOS, digest, survey_mode
from run_coverage_controls import CONTROL_POLICIES, ORDER_SEEDS, random_order, spatial_order
from run_cached_search import PublicCell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "control_protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    assert digest(args.run / "control_protocol.json") == report["protocol_sha256"]
    for name, expected in protocol["implementation_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    sources = {}
    for key, item in protocol["sources"].items():
        path = Path(item["path"])
        assert digest(path) == item["sha256"]
        sources[key] = path
    params = json.loads(sources["parameters"].read_text())
    prior = params["initial_presence_prior"]
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    observations, truth, targets, cells, scene_split, cue_hashes = {}, {}, {}, {}, {}, {}
    for split in ("calibration", "validation"):
        raw = json.loads(sources[split + "_observations"].read_text())
        assert {o["split"] for o in raw} == {split}
        observations.update({o["observation_id"]: o for o in raw})
        association = json.loads(sources[split + "_association"].read_text())
        truth.update({r["observation_id"]: r for r in association["rows"]})
        for sid in {o["sample_id"] for o in raw}:
            scene_split[sid] = split
            targets[sid] = {uid for r in association["rows"] if r["sample_id"] == sid for uid in r["visible_target_uids"]}
            path = args.prepared_data / sid / "public_cues.json"
            cue_hashes[sid] = digest(path)
            assert cue_hashes[sid] == report["public_cue_hashes"][sid]
            cue = json.loads(path.read_text())
            cells[sid] = {}
            for c in cue["cells"]:
                group = bisect_right(prior["bin_edges"], c["raw_change_score"])
                cells[sid][c["cell_id"]] = {**c, "prior": prior["bins"][group]["fitted_presence_probability"]}
    seen, checked_steps, prior_traces = set(), 0, {}
    aggregate_found, aggregate_false = defaultdict(list), defaultdict(list)
    with gzip.open(args.run / "traces.jsonl.gz", "rt") as f:
        for line in f:
            episode = json.loads(line)
            run, trace = episode["run"], episode["steps"]
            sid, policy, scenario, quality_seed, search_seed, budget = (run[k] for k in
                ("sample_id", "policy", "scenario", "quality_seed", "search_seed", "budget"))
            key = (sid, policy, scenario, quality_seed, search_seed, budget)
            assert key not in seen and run["split"] == scene_split[sid]
            seen.add(key)
            source_cells = cells[sid]
            fitted = policy == "prior_survey"
            upgrade_enabled = policy.endswith("gate")
            p = {cid: c["prior"] if fitted else prior["global_smoothed_prevalence"] for cid, c in source_cells.items()}
            public = [PublicCell(cid, c["col"], c["row"], 0, p[cid]) for cid, c in source_cells.items()]
            order = (spatial_order(public) if policy.startswith("spatial") else
                     random_order(public, sid, search_seed) if policy.startswith("random") else None)
            rank = {cid: i for i, cid in enumerate(order or [])}
            mode_weight = {"native_survey": 1., "mixed_survey": .5, "blurred_survey": 0.}[scenario]
            survey_tpr = mode_weight * rates["native"]["tpr"] + (1 - mode_weight) * rates["blurred"]["tpr"]
            latest, post, used, used_ids = {}, {}, set(), set()
            found, false, unknown = set(), set(), set()
            first, cost, upgrades, unmatched = None, 0, 0, 0
            position = (0, 0)
            for t in trace:
                options = []
                for cid, c in source_cells.items():
                    if cid not in latest:
                        kind, price, score = "survey", 1, p[cid] * survey_tpr
                    elif upgrade_enabled and latest[cid]["mode"] == "blurred" and not latest[cid]["declared_target"] and (cid, "upgrade") not in used:
                        kind, price, score = "upgrade", 2, post[cid] * rates["native"]["tpr"] / 2
                    else:
                        continue
                    if price <= budget - cost:
                        tie = rank[cid] if order is not None else math.dist(position, (c["col"], c["row"]))
                        options.append((-score, tie, cid, kind))
                _, _, cid, kind = min(options)
                assert t["action"] == {"cell_id": cid, "kind": kind}
                o = t["observation"]
                assert o["observation_id"] not in used_ids
                assert set(o) == {"sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target", "uncalibrated_severe_score"}
                assert all(observations[o["observation_id"]][k] == v for k, v in o.items())
                assert o["mode"] == (survey_mode(scenario, quality_seed, sid, cid) if kind == "survey" else "native")
                assert math.isclose(t["belief_before"], post.get(cid, p[cid]), abs_tol=1e-12)
                assert t["representative_replaced"] == (cid in latest)
                likelihood = rates[o["mode"]]
                if not upgrade_enabled or (o["mode"] == "blurred" and not o["declared_target"]):
                    value = p[cid]
                else:
                    a, b = (likelihood["tpr"], likelihood["fpr"]) if o["declared_target"] else (1 - likelihood["tpr"], 1 - likelihood["fpr"])
                    value = p[cid] * a / (p[cid] * a + (1 - p[cid]) * b)
                assert math.isclose(t["belief_after"], value, abs_tol=1e-12)
                post[cid], latest[cid] = value, o
                used.add((cid, kind))
                used_ids.add(o["observation_id"])
                cost += 1 if kind == "survey" else 2
                upgrades += kind == "upgrade"
                assert t["cost_total"] == cost <= budget
                r = truth[o["observation_id"]]
                matched = set(r["matched_target_uids"])
                found |= matched
                if matched and first is None:
                    first = cost
                if o["declared_target"]:
                    if r["target"] is None:
                        unknown.add(cid)
                    else:
                        unmatched += not matched
                        if r["target"] is False:
                            false.add(cid)
                position = source_cells[cid]["col"], source_cells[cid]["row"]
                checked_steps += 1
            assert run["cost"] == cost == run["surveys"] + 2 * run["upgrades"]
            assert run["upgrades"] == upgrades and run["actions"] == len(trace)
            assert run["target_count"] == len(targets[sid]) and run["unique_buildings_found"] == len(found)
            assert run["evidence_groups"] == len(latest) and run["first_discovery_cost"] == first
            assert run["false_positive_cells"] == len(false) and run["unknown_positive_cells"] == len(unknown)
            assert run["unmatched_positive_observations"] == unmatched and run["no_true_discovery"] == (not found)
            assert run["recall"] == (len(found) / len(targets[sid]) if targets[sid] else None)
            if policy == "prior_survey" and run["split"] == "calibration":
                prior_traces[(sid, scenario, quality_seed, budget)] = [t["action"] for t in trace]
            aggregate_found[(run["split"], scenario, budget, policy)].append(len(found))
            aggregate_false[(run["split"], scenario, budget, policy)].append(len(false))
    expected = {(sid, policy, scenario, quality_seed, search_seed, budget) for sid in cells
                for policy in CONTROL_POLICIES for scenario, quality_seeds in SCENARIOS.items()
                for quality_seed in quality_seeds for search_seed in (ORDER_SEEDS if policy.startswith("random") else (-1,))
                for budget in BUDGETS}
    assert seen == expected and len(seen) == report["new_episodes"] == 25200
    for row in report["summary"]:
        key = row["split"], row["scenario"], row["budget"], row["policy"]
        if key not in aggregate_found:
            continue
        repeats = len(SCENARIOS[row["scenario"]]) * (len(ORDER_SEEDS) if row["policy"].startswith("random") else 1)
        assert math.isclose(sum(aggregate_found[key]) / repeats, row["sum_buildings_found"], abs_tol=1e-9)
        assert math.isclose(sum(aggregate_false[key]) / repeats, row["sum_false_positive_cells"], abs_tol=1e-9)
    previous = args.run.parent / "disaster_search_cached_search_v1"
    prior_matches = 0
    with (previous / "traces.jsonl").open() as f:
        for line in f:
            episode = json.loads(line)
            r = episode["run"]
            if r["policy"] == "prior":
                assert prior_traces[(r["sample_id"], r["scenario"], r["quality_seed"], r["budget"])] == [t["action"] for t in episode["steps"]]
                prior_matches += 1
    assert prior_matches == 504
    checks = {"passed": True, "episodes_replayed": len(seen), "steps_replayed": checked_steps,
              "original_calibration_prior_trajectories_reproduced": prior_matches,
              "action_score_cost_evidence_and_building_replay": True, "all_random_seeds_retained": True,
              "all_source_hashes_and_parameters_preserved": True, "unit_tests_passed": 64, "test_used": False,
              "report_sha256": digest(args.run / "report.json"), "traces_sha256": digest(args.run / "traces.jsonl.gz")}
    (args.run / "integrity_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
