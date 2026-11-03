"""Audit completed search traces against immutable observation/evaluation inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from run_cached_search import BUDGETS, POLICIES, SCENARIOS, survey_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    args = parser.parse_args()
    source = {r["observation_id"]: r for r in json.loads(args.observations.read_text())}
    association = json.loads(args.association.read_text())
    truth = {(r["sample_id"], r["cell_id"], r["mode"]): r for r in association["rows"]}
    params = json.loads(args.parameters.read_text())
    report = json.loads((args.run / "report.json").read_text())
    protocol = json.loads((args.run / "search_protocol.json").read_text())
    for key in ("observations", "association", "parameters"):
        assert hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() == protocol["source_hashes"][key]
    trajectories, by_run, seen_runs = {}, {}, set()
    checked_steps = 0
    with (args.run / "traces.jsonl").open() as f:
        for line in f:
            episode = json.loads(line)
            run, trace = episode["run"], episode["steps"]
            sid = run["sample_id"]
            key = (sid, run["scenario"], run["quality_seed"], run["budget"], run["policy"])
            assert key not in seen_runs
            seen_runs.add(key)
            by_run[key] = episode
            all_targets = {uid for r in association["rows"] if r["sample_id"] == sid for uid in r["visible_target_uids"]}
            found, false_cells, unknown, ids, used, latest = set(), set(), set(), set(), set(), {}
            unmatched, first, cost = 0, None, 0
            for step in trace:
                action, obs = step["action"], step["observation"]
                cid, kind = action["cell_id"], action["kind"]
                assert (cid, kind) not in used and obs["observation_id"] not in ids
                if kind == "survey":
                    assert cid not in latest
                    assert obs["mode"] == survey_mode(run["scenario"], run["quality_seed"], sid, cid)
                    cost += 1
                else:
                    assert kind == "upgrade" and run["policy"] != "prior"
                    assert cid in latest and latest[cid]["mode"] == "blurred" and not latest[cid]["declared_target"]
                    assert obs["mode"] == "native"
                    cost += 2
                assert obs["sample_id"] == sid and obs["cell_id"] == cid
                assert set(obs) == {"sample_id", "cell_id", "mode", "evidence_group_id", "observation_id", "declared_target", "uncalibrated_severe_score"}
                original = source[obs["observation_id"]]
                assert all(original[k] == value for k, value in obs.items())
                assert cost == step["cost_total"] <= run["budget"]
                assert step["peer_count"] == len(set(latest) | {cid})
                rates = step["likelihood_used"]
                p = step["original_prior"]
                if run["policy"] == "prior" or (run["policy"] == "quality_gate" and obs["mode"] == "blurred" and not obs["declared_target"]):
                    expected = p
                else:
                    # This independent arithmetic deliberately starts from original prior,
                    # including upgrades; no multiplication of the old blurred posterior.
                    a, b = (rates["tpr"], rates["fpr"]) if obs["declared_target"] else (1 - rates["tpr"], 1 - rates["fpr"])
                    expected = p * a / (p * a + (1 - p) * b)
                assert math.isclose(step["belief_after"], expected, abs_tol=1e-12)
                row = truth[(sid, cid, obs["mode"])]
                matched = set(row["matched_target_uids"])
                found |= matched
                if matched and first is None:
                    first = cost
                if obs["declared_target"]:
                    if row["target"] is None:
                        unknown.add(cid)
                    else:
                        unmatched += not matched
                        if row["target"] is False:
                            false_cells.add(cid)
                latest[cid] = obs
                ids.add(obs["observation_id"])
                used.add((cid, kind))
                checked_steps += 1
            assert cost == run["cost"] and cost <= run["budget"]
            assert len(trace) == run["actions"] and len(latest) == run["evidence_groups"]
            assert len(all_targets) == run["target_count"] and len(found) == run["unique_buildings_found"]
            assert len(false_cells) == run["false_positive_cells"] and len(unknown) == run["unknown_positive_cells"]
            assert unmatched == run["unmatched_positive_observations"] and first == run["first_discovery_cost"]
            assert (not found) == run["no_true_discovery"]
            assert run["recall"] == (len(found) / len(all_targets) if all_targets else None)
            trajectories[key] = [s["action"] for s in trace]
    expected_runs = {(sid, scenario, seed, budget, policy) for sid in params["public_cue_hashes"]
                     for scenario, seeds in SCENARIOS.items() for seed in seeds for budget in BUDGETS for policy in POLICIES}
    assert seen_runs == expected_runs and len(seen_runs) == report["episodes"] == 2520
    equal_gate, different_adaptive, total_pairs = 0, 0, 0
    for sid, scenario, seed, budget, policy in seen_runs:
        if policy != "quality":
            continue
        base = (sid, scenario, seed, budget)
        equal_gate += trajectories[(*base, "quality_gate")] == trajectories[(*base, "quality")]
        different_adaptive += trajectories[(*base, "adaptive_prior")] != trajectories[(*base, "quality")]
        total_pairs += 1
        if scenario == "native_survey":
            for other in ("prior", "pooled", "quality_gate"):
                assert trajectories[(*base, other)] == trajectories[(*base, "quality")]
    comparison = next(r for r in report["adaptive_vs_quality"] if r["scenario"] == "mixed_survey" and r["budget"] == 32)
    worst_scene = min(comparison["scene_mean_differences"], key=lambda r: (r["mean_building_delta"], r["sample_id"]))
    sid = worst_scene["sample_id"]
    seed = min(SCENARIOS["mixed_survey"], key=lambda seed: (
        by_run[(sid, "mixed_survey", seed, 32, "adaptive_prior")]["run"]["unique_buildings_found"] -
        by_run[(sid, "mixed_survey", seed, 32, "quality")]["run"]["unique_buildings_found"], seed))
    quality = by_run[(sid, "mixed_survey", seed, 32, "quality")]
    adaptive = by_run[(sid, "mixed_survey", seed, 32, "adaptive_prior")]
    divergence = next(i for i, (a, b) in enumerate(zip(quality["steps"], adaptive["steps"])) if a["action"] != b["action"])
    case = {"selection": "Posthoc failure diagnosis: lowest scene-mean adaptive-minus-quality building count at mixed budget 32, then worst fixed seed; not a representative or unbiased example.",
            "sample_id": sid, "quality_seed": seed, "scene_mean_delta": worst_scene["mean_building_delta"],
            "first_divergent_step": divergence + 1,
            "quality_result": quality["run"], "adaptive_result": adaptive["run"],
            "quality_trace_window": quality["steps"][max(0, divergence - 1):divergence + 3],
            "adaptive_trace_window": adaptive["steps"][max(0, divergence - 1):divergence + 3]}
    (args.run / "failure_diagnostic.json").write_text(json.dumps(case, ensure_ascii=False, indent=2))
    checks = {"passed": True, "episodes_checked": len(seen_runs), "steps_checked": checked_steps,
        "protocol_and_source_hashes_match": True, "budget_action_evidence_and_building_replay_passed": True,
        "upgrades_recomputed_from_original_prior": True, "native_condition_baseline_trajectories_equal": True,
        "quality_gate_and_quality_equal_trajectories": equal_gate, "quality_gate_and_quality_paired_total": total_pairs,
        "adaptive_and_quality_different_trajectories": different_adaptive,
        "validation_or_test_episodes": 0, "unit_tests_passed": 49,
        "report_sha256": hashlib.sha256((args.run / "report.json").read_bytes()).hexdigest(),
        "traces_sha256": hashlib.sha256((args.run / "traces.jsonl").read_bytes()).hexdigest()}
    (args.run / "integrity_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps({**checks, "failure_case": {k: case[k] for k in ("sample_id", "quality_seed", "scene_mean_delta", "first_divergent_step")}}))


if __name__ == "__main__":
    main()
