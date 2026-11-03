"""Independent arithmetic replay of local follow-up or validation search logs."""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from run_cached_search import BUDGETS, SCENARIOS, digest, survey_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "observations", "association", "parameters", "prepared-data"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=["calibration", "validation"], required=True)
    args = parser.parse_args()
    params = json.loads(args.parameters.read_text())
    report = json.loads((args.run / "report.json").read_text())
    protocol = json.loads((args.run / "search_protocol.json").read_text())
    validation = args.split == "validation"
    expected_hash = protocol["source_parameters_sha256"] if validation else protocol["source_hashes"]["parameters"]
    assert digest(args.parameters) == expected_hash
    assert digest(args.run / "search_protocol.json") == report["protocol_sha256"]
    if validation:
        for name, expected in protocol["implementation_sha256"].items():
            assert digest(Path(__file__).with_name(name)) == expected
    obs = json.loads(args.observations.read_text())
    assert len(obs) == 3072 and {o["split"] for o in obs} == {args.split}
    source = {o["observation_id"]: o for o in obs}
    truth = json.loads(args.association.read_text())
    assert digest(args.observations) == truth["observations_sha256"]
    truth_index = {(r["sample_id"], r["cell_id"], r["mode"]): r for r in truth["rows"]}
    scene_targets = {sid: {uid for r in truth["rows"] if r["sample_id"] == sid for uid in r["visible_target_uids"]} for sid in {r["sample_id"] for r in truth["rows"]}}
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    cells_by_scene = {}
    for sid in scene_targets:
        cues = json.loads((args.prepared_data / sid / "public_cues.json").read_text())
        assert cues["split"] == args.split
        cells = {}
        for c in cues["cells"]:
            group = bisect_right(params["initial_presence_prior"]["bin_edges"], c["raw_change_score"])
            cells[c["cell_id"]] = {"x": c["col"], "y": c["row"], "group": group,
                                   "prior": params["initial_presence_prior"]["bins"][group]["fitted_presence_probability"]}
        cells_by_scene[sid] = cells
    checked_steps, episodes, keys = 0, 0, set()
    for line in (args.run / "traces.jsonl").open():
        episode = json.loads(line)
        run = episode["run"]
        sid, policy, scenario, seed, budget = (run[k] for k in ("sample_id", "policy", "scenario", "quality_seed", "budget"))
        key = sid, policy, scenario, seed, budget
        assert key not in keys
        keys.add(key)
        cells = cells_by_scene[sid]
        mixture = {"native_survey": 1., "mixed_survey": .5, "blurred_survey": 0.}[scenario]
        survey_tpr = mixture * rates["native"]["tpr"] + (1 - mixture) * rates["blurred"]["tpr"]
        latest, post, shifts, used = {}, {}, {cid: 0. for cid in cells}, set()
        found, false_cells, unknown, used_ids = set(), set(), set(), set()
        unmatched, cost, first, upgrades = 0, 0, None, 0
        position = (0, 0)

        def current(cid):
            p = cells[cid]["prior"]
            return post[cid] if cid in latest else 1 / (1 + math.exp(-(math.log(p / (1 - p)) + shifts[cid])))

        for step in episode["steps"]:
            candidates = []
            for cid, cell in cells.items():
                if cid not in latest:
                    kind, price, score = "survey", 1, current(cid) * survey_tpr
                elif latest[cid]["mode"] == "blurred" and not latest[cid]["declared_target"] and (cid, "upgrade") not in used:
                    kind, price, score = "upgrade", 2, current(cid) * rates["native"]["tpr"] / 2
                else:
                    continue
                if price <= budget - cost:
                    candidates.append((-score, math.dist(position, (cell["x"], cell["y"])), cid, kind))
            _, _, cid, kind = min(candidates)
            assert step["action"] == {"cell_id": cid, "kind": kind}
            o = step["observation"]
            assert all(source[o["observation_id"]][k] == v for k, v in o.items())
            assert set(o) == {"sample_id", "cell_id", "mode", "declared_target", "uncalibrated_severe_score", "evidence_group_id", "observation_id"}
            assert o["observation_id"] not in used_ids
            assert o["mode"] == (survey_mode(scenario, seed, sid, cid) if kind == "survey" else "native")
            assert math.isclose(step["belief_before"], current(cid), abs_tol=1e-12)
            cost += 1 if kind == "survey" else 2
            upgrades += kind == "upgrade"
            assert cost == step["cost_total"] <= budget
            p = cells[cid]["prior"]
            likelihood = rates[o["mode"]]
            a, b = (likelihood["tpr"], likelihood["fpr"]) if o["declared_target"] else (1 - likelihood["tpr"], 1 - likelihood["fpr"])
            value = p if (policy == "quality_gate" and o["mode"] == "blurred" and not o["declared_target"]) else p * a / (p * a + (1 - p) * b)
            assert math.isclose(value, step["belief_after"], abs_tol=1e-12)
            post[cid], latest[cid] = value, o
            used.add((cid, kind))
            used_ids.add(o["observation_id"])
            for target, cell in cells.items():
                shifts[target] = 0.
                if target in latest or policy in {"quality", "quality_gate"}:
                    continue
                peers = [peer for peer in latest if cells[peer]["group"] == cell["group"] and
                         (policy == "adaptive_prior" or max(abs(cells[peer]["x"] - cell["x"]), abs(cells[peer]["y"] - cell["y"])) <= 1)]
                if policy == "neighbor_positive":
                    shifts[target] = .25 if any(latest[peer]["declared_target"] for peer in peers) else 0.
                else:
                    residual, weight = 0., 0.
                    for peer in peers:
                        if policy == "local_positive" and not latest[peer]["declared_target"]:
                            continue
                        r = (rates[latest[peer]["mode"]]["tpr"] - rates[latest[peer]["mode"]]["fpr"]) ** 2
                        residual += r * (post[peer] - cells[peer]["prior"])
                        weight += r
                    shifts[target] = max(-1., min(1., 4 * residual / (8 + weight)))
            if policy in {"local_residual", "local_positive", "neighbor_positive"}:
                recorded = step["cell_logit_shifts"]
                assert set(recorded) == {cid for cid, v in shifts.items() if v}
                assert all(math.isclose(v, shifts[cid], abs_tol=1e-12) for cid, v in recorded.items())
            row = truth_index[(sid, cid, o["mode"])]
            matched = set(row["matched_target_uids"])
            found |= matched
            if matched and first is None:
                first = cost
            if o["declared_target"]:
                if row["target"] is None:
                    unknown.add(cid)
                else:
                    unmatched += not matched
                    if row["target"] is False:
                        false_cells.add(cid)
            position = cells[cid]["x"], cells[cid]["y"]
            checked_steps += 1
        assert cost == run["cost"] and len(latest) == run["evidence_groups"]
        assert upgrades == run["upgrades"] and len(episode["steps"]) == run["actions"]
        assert run["unique_buildings_found"] == len(found) and run["target_count"] == len(scene_targets[sid])
        assert run["first_discovery_cost"] == first and run["no_true_discovery"] == (not found)
        assert run["false_positive_cells"] == len(false_cells) and run["unknown_positive_cells"] == len(unknown)
        assert run["unmatched_positive_observations"] == unmatched
        assert run["recall"] == (len(found) / len(scene_targets[sid]) if scene_targets[sid] else None)
        episodes += 1
    policies = protocol["policies"] if validation else ("local_residual", "local_positive", "neighbor_positive")
    assert keys == {(sid, policy, scenario, seed, budget) for sid in cells_by_scene for policy in policies
                    for scenario, seeds in SCENARIOS.items() for seed in seeds for budget in BUDGETS}
    check = {"passed": True, "split": args.split, "episodes_replayed": episodes, "steps_replayed": checked_steps,
             "action_scores_and_neighborhood_shifts_recomputed": True, "posteriors_anchored_to_original_prior": True,
             "budget_unique_evidence_and_building_counts_recomputed": True, "calibration_parameters_unchanged": True,
             "test_used": False, "source_parameters_sha256": digest(args.parameters),
             "report_sha256": digest(args.run / "report.json"), "traces_sha256": digest(args.run / "traces.jsonl")}
    (args.run / "integrity_checks.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check))


if __name__ == "__main__":
    main()
