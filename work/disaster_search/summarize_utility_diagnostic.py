"""Record bounded diagnostic conclusions without fitting a new policy."""
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.run / (name + ".json") for name in ["component_calibration", "oracle_summary", "selected_action_summary", "ranking_summary", "audit", "protocol"]}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if data["audit"]["stage"] != "UTILITY_FAILURE_AND_ORACLE_REPLAY_PASSED":
        raise ValueError("Full replay required before interpretation")
    def action(policy, event=None):
        rows = data["selected_action_summary"]["event_macro_means" if event is None else "event_means"]
        return next(r for r in rows if r["fit_scope"] == "event_holdout" and r["tie_rule"] == "fixed_random" and r["policy"] == policy and (event is None or r["event"] == event))
    def oracle(kind):
        return next(r for r in data["oracle_summary"]["event_macro_means"] if r["budget"] == 32 and r["oracle_kind"] == kind)
    source_mean_path = Path(data["protocol"]["sources"]["unified_means"]["path"])
    source_means = json.loads(source_mean_path.read_text())
    gate_events = {}
    for event in ["hurricane-harvey", "santa-rosa-wildfire", "mexico-earthquake"]:
        values = [r["distinct_found"] for r in source_means if r["event"] == event and r["fit_scope"] == "development_fit" and r["policy"] == "prior_gate" and r["tie_rule"] == "fixed_random" and r["scenario"] == "mixed_survey" and r["upgrade_cost"] == 2 and r["budget"] == 32 and r["threshold"] == .1]
        if len(values) != 24:
            raise ValueError("Incomplete unchanged gate reference")
        gate_events[event] = sum(values) / len(values)
    gate = sum(gate_events.values()) / 3
    survey_oracle, full_oracle = oracle("survey_only"), oracle("survey_and_upgrade")
    raw, count, negative = action("prior_survey"), action("count_survey"), action("count_negative")
    santa = [r for r in data["component_calibration"]["rows"] if r["fit_scope"] == "event_holdout" and r["event"] == "santa-rosa-wildfire"]
    harvey = [r for r in data["component_calibration"]["rows"] if r["fit_scope"] == "event_holdout" and r["event"] == "hurricane-harvey"]
    error_direction = [{"event": event, "expected_known_local": action("count_all", event)["expected_known_local"],
        "actual_known_local": action("count_all", event)["actual_known_local"],
        "expected_minus_observed": action("count_all", event)["local_estimation_excess"]}
        for event in ["hurricane-harvey", "santa-rosa-wildfire", "mexico-earthquake"]]
    sensor_loss = full_oracle["visible_targets"] - full_oracle["accessible_union"]
    budget_loss = full_oracle["accessible_union"] - full_oracle["lower"]
    private_information_gap = full_oracle["lower"] - gate
    if abs(sensor_loss + budget_loss + private_information_gap - (full_oracle["visible_targets"] - gate)) > 1e-10:
        raise ValueError("Diagnostic ceiling-gap identity failed")
    evidence = {"stage": "POSTHOC_INTERPRETATION_OF_COMPLETED_FROZEN_DIAGNOSTIC",
        "source_sha256": sha(Path(__file__).read_bytes()),
        "inputs_sha256": {name: sha(path.read_bytes()) for name, path in paths.items()},
        "primary": {"scenario": "mixed_survey", "budget": 32, "upgrade_cost": 2, "threshold": .1},
        "santa_source_prediction": [r["predicted_mixed"] for r in santa],
        "santa_held_observed": [r["held_observed_mixed"] for r in santa],
        "harvey_source_prediction": [r["predicted_mixed"] for r in harvey],
        "harvey_held_observed": [r["held_observed_mixed"] for r in harvey],
        "selected_count_all_error_by_event": error_direction,
        "known_cell_estimate_scope": "Expected and observed local counts on known factual actions, event-equal means; not a population calibration estimate or global unique reward.",
        "negative_revisit_opportunity": {k: negative[k] for k in ["upgrade_expected_known_local", "upgrade_known_local", "upgrade_known_global", "opportunity_cases", "opportunity_upgrade_global", "opportunity_alternative_global", "opportunity_actual_difference"]},
        "ranking_gap_identity": {"count_minus_raw_local": count["actual_local"] - raw["actual_local"],
            "benefit_of_fewer_duplicates": raw["cross_cell_duplicates"] - count["cross_cell_duplicates"],
            "count_minus_raw_global": count["actual_global"] - raw["actual_global"]},
        "ceilings": {"visible_target_mean": full_oracle["visible_targets"], "full_accessible_cache_union": full_oracle["accessible_union"],
            "full_budget_oracle_lower": full_oracle["lower"], "full_budget_oracle_upper": full_oracle["upper"],
            "survey_only_oracle_lower": survey_oracle["lower"], "survey_only_oracle_upper": survey_oracle["upper"],
            "raw_survey_reference": raw["actual_global"], "raw_gate_reference": gate,
            "unavailable_in_cache": sensor_loss, "accessible_but_budget_limited": budget_loss,
            "perfect_information_gap_vs_raw_gate": private_information_gap},
        "ceilings_interpretation": "Gap is against an oracle with private future/instance knowledge. It does not estimate an executable policy's attainable gain or justify new data/model claims.",
        "new_policy_selected": False, "new_model_fit": False, "independent_validation": False}
    atomic_json(args.run / "evidence_summary.json", evidence)
    freeze(args.run / "next_step_decision.json", {"stage": "DIAGNOSTIC_COMPLETED_EVENT_DEPENDENT_RANKING_AND_OPPORTUNITY_COST_SUPPORTED",
        "basis_audit_sha256": sha(paths["audit"].read_bytes()), "basis_evidence_sha256": sha((args.run / "evidence_summary.json").read_bytes()),
        "supported": ["The frozen source-bin ordering misranks held Santa Rosa; no globally monotone relationship is established because Harvey differs.",
            "Factual count-negative upgrades have lower one-step global yield than two public-ranked unseen surveys on average in held development events; not a full-policy causal estimate.",
            "Cross-cell duplicates are real, but count survey has fewer duplicates than raw survey and still loses; duplicate removal alone does not explain the failure.",
            "The current cache has substantial perfect-information headroom; Mexico remains zero and cannot be repaired by planning in this cache."],
        "decision": "Keep the old positive-revisit policy stopped. Do not deepen planning or present this diagnostic as a new method. A follow-up may test a source-fitted rank-preserving survey-utility calibration as a baseline control, with an independently specified revisit opportunity-cost model; first distinguish it from simply restoring raw ordering.",
        "next_controls_required": ["Frozen raw survey and raw negative gate", "Current frozen count-only/negative controls", "Rank-preserving calibration only as a baseline repair control, not claimed novelty", "Specified budget-compatible VAS/MF-ENS-related strong baselines before a method claim"],
        "gate": "Only lock one new candidate after protocol, nearest-work mechanism comparison, and independent-data admission plan. If correction only reproduces public raw ranking, label baseline repair; do not promote it as a research method.",
        "no_revival_of_failed_variants": "Do not repackage the earlier boundary discount or short-prefix switching failures as new mechanisms.",
        "new_followup_experiment_started": False, "new_model_fit_enabled": False,
        "validation_or_confirmation_access_enabled": False, "new_policy_selected": False,
        "paid_gpu_needed": False, "novelty_established": False})
    print(json.dumps({"evidence_written": True, "oracle_cases": data["oracle_summary"]["cases"], "new_policy_selected": False}))


if __name__ == "__main__":
    main()
