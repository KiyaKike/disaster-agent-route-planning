"""Descriptive score-gap identities and application of the frozen stop rule.

This reads completed summaries only. The decomposition is an algebraic
identity between policy scores, not a causal partition of independent errors.
"""
from collections import defaultdict
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from run_utility_tie_controls import is_primary

KEYS = ["fit_scope", "scenario", "upgrade_cost", "budget", "threshold", "tie_rule"]


def decompose(rows, event_level=False):
    fields = KEYS + (["event"] if event_level else [])
    groups = defaultdict(dict)
    for r in rows:
        groups[tuple(r[k] for k in fields)][r["policy"]] = r["distinct_found"]
    output = []
    for key, values in sorted(groups.items()):
        ranking = values["count_survey"] - values["prior_survey"]
        negative = values["count_negative"] - values["count_survey"]
        positive = values["count_all"] - values["count_negative"]
        raw_gate = values["prior_gate"] - values["prior_survey"]
        for policy in ["count_negative", "count_all"]:
            direct = values[policy] - values["prior_gate"]
            terms = {"count_vs_raw_survey": ranking, "count_negative_vs_survey": negative,
                "count_positive_vs_negative": positive if policy == "count_all" else 0.,
                "minus_raw_gate_vs_survey": -raw_gate}
            residual = direct - sum(terms.values())
            if abs(residual) > 1e-10:
                raise ValueError("Score-gap identity did not close")
            output.append({**dict(zip(fields, key)), "candidate": policy, "baseline": "prior_gate",
                "candidate_found": values[policy], "baseline_found": values["prior_gate"],
                "direct_score_gap": direct, "terms": terms, "closure_residual": residual})
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run / n for n in ["summary.json", "paired_effects.json", "audit.json", "protocol.json"]]
    summary = json.loads(paths[0].read_text())
    effects = json.loads(paths[1].read_text())
    audit = json.loads(paths[2].read_text())
    if audit["stage"] != "DEVELOPMENT72_UNIFIED_REVISIT_FULL_REPLAY_PASSED":
        raise ValueError("Completed replay required")
    macros = decompose(summary["event_macro_means"])
    events = decompose(summary["event_means"], event_level=True)
    atomic_json(args.run / "score_gap_decomposition.json", {"stage": "POST_HOC_DESCRIPTIVE_SCORE_IDENTITY",
        "source_sha256": sha(Path(__file__).read_bytes()),
        "inputs_sha256": {p.name: sha(p.read_bytes()) for p in paths},
        "interpretation": "Exact identity of matched-policy performance differences; not independent causal error contributions. No new scores, model fits or selected primary setting.",
        "primary": [r for r in macros if is_primary(r)], "event_macro": macros, "event_level": events})
    primary = [r for r in effects["event_macro_effects"] if is_primary(r)]
    all_vs_survey = [r for r in primary if r["comparison"] == "count_all_vs_count_survey"]
    negative_vs_survey = [r for r in primary if r["comparison"] == "count_negative_vs_count_survey"]
    if len(all_vs_survey) != 6 or len(negative_vs_survey) != 6:
        raise ValueError("Incomplete primary tie/fit comparisons")
    stop_positive = all(r["delta_found"] < 0 for r in all_vs_survey)
    held_negative_fails = all(r["delta_found"] < 0 for r in negative_vs_survey if r["fit_scope"] == "event_holdout")
    if not stop_positive or not held_negative_fails:
        raise ValueError("Expected descriptive decision basis changed; review rather than silently changing the decision")
    freeze(args.run / "next_step_decision.json", {
        "stage": "CURRENT_COUNT_POSITIVE_REVISIT_POLICY_STOPPED_NOT_A_RESEARCH_METHOD",
        "basis_audit_sha256": sha(paths[2].read_bytes()), "basis_effects_sha256": sha(paths[1].read_bytes()),
        "basis_prior_decision_sha256": json.loads(paths[3].read_text())["sources"]["decision"]["sha256"],
        "decision": "Apply the prior frozen stop rule: stop promoting the current count-based positive-revisit policy. Keep raw-change survey and simple negative-gate policies as reference controls; no new method selected.",
        "evidence": {"all_revisits_below_survey_only_primary_cases": 6,
            "count_negative_below_survey_only_in_all3_held_tie_rules": True,
            "raw_gate_random_primary_gain": next(r["delta_found"] for r in primary if r["comparison"] == "prior_gate_vs_prior_survey" and r["fit_scope"] == "development_fit" and r["tie_rule"] == "fixed_random"),
            "raw_gate_helps_one_detectable_event_and_harms_the_other": True},
        "interpretation": "Raw-gate gain is small and event-dependent; fitted count-policy loss already exists without upgrades. Positive-only local findings are not net budget gain. Cost1 gains do not replace prespecified cost2 primary.",
        "next_focus": "Diagnose cross-event utility ranking and revisit valuation separately before designing another policy. The current summary-gap identity is completed descriptive evidence; a new diagnostic experiment still needs a prospective protocol.",
        "new_experiment_started": False, "new_model_fit_enabled": False, "new_controller_training_enabled": False,
        "validation_or_confirmation_access_enabled": False, "paid_gpu_needed": False,
        "new_method_selected": False, "independent_efficacy_claim": False,
        "limits": ["This stops this fitted policy, not all possible positive revisits.",
            "Held development events were previously inspected; raw policies are identical across fit scopes by construction, not a second validation.",
            "Mexico has zero cached detected targets; all scenes retained.",
            "Training-event exposure and full-image geographic independence remain unresolved."]})
    print(json.dumps({"gap_identities_checked": len(macros) + len(events),
        "positive_policy_stopped": stop_positive, "primary_cases_vs_survey": len(all_vs_survey)}))


if __name__ == "__main__":
    main()
