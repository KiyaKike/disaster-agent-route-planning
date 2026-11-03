"""Post-run action/privacy/budget audit and failed-mechanism diagnosis."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from cross_cell_count_smoke import (CELLS, PublicState, discount_factor, matched,
    quality, SCENARIOS)


def audit_trace(record):
    state, spent = PublicState(), 0
    allowed = {"cell_id","kind","cost","base_utility","discount","utility",
               "boundary_contact","mode","declared_target","edges","cumulative_cost"}
    counts = Counter()
    for a in record["trace"]:
        if set(a) != allowed:
            raise ValueError("Trajectory has extra/private or missing action fields")
        eligible = state.candidates(record["budget"]-spent,record["upgrade_cost"])
        if (a["cell_id"],a["kind"],a["cost"]) not in eligible:
            raise ValueError("Ineligible action or overspent budget")
        mode = "native" if a["kind"] == "upgrade" else quality(record["scenario"],record["quality_seed"],record["sample_id"],a["cell_id"])
        if a["mode"] != mode or a["boundary_contact"] != state.boundary_contact(a["cell_id"]):
            raise ValueError("Future quality mismatch or edge feature not reconstructible from paid prefix")
        spent += a["cost"]
        if spent != a["cumulative_cost"] or spent > record["budget"]:
            raise ValueError("Incorrect budget accounting")
        counts["actions"] += 1
        counts["upgrades"] += a["kind"] == "upgrade"
        counts["positive_blur_upgrades"] += bool(a["kind"] == "upgrade" and state.latest[a["cell_id"]]["declared_target"])
        state.observe(a)
    return counts


def candidate_exposure(record, model):
    state, affected_candidates, steps = PublicState(), 0, 0
    for a in record["trace"]:
        for cell,kind,cost in state.candidates(record["budget"]-(a["cumulative_cost"]-a["cost"]),record["upgrade_cost"]):
            old = state.latest.get(cell)
            contact = state.boundary_contact(cell)
            positive = old["declared_target"] if old else None
            affected_candidates += discount_factor(model,kind,positive,contact) < 1
        steps += 1
        state.observe(a)
    return {"candidate_action_occurrences_with_discount":affected_candidates,"decision_steps":steps}


def diagnostic_actions(trace, lookup, sid):
    found, rows = set(), []
    for a in trace:
        ids=matched(lookup[sid,a["cell_id"],a["mode"]])
        rows.append({"cell_id":a["cell_id"],"kind":a["kind"],"mode":a["mode"],
                     "cost":a["cumulative_cost"],"new_buildings_evaluator_only":len(ids-found),
                     "already_found_matches_evaluator_only":len(ids & found)})
        found |= ids
    return rows,len(found)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--perception",type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.run/"report.json").read_text())
    for name,field in [("trajectories.jsonl","trajectory_sha256"),("metrics.jsonl","metrics_sha256")]:
        if sha((args.run/name).read_bytes()) != report[field]: raise ValueError("Output changed")
    discount=json.loads((args.run/"discount_parameters.json").read_text())
    totals=Counter(); pairs=defaultdict(dict); exposure=Counter()
    for line in (args.run/"trajectories.jsonl").open():
        r=json.loads(line)
        totals.update(audit_trace(r)); totals["trajectories"] += 1
        if r["policy"] == "boundary_count": exposure.update(candidate_exposure(r,discount))
        if r["policy"] in {"local_count","boundary_count"}:
            key=(r["sample_id"],r["scenario"],r["upgrade_cost"],r["quality_seed"],r["budget"])
            pairs[key][r["policy"]]=r
    if totals["trajectories"] != report["trajectories"]: raise ValueError("Incomplete trajectory file")
    association=json.loads((args.perception/"association_report.json").read_text())
    lookup={(r["sample_id"],r["cell_id"],r["mode"]):r for r in association["rows"]}
    changed, examples = 0,[]
    for key,pair in pairs.items():
        local,boundary=pair["local_count"],pair["boundary_count"]
        a,b=local["trace"],boundary["trace"]
        divergent=next((i for i,(x,y) in enumerate(zip(a,b)) if (x["cell_id"],x["kind"]) != (y["cell_id"],y["kind"])),None)
        if divergent is None: continue
        changed += 1
        if key[1] != "mixed_survey" or key[2] != 2 or key[4] != 32: continue
        ad,an=diagnostic_actions(a,lookup,key[0]); bd,bn=diagnostic_actions(b,lookup,key[0])
        prefix=PublicState()
        for action in b[:divergent]: prefix.observe(action)
        rejected=a[divergent]
        contact=prefix.boundary_contact(rejected["cell_id"])
        old=prefix.latest.get(rejected["cell_id"])
        factor=discount_factor(discount,rejected["kind"],old["declared_target"] if old else None,contact)
        examples.append({"sample_id":key[0],"quality_seed":key[3],"first_divergence_action_number":divergent+1,
            "common_prefix_cost":a[divergent-1]["cumulative_cost"] if divergent else 0,
            "local_selected_cell":rejected["cell_id"],"boundary_selected_cell":b[divergent]["cell_id"],
            "local_candidate_boundary_contact":contact,"local_candidate_discount_if_boundary_policy":factor,
            "local_distinct_found":an,"boundary_distinct_found":bn,
            "local_evaluator_only_actions":ad,"boundary_evaluator_only_actions":bd})
    if changed != report["paired_configs_with_changed_action_order"]: raise ValueError("Pair count differs")
    count_params=Path("output/disaster_search_calibration_v2/calibration_parameters.json")
    old_sha="db3c3a77647b18009e2e192e187084c5148f39c24fb952a0f7b40420ea3fc600"
    if sha(count_params.read_bytes()) != old_sha: raise ValueError("Legacy fitted parameters changed")
    rows={r["policy"]:r for r in report["primary_setting_diagnostic"]}
    delta=rows["boundary_count"]["distinct_found"]-rows["local_count"]["distinct_found"]
    result={"stage":"BUDGET_AND_PUBLIC_PREFIX_AUDIT_PASSED", "runner_sha256":sha(Path(__file__).read_bytes()),
        "all_action_fields_whitelisted":True,"all_budgets_and_quality_maps_verified":True,
        "boundary_features_reconstructed_from_paid_prefix":True,"counts":dict(totals),
        "boundary_candidate_exposure":dict(exposure),"changed_paired_action_orders":changed,
        "chosen_discounted_actions":report["discounted_actions_across_paired_configs"],
        "exposure_interpretation":"Chosen actions can all have factor 1 because discounted candidates are deferred; candidate exposure, action order and evaluator utility must be checked separately.",
        "primary_event_macro_absolute_delta":delta,
        "primary_relative_delta":delta/rows["local_count"]["distinct_found"],
        "decision":"DO_NOT_PROMOTE_BOUNDARY_DISCOUNT. Keep as a failed engineering ablation; no tuning on validation/confirmation.",
        "legacy_parameters_preserved":True,"independent_efficacy_claim":False,
        "validation_or_confirmation_scored":False,"source_inputs_preserved":all(report["inputs_preserved"].values())}
    atomic_json(args.run/"action_budget_audit.json",result)
    atomic_json(args.run/"failure_examples_evaluator_only.json",examples)
    print(json.dumps(result))


if __name__ == "__main__": main()
