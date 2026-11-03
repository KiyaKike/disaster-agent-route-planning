"""Fit fixed paired/count baselines on 72 development scenes; no policy scoring.

Leave-one-development-event-out count prediction diagnoses transfer only. It is
not independent validation, and contains no global distinct-building planner.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from build_xbd_mini import atomic_json,sha
from paired_count_baseline import fit_local_count,action_score
from plan_search_v2 import freeze


def count_prediction_diagnostic(model,rows,scores):
    pairs=defaultdict(dict)
    for row in rows:pairs[row["sample_id"],row["cell_id"]][row["mode"]]=row
    records=defaultdict(list);unsupported=0;unknown=0
    for (sid,cid),pair in pairs.items():
        b,n=pair["blurred"],pair["native"]
        if n["target"] is None:
            unknown+=1;continue
        bm,nm=set(b["matched_target_uids"]),set(n["matched_target_uids"])
        for mode,ids in [("native",nm),("blurred",bm)]:
            predicted=action_score(model,scores[sid][cid],"survey",scenario=mode+"_survey")
            records[mode].append((predicted,len(ids)))
        predicted=action_score(model,scores[sid][cid],"upgrade",blurred_positive=b["predicted"],cost=1)
        if predicted is None:unsupported+=1
        else:records["upgrade"].append((predicted,len(nm-bm)))
    result={}
    for mode,values in records.items():
        prediction,truth=np.asarray(values,dtype=float).T
        result[mode]={"cells":len(values),"mean_predicted_count":float(prediction.mean()),"mean_observed_count":float(truth.mean()),
            "mean_absolute_error":float(np.abs(prediction-truth).mean()),"mean_squared_error":float(((prediction-truth)**2).mean())}
    return {"count_prediction":result,"unsupported_upgrade_conditions":unsupported,"unknown_pairs_excluded":unknown}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["perception","data","design","out"]:parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths={"association":args.perception/"association_report.json","cues":args.perception/"public_change_scores.json",
           "manifest":args.data/"manifest.json","contract":args.design/"experiment_contract_v2_2.json"}
    protocol={"version":"development72_paired_count_fit_v1","sources":{k:sha(p.read_bytes()) for k,p in paths.items()},
        "runner_sha256":sha(Path(__file__).read_bytes()),"fit_helper_sha256":sha(Path(__file__).with_name("paired_count_baseline.py").read_bytes()),
        "joint_pseudocount_per_outcome":.5,"count_shrinkage_pseudo_cells":8,"public_cue_bins":4,
        "fit":"Same frozen paired/count definitions as six-scene smoke. Unknown cell pairs excluded, B+ upgrade allowed, no independence multiplication.",
        "diagnostic":"Hold out each development event, refit using the other two events, predict local survey/upgrade counts. No threshold/parameter selection from these diagnostics.",
        "cross_cell_utility":"Not solved. Boundary-only discount retained as failed ablation and not refit.",
        "validation_or_confirmation_access":False,"new_controller_training":False,"policy_scores":False}
    freeze(args.out/"fit_protocol.json",protocol)
    association=json.loads(paths["association"].read_text());scores=json.loads(paths["cues"].read_text())
    manifest=json.loads(paths["manifest"].read_text());selected=json.loads((args.design/"next_development_batch.json").read_text())
    expected={r["sample_id"] for r in selected}
    if len(expected)!=72 or {s["sample_id"] for s in manifest["samples"]}!=expected or set(scores)!=expected:
        raise ValueError("Fit requires the exact 72 development scenes")
    rows=association["rows"]
    if association["stage"]!="DEVELOPMENT72_SENSOR_QA" or len(rows)!=9216 or {r["sample_id"] for r in rows}!=expected:
        raise ValueError("Unexpected perception stage or identities")
    model=fit_local_count(rows,scores)
    model["stage"]="DEVELOPMENT72_AGGREGATE_BASELINE_FIT_NOT_VALIDATED"
    model["limits"][2]="Fit on 72 development scenes; event-held-out count diagnostics do not constitute independent policy validation."
    model["source_hashes"]=protocol["sources"]
    atomic_json(args.out/"component_parameters.json",model)
    events={r["sample_id"]:r["event"] for r in selected}
    transfer={}
    for event in sorted(set(events.values())):
        training=[r for r in rows if events[r["sample_id"]]!=event]
        held=[r for r in rows if events[r["sample_id"]]==event]
        fitted=fit_local_count(training,scores)
        transfer[event]={"training_events":sorted(set(events.values())-{event}),"training_scenes":48,"held_out_scenes":24,
                         "training_paired_truth_support":{t:sum(c.values()) for t,c in fitted["paired_sensor"]["counts"].items()},
                         "training_upgrade_support":fitted["global_upgrade"],
                         **count_prediction_diagnostic(fitted,held,scores)}
    result={"stage":model["stage"],"source_scenes":model["source_scenes"],"known_pairs":model["paired_known_cells"],
        "unknown_pairs_excluded":model["unknown_pairs_excluded"],"global_survey_mean_count":model["global_survey_mean_count"],
        "global_upgrade":model["global_upgrade"],"paired_sensor":model["paired_sensor"],
        "development_fit_diagnostic":count_prediction_diagnostic(model,rows,scores),
        "leave_one_development_event_out":transfer,
        "independent_efficacy_claim":False,"search_policy_evaluated":False,"full_216_cohort_admitted":False,
        "new_method_selected":False,"cross_cell_distinct_utility_solved":False}
    atomic_json(args.out/"report.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["paired_sensor","leave_one_development_event_out","development_fit_diagnostic"]}))


if __name__ == "__main__":main()
