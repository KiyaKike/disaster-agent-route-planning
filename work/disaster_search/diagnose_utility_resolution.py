"""Post-hoc public-feature diagnostic; no labels, future observations or policy."""
import json
from pathlib import Path

from build_xbd_mini import atomic_json,sha
from cross_cell_count_smoke import CELLS
from run_budget_planning import planning_inputs


def diagnose(models,features):
    records=[]
    for sid,feature in features.items():
        event=sid.rsplit("_",1)[0]
        for scope,model in [("development_fit",models["development_fit"]),("event_holdout",models["held_"+event])]:
            inputs=planning_inputs(feature,model,"mixed_survey",2)
            scores=[v[0] for v in inputs["greedy_utility"]];best=max(scores)
            tied=[i for i,x in enumerate(scores) if x==best]
            raw_best=max(CELLS,key=lambda c:feature[c]["change"])
            records.append({"sample_id":sid,"event":event,"fit_scope":scope,"distinct_survey_utility_values":len(set(scores)),
                "top_score_tie_cells":len(tied),"raw_change_distinct_values":len({f["change"] for f in feature.values()}),
                "count_row_tie_choice":CELLS[tied[0]],"raw_change_choice":raw_best,"choices_differ":CELLS[tied[0]]!=raw_best})
    summaries={}
    for scope in ["development_fit","event_holdout"]:
        group=[r for r in records if r["fit_scope"]==scope]
        summaries[scope]={"scenes":len(group),
            "mean_utility_distinct_values":sum(r["distinct_survey_utility_values"] for r in group)/len(group),
            "mean_top_tie_cells":sum(r["top_score_tie_cells"] for r in group)/len(group),
            "mean_raw_change_distinct_values":sum(r["raw_change_distinct_values"] for r in group)/len(group),
            "initial_choice_differs_from_raw_change_scenes":sum(r["choices_differ"] for r in group)}
    return {"stage":"POST_HOC_PUBLIC_UTILITY_RESOLUTION_DIAGNOSTIC_ONLY","not_a_policy_result":True,
        "labels_or_future_observations_read":False,"per_scene":records,"by_scope":summaries,
        "interpretation":"Four-bin utility creates ties resolved by row order. This identifies lost score resolution, not proof a different tie rule improves discovery. No policy/parameters fitted."}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    model_path=args.previous/"models.json";feature_path=args.previous/"public_features.json"
    result=diagnose(json.loads(model_path.read_text()),json.loads(feature_path.read_text()))
    result["source_sha256"]={"models":sha(model_path.read_bytes()),"features":sha(feature_path.read_bytes()),"runner":sha(Path(__file__).read_bytes())}
    atomic_json(args.out/"utility_resolution.json",result);print(json.dumps(result["by_scope"]))


if __name__=="__main__":main()
