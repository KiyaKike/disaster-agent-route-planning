"""Replay budget/quality/eligibility and re-score all development trajectories.

Descriptive paired scene effects, with event means and cost sensitivity; never
treat repeated seeds as independent disasters or select a new main method.
"""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

from build_xbd_mini import atomic_json,sha
from cross_cell_count_smoke import quality
from plan_search_v2 import freeze
from run_development_budget_controls import (CELLS_INDEX,THRESHOLDS,SUMMARY_FIELDS,private_evaluator,score_trace)


ACTION_FIELDS={"cell_id","kind","mode","declared_target","cost","cumulative_cost","utility"}


def audit_actions(record,observations):
    states={};spent=0;counts=Counter()
    policy=record["policy"]
    for action in record["trace"]:
        if set(action)!=ACTION_FIELDS:raise ValueError("Private or missing saved action field")
        cid=action["cell_id"]
        if cid not in CELLS_INDEX:raise ValueError("Unknown grid cell")
        old=states.get(cid)
        if old is None:
            if action["kind"]!="survey":raise ValueError("Native shortcut before survey")
            cost=1;mode=quality(record["scenario"],record["quality_seed"],record["sample_id"],cid)
        elif old[0]=="blurred":
            if action["kind"]!="upgrade":raise ValueError("Repeated survey")
            if policy in {"random_survey","coverage_survey","prior_survey","count_survey","usable_coverage","usable_prior"}:
                raise ValueError("Survey-only ablation upgraded")
            if old[1] and policy in {"count_negative","prior_gate"}:raise ValueError("Negative-only ablation upgraded positive")
            cost=record["upgrade_cost"];mode="native"
        else:raise ValueError("Repeated native observation")
        if cost!=action["cost"] or mode!=action["mode"]:raise ValueError("Changed cost or hidden quality map")
        if action["declared_target"] is not observations[cid,mode]:raise ValueError("Changed frozen paid outcome")
        spent+=cost
        if spent!=action["cumulative_cost"] or spent>record["budget"]:raise ValueError("Budget violation")
        counts["actions"]+=1
        counts["upgrades"]+=action["kind"]=="upgrade"
        counts["positive_upgrades"]+=bool(old and old[1] and action["kind"]=="upgrade")
        states[cid]=(mode,action["declared_target"])
    return counts


def paired_effects(scene_means):
    by_config=defaultdict(dict)
    for r in scene_means:
        key=tuple(r[k] for k in ["fit_scope","scenario","upgrade_cost","budget","threshold","sample_id","event"])
        by_config[key][r["policy"]]=r
    comparisons=[("count_all","count_negative"),("count_all","count_survey"),("count_all","presence"),
        ("count_all","prior_survey"),("count_all","coverage_survey"),("count_all","random_survey"),
        ("count_all","random_upgrade"),("usable_prior","prior_survey"),("usable_coverage","coverage_survey"),
        ("count_usable","count_all")]
    rows=[];event_groups=defaultdict(list)
    for config,policies in by_config.items():
        for proposed,baseline in comparisons:
            if proposed not in policies or baseline not in policies:continue
            a,b=policies[proposed],policies[baseline]
            record={**dict(zip(["fit_scope","scenario","upgrade_cost","budget","threshold","sample_id","event"],config)),
                "comparison":proposed+"_vs_"+baseline,"candidate_found":a["distinct_found"],"baseline_found":b["distinct_found"],
                "delta_found":a["distinct_found"]-b["distinct_found"],"delta_fp_cells":a["false_positive_cells"]-b["false_positive_cells"],
                "candidate_positive_upgrades":a["positive_upgrades"],
                "candidate_new_from_positive_upgrades":a["new_buildings_from_positive_upgrades"]}
            rows.append(record)
            key=tuple(record[k] for k in ["fit_scope","scenario","upgrade_cost","budget","threshold","comparison","event"])
            event_groups[key].append(record)
    event_rows=[];macro=defaultdict(list)
    for key,group in sorted(event_groups.items()):
        record={**dict(zip(["fit_scope","scenario","upgrade_cost","budget","threshold","comparison","event"],key)),
            **{k:sum(r[k] for r in group)/len(group) for k in ["candidate_found","baseline_found","delta_found","delta_fp_cells",
                "candidate_positive_upgrades","candidate_new_from_positive_upgrades"]},
            "scenes":len(group),"positive_scene_effects":sum(r["delta_found"]>1e-12 for r in group),
            "negative_scene_effects":sum(r["delta_found"]< -1e-12 for r in group),
            "zero_scene_effects":sum(abs(r["delta_found"])<=1e-12 for r in group)}
        event_rows.append(record);macro[key[:-1]].append(record)
    macro_rows=[]
    for key,group in sorted(macro.items()):
        record={**dict(zip(["fit_scope","scenario","upgrade_cost","budget","threshold","comparison"],key)),
            **{k:sum(r[k] for r in group)/len(group) for k in ["candidate_found","baseline_found","delta_found","delta_fp_cells"]},
            "positive_events":sum(r["delta_found"]>1e-12 for r in group),"events":len(group)}
        record["relative_delta"]=record["delta_found"]/record["baseline_found"] if record["baseline_found"] else None
        macro_rows.append(record)
    return {"paired_scene_effects":rows,"event_effects":event_rows,"event_macro_effects":macro_rows}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["run","perception","fit"]:parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.run/"report.json").read_text());protocol=json.loads((args.run/"protocol.json").read_text())
    if protocol["runner_sha256"]!=sha(Path(__file__).with_name("run_development_budget_controls.py").read_bytes()):raise ValueError("Runner changed")
    for name,digest in protocol["helper_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Helper changed")
    for name,key in [("trajectories.jsonl.gz","trajectory_sha256"),("metrics.jsonl.gz","metrics_sha256")]:
        if sha((args.run/name).read_bytes())!=report[key]:raise ValueError("Run output changed")
    obs=defaultdict(dict)
    for r in json.loads((args.perception/"observations.json").read_text()):obs[r["sample_id"]][r["cell_id"],r["mode"]]=r["declared_target"]
    rows=defaultdict(list)
    for r in json.loads((args.perception/"association_report.json").read_text())["rows"]:rows[r["sample_id"]].append(r)
    evaluators={sid:private_evaluator(values) for sid,values in rows.items()}
    counts=Counter();scene_counts=Counter();seen=set()
    with gzip.open(args.run/"trajectories.jsonl.gz","rt") as traces,gzip.open(args.run/"metrics.jsonl.gz","rt") as metrics:
        for line in traces:
            record=json.loads(line);sid=record["sample_id"]
            if sid not in obs:raise ValueError("Nondevelopment scene")
            key=tuple(record[k] for k in ["sample_id","fit_scope","scenario","upgrade_cost","budget","quality_seed","order_seed","policy"])
            if key in seen:raise ValueError("Duplicate trajectory configuration")
            seen.add(key)
            counts.update(audit_actions(record,obs[sid]));counts["trajectories"]+=1;scene_counts[sid]+=1
            evaluator,targets=evaluators[sid]
            rescored=score_trace(record["trace"],evaluator,targets)
            for expected in rescored:
                line=metrics.readline()
                if not line:raise ValueError("Missing metric row")
                row=json.loads(line)
                if any(row[k]!=record[k] for k in ["sample_id","fit_scope","scenario","upgrade_cost","budget","quality_seed","order_seed","policy"]):raise ValueError("Metric config mismatch")
                if any(row[k]!=value for k,value in expected.items()):raise ValueError("Evaluator replay differs")
                counts["threshold_scores"]+=1
        if metrics.readline():raise ValueError("Unexpected extra metrics")
    if counts["trajectories"]!=293760 or counts["threshold_scores"]!=881280 or set(scene_counts.values())!={4080}:
        raise ValueError("Incomplete fixed condition grid")
    models=json.loads((args.run/"models.json").read_text())
    for name,model in models.items():
        if name.startswith("held_") and any(sid.rsplit("_",1)[0]==name.removeprefix("held_") for sid in model["fit_ids"]):
            raise ValueError("Held event included in parameter fit")
    previous=json.loads((args.fit/"completion_receipt.json").read_text())
    if any(sha((args.fit/name).read_bytes())!=digest for name,digest in previous["files"].items()):raise ValueError("Previous output changed")
    effects=paired_effects(json.loads((args.run/"scene_means.json").read_text()))
    atomic_json(args.run/"paired_effects.json",effects)
    primary=[r for r in effects["event_macro_effects"] if r["scenario"]=="mixed_survey" and r["upgrade_cost"]==2 and r["budget"]==32 and r["threshold"]==.1]
    result={"stage":"DEVELOPMENT72_BUDGET_AND_SCORE_REPLAY_PASSED","counts":dict(counts),
        "fixed_configs_per_scene":4080,"all_action_fields_whitelisted":True,"all_budgets_quality_maps_and_eligibility_checked":True,
        "all_trajectory_threshold_scores_replayed":True,"event_holdout_fit_ids_checked":True,
        "previous_development_fit_outputs_unchanged":True,"primary_paired_effects":primary,
        "validation_or_confirmation_scored":False,"independent_efficacy_claim":False,"new_method_selected":False,
        "statistics":"Paired scene effects after averaging seeds, then event means. No seeds/cells counted as independent disasters and no inferential confidence claims.",
        "runner_sha256":sha(Path(__file__).read_bytes())}
    atomic_json(args.run/"audit.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
