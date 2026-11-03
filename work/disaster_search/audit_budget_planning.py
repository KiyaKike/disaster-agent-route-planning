"""Budget replay, previous-greedy equivalence and paired planning effects."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

from audit_development_budget_controls import ACTION_FIELDS,audit_actions
from build_xbd_mini import atomic_json,sha
from run_development_budget_controls import private_evaluator,score_trace


def trajectory_key(r):
    return tuple(r[k] for k in ["sample_id","fit_scope","scenario","upgrade_cost","budget","quality_seed","order_seed"])


def action_signature(trace):
    return [(r["cell_id"],r["kind"],r["mode"],r["declared_target"],r["cost"],r["cumulative_cost"]) for r in trace]


def effects(scene_means,old_means):
    groups=defaultdict(dict)
    keys=["fit_scope","scenario","upgrade_cost","budget","threshold","sample_id","event"]
    for r in old_means:groups[tuple(r[k] for k in keys)][r["policy"]]=r
    for r in scene_means:groups[tuple(r[k] for k in keys)][r["policy"]]=r
    comparisons=[(a,b) for a in ["two_step","batch_tail"] for b in ["proxy_greedy","count_negative","count_survey","presence","prior_survey","prior_gate","random_survey","coverage_survey"]]
    scene_rows=[];events=defaultdict(list)
    for key,policies in groups.items():
        for a,b in comparisons:
            if a not in policies or b not in policies:continue
            ar,br=policies[a],policies[b]
            row={**dict(zip(keys,key)),"comparison":a+"_vs_"+b,"candidate_found":ar["distinct_found"],"baseline_found":br["distinct_found"],
                "delta_found":ar["distinct_found"]-br["distinct_found"],"delta_fp":ar["false_positive_cells"]-br["false_positive_cells"],
                "delta_upgrades":ar["upgrades"]-br["upgrades"],"delta_positive_upgrades":ar["positive_upgrades"]-br["positive_upgrades"]}
            scene_rows.append(row);events[key[:-2]+(row["comparison"],key[-1])].append(row)
    event_rows=[];macros=defaultdict(list)
    fields=["candidate_found","baseline_found","delta_found","delta_fp","delta_upgrades","delta_positive_upgrades"]
    for key,rows in sorted(events.items()):
        row={**dict(zip(keys[:5]+["comparison","event"],key)),**{f:sum(r[f] for r in rows)/len(rows) for f in fields},
            "scenes":len(rows),"scene_positive":sum(r["delta_found"]>1e-12 for r in rows),
            "scene_negative":sum(r["delta_found"]< -1e-12 for r in rows),"scene_zero":sum(abs(r["delta_found"])<=1e-12 for r in rows)}
        event_rows.append(row);macros[key[:-1]].append(row)
    macro_rows=[]
    for key,rows in sorted(macros.items()):
        row={**dict(zip(keys[:5]+["comparison"],key)),**{f:sum(r[f] for r in rows)/len(rows) for f in fields},
            "positive_events":sum(r["delta_found"]>1e-12 for r in rows),"events":len(rows)}
        row["relative_delta"]=row["delta_found"]/row["baseline_found"] if row["baseline_found"] else None
        macro_rows.append(row)
    return {"scene_effects":scene_rows,"event_effects":event_rows,"event_macro_effects":macro_rows}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["run","previous","perception"]:parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();report=json.loads((args.run/"report.json").read_text());protocol=json.loads((args.run/"protocol.json").read_text())
    if protocol["runner_sha256"]!=sha(Path(__file__).with_name("run_budget_planning.py").read_bytes()):raise ValueError("Runner changed")
    for name,digest in protocol["helper_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Helper changed")
    for name,digest in protocol["primary_source_sha256"].items():
        if sha((args.run/"sources"/name).read_bytes())!=digest:raise ValueError("Primary source changed")
    for name,key in [("trajectories.jsonl.gz","trajectory_sha256"),("metrics.jsonl.gz","metrics_sha256")]:
        if sha((args.run/name).read_bytes())!=report[key]:raise ValueError("Output changed")
    previous=json.loads((args.previous/"completion_receipt.json").read_text())
    for name,digest in previous["files"].items():
        if sha((args.previous/name).read_bytes())!=digest:raise ValueError("Previous experiment output changed")
    old={}
    for line in gzip.open(args.previous/"trajectories.jsonl.gz","rt"):
        row=json.loads(line)
        if row["policy"]=="count_all":old[trajectory_key(row)]=action_signature(row["trace"])
    obs=defaultdict(dict);rows=defaultdict(list)
    for r in json.loads((args.perception/"observations.json").read_text()):obs[r["sample_id"]][r["cell_id"],r["mode"]]=r["declared_target"]
    for r in json.loads((args.perception/"association_report.json").read_text())["rows"]:rows[r["sample_id"]].append(r)
    evaluators={sid:private_evaluator(values) for sid,values in rows.items()}
    totals=Counter();configs=set();scene_counts=Counter();greedy={};planning={}
    with gzip.open(args.run/"trajectories.jsonl.gz","rt") as traces,gzip.open(args.run/"metrics.jsonl.gz","rt") as metrics:
        for line in traces:
            row=json.loads(line);sid=row["sample_id"];key=trajectory_key(row)
            if key+(row["policy"],) in configs:raise ValueError("Duplicate config")
            configs.add(key+(row["policy"],));scene_counts[sid]+=1
            if any(set(a)!=ACTION_FIELDS|{"greedy_utility"} for a in row["trace"]):raise ValueError("Extra/private action fields")
            stripped={**row,"trace":[{k:v for k,v in a.items() if k in ACTION_FIELDS} for a in row["trace"]]}
            totals.update(audit_actions(stripped,obs[sid]));totals["trajectories"]+=1
            signature=action_signature(row["trace"])
            if row["policy"]=="proxy_greedy":
                if signature!=old[key]:raise ValueError("New greedy differs from original count_all")
                greedy[key]=signature;totals["original_greedy_trajectories_reproduced"]+=1
            else:planning[key+(row["policy"],)]=signature
            for expected in score_trace(row["trace"],*evaluators[sid]):
                line=metrics.readline()
                if not line:raise ValueError("Missing score")
                saved=json.loads(line)
                if trajectory_key(saved)!=key or saved["policy"]!=row["policy"]:raise ValueError("Mismatched score config")
                if any(saved[k]!=v for k,v in expected.items()):raise ValueError("Score replay differs")
                totals["threshold_scores"]+=1
        if metrics.readline():raise ValueError("Extra metric")
    if totals["trajectories"]!=30240 or totals["threshold_scores"]!=90720 or set(scene_counts.values())!={420}:
        raise ValueError("Incomplete fixed grid")
    changes=Counter();native_changed=0
    for key,signature in planning.items():
        changed=signature!=greedy[key[:-1]]
        changes[key[-1]]+=changed
        if key[2]=="native_survey":native_changed+=changed
    if native_changed:raise ValueError("Native-only symmetry/tie invariant failed")
    paired=effects(json.loads((args.run/"scene_means.json").read_text()),json.loads((args.previous/"scene_means.json").read_text()))
    atomic_json(args.run/"paired_effects.json",paired)
    primary=[r for r in paired["event_macro_effects"] if r["scenario"]=="mixed_survey" and r["upgrade_cost"]==2 and r["budget"]==32 and r["threshold"]==.1]
    result={"stage":"PLANNING_BUDGET_AND_SCORE_REPLAY_PASSED","counts":dict(totals),"original_greedy_all_10080_configs_reproduced":True,
        "changed_trajectories_vs_greedy":dict(changes),"native_only_changed":native_changed,
        "all_budgets_eligibility_paid_outcomes_and_quality_maps_checked":True,"all_scores_replayed":True,
        "previous_outputs_preserved":True,"primary_effects":primary,"new_method_selected":False,
        "independent_efficacy_claim":False,"validation_or_confirmation_scored":False,"faithful_ens_or_mfens_reproduction":False,
        "runner_sha256":sha(Path(__file__).read_bytes())}
    atomic_json(args.run/"audit.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
