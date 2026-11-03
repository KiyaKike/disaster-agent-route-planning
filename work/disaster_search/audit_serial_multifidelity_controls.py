"""Full decision/score replay and exact historical one-stage reproduction."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from audit_development_budget_controls import audit_actions
from build_xbd_mini import atomic_json, sha
from run_rank_calibration_control import assert_fit_ids
from run_budget_planning import planning_inputs
from run_development_budget_controls import (PaidBinaryEnvironment, QUALITY_SEEDS, ORDER_SEEDS,
    private_evaluator, score_trace, SUMMARY_FIELDS)
from run_utility_tie_controls import (TIES, CONFIG_FIELDS, aggregate_ties, summarize_ties,
    tie_ranks, verify_receipt)
from serial_multifidelity_baseline import POLICIES, compress_inputs, run_episode

KEYS = ["fit_scope", "scenario", "upgrade_cost", "budget", "threshold", "tie_rule"]
COMPARISONS = [("serial_two_stage", "serial_one_stage"), ("serial_two_stage", "prior_survey"),
    ("serial_two_stage", "prior_gate"), ("serial_two_stage", "count_survey"),
    ("serial_two_stage", "count_negative"), ("serial_two_stage", "count_all")]
FIELDS = ["candidate_found", "baseline_found", "delta_found", "delta_fp", "delta_upgrades"]


def primary_batch_rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["policy"] == "batch_tail" and (row["scenario"], row["upgrade_cost"], row["budget"]) == ("mixed_survey", 2, 32):
                yield row


def paired_effects(rows):
    groups = defaultdict(dict)
    for r in rows:
        key = tuple(r[k] for k in KEYS + ["sample_id", "event"])
        if r["policy"] in groups[key]:
            raise ValueError("Duplicate scene mean")
        groups[key][r["policy"]] = r
    scenes, events = [], defaultdict(list)
    for key, policies in groups.items():
        for a,b in COMPARISONS:
            if a not in policies or b not in policies:
                continue
            ar,br = policies[a],policies[b]
            row = {**dict(zip(KEYS+["sample_id","event"],key)),"comparison":a+"_vs_"+b,
                "candidate_found":ar["distinct_found"],"baseline_found":br["distinct_found"],
                "delta_found":ar["distinct_found"]-br["distinct_found"],
                "delta_fp":ar["false_positive_cells"]-br["false_positive_cells"],
                "delta_upgrades":ar["upgrades"]-br["upgrades"]}
            scenes.append(row)
            events[key[:6]+(row["comparison"],key[-1])].append(row)
    event_rows, macros = [],defaultdict(list)
    for key, part in sorted(events.items()):
        row = {**dict(zip(KEYS+["comparison","event"],key)),
            **{f:sum(r[f] for r in part)/len(part) for f in FIELDS},"scenes":len(part),
            "positive_scenes":sum(r["delta_found"]>1e-12 for r in part),
            "negative_scenes":sum(r["delta_found"]< -1e-12 for r in part)}
        event_rows.append(row)
        macros[key[:-1]].append(row)
    macro_rows=[]
    for key,part in sorted(macros.items()):
        row={**dict(zip(KEYS+["comparison"],key)),**{f:sum(r[f] for r in part)/len(part) for f in FIELDS},
            "events":len(part),"positive_events":sum(r["delta_found"]>1e-12 for r in part),
            "negative_events":sum(r["delta_found"]< -1e-12 for r in part)}
        row["relative_delta"]=row["delta_found"]/row["baseline_found"] if row["baseline_found"] else None
        macro_rows.append(row)
    return {"scene_effects":scenes,"event_effects":event_rows,"event_macro_effects":macro_rows}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.run/"protocol.json").read_text())
    report=json.loads((args.run/"report.json").read_text())
    paths={k:Path(r["path"]) for k,r in protocol["sources"].items()}
    for k,p in paths.items():
        if sha(p.read_bytes())!=protocol["sources"][k]["sha256"]:
            raise ValueError("Source artifact changed: "+k)
    for n,d in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(n).read_bytes())!=d:
            raise ValueError("Frozen source code changed: "+n)
    for key in ["model_receipt","previous_receipt","ties_receipt","simple_receipt"]:
        verify_receipt(paths[key].parent)
    for name,key in [("trajectories.jsonl.gz","trajectory_sha256"),("metrics.jsonl.gz","metrics_sha256")]:
        if sha((args.run/name).read_bytes())!=report[key]:raise ValueError("Output changed")
    selected=json.loads(paths["selection"].read_text())
    event_by_id={r["sample_id"]:r["event"] for r in selected}
    features=json.loads(paths["features"].read_text())
    models=json.loads(paths["models"].read_text())
    assert_fit_ids(models,selected)
    by_obs,obs,by_rows=defaultdict(list),defaultdict(dict),defaultdict(list)
    observations=json.loads(paths["observations"].read_text())
    for r in observations:
        if sha(Path(r["cache_file"]).read_bytes())!=r["cache_sha256"]:
            raise ValueError("Prediction mask changed")
        by_obs[r["sample_id"]].append(r)
        obs[r["sample_id"]][r["cell_id"],r["mode"]]=r["declared_target"]
    for r in json.loads(paths["association"].read_text())["rows"]:
        by_rows[r["sample_id"]].append(r)
    evaluators={s:private_evaluator(rows) for s,rows in by_rows.items()}
    old=primary_batch_rows(paths["old_batch_trajectories"])
    counts,scene_counts,seen=Counter(),Counter(),set()
    rebuilt,local,diagnostics=[],[],[]
    previous_sid=None
    planners={}
    with gzip.open(args.run/"trajectories.jsonl.gz","rt") as traces,gzip.open(args.run/"metrics.jsonl.gz","rt") as metrics:
        for line in traces:
            r=json.loads(line)
            sid=r["sample_id"]
            if sid not in event_by_id or r["event"]!=event_by_id[sid]:raise ValueError("Foreign scene/event")
            if sid!=previous_sid:
                if previous_sid is not None:
                    rebuilt.extend({"sample_id":previous_sid,"event":event_by_id[previous_sid],**row} for row in aggregate_ties(local))
                    for inputs,model,membership in planners.values():
                        for cached in [model.tail,model.after_surveys,model.future,model.q]:cached.cache_clear()
                    print(json.dumps({"stage":"SCENE_REPLAY_COMPLETE","sample_id":previous_sid,"trajectories":counts["trajectories"]}),flush=True)
                previous_sid,local,planners=sid,[],{}
            key=tuple(r[k] for k in CONFIG_FIELDS)
            if key in seen:raise ValueError("Repeated configuration")
            seen.add(key)
            if (r["scenario"],r["upgrade_cost"],r["budget"])!=("mixed_survey",2,32):raise ValueError("Unplanned config")
            if r["policy"] not in POLICIES or r["tie_rule"] not in TIES or r["quality_seed"] not in QUALITY_SEEDS:raise ValueError("Unplanned policy/seed/rule")
            if r["order_seed"] not in (ORDER_SEEDS if r["tie_rule"]=="fixed_random" else ORDER_SEEDS[:1]):raise ValueError("Unplanned order seed")
            if r["fit_scope"] not in ["development_fit","event_holdout"]:raise ValueError("Unplanned fit scope")
            counts.update(audit_actions(r,obs[sid]))
            if r["fit_scope"] not in planners:
                model_key="development_fit" if r["fit_scope"]=="development_fit" else "held_"+r["event"]
                inputs=planning_inputs(features[sid],models[model_key],"mixed_survey",2)
                model,membership=compress_inputs(inputs,2)
                planners[r["fit_scope"]]=inputs,model,membership
            inputs,model,membership=planners[r["fit_scope"]]
            env=PaidBinaryEnvironment(by_obs[sid],sid,"mixed_survey",r["quality_seed"],2)
            trace,stats=run_episode(env,inputs,32,2,tie_ranks(features[sid],r["tie_rule"],r["order_seed"]),r["policy"],model,membership)
            if trace!=r["trace"] or stats!=r["planning_diagnostics"]:raise ValueError("Full decision/diagnostic replay differs")
            counts["complete_decision_replays"]+=1
            diagnostics.append({**{k:v for k,v in r.items() if k not in ["trace","planning_diagnostics"]},**stats})
            if r["policy"]=="serial_one_stage":
                reference=next(old)
                if any(reference[k]!=r[k] for k in CONFIG_FIELDS if k!="policy"):raise ValueError("Historical config differs")
                if [{k:v for k,v in a.items() if k!="greedy_utility"} for a in reference["trace"]]!=r["trace"]:raise ValueError("Historical batch-tail trace not reproduced")
                if any(stats.values()):raise ValueError("One-stage control has two-stage diagnostics")
                counts["historical_one_stage_exact_trajectories"]+=1
            counts["selected_actions_with_positive_two_stage_gain"]+=stats["selected_actions_with_positive_two_stage_gain"]
            counts["trajectories"]+=1
            scene_counts[sid]+=1
            for expected in score_trace(r["trace"],*evaluators[sid]):
                saved=json.loads(next(metrics))
                if tuple(saved[k] for k in CONFIG_FIELDS)!=key or any(saved[k]!=v for k,v in expected.items()):raise ValueError("Full score replay differs")
                counts["threshold_scores"]+=1
                local.append(saved)
        if metrics.readline() or next(old,None) is not None:raise ValueError("Extra metrics or historical control rows")
    rebuilt.extend({"sample_id":previous_sid,"event":event_by_id[previous_sid],**r} for r in aggregate_ties(local))
    if counts["trajectories"]!=17280 or counts["threshold_scores"]!=51840 or counts["historical_one_stage_exact_trajectories"]!=8640:
        raise ValueError("Incomplete run")
    if set(scene_counts)!=set(event_by_id) or set(scene_counts.values())!={240}:raise ValueError("Incomplete per-scene grid")
    if rebuilt!=json.loads((args.run/"scene_means.json").read_text()):raise ValueError("Scene aggregation differs")
    if diagnostics!=json.loads((args.run/"planning_diagnostics.json").read_text()):raise ValueError("Diagnostic export differs")
    spatial={r["sample_id"]:r for r in json.loads(paths["spatial"].read_text())}
    if summarize_ties(rebuilt,spatial)!=json.loads((args.run/"summary.json").read_text()):raise ValueError("Event/spatial aggregation differs")
    refs=[r for r in json.loads(paths["simple_scenes"].read_text()) if (r["scenario"],r["upgrade_cost"],r["budget"])==("mixed_survey",2,32)
        and r["policy"] in ["prior_survey","prior_gate","count_survey","count_negative","count_all"]]
    effects=paired_effects(rebuilt+refs)
    atomic_json(args.run/"paired_effects.json",effects)
    audit={"stage":"SERIAL_TWO_STAGE_PRIMARY_FULL_REPLAY_PASSED","counts":dict(counts),
        "all_decisions_and_threshold_scores_replayed":True,"all_aggregations_replayed":True,"prediction_masks_verified":len(observations),
        "source_models_unchanged":True,"held_event_fit_ids_checked":True,"historical_one_stage_reproduced":True,
        "faithful_original_mfens_reproduction":False,"validation_or_confirmation_scored":False,
        "new_fit":False,"new_inference_calls":0,"gpu_rented":False}
    atomic_json(args.run/"audit.json",audit)
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    main()
