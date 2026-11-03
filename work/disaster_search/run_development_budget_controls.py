"""Frozen budget controls on 72 development scenes; no independent validation.

Online code receives coarse image features, aggregate models, and paid binary
observations only. Evaluator identities are encoded privately for post-hoc
distinct-building scoring. No boundary-discount tuning or new controller.
"""
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from cross_cell_count_smoke import CELLS,quality
from paired_count_baseline import fit_local_count,joint_posterior,native_positive_given_blur
from plan_search_v2 import freeze

POLICIES=["random_survey","coverage_survey","prior_survey","random_upgrade","prior_gate",
          "count_survey","count_negative","count_all","presence","usable_coverage","usable_prior","count_usable"]
QUALITY_SEEDS=list(range(20261025,20261030))
ORDER_SEEDS=list(range(20261030,20261040))
SCENARIOS=["native_survey","mixed_survey","blurred_survey"]
THRESHOLDS=[.05,.1,.25]
CELLS_INDEX={cell:i for i,cell in enumerate(CELLS)}


def public_rgb_features(data,manifest):
    """Only the same BOX-downsampled 128px paired RGB available initially."""
    expected={r["path"]:r["sha256"] for r in manifest["files"] if r["path"].endswith(".png")}
    result={}
    for sample in manifest["samples"]:
        images={}
        for period in ["pre","post"]:
            path=sample["files"][period+"_image"]
            if sha((data/path).read_bytes())!=expected[path]:raise ValueError("RGB hash mismatch")
            with Image.open(data/path) as im:images[period]=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=float)
        delta=np.abs(images["pre"]-images["post"]).mean(axis=2)/255
        usable=np.any(images["pre"]>0,axis=2)&np.any(images["post"]>0,axis=2)
        result[sample["sample_id"]]={}
        for r in range(8):
            for c in range(8):
                window=(slice(16*r,16*(r+1)),slice(16*c,16*(c+1)))
                result[sample["sample_id"]][f"r{r:02d}c{c:02d}"]={
                    "change":float(np.percentile(delta[window],75)),"coarse_nonzero_fraction":float(usable[window].mean())}
    return result


def fit_presence(rows,scores,count_model):
    """Offline truth-only Bernoulli prior, with fixed eight-cell shrinkage."""
    known=[r for r in rows if r["mode"]=="native" and r["target"] is not None]
    if not known:raise ValueError("No known truth cells")
    global_prior=(sum(r["target"] for r in known)+.5)/(len(known)+1)
    bins=[]
    for number in range(4):
        subset=[r for r in known if bisect_right(count_model["public_quantile_edges"],scores[r["sample_id"]][r["cell_id"]])==number]
        bins.append({"known_cells":len(subset),"prior":(sum(r["target"] for r in subset)+8*global_prior)/(len(subset)+8)})
    return {"stage":"DEVELOPMENT_ONLY_EXISTENCE_ABLATION","bins":bins,"global_prior":global_prior,
            "public_quantile_edges":count_model["public_quantile_edges"],"paired_sensor":count_model["paired_sensor"],
            "count_shrinkage_pseudo_cells":8,"truth_prior_pseudocount_per_class":.5}


def presence_utilities(model,raw,scenario,cost):
    prior=model["bins"][bisect_right(model["public_quantile_edges"],raw)]["prior"]
    joint=model["paired_sensor"]
    target=joint["probabilities"]["1"]
    native=target["01"]+target["11"]
    blurred=target["10"]+target["11"]
    weight={"native_survey":1.,"mixed_survey":.5,"blurred_survey":0.}[scenario]
    survey=prior*(weight*native+(1-weight)*blurred)
    upgrade=joint_posterior(prior,joint,blurred=False)*native_positive_given_blur(joint,True,False)/cost
    # Existence-only objective rewards first successful target-cell declaration.
    # B+ already declares that cell, so its additional existence reward is zero.
    return [survey,upgrade,0.]


def utility_table(features,count_model,presence_model,scenario,upgrade_cost,policy):
    table=[]
    native_weight={"native_survey":1.,"mixed_survey":.5,"blurred_survey":0.}[scenario]
    for i,cell in enumerate(CELLS):
        f=features[cell];raw=f["change"];usable=f["coarse_nonzero_fraction"]
        if not math.isfinite(raw) or not 0<=usable<=1:raise ValueError("Invalid public feature")
        if policy in {"random_survey","random_upgrade"}:values=[1.,1.,1.]
        elif policy=="coverage_survey":values=[64-i,None,None]
        elif policy=="prior_survey":values=[raw,None,None]
        elif policy=="prior_gate":values=[raw,raw/upgrade_cost,None]
        elif policy=="usable_coverage":values=[usable,None,None]
        elif policy=="usable_prior":values=[usable*raw,None,None]
        elif policy=="presence":values=presence_utilities(presence_model,raw,scenario,upgrade_cost)
        else:
            b=count_model["bins"][bisect_right(count_model["public_quantile_edges"],raw)]
            survey=native_weight*b["survey_mean_count"]["native"]+(1-native_weight)*b["survey_mean_count"]["blurred"]
            upgrades=[b["upgrade"][str(k)]["mean_new_within_cell"] for k in [0,1]]
            values=[survey]+[None if value is None else value/upgrade_cost for value in upgrades]
            if policy=="count_survey":values[1:]=[None,None]
            elif policy=="count_negative":values[2]=None
            elif policy=="count_usable":values=[value*usable if value is not None else None for value in values]
            elif policy!="count_all":raise ValueError("Unknown policy")
        if policy=="random_survey":values[1:]=[None,None]
        table.append(values)
    return table


class PaidBinaryEnvironment:
    def __init__(self,observations,sid,scenario,seed,upgrade_cost):
        self._rows={(r["cell_id"],r["mode"]):r for r in observations if r["sample_id"]==sid}
        self._sid,self._scenario,self._seed,self._cost=sid,scenario,seed,upgrade_cost
        self._latest={}

    def observe(self,cell,kind):
        old=self._latest.get(cell)
        if old is None and kind=="survey":mode=quality(self._scenario,self._seed,self._sid,cell);cost=1
        elif old=="blurred" and kind=="upgrade":mode="native";cost=self._cost
        else:raise ValueError("Ineligible action")
        self._latest[cell]=mode
        row=self._rows[cell,mode]
        if type(row["declared_target"]) is not bool:raise ValueError("Expected frozen boolean")
        return {"cell_id":cell,"mode":mode,"declared_target":row["declared_target"],"cost":cost}


def run_episode(environment,table,policy,budget,upgrade_cost,order_seed):
    # 0 unvisited; 1 blurred negative; 2 blurred positive; 3 native terminal.
    state=[0]*64;rng=random.Random(order_seed);spent=0;trace=[]
    while spent<budget:
        chosen=None;best=-math.inf;action_cost=None
        for i,status in enumerate(state):
            if status==3:continue
            cost=1 if status==0 else upgrade_cost
            if cost>budget-spent:continue
            value=table[i][status]
            if value is None:continue
            score=rng.random() if policy.startswith("random_") else value
            if not math.isfinite(score):raise ValueError("Non-finite utility")
            # Strict > gives deterministic row/column tie break and no future-quality lookahead.
            if score>best:chosen=i;best=score;action_cost=cost
        if chosen is None:break
        kind="survey" if state[chosen]==0 else "upgrade"
        result=environment.observe(CELLS[chosen],kind)
        if result["cost"]!=action_cost:raise ValueError("Environment cost mismatch")
        spent+=action_cost
        trace.append({"cell_id":result["cell_id"],"kind":kind,"mode":result["mode"],
            "declared_target":result["declared_target"],"cost":action_cost,"cumulative_cost":spent,"utility":best})
        state[chosen]=3 if result["mode"]=="native" else 2 if result["declared_target"] else 1
    return trace


def private_evaluator(rows):
    """Evaluator-only bitsets; nothing from here is passed to the planner."""
    uids=sorted({uid for r in rows for uid in r["visible_target_uids"]})
    indices={uid:i for i,uid in enumerate(uids)}
    result={}
    for r in rows:
        matches=[]
        for threshold in THRESHOLDS:
            ids={v["uid"] for v in r["overlap_records"] if v["visible_pixels"]>=4 and v["severe_overlap_pixels"]>=4 and v["visible_footprint_coverage"]>=threshold}
            matches.append(sum(1<<indices[uid] for uid in ids))
        result[r["cell_id"],r["mode"]]={"matches":matches,"truth":r["target"]}
    return result,len(uids)


def score_trace(trace,evaluator,targets):
    output=[]
    for index,threshold in enumerate(THRESHOLDS):
        found=0;cell_found=defaultdict(int);first=None;fp=set();unknown=set();unmatched=set()
        duplicates=0;positive_upgrades=0;negative_upgrades=0;local_new=0;upgrade_new=0;positive_upgrade_new=0;latest={}
        for a in trace:
            row=evaluator[a["cell_id"],a["mode"]];ids=row["matches"][index];new=ids&~found
            local=ids&~cell_found[a["cell_id"]]
            local_new+=local.bit_count();duplicates+=(local&found).bit_count()
            if new and first is None:first=a["cumulative_cost"]
            if a["kind"]=="upgrade":
                positive=latest[a["cell_id"]]
                positive_upgrades+=positive;negative_upgrades+=not positive
                upgrade_new+=new.bit_count()
                if positive:positive_upgrade_new+=new.bit_count()
            if a["declared_target"]:
                if row["truth"] is False:fp.add(a["cell_id"])
                if row["truth"] is None:unknown.add(a["cell_id"])
                if not ids:unmatched.add(a["cell_id"])
            found|=ids;cell_found[a["cell_id"]]|=ids;latest[a["cell_id"]]=a["declared_target"]
        output.append({"threshold":threshold,"distinct_found":found.bit_count(),"visible_targets":targets,
            "recall":found.bit_count()/targets if targets else None,"false_positive_cells":len(fp),
            "unknown_positive_cells":len(unknown),"unmatched_positive_cells":len(unmatched),
            "cross_cell_duplicate_match_events":duplicates,"local_new_match_events":local_new,
            "positive_upgrades":positive_upgrades,"negative_upgrades":negative_upgrades,
            "upgrades":positive_upgrades+negative_upgrades,"new_buildings_from_upgrades":upgrade_new,
            "new_buildings_from_positive_upgrades":positive_upgrade_new,"first_discovery_cost":first,
            "first_discovery_censored":first is None,"actions":len(trace),
            "spent":trace[-1]["cumulative_cost"] if trace else 0})
    return output


SUMMARY_FIELDS=["distinct_found","false_positive_cells","cross_cell_duplicate_match_events","upgrades",
                "positive_upgrades","negative_upgrades","new_buildings_from_upgrades","new_buildings_from_positive_upgrades",
                "spent","actions","first_discovery_censored","planning_s"]


def aggregate_scene_metrics(metrics):
    groups=defaultdict(list)
    for r in metrics:
        key=tuple(r[k] for k in ["fit_scope","policy","scenario","upgrade_cost","budget","threshold"])
        groups[key].append(r)
    result=[]
    for key,rows in sorted(groups.items()):
        values={k:sum(r[k] for r in rows)/len(rows) for k in SUMMARY_FIELDS}
        result.append({**dict(zip(["fit_scope","policy","scenario","upgrade_cost","budget","threshold"],key)),
                       **values,"repeats":len(rows),"visible_targets":rows[0]["visible_targets"],
                       "recall":values["distinct_found"]/rows[0]["visible_targets"] if rows[0]["visible_targets"] else None})
    return result


def summarize(scene_means,spatial):
    groups=defaultdict(list);spatial_groups=defaultdict(list);strata=defaultdict(list)
    fields=["fit_scope","policy","scenario","upgrade_cost","budget","threshold"]
    for r in scene_means:
        key=tuple(r[k] for k in fields)
        groups[key+(r["event"],)].append(r)
        spatial_groups[key+(r["event"],spatial[r["sample_id"]]["analysis_group"])].append(r)
        strata[key+(r["event"],spatial[r["sample_id"]]["training_envelope_overlap"])].append(r)
    def finish(collection,labels):
        return [{**dict(zip(fields+labels,key)),**{f:sum(r[f] for r in rows)/len(rows) for f in SUMMARY_FIELDS},"scenes":len(rows)}
                for key,rows in sorted(collection.items())]
    event_means=finish(groups,["event"])
    macro=defaultdict(list)
    for r in event_means:macro[tuple(r[k] for k in fields)].append(r)
    macro_means=[{**dict(zip(fields,key)),**{f:sum(r[f] for r in rows)/len(rows) for f in SUMMARY_FIELDS},"events":len(rows)}
                 for key,rows in sorted(macro.items())]
    return {"event_means":event_means,"event_macro_means":macro_means,
            "spatial_group_means":finish(spatial_groups,["event","analysis_group"]),
            "training_envelope_strata":finish(strata,["event","training_envelope_overlap"])}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["data","perception","fit","design","spatial","out"]:parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths={"manifest":args.data/"manifest.json","observations":args.perception/"observations.json",
        "association":args.perception/"association_report.json","scores":args.perception/"public_change_scores.json",
        "parameters":args.fit/"component_parameters.json","selection":args.design/"next_development_batch.json",
        "contract":args.design/"experiment_contract_v2_2.json","spatial":args.spatial,
        "previous_receipt":args.fit/"completion_receipt.json"}
    sources={k:sha(p.read_bytes()) for k,p in paths.items()}
    freeze(args.out/"protocol.json",{"stage":"DEVELOPMENT72_BUDGETED_SIMPLE_CONTROLS","sources":sources,
        "runner_sha256":sha(Path(__file__).read_bytes()),"helper_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in ["paired_count_baseline.py","cross_cell_count_smoke.py"]},
        "policies":POLICIES,"quality_seeds":QUALITY_SEEDS,"order_seeds":ORDER_SEEDS,"quality_scenarios":SCENARIOS,
        "budgets":[8,16,32],"upgrade_costs":[1,2,4],"measurement_thresholds":THRESHOLDS,
        "primary":{"scenario":"mixed_survey","budget":32,"upgrade_cost":2,"threshold":.1},
        "fit_scopes":{"development_fit":"Parameters fitted on all 72, scored in sample. Entire cost/quality/budget grid.",
                      "event_holdout":"Fit using the other two development events, primary cost/quality/budget only; previously inspected development events, not independent validation."},
        "event_holdout_policies":["prior_survey","count_survey","count_negative","count_all","presence","count_usable"],
        "presence_prior":"Known truth in count-model's four public change bins; eight-cell shrinkage toward Jeffreys-smoothed global prior. Joint paired sensor fixed at 0.5 per outcome.",
        "presence_reward":"Expected first target-cell detection. B+ additional existence reward is zero; count_all permits B+ additional building reward. No global posterior propagation.",
        "count_utility":"Fixed development-fitted same-cell new matched count/cost; not exact global distinct-building utility.",
        "usable_controls":"Both 128px BOX coarse RGB images nonzero; fraction per16px cell. No cell removal. usable_coverage: descending fraction then row order; usable_prior: fraction*raw change; count_usable: fraction*count utility.",
        "random_upgrade":"Uniform among currently affordable eligible actions; does not choose quality or bypass survey. No travel costs.",
        "prior_gate":"Raw change/cost, upgrade only blurred negatives. No fitted threshold.",
        "unsupported_upgrade":"Skip action with None fitted mean, never fabricate supported value.",
        "eligibility":"Survey unvisited cost1; one native upgrade after any blurred observation; no native repeat. Ablations may suppress upgrades explicitly.",
        "statistics":"Average order/quality repeats within scenes, then scenes within events and events equally. Spatial groups and detector-training envelope overlap strata descriptive only; full-image independence unverified.",
        "new_method_selected":False,"validation_or_confirmation_access":False,
        "no_parameter_tuning_after_this_run":True,"no_claim_of_independent_improvement":True})
    manifest=json.loads(paths["manifest"].read_text());selected=json.loads(paths["selection"].read_text())
    expected={r["sample_id"] for r in selected}
    if len(expected)!=72 or any(r["split"]!="development" for r in selected) or {r["sample_id"] for r in manifest["samples"]}!=expected:
        raise ValueError("Exact 72 development identities required")
    previous=json.loads(paths["previous_receipt"].read_text())
    if previous["stage"]!="DEVELOPMENT72_MATERIALIZATION_SENSOR_AND_AGGREGATE_FIT_COMPLETE":raise ValueError("Development QA incomplete")
    for name,digest in previous["files"].items():
        if sha((args.fit/name).read_bytes())!=digest:raise ValueError("Previous development output changed")
    observations=json.loads(paths["observations"].read_text());association=json.loads(paths["association"].read_text())
    if len(observations)!=9216 or {r["sample_id"] for r in observations}!=expected:raise ValueError("Invalid observations")
    for r in observations:
        if sha(Path(r["cache_file"]).read_bytes())!=r["cache_sha256"]:raise ValueError("Prediction mask changed")
    rows=association["rows"]
    if association["stage"]!="DEVELOPMENT72_SENSOR_QA" or len(rows)!=9216 or {r["sample_id"] for r in rows}!=expected:
        raise ValueError("Invalid evaluator")
    features=public_rgb_features(args.data,manifest);scores={sid:{cid:v["change"] for cid,v in vals.items()} for sid,vals in features.items()}
    if scores!=json.loads(paths["scores"].read_text()):raise ValueError("Coarse cue computation differs")
    freeze(args.out/"public_features.json",features)
    model=json.loads(paths["parameters"].read_text());presence=fit_presence(rows,scores,model)
    fitted={"development_fit":{"count":model,"presence":presence}}
    for event in sorted({r["event"] for r in selected}):
        allowed={r["sample_id"] for r in selected if r["event"]!=event}
        training=[r for r in rows if r["sample_id"] in allowed]
        count=fit_local_count(training,scores)
        fitted["held_"+event]={"count":count,"presence":fit_presence(training,scores,count),"fit_ids":sorted(allowed)}
    atomic_json(args.out/"models.json",fitted)
    by_obs=defaultdict(list);by_rows=defaultdict(list)
    for r in observations:by_obs[r["sample_id"]].append(r)
    for r in rows:by_rows[r["sample_id"]].append(r)
    spatial={r["sample_id"]:r for r in json.loads(args.spatial.read_text())}
    scene_means=[];trajectories=0;metric_rows=0;planning_total=0.;started=time.perf_counter()
    configs_held=[("mixed_survey",2,32)]
    controls_held=["prior_survey","count_survey","count_negative","count_all","presence","count_usable"]
    with gzip.open(args.out/"trajectories.jsonl.gz","wt",compresslevel=3) as traces,gzip.open(args.out/"metrics.jsonl.gz","wt",compresslevel=3) as metrics:
        for sample in selected:
            sid,event=sample["sample_id"],sample["event"]
            evaluator,targets=private_evaluator(by_rows[sid]);local_metrics=[]
            for fit_scope,parameters,configs,policies in [
                ("development_fit",fitted["development_fit"],[(s,c,b) for s in SCENARIOS for c in [1,2,4] for b in [8,16,32]],POLICIES),
                ("event_holdout",fitted["held_"+event],configs_held,controls_held)]:
                for scenario,upgrade_cost,budget in configs:
                    for policy in policies:
                        table=utility_table(features[sid],parameters["count"],parameters["presence"],scenario,upgrade_cost,policy)
                        for quality_seed in QUALITY_SEEDS:
                            seeds=ORDER_SEEDS if policy.startswith("random_") else ORDER_SEEDS[:1]
                            for order_seed in seeds:
                                env=PaidBinaryEnvironment(by_obs[sid],sid,scenario,quality_seed,upgrade_cost)
                                t=time.perf_counter();trace=run_episode(env,table,policy,budget,upgrade_cost,order_seed);elapsed=time.perf_counter()-t
                                config={"sample_id":sid,"event":event,"fit_scope":fit_scope,"policy":policy,
                                    "scenario":scenario,"upgrade_cost":upgrade_cost,"budget":budget,"quality_seed":quality_seed,"order_seed":order_seed}
                                traces.write(json.dumps({**config,"trace":trace},separators=(",",":"))+"\n")
                                trajectories+=1;planning_total+=elapsed
                                for result in score_trace(trace,evaluator,targets):
                                    record={**config,**result,"planning_s":elapsed}
                                    local_metrics.append(record);metric_rows+=1
                                    metrics.write(json.dumps(record,separators=(",",":"))+"\n")
            for summary in aggregate_scene_metrics(local_metrics):scene_means.append({"sample_id":sid,"event":event,**summary})
            print(json.dumps({"stage":"SCENE_COMPLETE","sample_id":sid,"trajectories":trajectories}),flush=True)
    atomic_json(args.out/"scene_means.json",scene_means)
    summary=summarize(scene_means,spatial);atomic_json(args.out/"summary.json",summary)
    primary=[r for r in summary["event_macro_means"] if r["scenario"]=="mixed_survey" and r["budget"]==32 and r["upgrade_cost"]==2 and r["threshold"]==.1]
    preserved={k:sha(p.read_bytes())==sources[k] for k,p in paths.items()}
    if not all(preserved.values()):raise ValueError("Source outputs changed")
    report={"stage":"DEVELOPMENT72_BUDGETED_CONTROLS_COMPLETE","scenes":72,"trajectories":trajectories,
        "metric_threshold_rows":metric_rows,"primary":primary,"planning_wall_s":planning_total,"total_wall_s":time.perf_counter()-started,
        "source_inputs_preserved":preserved,"independent_efficacy_claim":False,"validation_or_confirmation_scored":False,
        "new_method_selected":False,"paid_gpu_used":False,"new_inference_calls":0,
        "trajectory_sha256":sha((args.out/"trajectories.jsonl.gz").read_bytes()),"metrics_sha256":sha((args.out/"metrics.jsonl.gz").read_bytes())}
    atomic_json(args.out/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
