"""Serial/costed ENS-inspired batch-tail control; not ENS/MF-ENS reproduction.

The unchanged reward model is a same-cell incremental-count proxy. Branches
use only frozen aggregate paired-observation parameters. Other cells are not
updated by fantasies. True instances and future cached outcomes are private.
"""
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
import gzip
import json
import math
from pathlib import Path
import time

from build_xbd_mini import atomic_json,sha
from cross_cell_count_smoke import CELLS
from plan_search_v2 import freeze
from run_development_budget_controls import (PaidBinaryEnvironment,QUALITY_SEEDS,SCENARIOS,
    THRESHOLDS,ORDER_SEEDS,utility_table,private_evaluator,score_trace,aggregate_scene_metrics,summarize)

POLICIES=["proxy_greedy","two_step","batch_tail"]
EPS=1e-12


def planning_inputs(features,models,scenario,upgrade_cost):
    count,presence=models["count"],models["presence"]
    utilities=utility_table(features,count,presence,scenario,upgrade_cost,"count_all")
    rewards=[];blur_probabilities=[]
    for cell,values in zip(CELLS,utilities):
        rewards.append([values[0]]+[None if x is None else x*upgrade_cost for x in values[1:]])
        prior=presence["bins"][bisect_right(presence["public_quantile_edges"],features[cell]["change"])]["prior"]
        joint=presence["paired_sensor"]["probabilities"]
        probability=sum(weight*(joint[truth]["10"]+joint[truth]["11"])
                        for truth,weight in [("1",prior),("0",1-prior)])
        if not 0<=probability<=1:raise ValueError("Invalid aggregate blurred probability")
        blur_probabilities.append(probability)
    return {"reward":rewards,"greedy_utility":utilities,"blur_positive":blur_probabilities,
            "native_survey_probability":{"native_survey":1.,"mixed_survey":.5,"blurred_survey":0.}[scenario]}


class BatchTail:
    """Exact costed batch value for currently eligible actions; no new unlocks."""
    def __init__(self,items,upgrade_cost,kind):
        self.kind=kind;self.cost=upgrade_cost
        self.items=items
        if any(c not in {1,upgrade_cost} or r<0 or not math.isfinite(r) for c,r in items):
            raise ValueError("Invalid batch item")
        # For cost=1 either category assignment works; all items go into S.
        self.survey=list(accumulate(sorted((r for c,r in items if c==1),reverse=True),initial=0.))
        self.upgrade=list(accumulate(sorted((r for c,r in items if c!=1),reverse=True),initial=0.))

    def value(self,budget):
        if budget<=0:return 0.
        if self.kind=="two_step":
            return max((r for c,r in self.items if c<=budget),default=0.)
        if self.kind!="batch_tail":raise ValueError("Unknown tail type")
        return max(self.upgrade[k]+self.survey[min(len(self.survey)-1,budget-self.cost*k)]
                   for k in range(min(len(self.upgrade)-1,budget//self.cost)+1))


def action_values(state,inputs,remaining,upgrade_cost,policy):
    if len(state)!=len(inputs["reward"]):raise ValueError("State/model size differs")
    if policy not in POLICIES:raise ValueError("Unknown policy")
    actions=[]
    for i,status in enumerate(state):
        if status==3:continue
        cost=1 if status==0 else upgrade_cost
        reward=inputs["reward"][i][status]
        if reward is not None and cost<=remaining:
            actions.append((i,status,cost,reward))
    if policy=="proxy_greedy":
        return [(i,inputs["greedy_utility"][i][status],inputs["greedy_utility"][i][status])
                for i,status,cost,reward in actions]
    # Cells with identical public parameters and state are interchangeable under
    # this proxy model. Retain the lowest index; no top-K candidate truncation.
    representatives={}
    for i,status,cost,reward in actions:
        key=(status,tuple(inputs["reward"][i]),inputs["blur_positive"][i] if status==0 else None)
        representatives.setdefault(key,(i,status,cost,reward))
    values=[]
    for i,status,cost,reward in representatives.values():
        items=[(other_cost,other_reward) for j,_,other_cost,other_reward in actions if j!=i]
        # actions was affordable under current budget; an item too expensive now
        # cannot become affordable in the strictly smaller tail budget.
        tail=BatchTail(items,upgrade_cost,policy)
        budget=remaining-cost
        baseline=tail.value(budget)
        if status==0:
            native=inputs["native_survey_probability"]
            positive=inputs["blur_positive"][i]
            branches=[]
            for result in [1,2]:
                upgrade_reward=inputs["reward"][i][result]
                branch=baseline
                if upgrade_reward is not None and upgrade_cost<=budget:
                    branch=max(branch,upgrade_reward+tail.value(budget-upgrade_cost))
                branches.append(branch)
            future=native*baseline+(1-native)*((1-positive)*branches[0]+positive*branches[1])
        else:future=baseline
        values.append((i,reward+future,inputs["greedy_utility"][i][status]))
    return values


def choose(state,inputs,remaining,upgrade_cost,policy):
    candidates=action_values(state,inputs,remaining,upgrade_cost,policy)
    best=None
    for i,value,greedy in candidates:
        if not math.isfinite(value):raise ValueError("Nonfinite planning value")
        if best is None or (value>best[1] if policy=="proxy_greedy" else
            value>best[1]+EPS or (abs(value-best[1])<=EPS and (greedy>best[2] or (greedy==best[2] and i<best[0])))):
            best=(i,value,greedy)
    return best


def run_episode(environment,inputs,policy,budget,upgrade_cost):
    state=[0]*len(inputs["reward"]);spent=0;trace=[]
    while spent<budget:
        selected=choose(state,inputs,budget-spent,upgrade_cost,policy)
        if selected is None:break
        i,value,greedy=selected
        kind="survey" if state[i]==0 else "upgrade"
        expected_cost=1 if state[i]==0 else upgrade_cost
        result=environment.observe(CELLS[i],kind)
        if result["cost"]!=expected_cost:raise ValueError("Planning/environment cost mismatch")
        spent+=expected_cost
        trace.append({"cell_id":CELLS[i],"kind":kind,"mode":result["mode"],"declared_target":result["declared_target"],
            "cost":expected_cost,"cumulative_cost":spent,"utility":value,"greedy_utility":greedy})
        state[i]=3 if result["mode"]=="native" else 2 if result["declared_target"] else 1
    return trace


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["previous","perception","design","spatial","out"]:parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths={"models":args.previous/"models.json","features":args.previous/"public_features.json",
        "previous_receipt":args.previous/"completion_receipt.json","previous_scene_means":args.previous/"scene_means.json",
        "observations":args.perception/"observations.json","association":args.perception/"association_report.json",
        "selection":args.design/"next_development_batch.json","contract":args.design/"experiment_contract_v2_2.json","spatial":args.spatial,
        "adaptation":args.out/"adaptation.md"}
    sources={k:sha(p.read_bytes()) for k,p in paths.items()}
    freeze(args.out/"protocol.json",{"stage":"DEVELOPMENT72_COSTED_PLANNING_CONTROLS","sources":sources,
        "primary_source_sha256":{name:sha((args.out/"sources"/name).read_bytes()) for name in ["jiang17d.pdf","nguyen21f.pdf","ens_page.json","mfens_page.json"]},
        "runner_sha256":sha(Path(__file__).read_bytes()),"helper_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in ["run_development_budget_controls.py","paired_count_baseline.py","cross_cell_count_smoke.py"]},
        "primary":{"scenario":"mixed_survey","budget":32,"upgrade_cost":2,"threshold":.1},
        "policies":POLICIES,"quality_seeds":QUALITY_SEEDS,"scenarios":SCENARIOS,"upgrade_costs":[1,2,4],"budgets":[8,16,32],"thresholds":THRESHOLDS,
        "fit_scopes":{"development_fit":"All 72 fit, full settings grid.","event_holdout":"Fit on other two dev events; primary cost/quality/budget only. Previously seen development data, not independent validation."},
        "reward":"Unchanged count_all same-cell reward and paired/presence parameters; no global dedup proxy, spatial update or new fit. Reward is unconditional expected count, not rate, inside lookahead.",
        "branches":"Hypothetical survey: native terminal with known scenario mixture probability; blurred-negative/positive according to frozen paired likelihood mixed under frozen binned truth prior. Conditional upgrade rewards already fitted by B result. Never multiply marginal B,N likelihoods.",
        "two_step":"First action plus best affordable one later action; exact two-action expectation under frozen proxy, budget beyond second action ignored.",
        "batch_tail":"First action plus expected optimal costed batch of currently eligible subsequent actions, one item per cell. Future survey-upgrade chains inside batch omitted; only first fantasy survey unlocks its conditional upgrade. Replan after each paid observation.",
        "candidate_grouping":"Equivalent state/reward/branch-probability public parameter classes; exact symmetry under proxy. No shortlist or hidden-data pruning.",
        "ties":"Planning Q within 1e-12: higher original greedy reward/cost, then cell index. Greedy retains strict > and exact previous utilities.",
        "assumptions":["Proxy adds counts over cells despite possible duplicate true buildings.","Other-cell distributions are unchanged by a fantasy; no learned spatial coupling.","Joint declaration probabilities and shrunk count means are aggregate approximations, not a validated coherent instance-level generative model.","Batch tail is not an exact multi-step optimum and does not implement MF-ENS's exploratory L batch."],
        "no_parameter_tuning_after_run":True,"new_method_selected":False,"faithful_ens_or_mfens_reproduction":False,
        "validation_or_confirmation_access":False,"new_inference_calls":0,"paid_gpu":False})
    previous=json.loads(paths["previous_receipt"].read_text())
    for name,digest in previous["files"].items():
        if sha((args.previous/name).read_bytes())!=digest:raise ValueError("Previous budget-control output changed")
    selected=json.loads(paths["selection"].read_text());expected={r["sample_id"] for r in selected}
    if len(expected)!=72 or any(r["split"]!="development" for r in selected):raise ValueError("Exact 72 development scenes required")
    models=json.loads(paths["models"].read_text());features=json.loads(paths["features"].read_text())
    observations=json.loads(paths["observations"].read_text());association=json.loads(paths["association"].read_text())
    if len(observations)!=9216 or {r["sample_id"] for r in observations}!=expected or set(features)!=expected:
        raise ValueError("Incomplete or foreign cache/feature inputs")
    if association["stage"]!="DEVELOPMENT72_SENSOR_QA" or {r["sample_id"] for r in association["rows"]}!=expected:
        raise ValueError("Unexpected evaluator")
    by_obs=defaultdict(list);by_rows=defaultdict(list)
    for r in observations:by_obs[r["sample_id"]].append(r)
    for r in association["rows"]:by_rows[r["sample_id"]].append(r)
    spatial={r["sample_id"]:r for r in json.loads(args.spatial.read_text())}
    scene_means=[];trajectories=0;metrics_count=0;planning_s=0.;started=time.perf_counter()
    with gzip.open(args.out/"trajectories.jsonl.gz","wt",compresslevel=3) as traces,gzip.open(args.out/"metrics.jsonl.gz","wt",compresslevel=3) as metrics:
        for sample in selected:
            sid,event=sample["sample_id"],sample["event"]
            evaluator,targets=private_evaluator(by_rows[sid]);local=[]
            for scope,parameters,configs in [
                ("development_fit",models["development_fit"],[(s,c,b) for s in SCENARIOS for c in [1,2,4] for b in [8,16,32]]),
                ("event_holdout",models["held_"+event],[("mixed_survey",2,32)])]:
                for scenario,cost,budget in configs:
                    inputs=planning_inputs(features[sid],parameters,scenario,cost)
                    for seed in QUALITY_SEEDS:
                        for policy in POLICIES:
                            environment=PaidBinaryEnvironment(by_obs[sid],sid,scenario,seed,cost)
                            t=time.perf_counter();trace=run_episode(environment,inputs,policy,budget,cost);elapsed=time.perf_counter()-t
                            config={"sample_id":sid,"event":event,"fit_scope":scope,"policy":policy,
                                "scenario":scenario,"upgrade_cost":cost,"budget":budget,"quality_seed":seed,"order_seed":ORDER_SEEDS[0]}
                            traces.write(json.dumps({**config,"trace":trace},separators=(",",":"))+"\n")
                            trajectories+=1;planning_s+=elapsed
                            for result in score_trace(trace,evaluator,targets):
                                record={**config,**result,"planning_s":elapsed};local.append(record);metrics_count+=1
                                metrics.write(json.dumps(record,separators=(",",":"))+"\n")
            scene_means.extend({"sample_id":sid,"event":event,**row} for row in aggregate_scene_metrics(local))
            print(json.dumps({"stage":"SCENE_COMPLETE","sample_id":sid,"trajectories":trajectories}),flush=True)
    atomic_json(args.out/"scene_means.json",scene_means)
    summary=summarize(scene_means,spatial);atomic_json(args.out/"summary.json",summary)
    primary=[r for r in summary["event_macro_means"] if r["scenario"]=="mixed_survey" and r["upgrade_cost"]==2 and r["budget"]==32 and r["threshold"]==.1]
    preserved={k:sha(p.read_bytes())==sources[k] for k,p in paths.items()}
    if not all(preserved.values()):raise ValueError("Source artifacts changed")
    report={"stage":"DEVELOPMENT72_COSTED_PLANNING_COMPLETE","scenes":72,"trajectories":trajectories,
        "threshold_scores":metrics_count,"primary":primary,"planning_wall_s":planning_s,"total_wall_s":time.perf_counter()-started,
        "sources_preserved":preserved,"new_method_selected":False,"independent_efficacy_claim":False,
        "validation_or_confirmation_scored":False,"faithful_ens_or_mfens_reproduction":False,"new_inference_calls":0,"gpu_rented":False,
        "trajectory_sha256":sha((args.out/"trajectories.jsonl.gz").read_bytes()),"metrics_sha256":sha((args.out/"metrics.jsonl.gz").read_bytes())}
    atomic_json(args.out/"report.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()
