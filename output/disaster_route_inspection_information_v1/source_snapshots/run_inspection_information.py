"""Frozen-before-execution controlled sweep; no real-data efficacy claim."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import time

from inspection_information_core import make_policies, exhaustive_evaluate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"output/disaster_route_inspection_information_v1"
GRID={"rho":[0,.25,.5,.75,1],"sensor_q":[.5,.65,.8,1],"budget":list(range(6,15)),"extra_replan_fee":[0,1,2,3]}


def dump(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def run():
    OUT.mkdir(exist_ok=False)
    protocol={"date":"2026-11-02","role":"Synthetic controlled inspection-information boundary;not a new method or real UAV experiment.",
      "grid":GRID,"home":[0,0],"sites":[[1,0],[2,0],[0,2]],
      "costs":"Manhattan movement +1 per local inspection +1 initial planning + declared optional extra update fee;closed return charged.",
      "reward":"One unit for each correctly detected positive at a paid observation;no separate target arrival reward,no repeats.",
      "sensor":"Conditional independent symmetric local errors:TPR=q,FPR=1-q. Fixed known model in this reference.",
      "joint":"rho mixture of (Z,Z,1-Z),Z~Bernoulli(.5),and three independent Bernoulli(.5) states.All marginals identical.",
      "policies":["exact static","static with identical extra fee, no feedback","best forced update immediately after first observation","best static or one update under known model"],
      "scope":"At most one update after first observation;not unrestricted adaptive optimum. Model known and calibrated by construction.",
      "scenarios":"All world and local observation configurations integrated exactly;not independent sampled scenes.",
      "predeclared_examples":[[0,1,10,1],[1,1,10,1],[1,.5,10,1],[0,1,12,1],[1,1,10,2]],
      "real_data_modified":False,"gpu_rental":False,"new_model_inference":0}
    dump(OUT/"protocol.json",protocol)
    snap=OUT/"source_snapshots";snap.mkdir()
    for name in ("inspection_information_core.py","run_inspection_information.py","test_inspection_information.py"):
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    started=time.perf_counter();records=[];leaves=0
    for rho,q,budget,fee in itertools.product(GRID["rho"],GRID["sensor_q"],GRID["budget"],GRID["extra_replan_fee"]):
        policies=make_policies(rho,q,budget,fee)
        for name,p in policies.items():
            result=exhaustive_evaluate(p,rho,q,budget);leaves+=result["exact_positive_probability_leaves"]
            records.append({"rho":rho,"q":q,"budget":budget,"fee":fee,"policy":name,"plan":asdict(p),"exact":result})
    groups={}
    for r in records:groups.setdefault((r["rho"],r["q"],r["budget"],r["fee"]),{})[r["policy"]]=r
    noinfo=0;positive=0;forced_worse=0
    for (rho,q,budget,fee),g in groups.items():
        static=g["static"]["exact"]["reward"];chosen=g["best_static_or_one_update"]["exact"]["reward"]
        forced=g["forced_one_update"]["exact"]["reward"];charged=g["static_with_same_extra_fee"]["exact"]["reward"]
        assert chosen>=static-1e-10 and forced>=charged-1e-10
        positive+=int(chosen>static+1e-10);forced_worse+=int(forced<static-1e-10)
        if rho==0 or q==.5:
            assert abs(chosen-static)<1e-10 and abs(forced-charged)<1e-10
            noinfo+=1
    summary={"conditions":len(groups),"policy_evaluations":len(records),"exact_weighted_replay_leaves":leaves,
        "no_information_conditions":noinfo,"no_information_adaptive_advantage":0,
        "known_model_best_one_update_positive_conditions":positive,"forced_update_worse_than_static_conditions":forced_worse,
        "budget_violations":0,"elapsed_cpu_wall_seconds":time.perf_counter()-started,
        "examples":[{"rho":rho,"q":q,"budget":budget,"fee":fee,
          **{name:groups[(rho,q,budget,fee)][name]["exact"]["reward"] for name in policies}}
                    for rho,q,budget,fee in protocol["predeclared_examples"]],
        "interpretation":"Counts describe a chosen synthetic parameter grid,not prevalence or efficacy on disaster scenes.",
        "new_method":False,"real_disaster_information_value_established":False}
    dump(OUT/"results.json",records);dump(OUT/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":run()
