"""Fixed paired solver x stopping-rule experiment, reusing the sealed V1 worlds."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_route_loop import ROOT, HERE, maps, read, save, digest, checked
from route_core import Config, Observer, NOISE, POLICIES, evaluate
from route_factor_core import Planner, execute

PARENT = ROOT/'output/disaster_route_loop_v1'
OUT = ROOT/'output/disaster_route_factor_v2'
SOURCES = ['route_factor_core.py','test_route_factor.py','run_route_factor.py','audit_route_factor.py','report_route_factor.py']
SOLVERS = ('greedy','insertion_2opt')
STOPS = ('cap30','time_only')
KEY = ('region','mode','noise','budget_s','solver','stopping','policy')


def verify_parent():
    receipt=read(PARENT/'completion_receipt.json')
    for name,sha in receipt['files'].items():checked(PARENT/name,sha)
    for name,sha in receipt['external_files'].items():checked(name,sha)
    return receipt


def verify():
    p=read(OUT/'protocol.json')
    checked(PARENT/'completion_receipt.json',p['parent_receipt_sha256'])
    for name,sha in p['source_sha256'].items():checked(HERE/name,sha)
    for name,sha in p['input_sha256'].items():checked(name,sha)
    checked(OUT/'test_log.txt',p['test_log_sha256'])
    return p


def plan():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    verify_parent()
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    (OUT/'test_log.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0 and 'Ran 23 tests' in tests.stdout+tests.stderr
    inputs={str(PARENT/name):digest(PARENT/name) for name in ('synthetic_worlds.json','public_geometry.json','protocol.json','routes.jsonl','metrics.json')}
    old=read(PARENT/'protocol.json')
    for name,sha in old['external_sha256'].items():inputs[name]=sha
    p={'client_date':'2026-11-02','stage':'PAIRED_SOLVER_X_STOPPING_V2','parent_receipt_sha256':digest(PARENT/'completion_receipt.json'),
       'source_sha256':{name:digest(HERE/name) for name in SOURCES},'input_sha256':inputs,'test_log_sha256':digest(OUT/'test_log.txt'),
       'question':'Does feedback gain survive a stronger common closed-route solver and removal of the inherited 30-inspection cap?',
       'factorial':{'solvers':SOLVERS,'stopping':STOPS,'policies':POLICIES,'budgets_s':[600,1200,1800],'primary_budget_s':1200,
                    'regions':2,'prior_modes':3,'noise_modes':3,'seeds_per_region_mode':10,'existing_worlds':60,'routes':8640},
       'worlds':'Exactly the sealed V1 synthetic targets and uniform observation draws. No new worlds, labels, fitting or outcome-based selection.',
       'stopping':'cap30 reproduces V1;time_only permits at most the number of eligible cells, forbids repeat inspections and otherwise stops by the same time budget.',
       'strong_solver':'Two deterministic candidates: original greedy seed and empty seed;each uses reward/marginal closed-tour-time insertion, up to eight improving 2-opt reversals, then insertion refill. Select greatest sum of public expected targets, then shortest distance, then lexicographic indices. No global optimality claim.',
       'costs':{'speed_m_s':5,'observe_s':20,'planning_latency_s':1,'all_return_charged':True,
                'latency_note':'Inherited assumed one second per planning call, even cache hits;not a measured deployment cost. Strong solver has greater actual CPU work, recorded separately;no equal-compute or efficiency claim.'},
       'memoization':'Same 20000-state LRU for both solvers. Key includes public position,visited cells,elapsed time,belief,costs and limit;no targets/future signals. Every invocation still pays simulated latency. Cache work and hits are reported.',
       'information':'Public geometry and prior plus past paid observation bits only;known V1 observation likelihood and 3x3 latent structure remain favorable assumptions.',
       'primary':'Online minus post_static detected synthetic target cells at1200s for every region/prior/noise condition,per solver/stopping. Also online minus frozen-update control.',
       'solver_effect':'Paired change of each policy under stronger solver;difference of online-poststatic gains between solvers. Direction is reported regardless of benefit.',
       'legacy_replay':'All2160 greedy/cap30 action paths,steps,planning prefixes and non-runtime metrics must exactly reproduce sealed V1.',
       'failure_handling':'First arrival/detection null when absent;separate failure rates;restricted time summaries use budget on failure. No-target recall null.',
       'examples':'Both regions,aligned/moderate/seed2026102800/budget1200;strong solver,time_only,all three main policies. Keep fixed even if failure.',
       'claims':'Post-hoc development sensitivity experiment on two geometries and reused synthetic worlds;not independent disaster confirmation,real PS efficacy or a new routing method.',
       'new_training':False,'new_perception_calls':0,'new_downloads':False,'gpu_rental':False,'real_target_labels_written':False}
    save(OUT/'protocol.json',p)
    save(OUT/'selection_receipt.json',{'protocol_sha256':digest(OUT/'protocol.json'),'routes_executed':False,'new_worlds_generated':False})
    print(json.dumps({'protocol':'FROZEN','routes':8640,'tests_passed':23}),flush=True)


def run():
    verify();assert not (OUT/'routes.jsonl.gz').exists()
    begin=time.monotonic();mm=maps();worlds=read(PARENT/'synthetic_worlds.json');planner=Planner();metrics=[];examples=[]
    legacy={}
    for line in (PARENT/'routes.jsonl').open():
        row=json.loads(line);legacy[row['world_id'],row['noise'],row['budget_s'],row['policy']]=row
    reproduced=0
    with gzip.open(OUT/'routes.jsonl.gz','wt',encoding='utf-8',compresslevel=4) as output:
        for region,g in mm.items():
            for solver in SOLVERS:
                for stopping in STOPS:
                    for world in [w for w in worlds if w['region']==region]:
                        target=np.array([sid in set(world['target_ids']) for sid in g.ids],bool)
                        draw=np.array(world['uniform_observation_draws'])
                        for noise,(sensitivity,fpr) in NOISE.items():
                            bits=draw<np.where(target,sensitivity,fpr)
                            for budget in (600,1200,1800):
                                for policy in POLICIES:
                                    config=Config(budget_s=budget,max_visits=30 if stopping=='cap30' else len(g.ids))
                                    observer=Observer(bits,NOISE[noise]);r=execute(g,policy,config,observer,solver,planner)
                                    assert observer.accessed==[s['cell_index'] for s in r['steps']]
                                    r.update(world_id=world['world_id'],mode=world['mode'],seed=world['seed'],noise=noise,stopping=stopping)
                                    r['evaluation']=evaluate(r,target)
                                    if solver=='greedy' and stopping=='cap30':
                                        old=legacy[world['world_id'],noise,budget,policy]
                                        for name,value in old.items():
                                            if name in ('measured_planning_wall_s','planning_events'):continue
                                            assert r[name]==value,('Legacy mismatch',name,world['world_id'],policy)
                                        assert len(old['planning_events'])==len(r['planning_events'])
                                        for a,b in zip(old['planning_events'],r['planning_events']):
                                            for name,value in a.items():assert b[name]==value,('Legacy event mismatch',name)
                                        reproduced+=1
                                    output.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
                                    key={k:r[k] for k in (*KEY,'world_id','seed')}
                                    values={k:r[k] for k in ('visits','total_distance_m','total_time_s','planning_calls','simulated_planning_latency_s','measured_planning_wall_s','uncached_solver_compute_s','public_plan_cache_hits')}
                                    metrics.append({**key,**r['evaluation'],**values})
                                    if solver=='insertion_2opt' and stopping=='time_only' and world['mode']=='aligned' and noise=='moderate' and budget==1200 and world['seed']==2026102800 and policy!='online_frozen':examples.append(r)
                    print(json.dumps({'region':region,'solver':solver,'stopping':stopping,'routes_completed':len(metrics),'elapsed_s':round(time.monotonic()-begin,1)},ensure_ascii=False),flush=True)
    assert len(metrics)==8640 and reproduced==2160 and len(examples)==6
    save(OUT/'metrics.json',metrics);save(OUT/'examples.json',examples)
    save(OUT/'runtime.json',{'status':'COMPLETE','routes':len(metrics),'legacy_exact_reproductions':reproduced,
         'loop_seconds':time.monotonic()-begin,'planner_invocations':sum(r['planning_calls'] for r in metrics),
         'public_cache_hits':sum(r['public_plan_cache_hits'] for r in metrics),
         'sum_uncached_solver_compute_s':sum(r['uncached_solver_compute_s'] for r in metrics),
         'new_perception_calls':0,'gpu_rented':False,'new_real_labels':0,'new_worlds':0})


def summarize():
    verify();rows=read(OUT/'metrics.json');groups=defaultdict(list);means=[];pairs=[];effects=[]
    for r in rows:groups[tuple(r[k] for k in KEY)].append(r)
    lookup={(r['world_id'],r['noise'],r['budget_s'],r['solver'],r['stopping'],r['policy']):r for r in rows}
    scalar=('detected_target_cells','visited_target_cells','false_positive_observations','visits','total_distance_m','total_time_s','first_detection_or_budget_s','planning_calls','measured_planning_wall_s')
    for group,rs in sorted(groups.items()):
        assert len(rs)==10
        entry=dict(zip(KEY,group));entry.update(runs=10,**{k:float(np.mean([r[k] for r in rs])) for k in scalar})
        entry.update(no_detection_rate=sum(r['first_detection_censored'] for r in rs)/10,
                     no_arrival_rate=sum(r['first_target_arrival_s'] is None for r in rs)/10,
                     first_arrival_or_budget_s=float(np.mean([r['first_target_arrival_s'] if r['first_target_arrival_s'] is not None else r['budget_s'] for r in rs])),
                     at_30_visit_count=sum(r['visits']==30 for r in rs),
                     mean_unused_budget_s=float(np.mean([r['budget_s']-r['total_time_s'] for r in rs])))
        for k in ('visited_target_recall','detected_target_recall'):
            valid=[r[k] for r in rs if r[k] is not None];entry[k]=float(np.mean(valid)) if valid else None
        means.append(entry)
        if group[-1]=='online':
            ordered=sorted(rs,key=lambda r:r['seed'])
            for comparator in ('post_static','pre_static','online_frozen'):
                other=[lookup[r['world_id'],r['noise'],r['budget_s'],r['solver'],r['stopping'],comparator] for r in ordered]
                delta=[r['detected_target_cells']-b['detected_target_cells'] for r,b in zip(ordered,other)]
                pairs.append({**dict(zip(KEY[:-1],group[:-1])),'comparator':comparator,'paired_runs':10,'individual_detected_deltas':delta,
                              'mean_detected_delta':float(np.mean(delta)),
                              'win_tie_loss':[sum(d>0 for d in delta),sum(d==0 for d in delta),sum(d<0 for d in delta)],
                              'mean_restricted_first_detection_delta_s':float(np.mean([r['first_detection_or_budget_s']-b['first_detection_or_budget_s'] for r,b in zip(ordered,other)])),
                              'mean_no_detection_rate_delta':float(np.mean([int(r['first_detection_censored'])-int(b['first_detection_censored']) for r,b in zip(ordered,other)]))})
        if group[4]=='insertion_2opt':
            old=[lookup[r['world_id'],r['noise'],r['budget_s'],'greedy',r['stopping'],r['policy']] for r in rs]
            effects.append({**entry,'solver_gain_detected':float(np.mean([r['detected_target_cells']-b['detected_target_cells'] for r,b in zip(rs,old)]))})
    primary=[]
    for solver in SOLVERS:
        for stopping in STOPS:
            rs=[r for r in pairs if r['solver']==solver and r['stopping']==stopping and r['budget_s']==1200 and r['comparator']=='post_static']
            primary.append({'solver':solver,'stopping':stopping,'positive':sum(r['mean_detected_delta']>0 for r in rs),'zero':sum(r['mean_detected_delta']==0 for r in rs),'negative':sum(r['mean_detected_delta']<0 for r in rs)})
    save(OUT/'condition_means.json',means);save(OUT/'paired_comparisons.json',pairs);save(OUT/'solver_effects.json',effects)
    save(OUT/'summary.json',{'status':'SIMULATION_ONLY','primary_budget_s':1200,'primary_condition_counts':primary,
         'budget_violations':sum(r['total_time_s']>r['budget_s']+1e-8 for r in rows),
         'real_efficacy_established':False,'novel_method_established':False,'reused_worlds':True})
    print(json.dumps({'summary':primary},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('plan','run','summarize'),required=True);a=parser.parse_args()
    assert not (OUT/'completion_receipt.json').exists(),'Completed output frozen'
    {'plan':plan,'run':run,'summarize':summarize}[a.stage]()
