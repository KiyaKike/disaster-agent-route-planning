"""Freeze then compare one-time route revisions from identical paid histories."""
import argparse
from collections import defaultdict
import gzip
import json
import subprocess
import sys
import time

import numpy as np

from run_route_loop import ROOT,HERE,maps,read,save,digest,checked
from run_route_factor import OUT as FACTOR,PARENT as BASE,verify as verify_factor,verify_parent as verify_base
from route_core import NOISE
from route_factor_core import Planner
from route_snapshot_core import Session,PrivateWorld,BRANCHES,finish_branch,score_branch

OUT=ROOT/'output/disaster_route_snapshot_v3'
SOURCES=['route_snapshot_core.py','test_route_snapshot.py','run_route_snapshot.py','audit_route_snapshot.py','report_route_snapshot.py']
CHECKPOINTS=(5,10,20)
GROUP=('region','mode','noise','checkpoint','branch')


def verify_parents():
    receipt=read(FACTOR/'completion_receipt.json')
    for name,sha in receipt['files'].items():checked(FACTOR/name,sha)
    verify_factor();verify_base()


def verify():
    p=read(OUT/'protocol.json')
    for name,sha in p['source_sha256'].items():checked(HERE/name,sha)
    for name,sha in p['input_sha256'].items():checked(name,sha)
    checked(OUT/'test_log.txt',p['test_log_sha256'])
    return p


def world_registry():
    mm=maps();registry={};index={}
    for w in read(BASE/'synthetic_worlds.json'):
        g=mm[w['region']];targets=np.array([sid in set(w['target_ids']) for sid in g.ids],bool)
        draw=np.array(w['uniform_observation_draws'])
        for noise,(se,fp) in NOISE.items():
            token=__import__('hashlib').sha256((w['world_id']+'|'+noise).encode()).hexdigest()
            registry[token]=PrivateWorld(token,targets,draw<np.where(targets,se,fp),(se,fp))
            index[w['world_id'],noise]=token
    return mm,registry,index


def plan():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists();verify_parents()
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_route*.py','-v'],capture_output=True,text=True)
    (OUT/'test_log.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0 and 'Ran 39 tests' in tests.stdout+tests.stderr
    # Selection feasibility uses public geometry and priors only.
    mm=maps();planner=Planner();counts={}
    for region,g in mm.items():
        dummy=PrivateWorld('plan-only-'+region,np.zeros(len(g.ids),bool),np.zeros(len(g.ids),bool),(.8,.1))
        session=Session(g,dummy);session.plan(planner,'prior');counts[region]=len(session.state['remaining_route'])
        assert counts[region]>max(CHECKPOINTS)
    inputs={str(BASE/n):digest(BASE/n) for n in ('synthetic_worlds.json','public_geometry.json','completion_receipt.json')}
    for name in ('completion_receipt.json','routes.jsonl.gz','metrics.json','protocol.json'):
        inputs[str(FACTOR/name)]=digest(FACTOR/name)
    for name,sha in read(BASE/'protocol.json')['external_sha256'].items():inputs[name]=sha
    p={'client_date':'2026-11-02','stage':'SAME_STATE_ONE_TIME_ROUTE_REVISION_V3',
       'source_sha256':{n:digest(HERE/n) for n in SOURCES},'input_sha256':inputs,'test_log_sha256':digest(OUT/'test_log.txt'),
       'inspiration':{'paper':'DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale',
                      'url':'https://arxiv.org/abs/2609.22978','sections':['6.1','6.2','6.4','6.5'],
                      'scope':'Checkpoint/restore and evaluator separation inspire this implementation. Counterfactual route forks are our adaptation,not a DSec routing algorithm.'},
       'question':'After exactly the same paid observation history,is one immediate route revision worth its cost?',
       'selection':{'behavior':'Strong solver,post_static,time_only,budget1200;track Bayes beliefs without changing the collection path.',
                    'worlds':'Reuse every one of the60 sealed V1 worlds and3 noise modes,including all favorable and adverse conditions.',
                    'checkpoints_after_observations':CHECKPOINTS,'primary_checkpoint':10,'public_initial_route_visits':counts,
                    'snapshots':540,'branches_per_snapshot':3,'suffix_rollouts':1620,'parent_episodes':180,
                    'no_outcome_based_selection':True},
       'branches':{'continue':'Keep the remaining original route;no new planning call.',
                   'replan_feedback':'Pay1s,replan once using current posterior,then commit to that suffix until return.',
                   'replan_prior':'Pay1s,replan once using original post prior and current position/time/visited set,then commit;control for rescheduling without feedback.'},
       'costs':{'budget_s':1200,'speed_m_s':5,'observe_s':20,'planning_latency_s':1,'stopping':'time_only,no repeats',
                'rule':'The paid prefix is copied,not executed or charged again. Every suffix includes its own real movement,inspection,new planning fee and return. No separate duplicate turning/transfer penalty.'},
       'snapshot':'JSON roundtrip preserves complete public history,posterior,position,spent cost,remaining route and private observation cursor. Copies have separate mutable state.',
       'information_boundary':'Planner receives geometry and a copied public state only. Private world references,targets and future signals are evaluator/environment-side. This is interface-level isolation for fixed Python policies,not a malicious-code sandbox.',
       'primary_metric':'Paired difference in correctly detected synthetic target cells after checkpoint10,replan_feedback minus continue,perregion/prior/noise condition.',
       'secondary':['checkpoint5 and20','feedback minus prior replan','prior replan minus continue','future coverage','false positives','next-arrival/detection delay','suffix distance/time','failure rate','public predicted versus realized gain'],
       'predicted_gain':'Before executing future actions,sum sensitivity*(.02+.58*current block posterior) over each chosen suffix. Descriptive forecast diagnostic only,no trigger fitted or deployed.',
       'failure_handling':'Next detection delay null on failure,restricted delay uses original remaining budget;separate failure rate. Prefix detections excluded from future reward.',
       'pairing':'All branches share world and paid prefix. Checkpoints within an episode and noise/prior variants are correlated;540 snapshots are not540 independent events.',
       'examples':'Both regions,aligned/moderate/first seed2026102800/checkpoint10;show all branches even if zero detections.',
       'claim_gate':'Development counterfactual simulation only. One-time revision effect does not equal full online-policy improvement. No thresholds,learner or real-world efficacy claim.',
       'new_training':False,'new_perception_calls':0,'new_downloads':False,'gpu_rental':False,'new_real_labels':0}
    save(OUT/'protocol.json',p);save(OUT/'selection_receipt.json',{'protocol_sha256':digest(OUT/'protocol.json'),'forks_executed':False})
    print(json.dumps({'protocol':'FROZEN','snapshots':540,'suffix_rollouts':1620,'tests_passed':39}),flush=True)


def run():
    verify();assert not (OUT/'fork_results.jsonl.gz').exists()
    started=time.monotonic();mm,registry,index=world_registry();planner=Planner();metrics=[];snapshot_index=[];examples=[]
    parent={}
    for line in gzip.open(FACTOR/'routes.jsonl.gz','rt',encoding='utf-8'):
        r=json.loads(line)
        if r['solver']=='insertion_2opt' and r['stopping']=='time_only' and r['budget_s']==1200 and r['policy']=='post_static':
            parent[r['world_id'],r['noise']]=r
    reproduced=0
    with gzip.open(OUT/'private_snapshots.jsonl.gz','wt',encoding='utf-8') as private_out, (OUT/'snapshots_public.jsonl').open('w') as public_out, gzip.open(OUT/'fork_results.jsonl.gz','wt',encoding='utf-8') as results_out:
        for w in read(BASE/'synthetic_worlds.json'):
            g=mm[w['region']]
            for noise in NOISE:
                world=registry[index[w['world_id'],noise]];session=Session(g,world);session.plan(planner,'prior')
                original=parent[w['world_id'],noise]
                assert session.state['remaining_route']==[h['cell_index'] for h in original['steps']]
                for k in range(1,max(CHECKPOINTS)+1):
                    session.step()
                    if k not in CHECKPOINTS:continue
                    snapshot=json.loads(json.dumps(session.snapshot()));sid=snapshot['sha256']
                    metadata={'snapshot_id':sid,'world_id':w['world_id'],'region':w['region'],'mode':w['mode'],'seed':w['seed'],'noise':noise,'checkpoint':k}
                    snapshot_index.append(metadata)
                    private_out.write(json.dumps({'metadata':metadata,'snapshot':snapshot},ensure_ascii=False,separators=(',',':'))+'\n')
                    public_out.write(json.dumps({'snapshot_id':sid,'public':snapshot['payload']['public']},ensure_ascii=False,separators=(',',':'))+'\n')
                    for branch in BRANCHES:
                        r=finish_branch(g,snapshot,registry,branch,planner);score=score_branch(r,world)
                        assert r['final_public']['history'][:k]==session.state['history']
                        if branch=='continue':
                            final=r['final_public'];assert final['visited_indices']==[h['cell_index'] for h in original['steps']]
                            assert final['elapsed_s']==original['total_time_s'] and final['distance_m']==original['total_distance_m']
                            assert score['total_detected_target_cells']==original['evaluation']['detected_target_cells']
                            for a,b in zip(final['history'],original['steps']):
                                for key in ('cell_index','cell_id','leg_m','arrival_s','observation_complete_s','observed_positive'):
                                    assert a[key]==b[key]
                            reproduced+=1
                        r.update(metadata);r['evaluation']=score
                        results_out.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
                        metrics.append({**metadata,'branch':branch,**score,'public_expected_future_detections':r['public_expected_future_detections'],
                                        'fork_elapsed_s':r['fork_elapsed_s'],'branch_planner_wall_s':r['branch_planner_wall_s']})
                        if w['mode']=='aligned' and noise=='moderate' and w['seed']==2026102800 and k==10:examples.append(r)
            if len(snapshot_index)%90==0:print(json.dumps({'snapshots':len(snapshot_index),'branches':len(metrics),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    assert len(snapshot_index)==540 and len(metrics)==1620 and reproduced==540 and len(examples)==6
    save(OUT/'snapshot_index.json',snapshot_index);save(OUT/'metrics.json',metrics);save(OUT/'examples.json',examples)
    save(OUT/'runtime.json',{'status':'COMPLETE','snapshots':540,'suffix_rollouts':1620,'continued_parent_reproductions':540,
                           'unique_parent_episodes':180,'loop_seconds':time.monotonic()-started,
                           'sum_branch_planner_wall_s':sum(r['branch_planner_wall_s'] for r in metrics),
                           'new_training':False,'gpu_rented':False,'new_perception_calls':0,'new_real_labels':0})


def summarize():
    verify();rows=read(OUT/'metrics.json');lookup={(r['snapshot_id'],r['branch']):r for r in rows};groups=defaultdict(list);means=[];pairs=[]
    for r in rows:groups[tuple(r[k] for k in GROUP)].append(r)
    fields=['future_detected_target_cells','future_visited_target_cells','future_false_positive_observations','future_observations',
            'next_detection_or_remaining_budget_s','suffix_distance_m','suffix_elapsed_s','total_time_s','additional_planning_calls','public_expected_future_detections']
    for group,rs in sorted(groups.items()):
        assert len(rs)==10
        means.append({**dict(zip(GROUP,group)),'runs':10,**{key:float(np.mean([r[key] for r in rs])) for key in fields},
                      'no_future_detection_rate':sum(r['no_future_detection'] for r in rs)/10})
        if group[-1]!='replan_feedback':continue
        for first,second in [('replan_feedback','continue'),('replan_prior','continue'),('replan_feedback','replan_prior')]:
            ordered=sorted(rs,key=lambda r:r['seed']);a=[lookup[r['snapshot_id'],first] for r in ordered];b=[lookup[r['snapshot_id'],second] for r in ordered]
            delta=[x['future_detected_target_cells']-y['future_detected_target_cells'] for x,y in zip(a,b)]
            predicted=[x['public_expected_future_detections']-y['public_expected_future_detections'] for x,y in zip(a,b)]
            pairs.append({**dict(zip(GROUP[:-1],group[:-1])),'first':first,'second':second,'paired_snapshots':10,
                          'individual_detected_deltas':delta,'mean_detected_delta':float(np.mean(delta)),
                          'individual_public_predicted_deltas':predicted,'mean_public_predicted_delta':float(np.mean(predicted)),
                          'win_tie_loss':[sum(d>0 for d in delta),sum(d==0 for d in delta),sum(d<0 for d in delta)],
                          'mean_suffix_distance_delta_m':float(np.mean([x['suffix_distance_m']-y['suffix_distance_m'] for x,y in zip(a,b)])),
                          'mean_future_observations_delta':float(np.mean([x['future_observations']-y['future_observations'] for x,y in zip(a,b)])),
                          'mean_next_detection_penalty_delta_s':float(np.mean([x['next_detection_or_remaining_budget_s']-y['next_detection_or_remaining_budget_s'] for x,y in zip(a,b)])),
                          'mean_no_detection_rate_delta':float(np.mean([int(x['no_future_detection'])-int(y['no_future_detection']) for x,y in zip(a,b)]))})
    primary=[p for p in pairs if p['checkpoint']==10 and p['first']=='replan_feedback' and p['second']=='continue']
    diagnostics=[]
    for noise in NOISE:
        rs=[r for r in rows if r['checkpoint']==10 and r['branch']=='replan_feedback' and r['noise']==noise]
        data=[]
        for a in rs:
            b=lookup[a['snapshot_id'],'continue'];data.append({'snapshot_id':a['snapshot_id'],
                'predicted_delta':a['public_expected_future_detections']-b['public_expected_future_detections'],
                'realized_delta':a['future_detected_target_cells']-b['future_detected_target_cells']})
        positive=[r for r in data if r['predicted_delta']>0]
        diagnostics.append({'noise':noise,'snapshots':len(rs),'positive_predicted_gain':len(positive),
                            'actual_losses_among_positive_predictions':sum(r['realized_delta']<0 for r in positive),'data':data})
    summary={'status':'DEVELOPMENT_SIMULATION_ONLY','primary_checkpoint':10,
             'primary_condition_counts':{'positive':sum(p['mean_detected_delta']>0 for p in primary),
                                         'zero':sum(p['mean_detected_delta']==0 for p in primary),'negative':sum(p['mean_detected_delta']<0 for p in primary)},
             'primary_snapshot_win_tie_loss':[sum(p['win_tie_loss'][i] for p in primary) for i in range(3)],
             'budget_violations':sum(r['total_time_s']>1200+1e-8 for r in rows),'trigger_fitted':False,'new_policy_efficacy_established':False,
             'scope':'One-time route revision after shared static collection history;not the previous always-online policy and not independent real-disaster replications.'}
    save(OUT/'condition_means.json',means);save(OUT/'paired_comparisons.json',pairs);save(OUT/'prediction_diagnostics.json',diagnostics);save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('plan','run','summarize'),required=True);a=parser.parse_args()
    assert not (OUT/'completion_receipt.json').exists(),'Completed output frozen'
    {'plan':plan,'run':run,'summarize':summarize}[a.stage]()
