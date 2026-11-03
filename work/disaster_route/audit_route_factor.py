"""Independent feasibility, signal, belief, reward and factorial-summary checks."""
from collections import Counter
import gzip
import json
import math

import numpy as np

from run_route_factor import OUT,PARENT,KEY,read,save,verify,verify_parent


def distance(a,b):
    return math.hypot(float(a[0]-b[0]),float(a[1]-b[1]))


def same(a,b):
    if isinstance(a,(float,int)) and not isinstance(a,bool):
        assert abs(a-b)<1e-7,(a,b)
    else:
        assert a==b,(a,b)


def greedy_reference(xy,home,position,visited,elapsed,belief,blocks,budget,limit,latency):
    route=[];used=set(visited);back=np.sqrt(((xy-home)**2).sum(axis=1))/5
    for _ in range(limit):
        travel=np.sqrt(((xy-position)**2).sum(axis=1))/5
        cost=travel+20+latency
        valid=elapsed+cost+back<=budget+1e-9
        if used:valid[list(used)]=False
        if not valid.any():break
        utility=np.where(valid,(.02+.58*belief[blocks])/cost,-np.inf)
        index=int(np.argmax(utility));route.append(index);used.add(index)
        elapsed+=float(cost[index]);position=xy[index]
    return route


def main():
    verify();verify_parent()
    public=read(PARENT/'public_geometry.json');worlds={w['world_id']:w for w in read(PARENT/'synthetic_worlds.json')}
    metrics=read(OUT/'metrics.json');lookup={(r['world_id'],r['noise'],r['budget_s'],r['solver'],r['stopping'],r['policy']):r for r in metrics}
    assert len(metrics)==len(lookup)==8640
    legacy={(r['world_id'],r['noise'],r['budget_s'],r['policy']):r for r in read(PARENT/'metrics.json')}
    noise={'perfect':(1.,0.),'moderate':(.8,.1),'poor':(.6,.25)}
    checked_plans={};groups=Counter();steps_checked=events_checked=legacy_checked=routes=0
    for line in gzip.open(OUT/'routes.jsonl.gz','rt',encoding='utf-8'):
        r=json.loads(line);g=public[r['region']];xy=np.array(g['xy_m']);blocks=np.array(g['block_indices']);ids=g['ids']
        home_index=ids.index(g['start_id']);home=xy[home_index];position=home.copy();world=worlds[r['world_id']]
        targets=np.array([sid in set(world['target_ids']) for sid in ids],bool);draw=np.array(world['uniform_observation_draws'])
        se,fp=noise[r['noise']];bits=draw<np.where(targets,se,fp)
        belief=np.array(g['prior_pre_hot_probability'] if r['policy']=='pre_static' else g['prior_post_hot_probability'])
        static=r['policy'] in ('pre_static','post_static');elapsed=1. if static else 0.;visited=set();travel=0.
        cap=30 if r['stopping']=='cap30' else len(ids);events=r['planning_events'];nt=int(targets.sum());arrivals=[];correct=[];falsepos=0
        assert len(events)==r['planning_calls']==(1 if static else len(r['steps']))
        static_path=None
        for j,s in enumerate(r['steps']):
            if not static or j==0:
                event=events[0 if static else j];suffix=event['planned_suffix_indices'];latency=0. if static else 1.
                assert event['after_observations']==j
                assert len(suffix)==len(set(suffix))<=cap-j and not(set(suffix)&visited)
                assert event['first_five_suffix_ids']==[ids[i] for i in suffix[:5]] and event['planned_suffix_visits']==len(suffix)
                nodes=[position,*[xy[i] for i in suffix],home]
                length=sum(distance(a,b) for a,b in zip(nodes,nodes[1:]))
                assert elapsed+len(suffix)*(20+latency)+length/5<=r['budget_s']+1e-7
                reward=float(sum(.02+.58*belief[blocks[i]] for i in suffix))
                same(reward,event['planned_expected_targets'])
                cache_key=(r['region'],position.tobytes(),tuple(sorted(visited)),elapsed,belief.tobytes(),r['budget_s'],cap-j,latency)
                if cache_key not in checked_plans:
                    checked_plans[cache_key]=greedy_reference(xy,home,position,visited,elapsed,belief,blocks,r['budget_s'],cap-j,latency)
                greedy=checked_plans[cache_key]
                if r['solver']=='greedy':assert suffix==greedy
                else:assert reward+1e-8>=sum(.02+.58*belief[blocks[i]] for i in greedy)
                if static:static_path=suffix
                else:assert s['cell_index']==suffix[0]
                events_checked+=1
            if static:assert s['cell_index']==static_path[j]
            index=s['cell_index'];assert index not in visited and ids[index]==s['cell_id'] and s['step']==j+1
            latency=0. if static else 1.;leg=distance(position,xy[index]);arrival=elapsed+latency+leg/5;end=arrival+20
            for key,value in [('elapsed_before_step_s',elapsed),('planning_latency_s',latency),('leg_m',leg),('arrival_s',arrival),('observation_complete_s',end),('return_reserve_s',distance(xy[index],home)/5)]:same(s[key],value)
            assert end+distance(xy[index],home)/5<=r['budget_s']+1e-7
            assert s['observed_positive']==bool(bits[index]);b=blocks[index];same(s['block_belief_before'],belief[b])
            if r['policy']=='online':
                hot=fp+(se-fp)*.6;cold=fp+(se-fp)*.02
                if not bits[index]:hot,cold=1-hot,1-cold
                belief[b]=belief[b]*hot/(belief[b]*hot+(1-belief[b])*cold)
            same(s['block_belief_after'],belief[b])
            if targets[index]:arrivals.append(s['arrival_s'])
            if targets[index] and bits[index]:correct.append(s['observation_complete_s'])
            if not targets[index] and bits[index]:falsepos+=1
            elapsed=end;travel+=leg;visited.add(index);position=xy[index];steps_checked+=1
        if static:assert len(static_path)==len(r['steps'])
        back=distance(position,home);total=elapsed+back/5
        same(r['return_m'],back);same(r['total_distance_m'],travel+back);same(r['total_time_s'],total)
        same(r['movement_seconds'],(travel+back)/5);same(r['observation_seconds'],20*len(visited));same(r['simulated_planning_latency_s'],len(events))
        same(total,r['movement_seconds']+r['observation_seconds']+len(events));assert total<=r['budget_s']+1e-7
        assert len(visited)==r['visits']<=cap and r['closed_route_ids']==[g['start_id']]+[s['cell_id'] for s in r['steps']]+[g['start_id']]
        evaluation={'true_target_cells':nt,'visited_target_cells':len(arrivals),'detected_target_cells':len(correct),'false_positive_observations':falsepos,
                    'visited_target_recall':len(arrivals)/nt if nt else None,'detected_target_recall':len(correct)/nt if nt else None,
                    'first_target_arrival_s':arrivals[0] if arrivals else None,'first_correct_detection_s':correct[0] if correct else None,
                    'first_detection_or_budget_s':correct[0] if correct else r['budget_s'],'first_detection_censored':not bool(correct)}
        assert r['evaluation']==evaluation
        key=(r['world_id'],r['noise'],r['budget_s'],r['solver'],r['stopping'],r['policy']);m=lookup[key]
        for name,value in evaluation.items():same(m[name],value)
        for name in ('visits','total_distance_m','total_time_s','planning_calls','measured_planning_wall_s'):same(m[name],r[name])
        if r['solver']=='greedy' and r['stopping']=='cap30':
            old=legacy[r['world_id'],r['noise'],r['budget_s'],r['policy']]
            for name,value in old.items():
                if name!='measured_planning_wall_s':assert m[name]==value
            legacy_checked+=1
        groups[tuple(m[k] for k in KEY)]+=1;routes+=1
    assert routes==8640 and legacy_checked==2160 and len(groups)==864 and all(n==10 for n in groups.values())
    means=read(OUT/'condition_means.json');assert len(means)==864
    for m in means:
        rs=[r for r in metrics if all(r[k]==m[k] for k in KEY)]
        for key in ('detected_target_cells','visited_target_cells','false_positive_observations','visits','total_distance_m','total_time_s','first_detection_or_budget_s','planning_calls'):
            same(m[key],sum(r[key] for r in rs)/10)
        same(m['no_detection_rate'],sum(r['first_detection_censored'] for r in rs)/10)
        same(m['first_arrival_or_budget_s'],sum(r['first_target_arrival_s'] if r['first_target_arrival_s'] is not None else r['budget_s'] for r in rs)/10)
    pairs=read(OUT/'paired_comparisons.json');assert len(pairs)==648
    for p in pairs:
        rs=sorted([r for r in metrics if r['policy']=='online' and all(r[k]==p[k] for k in KEY[:-1])],key=lambda r:r['seed'])
        ds=[r['detected_target_cells']-lookup[r['world_id'],r['noise'],r['budget_s'],r['solver'],r['stopping'],p['comparator']]['detected_target_cells'] for r in rs]
        assert ds==p['individual_detected_deltas'];same(p['mean_detected_delta'],sum(ds)/10)
    effects=read(OUT/'solver_effects.json');assert len(effects)==432
    for effect in effects:
        rs=[r for r in metrics if all(r[k]==effect[k] for k in KEY)]
        ds=[r['detected_target_cells']-lookup[r['world_id'],r['noise'],r['budget_s'],'greedy',r['stopping'],r['policy']]['detected_target_cells'] for r in rs]
        same(effect['solver_gain_detected'],sum(ds)/10)
    verify_parent()
    save(OUT/'audit_report.json',{'status':'PASS','routes_checked':routes,'paid_observations_checked':steps_checked,'planned_suffixes_checked':events_checked,
         'unique_public_states_checked':len(checked_plans),'legacy_exact_metric_reproductions':legacy_checked,
         'condition_means_checked':864,'paired_comparisons_checked':648,'solver_effects_checked':432,'budget_violations':0,
         'original_results_and_sources_unchanged':True,'strong_prior_reward_lower_bound_checked':True,
         'scope':'Independent budget,timing,belief,signal,reward and old-greedy reconstruction;strong heuristic checked for feasibility and retained-seed lower bound,not global optimality.'})
    print(json.dumps({'audit':'PASS','routes':routes,'observations':steps_checked,'planned_suffixes':events_checked}),flush=True)


if __name__=='__main__':
    assert not (OUT/'completion_receipt.json').exists()
    main()
