"""Independent snapshot ledger, observation, fork reward and summary verification."""
from collections import Counter
import gzip
import hashlib
import json
import math

import numpy as np

from run_route_snapshot import OUT,BASE,FACTOR,GROUP,CHECKPOINTS,read,save,verify,verify_parents,maps
from route_core import Config
from route_factor_core import Planner


def near(a,b):
    if isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(a-b)<1e-7,(a,b)
    else:assert a==b,(a,b)


def distance(a,b):return math.hypot(float(a[0]-b[0]),float(a[1]-b[1]))


def main():
    verify();verify_parents();mm=maps();worlds={w['world_id']:w for w in read(BASE/'synthetic_worlds.json')}
    noise={'perfect':(1.,0.),'moderate':(.8,.1),'poor':(.6,.25)}
    private={};public={};steps_checked=plans_checked=0;solver=Planner()
    allowed={'region','budget_s','speed_m_s','observe_s','planning_latency_s','position_index','elapsed_s','distance_m',
             'visited_indices','belief_hot','remaining_route','planning_events','history','closed','return_m'}
    for line in (OUT/'snapshots_public.jsonl').open():
        row=json.loads(line);assert set(row)=={'snapshot_id','public'} and set(row['public'])==allowed
        public[row['snapshot_id']]=row['public']
    for line in gzip.open(OUT/'private_snapshots.jsonl.gz','rt',encoding='utf-8'):
        row=json.loads(line);snap=row['snapshot'];payload=snap['payload'];meta=row['metadata']
        canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        assert hashlib.sha256(canonical.encode()).hexdigest()==snap['sha256']==meta['snapshot_id']
        assert public[meta['snapshot_id']]==payload['public']
        assert meta['checkpoint'] in CHECKPOINTS and len(payload['public']['history'])==meta['checkpoint']
        assert payload['environment']['seen']==sorted(payload['public']['visited_indices'])
        token=hashlib.sha256((meta['world_id']+'|'+meta['noise']).encode()).hexdigest()
        assert payload['environment']['world_token']==token
        w=worlds[meta['world_id']];g=mm[meta['region']];se,fp=noise[meta['noise']]
        target=np.array([sid in set(w['target_ids']) for sid in g.ids],bool)
        bits=np.array(w['uniform_observation_draws'])<np.where(target,se,fp)
        world_payload={'token':token,'targets':target.tolist(),'observations':bits.tolist(),'likelihood':(se,fp)}
        world_canonical=json.dumps(world_payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        assert payload['environment']['world_sha256']==hashlib.sha256(world_canonical.encode()).hexdigest()
        private[meta['snapshot_id']]=row
    assert len(private)==len(public)==540
    counts=Counter((r['metadata']['region'],r['metadata']['mode'],r['metadata']['noise'],r['metadata']['checkpoint']) for r in private.values())
    assert len(counts)==54 and all(n==10 for n in counts.values())
    parent={}
    for line in gzip.open(FACTOR/'routes.jsonl.gz','rt',encoding='utf-8'):
        r=json.loads(line)
        if r['solver']=='insertion_2opt' and r['stopping']=='time_only' and r['budget_s']==1200 and r['policy']=='post_static':parent[r['world_id'],r['noise']]=r

    def replay(s,meta,check_choices):
        nonlocal steps_checked,plans_checked
        assert set(s)==allowed and s['budget_s']==1200 and (s['speed_m_s'],s['observe_s'],s['planning_latency_s'])==(5.,20.,1.)
        g=mm[meta['region']];w=worlds[meta['world_id']];targets=np.array([sid in set(w['target_ids']) for sid in g.ids],bool)
        se,fp=noise[meta['noise']];bits=np.array(w['uniform_observation_draws'])<np.where(targets,se,fp)
        belief=g.prior_post.copy();used=[];position=g.start_index;elapsed=travel=0.;cursor=0;pending=[]
        for j in range(len(s['history'])+1):
            while cursor<len(s['planning_events']) and s['planning_events'][cursor]['after_observations']==j:
                e=s['planning_events'][cursor];near(e['elapsed_before_s'],elapsed);assert e['position_index']==position
                elapsed+=1;near(e['elapsed_after_s'],elapsed)
                chosen=e['planned_route'];assert len(chosen)==len(set(chosen)) and not(set(chosen)&set(used))
                nodes=[position,*chosen,g.start_index];length=sum(distance(g.xy[a],g.xy[b]) for a,b in zip(nodes,nodes[1:]))
                assert elapsed+length/5+20*len(chosen)<=1200+1e-7
                if check_choices:
                    seen=np.zeros(len(g.ids),bool);seen[used]=True
                    expected_belief=belief if e['belief_source']=='feedback' else g.prior_post
                    # Reuse the unchanged V2 solver; independently reconstruct every input.
                    expected,_=solver.solve(g,'insertion_2opt',g.xy[position],seen,e['elapsed_after_s'],expected_belief,Config(max_visits=len(g.ids)),0.,len(g.ids)-len(used))
                    assert chosen==expected
                pending=list(chosen);cursor+=1;plans_checked+=1
            if j==len(s['history']):break
            h=s['history'][j];i=h['cell_index'];assert pending and pending.pop(0)==i and i not in used
            assert h['cell_id']==g.ids[i] and h['step']==j+1 and h['block_index']==int(g.blocks[i])
            leg=distance(g.xy[position],g.xy[i]);arrival=elapsed+leg/5;end=arrival+20
            for key,value in [('elapsed_before_step_s',elapsed),('leg_m',leg),('arrival_s',arrival),('observation_complete_s',end)]:near(h[key],value)
            assert end+distance(g.xy[i],g.xy[g.start_index])/5<=1200+1e-7
            assert h['observed_positive']==bool(bits[i]);b=int(g.blocks[i]);near(h['block_belief_before'],belief[b])
            a=fp+(se-fp)*.6;c=fp+(se-fp)*.02
            if not bits[i]:a,c=1-a,1-c
            belief[b]=belief[b]*a/(belief[b]*a+(1-belief[b])*c)
            near(h['block_belief_after'],belief[b]);used.append(i);position=i;elapsed=end;travel+=leg;steps_checked+=1
        assert cursor==len(s['planning_events']) and used==s['visited_indices']
        np.testing.assert_allclose(belief,s['belief_hot'],atol=1e-12,rtol=0)
        if s['closed']:
            assert not pending and not s['remaining_route'];back=distance(g.xy[position],g.xy[g.start_index])
            near(back,s['return_m']);elapsed+=back/5;travel+=back;position=g.start_index
        else:assert pending==s['remaining_route'] and s['return_m']==0
        assert position==s['position_index'];near(elapsed,s['elapsed_s']);near(travel,s['distance_m'])
        assert elapsed<=1200+1e-7
        near(s['elapsed_s'],s['distance_m']/5+len(s['history'])*20+len(s['planning_events']))
        return targets

    for row in private.values():replay(row['snapshot']['payload']['public'],row['metadata'],True)
    metrics=read(OUT/'metrics.json');lookup={(r['snapshot_id'],r['branch']):r for r in metrics};assert len(lookup)==len(metrics)==1620
    continued=0;seen_results=set()
    for line in gzip.open(OUT/'fork_results.jsonl.gz','rt',encoding='utf-8'):
        r=json.loads(line);row=private[r['snapshot_id']];meta=row['metadata'];snapshot=row['snapshot']['payload']['public'];s=r['final_public'];k=meta['checkpoint']
        assert s['history'][:k]==snapshot['history'] and s['planning_events'][:1]==snapshot['planning_events']
        assert r['fork_elapsed_s']==snapshot['elapsed_s'] and r['fork_observation_count']==k and r['fork_planning_calls']==1
        assert len(s['planning_events'])==1+int(r['branch']!='continue')
        if r['branch']!='continue':assert s['planning_events'][1]['belief_source']==('feedback' if r['branch']=='replan_feedback' else 'prior')
        targets=replay(s,meta,True);g=mm[meta['region']];future=s['history'][k:];se,_=noise[meta['noise']]
        assert r['planned_suffix_indices']==[h['cell_index'] for h in future]
        expected=sum(se*(.02+.58*snapshot['belief_hot'][g.blocks[i]]) for i in r['planned_suffix_indices'])
        near(expected,r['public_expected_future_detections'])
        detected=[h for h in future if targets[h['cell_index']] and h['observed_positive']]
        arrived=[h for h in future if targets[h['cell_index']]]
        all_detected=sum(bool(targets[h['cell_index']]) and h['observed_positive'] for h in s['history'])
        delay=detected[0]['observation_complete_s']-snapshot['elapsed_s'] if detected else None
        expected_score={'future_detected_target_cells':len(detected),'future_visited_target_cells':len(arrived),
            'future_false_positive_observations':sum(not targets[h['cell_index']] and h['observed_positive'] for h in future),
            'future_observations':len(future),'total_detected_target_cells':all_detected,
            'next_target_arrival_delay_s':arrived[0]['arrival_s']-snapshot['elapsed_s'] if arrived else None,
            'next_detection_delay_s':delay,'no_future_detection':not bool(detected),
            'next_detection_or_remaining_budget_s':delay if delay is not None else 1200-snapshot['elapsed_s'],
            'suffix_distance_m':s['distance_m']-sum(h['leg_m'] for h in snapshot['history']),
            'suffix_elapsed_s':s['elapsed_s']-snapshot['elapsed_s'],'total_distance_m':s['distance_m'],
            'total_time_s':s['elapsed_s'],'additional_planning_calls':int(r['branch']!='continue')}
        key=(r['snapshot_id'],r['branch']);m=lookup[key];seen_results.add(key)
        for name,value in expected_score.items():near(r['evaluation'][name],value);near(m[name],value)
        if r['branch']=='continue':
            old=parent[meta['world_id'],meta['noise']]
            assert s['visited_indices']==[h['cell_index'] for h in old['steps']]
            assert s['elapsed_s']==old['total_time_s'] and s['distance_m']==old['total_distance_m']
            assert all_detected==old['evaluation']['detected_target_cells'];continued+=1
    assert len(seen_results)==1620 and continued==540
    means=read(OUT/'condition_means.json');assert len(means)==162
    for m in means:
        rs=[r for r in metrics if all(r[k]==m[k] for k in GROUP)];assert len(rs)==10
        for key in ('future_detected_target_cells','future_visited_target_cells','future_false_positive_observations','future_observations','next_detection_or_remaining_budget_s','suffix_distance_m','suffix_elapsed_s','total_time_s','additional_planning_calls','public_expected_future_detections'):
            near(m[key],sum(r[key] for r in rs)/10)
        near(m['no_future_detection_rate'],sum(r['no_future_detection'] for r in rs)/10)
    pairs=read(OUT/'paired_comparisons.json');assert len(pairs)==162
    for p in pairs:
        rs=sorted([r for r in metrics if r['branch']==p['first'] and all(r[k]==p[k] for k in GROUP[:-1])],key=lambda r:r['seed'])
        other=[lookup[r['snapshot_id'],p['second']] for r in rs]
        ds=[a['future_detected_target_cells']-b['future_detected_target_cells'] for a,b in zip(rs,other)]
        assert ds==p['individual_detected_deltas'];near(p['mean_detected_delta'],sum(ds)/10)
        pred=[a['public_expected_future_detections']-b['public_expected_future_detections'] for a,b in zip(rs,other)]
        np.testing.assert_allclose(pred,p['individual_public_predicted_deltas'],atol=1e-12,rtol=0)
    for diagnostic in read(OUT/'prediction_diagnostics.json'):
        rs=[r for r in metrics if r['noise']==diagnostic['noise'] and r['checkpoint']==10 and r['branch']=='replan_feedback']
        assert len(rs)==diagnostic['snapshots']==60
        positive=[r for r in rs if r['public_expected_future_detections']>lookup[r['snapshot_id'],'continue']['public_expected_future_detections']]
        assert len(positive)==diagnostic['positive_predicted_gain']
        assert sum(r['future_detected_target_cells']<lookup[r['snapshot_id'],'continue']['future_detected_target_cells'] for r in positive)==diagnostic['actual_losses_among_positive_predictions']
    verify_parents()
    save(OUT/'audit_report.json',{'status':'PASS','public_private_snapshots_checked':540,'branches_checked':1620,
         'history_steps_replayed_including_shared_prefixes':steps_checked,'planning_events_replayed_including_shared_prefixes':plans_checked,
         'continue_parent_reproductions':continued,'unique_parent_episodes':180,'condition_means_checked':162,'paired_comparisons_checked':162,
         'budget_violations':0,'prefix_recharged_or_reobserved':0,'original_outputs_and_frozen_sources_unchanged':True,
         'boundary':'Interface-level public/private separation;no claim of malicious-code sandboxing.',
         'replay':'Timing,belief,observations,rewards and summaries independently reconstructed. Planner choices replayed using unchanged V2 solver and independently rebuilt inputs.'})
    print(json.dumps({'audit':'PASS','snapshots':540,'branches':1620,'history_steps':steps_checked,'parent_reproductions':continued}),flush=True)


if __name__=='__main__':
    assert not (OUT/'completion_receipt.json').exists()
    main()
