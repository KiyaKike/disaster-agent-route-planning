"""Independent route feasibility, allowed-information and paired-metric replay."""
from collections import defaultdict
import json,math
import numpy as np
from run_route_loop import HERE,ROOT,OUT,DATA,read,digest,checked,save,verify


def distance(a,b):return math.hypot(float(a[0]-b[0]),float(a[1]-b[1]))


def next_suffix(ids,xy,blocks,belief,home,position,visited,elapsed,budget,limit,latency):
    # Independent array formula; no calls to route_core greedy_route.
    result=[];position=position.copy();visited=set(visited)
    for _ in range(limit):
        move=np.sqrt((xy[:,0]-position[0])**2+(xy[:,1]-position[1])**2)
        back=np.sqrt((xy[:,0]-home[0])**2+(xy[:,1]-home[1])**2)
        cost=latency+move/5+20;valid=elapsed+cost+back/5<=budget+1e-9
        if visited:valid[list(visited)]=False
        if not valid.any():break
        utilities=np.where(valid,(.02+.58*belief[blocks])/cost,-np.inf)
        best=int(np.argmax(utilities))
        elapsed+=latency+distance(position,xy[best])/5+20;position=xy[best].copy();visited.add(best);result.append(best)
    return result


def main():
    p=verify();public=read(OUT/'public_geometry.json');source=read(DATA);worlds=read(OUT/'synthetic_worlds.json')
    assert len(worlds)==60 and sum(1 for _ in (OUT/'routes.jsonl').open())==2160
    fields={};noise={'perfect':(1.,0.),'moderate':(.8,.1),'poor':(.6,.25)}
    for name,g in public.items():
        raw=sorted([r for r in source[name]['cells'] if r['area_m2']>=1800],key=lambda r:r['cell_id'])
        assert [r['cell_id'] for r in raw]==g['ids']
        xy=np.array([[r['centroid_easting_m'],r['centroid_northing_m']] for r in raw]);np.testing.assert_array_equal(xy,g['xy_m'])
        origin=np.array([(xy[:,0].min()+xy[:,0].max())/2,xy[:,1].min()]);start=min(range(len(raw)),key=lambda i:(distance(xy[i],origin),raw[i]['cell_id']))
        assert g['start_id']==raw[start]['cell_id']
        keys=sorted({(r['row']//3,r['col']//3) for r in raw});blocks=np.array([keys.index((r['row']//3,r['col']//3)) for r in raw])
        np.testing.assert_array_equal(blocks,g['block_indices'])
        cue=[r['change_cue_score'] for r in raw];means=[]
        for j in range(len(keys)):
            values=[cue[i] for i in range(len(raw)) if blocks[i]==j and cue[i] is not None];means.append(sum(values)/len(values) if values else None)
        known=[v for v in means if v is not None];post=[]
        for value in means:
            if value is None:post.append(.2);continue
            rank=(sum(v<value for v in known)+(sum(v==value for v in known)-1)/2)/(len(known)-1) if len(known)>1 else .5
            post.append(.05+.3*rank)
        np.testing.assert_allclose(post,g['prior_post_hot_probability'],atol=1e-12,rtol=0)
        assert all(v==.2 for v in g['prior_pre_hot_probability'])
    for w in worlds:
        g=public[w['region']];post=np.array(g['prior_post_hot_probability']);rng=np.random.default_rng(w['generator_seed'])
        if w['mode']=='aligned':prob=post
        elif w['mode']=='reversed':
            order=np.argsort(post,kind='stable');prob=np.empty_like(post);prob[order]=post[order[::-1]]
        else:prob=np.full(len(post),.2)
        np.testing.assert_array_equal(prob,w['true_hot_probabilities'])
        hot=rng.random(len(prob))<prob;target=rng.random(len(g['ids']))<np.where(hot[np.array(g['block_indices'])],.6,.02);draw=rng.random(len(target))
        assert np.flatnonzero(hot).tolist()==w['hot_block_indices']
        assert [g['ids'][i] for i in np.flatnonzero(target)]==w['target_ids'];np.testing.assert_array_equal(draw,w['uniform_observation_draws'])
        fields[w['world_id']]=(target,draw)
    metrics=read(OUT/'metrics.json');lookup={(r['world_id'],r['noise'],r['budget_s'],r['policy']):r for r in metrics};assert len(lookup)==2160
    visited_steps=0;audited=[];event_count=0
    for line in (OUT/'routes.jsonl').open():
        r=json.loads(line);g=public[r['region']];ids=g['ids'];xy=np.array(g['xy_m']);blocks=np.array(g['block_indices']);home=xy[ids.index(g['start_id'])]
        target,draw=fields[r['world_id']];se,fp=noise[r['noise']];bits=draw<np.where(target,se,fp)
        belief=np.array(g['prior_pre_hot_probability'] if r['policy']=='pre_static' else g['prior_post_hot_probability'])
        static=r['policy'] in ['pre_static','post_static'];elapsed=1. if static else 0.;position=home.copy();visited=set();travel=0.;plans=0;arrival_target=[];correct=[];falsepos=0
        events=r['planning_events'];assert len(events)==r['planning_calls']
        # Check pre-planning before any observations and the stored planned suffix.
        if static:
            suffix=next_suffix(ids,xy,blocks,belief,home,position,visited,elapsed,r['budget_s'],30,0)
            assert events[0]['after_observations']==0 and events[0]['first_five_suffix_ids']==[ids[i] for i in suffix[:5]] and events[0]['planned_suffix_visits']==len(suffix)
            plans=1
        for j,s in enumerate(r['steps']):
            index=s['cell_index'];assert index not in visited and ids[index]==s['cell_id'] and s['step']==j+1
            latency=0. if static else 1.
            if static:assert index==suffix[j]
            else:
                planned=next_suffix(ids,xy,blocks,belief,home,position,visited,elapsed,r['budget_s'],30-j,1)
                e=events[j];assert e['first_five_suffix_ids']==[ids[i] for i in planned[:5]] and e['planned_suffix_visits']==len(planned) and e['after_observations']==j
                assert index==planned[0];plans+=1
            event_count+=int(not static or j==0)
            assert abs(s['elapsed_before_step_s']-elapsed)<1e-7 and s['planning_latency_s']==latency
            leg=distance(position,xy[index]);arrival=elapsed+latency+leg/5;end=arrival+20;reserve=distance(xy[index],home)/5
            for key,val in [('leg_m',leg),('arrival_s',arrival),('observation_complete_s',end),('return_reserve_s',reserve)]:assert abs(s[key]-val)<1e-7,(key,s[key],val)
            assert end+reserve<=r['budget_s']+1e-8
            assert s['observed_positive']==bool(bits[index])
            b=int(blocks[index]);assert abs(s['block_belief_before']-belief[b])<1e-12
            if r['policy']=='online':
                a=fp+(se-fp)*.6;c=fp+(se-fp)*.02
                hot=a if bits[index] else 1-a;cold=c if bits[index] else 1-c
                belief[b]=belief[b]*hot/(belief[b]*hot+(1-belief[b])*cold)
            assert abs(s['block_belief_after']-belief[b])<1e-12
            if target[index]:arrival_target.append(arrival)
            if target[index] and bits[index]:correct.append(end)
            if not target[index] and bits[index]:falsepos+=1
            position=xy[index];travel+=leg;elapsed=end;visited.add(index);visited_steps+=1
        return_m=distance(position,home);total=elapsed+return_m/5
        assert r['closed_route_ids']==[g['start_id']]+[s['cell_id'] for s in r['steps']]+[g['start_id']]
        assert len(r['steps'])==r['visits']<=30 and r['planning_calls']==plans
        for key,value in [('return_m',return_m),('total_distance_m',travel+return_m),('total_time_s',total),('movement_seconds',(travel+return_m)/5),('observation_seconds',20*len(visited)),('simulated_planning_latency_s',plans)]:assert abs(r[key]-value)<1e-7
        assert total<=r['budget_s']+1e-8 and abs(total-r['movement_seconds']-r['observation_seconds']-plans)<1e-7
        nt=int(target.sum());evaluation={'true_target_cells':nt,'visited_target_cells':len(arrival_target),'detected_target_cells':len(correct),'false_positive_observations':falsepos,
            'visited_target_recall':len(arrival_target)/nt if nt else None,'detected_target_recall':len(correct)/nt if nt else None,
            'first_target_arrival_s':arrival_target[0] if arrival_target else None,'first_correct_detection_s':correct[0] if correct else None,
            'first_detection_or_budget_s':correct[0] if correct else r['budget_s'],'first_detection_censored':not bool(correct)}
        assert evaluation==r['evaluation']
        metric=lookup[r['world_id'],r['noise'],r['budget_s'],r['policy']]
        for key,val in evaluation.items():assert metric[key]==val
        for key in ['visits','total_distance_m','total_time_s','planning_calls','simulated_planning_latency_s','measured_planning_wall_s']:assert metric[key]==r[key]
        audited.append(metric)
    means=read(OUT/'condition_means.json');assert len(means)==216
    for m in means:
        rs=[r for r in audited if all(r[k]==m[k] for k in ['region','mode','noise','budget_s','policy'])];assert len(rs)==10
        for key in ['detected_target_cells','visited_target_cells','false_positive_observations','visits','total_distance_m','total_time_s','first_detection_or_budget_s','planning_calls','simulated_planning_latency_s','measured_planning_wall_s']:
            assert abs(m[key]-sum(r[key] for r in rs)/10)<1e-7
        assert m['no_detection_rate']==sum(r['first_detection_censored'] for r in rs)/10
    paired=read(OUT/'paired_comparisons.json');assert len(paired)==162
    for pair in paired:
        rs=[r for r in audited if all(r[k]==pair[k] for k in ['region','mode','noise','budget_s']) and r['policy']=='online']
        delta=[r['detected_target_cells']-lookup[r['world_id'],r['noise'],r['budget_s'],pair['comparator']]['detected_target_cells'] for r in sorted(rs,key=lambda r:r['seed'])]
        assert delta==pair['individual_detected_deltas'] and sum(delta)/10==pair['mean_detected_delta']
    real=read(OUT/'PS_geometry_static_routes.json');geo=read(OUT/'PS_geometry_static_routes.geojson')
    assert len(real)==len(geo['features'])==4
    for r,feature in zip(real,geo['features']):
        g=public[r['region']];coords=feature['geometry']['coordinates'];assert coords[0]==coords[-1]
        assert coords==[g['lonlat'][g['ids'].index(sid)] for sid in r['closed_route_ids']]
        assert r['real_target_coverage'] is None and r['real_first_target_arrival_s'] is None and not r['synthetic_target_map_used']
        assert r['total_time_s']<=r['budget_s']+1e-8
    for n,h in p['old_perception_sources_sha256'].items():checked(ROOT/'work/disaster_search'/n,h)
    save(OUT/'audit_report.json',{'status':'PASS','simulation_worlds_regenerated':60,'routes_independently_replayed':2160,'paid_observation_steps_checked':visited_steps,
        'replanning_events_checked':event_count,'condition_summaries_replayed':216,'paired_condition_comparisons_replayed':162,'realPS_geometry_routes_checked':4,
        'closed_budget_violations':0,'future_signal_accesses_by_observer':0,'old_perception_frozen_sources_unchanged':len(p['old_perception_sources_sha256']),
        'original_real_target_table_sha256_unchanged':True,'synthetic_targets_created_in_separate_artifact':True,
        'independent_replay_note':'Independent feasibility,choice,belief,timing and reward formulas;uses recorded public geometry and independently regenerated synthetic truth. No real target efficacy claim.'})
    print({'audit':'PASS','routes':2160,'paid_steps':visited_steps,'replan_events':event_count},flush=True)


if __name__=='__main__':
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    main()
