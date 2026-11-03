"""Separated public feature construction, planning and private reference evaluation."""
import json,sys,math,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from probe_crasar_region import O,dump
from crasar_route_controls import plan_route
from route_headroom_core import exact_catalogue,reconstruct,subset_rewards

def checked_reference_counts(predicted,truth,valid):
    predicted=np.asarray(predicted,bool);truth=np.asarray(truth,bool);valid=np.asarray(valid,bool)
    if predicted.shape!=truth.shape or valid.shape!=truth.shape or np.any(truth&~valid):raise ValueError('Matched masks and valid positive references required')
    tp=int((predicted&truth).sum());fp=int((predicted&~truth&valid).sum())
    return {'tp':tp,'fp':fp,'ignored_unknown':int((predicted&~valid).sum()),'precision':tp/(tp+fp) if tp+fp else None}

def detected_at_coverage(covered,damage):
    covered=np.asarray(covered,float);damage=np.asarray(damage,float)
    if covered.shape!=damage.shape:raise ValueError('Matching coverage and damage arrays required')
    return (covered>=.8)&(damage>=.1*covered)

def make_inputs(target):
    geometry=[g for g in json.loads((O/'public_geometry.json').read_text()) if g['eligible']]
    c=np.zeros((16,len(geometry)));detected=c.copy();d=np.zeros(len(geometry));u=d.copy()
    masks=[]
    for g in geometry:
        im=Image.new('1',(512,512));ImageDraw.Draw(im).polygon([tuple(np.array(v)/2) for v in g['roi_polygon_px']],fill=1)
        masks.append(np.array(im))
    prior_d=c.copy();prior_u=c.copy()
    for j in range(16):
        row,col=divmod(j,4)
        with np.load(O/'fixed_model_cache'/f'post_{j:02d}.npz') as f:scores=f['mean_scores']
        norm=scores[1:]/np.maximum(scores[1:].sum(axis=0),1e-8)
        ordered=np.sort(norm,axis=0)
        damage=scores[0]*(norm[2]+norm[3] if target=='severe' else norm[1:].sum(axis=0));uncertainty=1-(ordered[-1]-ordered[-2])
        with np.load(O/'fixed_model_cache'/f'uas_{j:02d}.npz') as f:severe=np.isin(f['prediction'],[3,4] if target=='severe' else [2,3,4])
        uas=np.array(Image.open(O/'uas_roi.png').crop((col*256,row*256,(col+1)*256,(row+1)*256)).resize((128,128),Image.Resampling.NEAREST))
        available=np.any(uas!=0,axis=2)
        for k,mask in enumerate(masks):
            local=mask[row*128:(row+1)*128,col*128:(col+1)*128]
            denom=mask.sum()
            c[j,k]=np.sum(local&available)/denom
            detected[j,k]=np.sum(local&available&severe)/denom
            prior_d[j,k]=damage[local].sum()/denom
            prior_u[j,k]=uncertainty[local].sum()/denom
    d=prior_d.sum(axis=0);u=prior_u.sum(axis=0)
    xy=[[col*32+16,128-row*32-16] for row in range(4) for col in range(4)]+[[64,0]]
    public={'building_ids':[g['building_id'] for g in geometry],'xy_m_relative':xy,'window_building_fraction':c.tolist(),
        'prior_damage_score':d.tolist(),'prior_uncertainty_score':u.tolist(),
        'known_footprint_scope':'Published orthomosaic image availability treated as known provider metadata; not actual UAV view footprints or guaranteed visibility',
        'structure_scope':'Published Microsoft/custom footprint provenance; pre-disaster availability dates not independently verified'}
    dump(O/f'public_planning_inputs_{target}.json',public)
    dump(O/f'private_paid_observations_{target}.json',{'building_ids':public['building_ids'],'damage_model_pixel_fraction_per_window':detected.tolist(),
        'scope':'Window-local UAS model outputs; no reference labels; available only after paid access during replay'})

def plan_all(target):
    # This function reads no private reference or paid observations.
    p=json.loads((O/f'public_planning_inputs_{target}.json').read_text());protocol=json.loads((O/'perception_route_protocol.json').read_text())
    xy=np.array(p['xy_m_relative']);c=np.array(p['window_building_fraction']);d=np.array(p['prior_damage_score']);u=np.array(p['prior_uncertainty_score'])
    start=time.perf_counter();cat=exact_catalogue(xy,16);catalogue_wall=time.perf_counter()-start
    routes=[]
    for budget in protocol['budgets_proxy_s']:
      for policy in ['distance','damage','uncertainty','static_exact_damage','static_exact','value_cost_proxy']:
        uncertainty=np.ones_like(u) if policy=='static_exact_damage' else u
        r=plan_route('static_exact' if policy=='static_exact_damage' else policy,xy,c,d,uncertainty,
            speed=protocol['assumed_speed_mps'],observe=protocol['assumed_observation_s'],plan=protocol['assumed_planning_s'],budget=budget,catalogue=cat)
        r.update(policy=policy,budget_proxy_s=budget);routes.append(r)
    dump(O/f'routes_{target}.json',routes)
    dump(O/f'solver_metadata_{target}.json',{'shared_geometry_precompute_wall_seconds':catalogue_wall,'subsets':65536,
       'planning_wall_time_charged':'All policies pay same assumed1second planning charge; actual wall time reported separately; flight timing remains a proxy',
       'scope':'All controls and candidate plan from fixed prior. No adaptive information update or online-method gain claimed.'})

def evaluate(target):
    p=json.loads((O/f'public_planning_inputs_{target}.json').read_text());xy=p['xy_m_relative'];c=np.array(p['window_building_fraction'])
    paid=json.loads((O/f'private_paid_observations_{target}.json').read_text());s=np.array(paid['damage_model_pixel_fraction_per_window'])
    refs={r['building_id']:r for r in json.loads((O/'private_reference.json').read_text())}
    truth=np.array([refs[uid]['primary_target'] if target=='severe' else refs[uid]['labels']['uas'] in ['minor damage','major damage','destroyed'] for uid in p['building_ids']],bool)
    valid=np.array([refs[uid]['labels']['uas'] in ['no damage','minor damage','major damage','destroyed'] for uid in p['building_ids']],bool)
    protocol=json.loads((O/'perception_route_protocol.json').read_text());speed=protocol['assumed_speed_mps'];obs=protocol['assumed_observation_s'];plan=protocol['assumed_planning_s']
    scores=[]
    ledgers=[]
    for r in json.loads((O/f'routes_{target}.json').read_text()):
        path=r['route'];nodes=[16]+path+[16];dist=sum(math.dist(xy[a],xy[b]) for a,b in zip(nodes,nodes[1:]))
        elapsed=plan+dist/speed+len(path)*obs
        assert abs(dist-r['distance_m'])<1e-7 and abs(elapsed-r['elapsed_proxy_s'])<1e-7 and elapsed<=r['budget_proxy_s']+1e-7
        covered=np.zeros(len(truth));severe=covered.copy();first=None;elapsed_prefix=plan;current=16
        for j in path:
            elapsed_prefix+=math.dist(xy[current],xy[j])/speed+obs;current=j
            covered+=c[j];severe+=s[j]
            correct=truth&detected_at_coverage(covered,severe)
            if first is None and correct.any():first=elapsed_prefix
            # Explicit dual-map record: fixed prior plus paid evidence only.
            ledgers.append({'policy':r['policy'],'budget_proxy_s':r['budget_proxy_s'],'step':len([e for e in ledgers if e['policy']==r['policy'] and e['budget_proxy_s']==r['budget_proxy_s']])+1,
                'accessed_window':j,'observation_complete_proxy_s':elapsed_prefix,
                'prior_damage_score':p['prior_damage_score'],'prior_uncertainty_score':p['prior_uncertainty_score'],
                'evidence':{'observed_building_fraction':covered.tolist(),'observed_damage_model_pixel_fraction':severe.tolist()},
                'unobserved_coverage':np.maximum(0,1-covered).tolist(),'reference_labels_in_state':False,
                'online_replanning':False})
        predicted=detected_at_coverage(covered,severe)
        correct=truth&predicted
        counts=checked_reference_counts(predicted,truth,valid)
        scores.append({'policy':r['policy'],'budget_proxy_s':r['budget_proxy_s'],'route':path,
          'correctly_checked_target_buildings':int(correct.sum()),'eligible_target_buildings':int(truth.sum()),
          'target_recall':float(correct.sum()/truth.sum()) if truth.any() else None,
          'checked_buildings_geometry_only':int((covered>=.8).sum()),'first_correct_check_proxy_s':first,
          'false_positive_checked_buildings':counts['fp'],'checked_prediction_precision':counts['precision'],
          'ignored_unknown_label_checked_buildings':counts['ignored_unknown'],
          'distance_m':dist,'elapsed_proxy_s':elapsed,'budget_verified':True})
    full=detected_at_coverage(c.sum(axis=0),s.sum(axis=0))&truth
    # Evaluator-only ceiling: reference labels and all paid caches are never planner inputs.
    cat=exact_catalogue(xy,16)
    cv=np.column_stack([subset_rewards(c[:,k]) for k in range(c.shape[1])])
    sv=np.column_stack([subset_rewards(s[:,k]) for k in range(s.shape[1])])
    oracle_reward=(detected_at_coverage(cv,sv)&truth[None,:]).sum(axis=1)
    costs=plan+cat['closed']/speed+cat['cardinality']*obs
    oracles=[]
    for budget in protocol['budgets_proxy_s']:
        feasible=np.flatnonzero(costs<=budget+1e-8);best=int(oracle_reward[feasible].max())
        ties=feasible[oracle_reward[feasible]==best];mask=int(min(ties,key=lambda m:(cat['closed'][m],int(m))))
        oracles.append({'budget_proxy_s':budget,'correctly_checked_target_buildings':best,'route':reconstruct(cat,mask),
            'role':'Hindsight evaluator ceiling only; full future model cache and reference access; not a deployable control'})
    dump(O/f'private_oracle_{target}.json',oracles)
    main=[r for r in scores if r['budget_proxy_s']==protocol['main_budget_proxy_s']]
    summary={'region':json.loads((O/'protocol.json').read_text())['images']['uas']['Orthomosaic'],'target_definition':target,
      'endpoint_scope':('Primary severe' if truth.any() else 'Primary severe,undefined with zero targets') if target=='severe' else next(v for v in protocol['limitations'] if v.startswith('Secondary')),
      'eligible_buildings':len(truth),'known_label_buildings':int(valid.sum()),'ignored_unknown_label_buildings':int((~valid).sum()),
      'eligible_target_buildings':int(truth.sum()),'all_windows_correct_check_count':int(full.sum()),
      'full_access_scope':'Outcome after all16windows,not a general upper bound:adding nondamage pixels can dilute the ratio rule. Exact same-budget hindsight reference is the actual diagnostic upper bound.',
      'routes':len(scores),'budget_violations':0,'main_budget_results':main,
      'hindsight_same_budget_ceiling':oracles,
      'method_gain_established':False,'interpretation':'One development case with uncalibrated cross-sensor detector and proxy service/flight costs. Outcomes are an admission probe,not evidence of a general method contribution.',
      'online_information_gain_validated':False,'gpu_rental':False}
    dump(O/f'dual_map_paid_evidence_{target}.json',ledgers)
    dump(O/f'route_scores_{target}.json',scores);dump(O/f'experiment_summary_{target}.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    for target in ['severe','any_damage']:
        make_inputs(target);plan_all(target);evaluate(target)
