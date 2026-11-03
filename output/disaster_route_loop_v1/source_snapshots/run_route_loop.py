"""Freeze and execute paired closed-route simulations on existing PS geometry.

No imagery inference, training, external downloads or real target-label writes.
"""
import argparse,csv,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from route_core import *

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/disaster_route_loop_v1'
DATA=ROOT/'output/PS搜索决策_演示版/输入网格与显示图.json'
SCOPE=ROOT/'output/disaster_route_scope_v1/decision.json'
PERCEPTION=ROOT/'output/disaster_search_multievent36_v1/completion_receipt.json'
SOURCES=['route_core.py','test_route_core.py','run_route_loop.py','audit_route_loop.py','report_route_loop.py']
BUDGETS=(600,1200,1800)
SEEDS=tuple(range(2026102800,2026102810))


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def checked(p,h):
    if digest(p)!=h:raise ValueError('Changed frozen input: '+str(p))
def maps():return {region:public_map(region,part['cells']) for region,part in read(DATA).items()}
def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['source_sha256'].items():checked(HERE/n,h)
    for n,h in p['external_sha256'].items():checked(n,h)
    checked(OUT/'test_log.txt',p['test_log_sha256'])
    return p


def plan():
    if (OUT/'protocol.json').exists():raise ValueError('Protocol already frozen')
    assert 'Ran 12 tests' in (OUT/'test_log.txt').read_text() and (OUT/'test_log.txt').read_text().rstrip().endswith('OK')
    scope=read(SCOPE)
    for n,h in scope['existing_route_basis'].items():checked(ROOT/n,h)
    perception=read(PERCEPTION)
    for n,h in perception['source_code_sha256'].items():checked(ROOT/'work/disaster_search'/n,h)
    m=maps();public={}
    for region,g in m.items():
        public[region]={'ids':list(g.ids),'xy_m':g.xy.tolist(),'lonlat':g.lonlat.tolist(),'block_indices':g.blocks.tolist(),'block_ids':list(g.block_ids),
            'prior_pre_hot_probability':g.prior_pre.tolist(),'prior_post_hot_probability':g.prior_post.tolist(),
            'coarse_change_cue':[float(x) if np.isfinite(x) else None for x in g.cue],'start_id':g.ids[g.start_index]}
    save(OUT/'public_geometry.json',public)
    external={str(DATA):digest(DATA),str(SCOPE):digest(SCOPE),str(PERCEPTION):digest(PERCEPTION)}
    for n in ['work/ps_search_decision/engine.js','work/ps_search_decision/build_demo.py','output/PS搜索决策_演示版/参数与限制.json',
              'output/PS搜索决策_演示版/独立目标标注_空表.csv','output/PS变化先验_第一版/搜索网格_待复核.geojson']:
        external[str(ROOT/n)]=digest(ROOT/n)
    receipt=ROOT/'output/感知检查_阶段汇报_20261028/归档校验.json'
    if receipt.exists():external[str(receipt)]=digest(receipt)
    protocol={'stage':'PS_GEOMETRY_CLOSED_ROUTE_SIMULATION_V1','client_date':'2026-10-28',
        'question':'Given fixed pre/post public priors and noisy local observations,how do closed routes differ under equal movement+inspection+return budgets?',
        'source_sha256':{n:digest(HERE/n) for n in SOURCES},'external_sha256':external,'test_log_sha256':digest(OUT/'test_log.txt'),
        'public_geometry_sha256':digest(OUT/'public_geometry.json'),'regions':{r:{'eligible_cells':len(g.ids),'spatial_blocks':len(g.block_ids),'start_id':g.ids[g.start_index]} for r,g in m.items()},
        'main_policies':['pre_static','post_static','online'],'extra_control':'online_frozen,no belief update but same online scheduling latency',
        'pre_static':'Uniform block hotspot prior.2 plus geometry only;no postcue or feedback. This is a minimal predisaster-information baseline,not a learned real exposure map.',
        'post_static':'Same route heuristic,postcue rank prior,fixed tour after one initial planning call. Receives observed bits but cannot use them to change route.',
        'online':'Same heuristic and postcue prior;after each paid observation update only its3x3spatial block posterior by Bayes rule;replan feasible remaining suffix and execute its first visit.',
        'shared_solver':'Greedy expected target probability/(leg travel seconds+observation+future replanning latency),ID ties;return reserve checked at each planned stop;heuristic,not optimal routing.',
        'motion':'Independent regions;straight-line planar metric legs;start=eligible cell nearest bottom-middle;end=same depot. No terrain,obstacles,vehicle dynamics or certified airspace.',
        'costs':{'speed_m_s':5,'observe_s':20,'max_visits':30,'planning_latency_s':1,'budgets_s':list(BUDGETS),'primary_budget_s':1200,
                 'planning_latency_note':'Simulation constant,not measured deployment latency. Static pays once;online pays once per executed replan. Actual CPU solver wall time reported separately.'},
        'simulation':{'world_modes':list(WORLD_MODES),'seed_ids':list(SEEDS),'worlds':60,'observations':NOISE,'routes':2160,
            'hotspot_blocks':'3x3existing60mgrid cells,180mapprox;boundaryblocks can be partial;no physical damage annotation.',
            'post_hot_prior':'.05+.3*midrank(block mean valid change cue);unknown=.2. Simulation construction only;change scores are not calibrated real target probabilities.',
            'pre_hot_prior':.2,'target_probability_given_hot':HOT_TARGET_RATE,'target_probability_given_cold':COLD_TARGET_RATE,
            'aligned':'Latent hotspot probability follows constructedpostprior. Favors prior/model assumptions by design;not sufficient evidence alone.',
            'reversed':'Same distribution of hotspot probabilities assigned in reversepostpriororder;prior mismatch stress condition.',
            'independent':'All hotspot probabilities.2;postcue uninformative.',
            'pairing':'For each region/mode/seed,one target map and one uniformdraw per cell;noise settings and policies reuse those draws. No seed/outcome selection.',
            'likelihood':'Known constant simulation sensitivity/FPR integrated with cell target rates under blockhot/cold model;not estimated from remote-sensing pixelprecision or UAV deployment.'},
        'fixed_perception':{'reference_checkpoint':'StrongBaseline event fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6',
            'real_checkpoint_changed':False,'new_perception_calls':0,'role':'Preserved perception findings motivate uncertainty;these route tests use explicit fixed synthetic observation likelihoods,not new PS predictions.'},
        'primary_metric':'Mean distinct synthetic target cells both visited and signaledpositive at1200s,perregion/world/noise;paired deltas online-poststatic;allconditionsreported.',
        'other_metrics':['visitedtargetcoverage','detectedtargetrecall','firsttargetarrivaltime','firstcorrectdetectiontime','nodetectionrate','firstdetectionorbudgetcensorpenalty','totaldistance','totaltime','planningcalls','actualplannerwalltime'],
        'failure_handling':'No-detection time=null and counted as failure;restrictedtime indicator usesbudget,not0 or dropping runs. No-target recall=null.',
        'realPS_case':'Only two static geometry routes and visitedchangeproxy;no new target labels or real detection/route efficacy. Do not overlay simulatedtargets on sourceRGB as if observed.',
        'example':'For each region,aligned/moderate/firstseed/budget1200 fixed before runs;show all three mainpolicies,not best seed.',
        'claim_gate':'Engineeringprototype only. Budget and information-boundary violations invalidate run. Positive/negative/mismatch outcomes allretained;no guarantee online wins;none is a newmethodclaim.',
        'new_training':False,'gpu_rental':False,'new_downloads':False,'real_label_files_written':False,'old_reserved_sets_scored':False,
        'old_perception_sources_sha256':perception['source_code_sha256']}
    save(OUT/'protocol.json',protocol)
    save(OUT/'selection_receipt.json',{'protocol_sha256':digest(OUT/'protocol.json'),'synthetic_worlds_generated':False,'routes_executed':False})
    print(json.dumps({'protocol':'FROZEN','regions':protocol['regions'],'routes':2160,'synthetic_truth':True},ensure_ascii=False),flush=True)


def run():
    p=verify()
    if (OUT/'runtime.json').exists() or (OUT/'routes.jsonl').exists():raise ValueError('No silent route rerun')
    start=time.monotonic();mm=maps();world_rows=[];metrics=[];route_count=0
    with (OUT/'routes.jsonl').open('w') as output:
        for region,g in mm.items():
            for mode in WORLD_MODES:
                for seed in SEEDS:
                    # Seed differs by region but is common across policies/noise/budget.
                    world_seed=seed+(0 if region=='口岸' else 100000)
                    world=simulate_targets(g,mode,world_seed)
                    worldid=f'{region}/{mode}/{seed}'
                    world_rows.append({'world_id':worldid,'region':region,'mode':mode,'seed':seed,'generator_seed':world_seed,
                        'target_ids':[g.ids[i] for i in np.flatnonzero(world['targets'])],
                        'hot_block_indices':np.flatnonzero(world['hot_blocks']).tolist(),
                        'true_hot_probabilities':world['true_hot_probabilities'].tolist(),
                        'uniform_observation_draws':world['uniform_observation_draws'].tolist(),
                        'synthetic_truth_not_realPS':True})
                    for noise in NOISE:
                        bit=signals(world,noise)
                        for budget in BUDGETS:
                            for policy in POLICIES:
                                observer=Observer(bit,NOISE[noise]);r=execute(g,policy,Config(budget_s=budget),observer)
                                assert observer.accessed==[s['cell_index'] for s in r['steps']]
                                key={'world_id':worldid,'region':region,'mode':mode,'seed':seed,'noise':noise,'budget_s':budget,'policy':policy}
                                r.update(key);r['evaluation']=evaluate(r,world['targets']);output.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
                                metrics.append({**key,**r['evaluation'],**{k:r[k] for k in ['visits','total_distance_m','total_time_s','planning_calls','simulated_planning_latency_s','measured_planning_wall_s']}})
                                route_count+=1
            print(json.dumps({'region_completed':region,'routes_completed':route_count},ensure_ascii=False),flush=True)
    save(OUT/'synthetic_worlds.json',world_rows);save(OUT/'metrics.json',metrics)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    assert route_count==2160 and len(world_rows)==60
    # Existing PS demonstration has no independently labeled targets.
    real=[];features=[]
    for region,g in mm.items():
        for policy in ['pre_static','post_static']:
            observer=Observer(np.zeros(len(g.ids),bool),(.8,.1));route=execute(g,policy,Config(),observer)
            known=[g.cue[s['cell_index']] for s in route['steps'] if np.isfinite(g.cue[s['cell_index']])]
            public_steps=[{k:v for k,v in s.items() if k not in ['observed_positive','block_belief_before','block_belief_after']} for s in route['steps']]
            result={k:v for k,v in route.items() if k!='steps'};result['steps']=public_steps
            result.update(real_target_coverage=None,real_first_target_arrival_s=None,change_proxy_sum=float(sum(known)),
                change_proxy_note='Uncalibrated sum over knowncue visitedcells;unknownexcluded notzero;not true damage/targetreward.',
                known_cue_visits=len(known),unknown_cue_visits=route['visits']-len(known),synthetic_target_map_used=False)
            real.append(result)
            lookup={sid:i for i,sid in enumerate(g.ids)};coordinates=[g.lonlat[lookup[sid]].tolist() for sid in route['closed_route_ids']]
            features.append({'type':'Feature','geometry':{'type':'LineString','coordinates':coordinates},
                'properties':{k:v for k,v in result.items() if k not in ['steps','planning_events','closed_route_ids']}})
    save(OUT/'PS_geometry_static_routes.json',real);save(OUT/'PS_geometry_static_routes.geojson',{'type':'FeatureCollection','features':features})
    save(OUT/'runtime.json',{'status':'COMPLETE','simulation_routes':2160,'synthetic_worlds':60,'realPS_geometry_only_static_routes':4,
        'loop_seconds':time.monotonic()-start,'sum_planner_wall_seconds':sum(r['measured_planning_wall_s'] for r in metrics),
        'new_perception_calls':0,'gpu_rental':False,'new_training':False,'real_target_labels_created':0})


def summarize():
    verify();rows=read(OUT/'metrics.json');groups=defaultdict(list)
    for r in rows:groups[(r['region'],r['mode'],r['noise'],r['budget_s'],r['policy'])].append(r)
    means=[]
    keys=['detected_target_cells','visited_target_cells','false_positive_observations','visits','total_distance_m','total_time_s',
          'first_detection_or_budget_s','planning_calls','simulated_planning_latency_s','measured_planning_wall_s']
    for group,values in sorted(groups.items()):
        assert len(values)==10
        m={k:float(np.mean([r[k] for r in values])) for k in keys}
        m.update({k:float(np.mean([r[k] for r in values if r[k] is not None])) if any(r[k] is not None for r in values) else None
             for k in ['detected_target_recall','visited_target_recall','first_target_arrival_s','first_correct_detection_s']})
        means.append({'region':group[0],'mode':group[1],'noise':group[2],'budget_s':group[3],'policy':group[4],
            'runs':10,'no_detection_rate':sum(r['first_detection_censored'] for r in values)/10,
            'no_target_worlds':sum(r['true_target_cells']==0 for r in values),**m})
    paired=[];lookup={(r['world_id'],r['noise'],r['budget_s'],r['policy']):r for r in rows}
    for region in maps():
        for mode in WORLD_MODES:
            for noise in NOISE:
                for budget in BUDGETS:
                    for comparator in ['pre_static','post_static','online_frozen']:
                        a=[];b=[];time=[];fail=[]
                        for seed in SEEDS:
                            world=f'{region}/{mode}/{seed}';online=lookup[world,noise,budget,'online'];other=lookup[world,noise,budget,comparator]
                            a.append(online['detected_target_cells']-other['detected_target_cells'])
                            b.append(online['visited_target_cells']-other['visited_target_cells'])
                            time.append(online['first_detection_or_budget_s']-other['first_detection_or_budget_s'])
                            fail.append(int(online['first_detection_censored'])-int(other['first_detection_censored']))
                        paired.append({'region':region,'mode':mode,'noise':noise,'budget_s':budget,'comparator':comparator,'paired_runs':10,
                            'mean_detected_delta':float(np.mean(a)),'mean_visited_delta':float(np.mean(b)),
                            'mean_first_detection_or_budget_delta_s':float(np.mean(time)),
                            'mean_no_detection_rate_delta':float(np.mean(fail)),
                            'win_tie_loss':[sum(x>0 for x in a),sum(x==0 for x in a),sum(x<0 for x in a)],
                            'individual_detected_deltas':a})
    primary=[r for r in paired if r['budget_s']==1200 and r['comparator']=='post_static']
    save(OUT/'condition_means.json',means);save(OUT/'paired_comparisons.json',paired)
    summary={'status':'COMPLETE_SIMULATION_ONLY','primary_budget_s':1200,'main_policies':list(POLICIES[:3]),
        'primary_online_minus_post_static_by_condition':primary,
        'primary_condition_counts':{'positive':sum(r['mean_detected_delta']>0 for r in primary),'zero':sum(r['mean_detected_delta']==0 for r in primary),'negative':sum(r['mean_detected_delta']<0 for r in primary)},
        'every_route_budget_feasible':all(r['total_time_s']<=r['budget_s']+1e-8 for r in rows),
        'new_route_method_established':False,'real_target_efficacy_claim':False,'physical_flight_validation':False,
        'scope':'Two existing map geometries,synthetic target/observation worlds,knownlikelihood,planartravel;10seeds percondition;not independent disaster replications.'}
    save(OUT/'summary.json',summary)
    print(json.dumps({'primary_conditions':summary['primary_condition_counts'],'budget_feasible':summary['every_route_budget_feasible']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['plan','run','summarize'],required=True);a=p.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'plan':plan,'run':run,'summarize':summarize}[a.stage]()
