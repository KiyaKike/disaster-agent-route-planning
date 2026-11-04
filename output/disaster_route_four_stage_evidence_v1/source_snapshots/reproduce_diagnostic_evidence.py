"""Recompute the four-stage evidence tables and redraw two frozen-log figures.

Tables need only Python's standard library; figures need NumPy and Matplotlib.
This does not rerun VLM inference, original perception, or policy experiments.
"""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def mean(values):
    return sum(values)/len(values)


def rank_metrics(scores,labels):
    order=sorted(scores,key=lambda u:(-scores[u],u))
    positives=sum(labels.values());groups=defaultdict(list)
    for uid,p in scores.items():groups[p].append(labels[uid])
    tp=seen=0;ap=0.
    for p in sorted(groups,reverse=True):
        block=groups[p];hit=sum(block);tp+=hit;seen+=len(block)
        if positives:ap+=(hit/positives)*(tp/seen)
    discounts=[1/math.log2(i+2) for i in range(min(3,len(order)))]
    ideal=sum(discounts[:min(3,positives)])
    ndcg=sum(labels[u]*discounts[i] for i,u in enumerate(order[:3]))/ideal if ideal else None
    return {'ap':ap if positives else None,'brier':mean([(scores[u]-labels[u])**2 for u in order]),
            'top1':sum(labels[u] for u in order[:1]),'top2':sum(labels[u] for u in order[:2]),
            'top3':sum(labels[u] for u in order[:3]),'ndcg_at3':ndcg}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tables-only',action='store_true',help='Use Python standard library only')
    args=parser.parse_args();root,out=args.workspace.resolve(),args.output.resolve()
    if out.exists():raise FileExistsError('Use a new reproduction output directory')
    started=time.perf_counter();inputs={};checks=0
    def read(name):
        path=root/name;inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def dump(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def table(name,rows):
        with (out/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    # 1. Rational arithmetic on stored world-specific expected rewards.
    records=read('output/disaster_route_verification_screen_v1/results.json')
    values=lambda p:[Fraction(v['exact']) for v in p['values']]
    reward=lambda p:sum(values(p))/2
    equal=positive=harm=0
    for r in records:
        ps=r['policies'];equal+=values(ps['exact_bayes_reference'])==values(ps['disagreement_screen_candidate'])
        positive+=reward(ps['exact_bayes_reference'])>reward(ps['static_mixture'])
        harm+=reward(ps['always_probe'])<reward(ps['static_mixture'])
    controlled=[{'conditions':len(records),'candidate_equals_bayes':equal,
                 'optional_calibration_positive_gain':positive,'forced_calibration_harm':harm}]
    expected=read('output/disaster_route_verification_screen_v1/summary.json')
    assert (len(records),equal,positive,harm)==(expected['conditions'],expected['candidate_matches_exact_bayes_conditions'],
            expected['paid_calibration_improves_mixture_reward_conditions'],expected['always_calibrate_worse_than_static_conditions'])
    checks+=4
    # 2/3. Recompute prediction metrics from target scores, not summary tables.
    refs=read('output/disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json')
    labels={r['building_id']:r['view_binary']['uas'] for r in refs['rows']}
    initial=read('output/disaster_route_boone_context_run_v2/vlm_initial_scores.json')
    feature=read('output/disaster_route_boone_context_run_v2/satellite_features.json')
    ids=feature['building_ids'];b03=ids[2]
    prediction_rows=[];head_rows=[];bias_rows=[]
    for run in ('disaster_route_boone_context_run_v2','disaster_route_boone_structure_check_v1'):
        raw=read(f'output/{run}/prediction_rows.json');summary=read(f'output/{run}/summary.json')
        grouped=defaultdict(dict)
        for r in raw:
            key=(r['anchor_id'],r['method']);assert r['target_id'] not in grouped[key]
            grouped[key][r['target_id']]=r['score']
        aggregated=defaultdict(list)
        for (a,m),scores in grouped.items():
            assert set(scores)==set(ids)-{a}
            result=rank_metrics(scores,{u:labels[u] for u in scores});aggregated[m].append(result)
            if run=='disaster_route_boone_context_run_v2' and m=='V1_vlm_history':
                delta=[scores[u]-initial[u] for u in sorted(scores)]
                bias_rows.append({'anchor_display':f'B{ids.index(a)+1:02d}','anchor_label':labels[a],
                                  'all_shift_positive':all(d>0 for d in delta),'all_shift_negative':all(d<0 for d in delta),
                                  'direction_agrees_with_anchor':all((d>0)==bool(labels[a]) for d in delta),
                                  'mean_shift':mean(delta)})
            if run=='disaster_route_boone_structure_check_v1' and m in ('S5_pearson_change','C5_matched_pearson'):
                retained={u:p for u,p in scores.items() if u!=b03}
                after=rank_metrics(retained,{u:labels[u] for u in retained})
                head_rows.append({'anchor_display':f'B{ids.index(a)+1:02d}','method':m,
                                  'top1':result['top1'],'top2':result['top2'],'top3':result['top3'],
                                  'top1_without_B03':after['top1']})
        for method,data in aggregated.items():
            row={'run':run,'method':method,**{k:mean([x[k] for x in data]) for k in ('ap','brier','ndcg_at3')},
                 **{k+'_sum':sum(x[k] for x in data) for k in ('top1','top2','top3')}}
            saved=next(x for x in summary['table'] if x['reference_source']=='uas' and x['method']==method)
            for k in ('ap','brier'):
                assert abs(row[k]-saved[k])<1e-12;checks+=1
            assert abs(row['top3_sum']/12-saved['top3_positive'])<1e-12;checks+=1
            prediction_rows.append(row)
    bias_count=sum(r['direction_agrees_with_anchor'] for r in bias_rows)
    assert bias_count==12;checks+=1
    # Two-step results and dominance-node sensitivity from actual saved episodes.
    episodes=read('output/disaster_route_boone_two_step_v1/episodes.json')
    saved=read('output/disaster_route_boone_two_step_v1/summary.json')
    two=[]
    for row in saved['table']:
        part=[r for r in episodes if (r['method'],r['selector'])==(row['method'],row['selector'])]
        hits=sum(o['label'] for r in part for o in r['observations'][1:])
        assert hits==row['two_step_hit_sum'];checks+=1
        distance=sum(r['closed_distance_m'] for r in part)
        assert abs(distance-row['closed_distance_sum_m'])<1e-9;checks+=1
        two.append({'selector':row['selector'],'method':row['method'],'hit_sum':hits,
                    'hit_sum_excluding_B03_credit':sum(o['label'] for r in part for o in r['observations'][1:] if o['index']!=2),
                    'closed_distance_sum_m':distance})
    # 4. Reconstruct hits, closed distance, all cost items for all54 routes.
    base='output/disaster_route_crasar_pearson_isobudget_v1';macro=[]
    for case in ('case_rank03','case_rank04','case_rank06'):
        public=read(f'{base}/{case}/public_inputs.json');xy=public['xy_m'];home=len(public['building_ids'])
        for r in read(f'{base}/{case}/routes.json'):
            path=[home]+r['route']+[home]
            distance=sum(math.dist(xy[a],xy[b]) for a,b in zip(path,path[1:]))
            elapsed=distance/5+len(r['observations'])*10+r['planning_calls']
            hits=sum(o['source_label']==1 for o in r['observations'])
            assert elapsed<=r['budget_proxy_s']+1e-8 and abs(elapsed-r['elapsed_proxy_s'])<1e-9
            assert abs(distance-r['distance_m'])<1e-9 and hits==r['source_positive_checks'];checks+=4
            macro.append({'case':case,'budget_s':r['budget_proxy_s'],'policy':r['policy'],'source_positive_checks':hits,
                          'visits':len(r['route']),'distance_m':distance,'elapsed_proxy_s':elapsed,
                          'planning_calls':r['planning_calls']})
    policies=('distance','static_exact','no_update','spatial','matched_spatial','pearson')
    totals=[{'budget_s':b,**{p:sum(r['source_positive_checks'] for r in macro if (r['budget_s'],r['policy'])==(b,p)) for p in policies}} for b in (60,100,140)]
    costs=read('output/disaster_route_boone_context_run_v2/summary.json')
    inference=read(f'{base}/inference_count.json')
    resource={'boone_vlm_calls':costs['forward_calls'],'boone_forward_seconds':costs['measured_forward_seconds'],
              'boone_elapsed_seconds':costs['elapsed_seconds'],'boone_device':'MPS (local Apple GPU)',
              'external_vlm_calls':inference['initial_satellite_forward_calls'],
              'external_forward_seconds':sum(read(str(p.relative_to(root)))['seconds'] for p in sorted((root/base).glob('case_*/calls/*.json'))),
              'external_device':'MPS (local Apple GPU)','reproduction_device':'CPU; saved-score statistics and plots only',
              'paid_api_cost':0,'rented_gpu':False}
    out.mkdir(parents=True)
    for name,rows in [('controlled_counts.csv',controlled),('boone_prediction_metrics.csv',prediction_rows),
                      ('boone_topk_node_sensitivity.csv',head_rows),('vlm_direction_diagnostic.csv',bias_rows),
                      ('boone_two_step.csv',two),('external_routes.csv',macro),('external_budget_totals.csv',totals)]:table(name,rows)
    dump('summary.json',{'controlled':controlled[0],'vlm_same_direction_anchor_cases':bias_count,
                         'pearson_top1':sum(r['top1'] for r in head_rows if r['method']=='S5_pearson_change'),
                         'matched_top1':sum(r['top1'] for r in head_rows if r['method']=='C5_matched_pearson'),
                         'pearson_top1_without_B03':sum(r['top1_without_B03'] for r in head_rows if r['method']=='S5_pearson_change'),
                         'matched_top1_without_B03':sum(r['top1_without_B03'] for r in head_rows if r['method']=='C5_matched_pearson'),
                         'external_totals':totals,'reconciled_numeric_checks':checks})
    dump('resources.json',resource)
    if not args.tables_only:
        for script in ('diagnose_iona_ranking_route.py','audit_iona_ranking_diagnostic.py'):
            subprocess.run([sys.executable,str(root/'work/disaster_route'/script),'--workspace',str(root),
                            '--output',str(out/'figures')],check=True)
    dump('execution.json',{'status':'PASS','mode':'tables_only' if args.tables_only else 'tables_and_two_figures',
                          'elapsed_seconds':time.perf_counter()-started,'python':sys.version.split()[0],
                          'numeric_checks':checks,'new_model_calls':0,'new_route_runs':0,
                          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_hashes':inputs})
    print(json.dumps({'status':'PASS','numeric_checks':checks,'elapsed_seconds':time.perf_counter()-started,
                      'output':str(out)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
