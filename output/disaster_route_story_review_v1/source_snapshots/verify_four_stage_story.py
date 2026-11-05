"""Check the numeric premises of the four-stage story from sealed records.

The assertions verify observations and logged arithmetic, not untested causes.
Only Python's standard library is required. No new policy or model execution.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root,out=args.workspace.resolve(),args.output.resolve()
    if out.exists():raise FileExistsError('Use a new verification directory')
    inputs={};checks=[]
    def read(name):
        path=root/'output'/name
        inputs[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def record(key,values):checks.append({'claim_id':key,'numeric_premise':'VERIFIED','values':values})
    controlled=read('disaster_route_verification_screen_v1/results.json')
    world_values=lambda p:[Fraction(v['exact']) for v in p['values']]
    reward=lambda p:sum(world_values(p))/2
    harm=benefit=equal=0
    for row in controlled:
        p=row['policies']
        harm+=reward(p['always_probe'])<reward(p['static_mixture'])
        benefit+=reward(p['exact_bayes_reference'])>reward(p['static_mixture'])
        equal+=world_values(p['exact_bayes_reference'])==world_values(p['disagreement_screen_candidate'])
    assert (len(controlled),harm,benefit,equal)==(108,60,11,108)
    record('L1',{'conditions':108,'forced_calibration_harm':harm,'optional_gain':benefit,'candidate_bayes_equal':equal})
    summary=read('disaster_route_boone_context_run_v2/summary.json')
    metrics={r['method']:r for r in summary['table'] if r['reference_source']=='uas'}
    assert summary['forward_calls']==276
    assert abs(metrics['V1_vlm_history']['ap']-.3419642857142857)<1e-12
    assert metrics['V1_vlm_history']['ap']<metrics['S0_no_update']['ap']
    shifts=[r for r in read('disaster_route_boone_context_run_v2/score_shift_diagnostic.json') if r['method']=='V1_vlm_history']
    assert len(shifts)==12 and all(r['all_targets_move_same_direction'] for r in shifts)
    assert all(r['min_shift']>0 if r['anchor_uas_view_label']==1 else r['max_shift']<0 for r in shifts)
    record('L2_VLM',{'forward_calls':276,'history_ap':metrics['V1_vlm_history']['ap'],
                     'no_update_ap':metrics['S0_no_update']['ap'],'all_targets_follow_anchor_cases':12})
    table={r['method']:r for r in read('disaster_route_boone_structure_check_v1/summary.json')['table'] if r['reference_source']=='uas'}
    pairs=[('S2_visual_spatial','C2_matched36'),('S3_change6','C3_matched6'),('S4_change6_old_tau','C4_matched6_old_tau')]
    assert all(table[a]['ap']<table[b]['ap'] for a,b in pairs)
    assert table['S4_change6_old_tau']['top3_hit_sum']==12 and table['C4_matched6_old_tau']['top3_hit_sum']==11
    record('L2_FEATURES',{'ap_pairs':[{a:table[a]['ap'],b:table[b]['ap']} for a,b in pairs],
                         'counterexample_top3_old_tau':[12,11]})
    components=read('disaster_route_boone_feature_error_audit_v1/feature_distance_components.json')
    row=next(r for r in components if r['anchor_name']=='B11' and r['target_name']=='B09')
    color=row['post_RGB_histogram_squared_distance']+row['post_mean_std_squared_distance']
    fraction=color/row['total_squared_distance']
    assert fraction>.97
    structure=read('disaster_route_boone_structure_check_v1/structure_features.json')
    named={r['display_id']:r for r in structure['rows']}
    assert sum(named['B10']['channel_correlations'])/3<0
    record('L2_APPEARANCE',{'B11_B09_post_color_distance_fraction':fraction,
                           'B10_mean_pearson':sum(named['B10']['channel_correlations'])/3,
                           'physical_cause_verified':False})
    ids=[r['building_id'] for r in structure['rows']]
    refs={r['building_id']:r['view_binary']['uas'] for r in read('disaster_route_boone_context_pilot_v1/evaluator_only_view_references.json')['rows']}
    groups={}
    for r in read('disaster_route_boone_structure_check_v1/prediction_rows.json'):
        if r['method'] in ('S5_pearson_change','C5_matched_pearson'):
            groups.setdefault((r['anchor_id'],r['method']),{})[r['target_id']]=r['score']
    heads={m:{'original':0,'without_B03':0,'B03_chosen':0} for m in ('S5_pearson_change','C5_matched_pearson')}
    for (a,m),scores in groups.items():
        ranked=sorted(scores,key=lambda u:(-scores[u],u))
        retained=[u for u in ranked if u!=ids[2]]
        heads[m]['original']+=refs[ranked[0]]
        heads[m]['without_B03']+=refs[retained[0]]
        heads[m]['B03_chosen']+=ranked[0]==ids[2]
    a,b=heads.values()
    assert (a['original'],b['original'],a['without_B03'],b['without_B03'],a['B03_chosen'])==(9,1,3,1,7)
    assert table['S5_pearson_change']['ap']>table['C5_matched_pearson']['ap']
    assert table['S5_pearson_change']['top3_hit_sum']<table['C5_matched_pearson']['top3_hit_sum']
    record('L3_HEAD',{'heads':heads,'ap':[table[m]['ap'] for m in heads],
                     'top3':[table[m]['top3_hit_sum'] for m in heads]})
    two=read('disaster_route_boone_two_step_v1/episodes.json')
    grouped={(r['selector'],r['method']):[] for r in two}
    for r in two:grouped[r['selector'],r['method']].append(r)
    totals={m:sum(r['new_visit_hits'] for r in grouped['score_per_distance',m]) for m in ('pearson','matched_spatial')}
    assert totals=={'pearson':8,'matched_spatial':5}
    distances={m:sum(r['closed_distance_m'] for r in grouped['score_per_distance',m]) for m in totals}
    assert distances['pearson']>distances['matched_spatial']
    simple={m:sum(r['new_visit_hits'] for r in grouped['score',m]) for m in ('pearson','no_update')}
    assert simple=={'pearson':9,'no_update':11}
    record('L3_TWO_STEP',{'distance_rule_hits':totals,'closed_distances_m':distances,
                         'score_only_counterexample':simple})
    external=read('disaster_route_crasar_pearson_isobudget_v1/summary.json')['table']
    policies=('distance','static_exact','no_update','spatial','matched_spatial','pearson')
    budgets={str(b):{m:sum(r['positive_checks'] for r in external if r['budget_proxy_s']==b and r['policy']==m) for m in policies} for b in (60,100,140)}
    assert len(external)==54
    assert budgets['100']==dict(zip(policies,(3,5,3,4,3,3)))
    assert all(r['elapsed_proxy_s']<=r['budget_proxy_s']+1e-8 for r in external)
    assert all(budgets[str(b)]['static_exact']>budgets[str(b)]['pearson'] for b in (60,100,140))
    iona={r['policy']:r['positive_checks'] for r in external if r['case']=='case_rank04' and r['budget_proxy_s']==100}
    assert iona['static_exact']==0 and iona['spatial']==1
    record('L4',{'routes':54,'budget_totals':budgets,'Iona_100s_counterexample':iona})
    fork=read('disaster_route_iona_ranking_diagnostic_v1/iona_fork.json')
    points=read('disaster_route_iona_ranking_diagnostic_v1/ranking_route_points.json')
    assert fork['choice_without_this_last_update']=='I07' and fork['actual_choice']=='I01'
    assert fork['late_I07_observe_and_return_end_s']>100 and fork['pearson_distance_m']<fork['spatial_distance_m']
    lookup={(r['region'],r['method']):r for r in points}
    assert lookup['Iona','pearson']['ap']>lookup['Iona','no_update']['ap']
    assert lookup['Iona','pearson']['route_hits_100s']<lookup['Iona','no_update']['route_hits_100s']
    record('MECHANISM_VS_CAUSE',{'Iona_priority_reversal':'Verified at fixed logged state',
                               'Iona_late_return_end_s':fork['late_I07_observe_and_return_end_s'],
                               'registration_cause_verified':False,'heterogeneity_cause_verified':False})
    out.mkdir(parents=True)
    result={'status':'PASS','claim_groups_checked':len(checks),'checks':checks,
            'input_hashes':inputs,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Numerical premises verified; physical cause and generalization explicitly not certified',
            'new_model_calls':0,'new_policy_runs':0}
    (out/'numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','claim_groups_checked':len(checks)},indent=2))


if __name__=='__main__':main()
