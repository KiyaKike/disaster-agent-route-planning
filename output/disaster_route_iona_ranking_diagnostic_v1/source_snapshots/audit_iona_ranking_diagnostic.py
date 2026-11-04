"""Independent arithmetic audit of frozen-log figures; no imagery/model dependency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root,out=args.workspace.resolve(),args.output.resolve()
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    recipe=read(out/'execution_recipe.json')
    for name,digest in recipe['input_hashes'].items():assert sha(root/name)==digest,name
    base=root/'output/disaster_route_crasar_pearson_isobudget_v1'
    case=read(out/'iona_fork.json');public=read(base/'case_rank04/public_inputs.json')
    xy=public['xy_m'];route=next(r for r in read(base/'case_rank04/routes.json')
                                if r['budget_proxy_s']==100 and r['policy']=='pearson')
    update=next(u for u in route['updates'] if u['decision_number']==4)
    before,after={},{}
    for record in case['rows']:
        i=record['index'];distance=math.dist(xy[7],xy[i])
        delta=.5*math.exp(-distance**2/(2*50**2))*public['similarity'][7][i]*update['residual']
        expected=min(1.,max(0.,update['scores_before'][i]+delta))
        assert abs(record['score_after']-expected)<1e-12
        assert abs(record['score_change']-delta)<1e-12
        assert abs(record['priority_after']-expected/distance)<1e-12
        before[i]=update['scores_before'][i]/distance;after[i]=expected/distance
    assert max(before,key=before.get)==6 and max(after,key=after.get)==0
    # Reconstruct timing from paid observations and charges, not saved end-time result.
    i01_observation=route['observations'][3]
    end=i01_observation['observation_complete_proxy_s']+1+math.dist(xy[0],xy[6])/5+10+math.dist(xy[6],xy[-1])/5
    assert abs(end-case['late_I07_observe_and_return_end_s'])<1e-10 and end>100
    metrics_checked=0
    for point in read(out/'ranking_route_points.json'):
        folder=base/point['case'];ids=read(folder/'public_inputs.json')['building_ids']
        refs={r['building_id']:r['labels']['uas'] for r in read(root/'output/disaster_route_crasar_batch4_v1'/point['case']/'private_reference.json')}
        original=next(r for r in read(folder/'routes.json') if r['budget_proxy_s']==100 and r['policy']==point['method'])
        state=next(d for d in original['decisions'] if d['decision_number']==2)
        assert original['route'][0]==point['common_first_index']
        pairs=[(ids[i],state['scores'][i],int(refs[ids[i]]!='no damage')) for i in range(len(ids))
               if i not in state['observed_indices'] and refs[ids[i]] in ('no damage','minor damage','major damage','destroyed')]
        groups={}
        for _,score,label in pairs:groups.setdefault(score,[]).append(label)
        total=sum(v for _,_,v in pairs);hit=seen=0;ap=0.
        for score in sorted(groups,reverse=True):
            block=groups[score];positives=sum(block);hit+=positives;seen+=len(block)
            ap+=(positives/total)*(hit/seen)
        order=sorted(pairs,key=lambda r:(-r[1],r[0]))
        dcg=sum(r[2]/math.log2(i+2) for i,r in enumerate(order[:3]))
        ideal=sum(1/math.log2(i+2) for i in range(min(3,total)))
        assert abs(ap-point['ap'])<1e-12
        assert abs(dcg/ideal-point['ndcg_at3'])<1e-12
        assert order[0][2]==point['top1']
        assert original['source_positive_checks']==point['route_hits_100s']
        metrics_checked+=4
    for name in ('iona_fork.svg','ranking_route_scatter.svg'):
        assert ET.parse(out/name).getroot().tag.endswith('svg')
    availability=root/'docs/archive_availability.json'
    declared=set()
    if availability.exists():
        for record in read(availability):
            if record['receipt']=='output/disaster_route_crasar_pearson_isobudget_v1/completion_receipt.json':
                declared=set(record['absent_original_entries'])
    checked=missing=0
    for name,digest in read(base/'completion_receipt.json')['files'].items():
        path=base/name
        if not path.exists():
            assert name in declared,name
            missing+=1
        else:
            assert sha(path)==digest,name
            checked+=1
    result={'status':'PASS','ranking_points':12,'independent_metrics_checked':metrics_checked,
            'fork_candidate_updates_checked':len(case['rows']),
            'budget_exclusion_end_s':end,'historical_sealed_files_verified':checked,
            'declared_missing_historical_crops':missing,
            'figures_svg_valid':2,'new_model_calls':0,'new_route_executions':0}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
