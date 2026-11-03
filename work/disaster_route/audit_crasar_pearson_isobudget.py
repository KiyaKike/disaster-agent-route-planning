"""Replay closed budgets, paid histories, frozen priors and exact-static certificates."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from route_headroom_core import exact_catalogue, subset_rewards
from run_boone_context_experiment import dump, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root,out=args.workspace.resolve(),args.output.resolve()
    read=lambda p:json.loads(p.read_text())
    protocol=read(out/'protocol.json')
    old_prompt=read(root/'output/disaster_route_boone_context_run_v2/calls/prior_00.json')['prompt']
    counts={'routes':0,'paid_observations':0,'updates':0,'initial_model_calls':0,
            'pearson_features':0,'exact_static_certificates':0,'crop_hashes':0}
    summaries=read(out/'summary.json')['table']
    for region in read(out/'region_status.json'):
        if region['status']=='EXCLUDED_ORIGINAL_NON_RGB':
            assert region['case']=='case_rank05'
            assert region['original_bands']['pre']==1
            continue
        assert region['status']=='REPLAY_COMPLETE'
        dst=out/region['case']
        src=root/'output/disaster_route_crasar_batch4_v1'/region['case']
        public=read(dst/'public_inputs.json')
        ids,xy=public['building_ids'],np.array(public['xy_m'])
        n=len(ids)
        p0dict=read(dst/'vlm_initial_scores.json')
        p0=np.array([p0dict[u] for u in ids])
        sim=np.array(public['similarity'])
        references={r['building_id']:r['labels']['uas'] for r in read(src/'private_reference.json')}
        known={'no damage','minor damage','major damage','destroyed'}
        labels=[None if references[u] not in known else int(references[u]!='no damage') for u in ids]
        for name,digest in public['source_hashes'].items():
            assert sha(src/name)==digest
        geometry={g['building_id']:g['roi_polygon_px'] for g in read(src/'public_geometry.json')}
        images={s:np.asarray(Image.open(src/f'{s}_roi.png').convert('RGB')) for s in ('pre','post')}
        for row in read(dst/'crop_manifest.json')['rows']:
            uid=row['building_id']
            for crop in row['crops'].values():
                assert sha(dst/crop['path'])==crop['sha256']
                counts['crop_hashes']+=1
            mask=Image.new('L',(images['pre'].shape[1],images['pre'].shape[0]),0)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in geometry[uid]],fill=255)
            pixels={s:im[np.asarray(mask)>0].astype(float) for s,im in images.items()}
            corr=[np.corrcoef(pixels['pre'][:,c],pixels['post'][:,c])[0,1] for c in range(3)]
            np.testing.assert_allclose(corr,row['channel_correlations'],atol=1e-12,rtol=0)
            counts['pearson_features']+=1
            call=read(dst/f'calls/prior_{uid}.json')
            assert call['target_id']==uid and call['revision']==protocol['revision']
            assert call['prompt']==old_prompt
            assert call['image_paths']==[row['crops'][s]['path'] for s in ('pre','post')]
            assert abs(call['score']-p0dict[uid])<1e-12
            counts['initial_model_calls']+=1
        cat=exact_catalogue(xy,n)
        rewards=subset_rewards(p0)
        for route in read(dst/'routes.json'):
            budget=route['budget_proxy_s']
            assert route['label_access_indices']==route['route']
            assert len(set(route['route']))==len(route['route'])
            assert route['returned_home']
            current,elapsed,distance,visited=n,0.,0.,set()
            scores=p0.copy()
            updates={u['decision_number']:u for u in route['updates']}
            obs_iter=iter(route['observations'])
            if route['policy']=='static_exact':elapsed=1.
            for decision in route['decisions']:
                if route['policy']!='static_exact':elapsed+=1.
                assert abs(elapsed-decision['elapsed_proxy_s'])<1e-9
                assert decision['observed_indices']==sorted(visited)
                if decision['decision_number'] in updates:
                    update=updates[decision['decision_number']]
                    assert update['anchor_index']==current and current in visited
                    assert update['feedback']==labels[current] and labels[current] is not None
                    np.testing.assert_allclose(update['scores_before'],scores,atol=1e-12,rtol=0)
                    residual=labels[current]-scores[current]
                    assert abs(residual-update['residual'])<1e-12
                    expected=scores.copy();shadow=scores.copy();remaining=sorted(set(range(n))-visited)
                    for i in remaining:
                        d=math.dist(xy[current],xy[i]);w=math.exp(-d*d/(2*50**2))
                        scale=.5*sim[current,i] if route['policy']=='pearson' else update['gain']
                        expected[i]=min(1.,max(0.,scores[i]+scale*w*residual))
                        shadow[i]=min(1.,max(0.,scores[i]+.5*sim[current,i]*w*residual))
                    np.testing.assert_allclose(expected,update['scores_after'],atol=1e-12,rtol=0)
                    if route['policy']=='matched_spatial':
                        assert abs(np.abs(expected[remaining]-scores[remaining]).mean()-np.abs(shadow[remaining]-scores[remaining]).mean())<1e-12
                    scores=expected
                    counts['updates']+=1
                target=decision['target']
                if route['policy']!='static_exact':
                    feasible=[i for i in range(n) if i not in visited and elapsed+math.dist(xy[current],xy[i])/5+10+math.dist(xy[i],xy[n])/5<=budget+1e-9]
                    assert feasible==decision['feasible_indices']
                    if not feasible:
                        assert target is None
                        continue
                    expected_target=min(feasible,key=lambda i:(math.dist(xy[current],xy[i]),i)) if route['policy']=='distance' else min(feasible,key=lambda i:(-scores[i]/max(math.dist(xy[current],xy[i]),1e-6),i))
                    assert target==expected_target
                assert target not in visited
                obs=next(obs_iter)
                assert obs['index']==target and obs['source_label']==labels[target]
                leg=math.dist(xy[current],xy[target]);arrival=elapsed+leg/5
                elapsed=arrival+10;distance+=leg
                assert abs(arrival-obs['arrival_proxy_s'])<1e-9
                assert abs(elapsed-obs['observation_complete_proxy_s'])<1e-9
                assert elapsed+math.dist(xy[target],xy[n])/5<=budget+1e-8
                current=target;visited.add(target);counts['paid_observations']+=1
            back=math.dist(xy[current],xy[n]);elapsed+=back/5;distance+=back
            if route['policy']!='static_exact':assert len(route['decisions'])==route['planning_calls']
            assert abs(elapsed-route['elapsed_proxy_s'])<1e-9 and elapsed<=budget+1e-8
            assert abs(distance-route['distance_m'])<1e-9
            assert abs(elapsed-(distance/5+10*len(visited)+route['planning_calls']))<1e-9
            hits=sum(labels[i]==1 for i in visited)
            assert hits==route['source_positive_checks']
            assert sum(labels[i] is None for i in visited)==route['unknown_source_checks']
            summary=next(r for r in summaries if (r['case'],r['budget_proxy_s'],r['policy'])==(region['case'],budget,route['policy']))
            assert summary['positive_checks']==hits
            assert abs(summary['elapsed_proxy_s']-elapsed)<1e-9
            if route['policy']=='static_exact':
                cost=cat['closed']/5+cat['cardinality']*10+1
                possible=np.flatnonzero(cost<=budget+1e-9)
                actual=sum(p0[i] for i in visited)
                assert abs(actual-rewards[possible].max())<1e-9
                assert route['certificate']['gap']==0
                counts['exact_static_certificates']+=1
            counts['routes']+=1
    hashes=0
    for name in ('disaster_route_crasar_batch4_v1','disaster_route_boone_two_step_v1'):
        folder=root/'output'/name
        for path,digest in read(folder/'completion_receipt.json')['files'].items():
            assert sha(folder/path)==digest,str(folder/path)
            hashes+=1
    for name,digest in protocol['locked_source_hashes'].items():
        assert sha(root/'work/disaster_route'/name)==digest
    dump(out/'audit.json',{'status':'PASS',**counts,'budget_violations':0,
                           'old_sealed_files_unchanged':hashes,
                           'scope':'Archive paid-label paths and proxy budgets; no field registration or sensor-truth certificate'})
    print((out/'audit.json').read_text())


if __name__=='__main__':main()
