"""Fixed local-observation measurement; not a new adaptive search method."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from run_strongbaseline24 import ROOT,HERE,SOURCE,read,digest,checked,rawsha,save,freeze
from strongbaseline24_core import prepare_pair,decode_probabilities,restore_probability,FLIPS,load_network
from strongbaseline_local24_core import units,crop_pair,coarse_pair,static_order,acquisitions,counts,outcome,policy_result,exact_random,oracle_order,BUDGETS,PRIMARY_BUDGET

PARENT=ROOT/'output/disaster_search_strongbaseline24_v1'
OUT=ROOT/'output/disaster_search_strongbaseline_local24_v1'
NEW_SOURCES=['strongbaseline_local24_core.py','test_strongbaseline_local24.py','run_strongbaseline_local24.py','audit_strongbaseline_local24.py']


def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['source_code_sha256'].items():checked(HERE/n,h)
    for n,h in p['external_files'].items():checked(n,h)
    for n,h in p['upstream_files_sha256'].items():checked(SOURCE/n,h)
    for n,h in p['internal_inputs'].items():checked(OUT/n,h)
    return p


def plan():
    import torch
    parent=read(PARENT/'completion_receipt.json')
    assert digest(PARENT/'completion_receipt.json')=='a23d5e9a9c88a731ef7dd9f7cc40672041721e280987d4f78a1a6f067d7393f1'
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    for n,h in parent['source_code_sha256'].items():checked(HERE/n,h)
    assert 'Ran 357 tests' in (OUT/'test_log.txt').read_text() and (OUT/'test_log.txt').read_text().rstrip().endswith('OK')
    assert torch.backends.mps.is_available()
    jobs=read(PARENT/'rgb_jobs.json');refs=read(PARENT/'references.json');oldprotocol=read(PARENT/'protocol.json')
    assert len(jobs)==len(refs)==24 and [r['sample_id'] for r in jobs]==oldprotocol['selected_ids']
    freeze(OUT/'rgb_jobs.json',jobs);freeze(OUT/'references.json',refs)
    (OUT/'coarse').mkdir();public=[];coarse=[]
    for job in jobs:
        images=[np.asarray(Image.open(r['path'])).copy() for r in job['images']]
        cp=coarse_pair(*images);public.append({'sample_id':job['sample_id'],**static_order(*cp)})
        files=[]
        for mode,a in zip(['pre','post'],cp):
            path=OUT/'coarse'/(job['sample_id']+'_'+mode+'.png');Image.fromarray(a).save(path)
            files.append({'mode':mode,'path':str(path),'sha256':digest(path),'pixel_sha256':rawsha(a)})
        coarse.append({'sample_id':job['sample_id'],'images':files})
    freeze(OUT/'coarse_manifest.json',coarse);freeze(OUT/'public_static_orders.json',public)
    first=jobs[0]['sample_id'];control=next(r for r in read(PARENT/'cache_manifest.json') if r['sample_id']==first)
    freeze(OUT/'old_control.json',control)
    external={str(PARENT/'completion_receipt.json'):digest(PARENT/'completion_receipt.json')}
    for job in jobs:
        for r in job['images']:external[r['path']]=r['sha256']
    for ref in refs:
        external[ref['reference_file']]=ref['reference_sha256'];external[ref['label_path']]=ref['label_sha256']
    external[control['path']]=control['sha256']
    for n in ['summary.json','scene_scores.json','quality_and_error_observations.json']:external[str(PARENT/n)]=digest(PARENT/n)
    internal={n:digest(OUT/n) for n in ['rgb_jobs.json','references.json','coarse_manifest.json','public_static_orders.json','old_control.json','test_log.txt']}
    for row in coarse:
        for item in row['images']:internal[str(Path(item['path']).relative_to(OUT))]=item['sha256']
    sources={**parent['source_code_sha256'],**{n:digest(HERE/n) for n in NEW_SOURCES}}
    protocol={'stage':'FIXED24_LOCAL256_AREA_BUDGET_MEASUREMENT','client_date':'2026-10-28',
        'parent_receipt_sha256':digest(PARENT/'completion_receipt.json'),'source_code_sha256':sources,
        'external_files':external,'internal_inputs':internal,'upstream_files_sha256':oldprotocol['upstream_files_sha256'],
        'selected_ids':[r['sample_id'] for r in jobs],'scenes':24,'units':units(),'primary_budget':16,'budgets':[8,16,32,64],
        'cost_contract':'One128x128 paired-date native footprint costs1. Each256x256 paired observation costs4;1024full scene costs64. Primary16 buys4/16 tiles=25% native area. This is a prospective new1024-scene contract,not an oldEBD score comparison.',
        'coarse_prior':'One free64x64BOX RGB pre/post pair per scene,simulation assumption. Static score=mean absolute RGB change/255 on each16x16coarse block;row-major ties. Rank frozen before local predictions. No native pixels,labels or model outcomes passed to ranker.',
        'local_input':'Only purchased native256 RGB pre+post crop,converted to BGR/127-1;no extra context,resize,padding or coarse image inside model.',
        'decoder':'Unchanged author4flip sigmoid averaging,localization>.38,damage argmax over4channels then multiply. No threshold or input sweep.',
        'compute_cost':'Four forward calls per acquired tile,reported separately from area acquisition budget. Flips reuse same paid pixels;not claimed free compute. Offline all-tile precomputation is measurement scaffolding,not an executed affordable route.',
        'call_cap':1540,'call_inventory':{'old_full1024_regression':4,'new_native256_24x16x4':1536},
        'call_order':'4oldcontrol flips for firstsortedID;then sorted24scenes,rowmajor16tiles,4fixedflips;batch1 MPSfloat32;no warmup',
        'old_control':{'sample_id':first,'mean_probability_max_abs_tolerance':1e-5,'decoded_prediction_must_match_exactly':True},
        'primary_reward':'Distinct severe-reference pixels correctly predicted inside acquired disjoint tiles. Unobserved severe pixels remain FN;unknown/conflicting reference255 ignored. Count all24 scenes.',
        'baselines':['uniform_random_exact_expected_counts','static_public_coarse_change','unattainable_oracle_top_true_positive_gain'],
        'random':'Exact K/16 expectation for additive TP/FP and recall,not a selected random seed. Report precision/F1 as ratio of expected counts,not expectation of nonlinear metrics. Exact scene-hit probability uses hypergeometric zero-hit formula.',
        'oracle':'Offline upper bound uses all true tile gains for same fixed detector/cost. Not a feasible actor or new method.',
        'metrics':['all-grid256P/R/F1','per-scene detectability and input-retention','per-budget pooled severe-reference recall and TP/scene','equal-positive-scene recall','static/random/oracle spread'],
        'prospective_screen':{'minimum_reference_positive_scenes_with_local_TP':6,'minimum_all_grid_recall_fraction_of_previous_full1024':0.5,
            'note':'Engineering feasibility screen only,not a publication threshold or statistical test. If insufficient,stop planner work under this contract. Even if passed,only advance to prospectively defined multi-event measurement,not a method claim.'},
        'scope':'Already-used single-event development. All24 retained including existing black regions;no quality-based deletion or outcome-based reselection.',
        'tests_passed_before_predictions':357,'new_rgb':False,'training':False,'gpu_rental':False,'old_reserved_sets_enabled':False,
        'next_step':'Only if measurement survives,define fresh event-held-out metadata cohort and quality rules before payloads;no adaptive-controller implementation in this phase;oldEBD stopped variants remain stopped.'}
    freeze(OUT/'protocol.json',protocol)
    print(json.dumps({'status':'PLANNED','calls':1540,'local_tiles':384,'primary_budget':16,'area_fraction':0.25}),flush=True)


def predict():
    import torch
    p=verify()
    if (OUT/'calls').exists():raise ValueError('No silent restart of model calls')
    torch.manual_seed(321);np.random.seed(321);torch.set_num_threads(4)
    net,info=load_network(SOURCE/'utils/models.py',SOURCE/'strong_baseline_event.pt');net.to('mps')
    for n in ['calls','cache']:(OUT/n).mkdir()
    rows=[];caches=[];start=time.monotonic()
    def forward(array,role,sid,tile_id,flip_id):
        index=len(rows)+1
        if index>1540:raise ValueError('Call cap exceeded')
        row={'index':index,'role':role,'sample_id':sid,'tile_id':tile_id,'flip_id':flip_id,'input_shape':list(array.shape),
             'input_sha256':rawsha(array),'protocol_sha256':digest(OUT/'protocol.json'),'status':'STARTED'}
        path=OUT/'calls'/f'{index:04d}.json';save(path,row)
        x=torch.from_numpy(array).to('mps');torch.mps.synchronize();t=time.monotonic()
        with torch.inference_mode():logits=net(x)
        torch.mps.synchronize();elapsed=time.monotonic()-t
        assert tuple(logits.shape)==(1,5,*array.shape[-2:]) and torch.isfinite(logits).all()
        probability=torch.sigmoid(logits)[0].detach().cpu().numpy().copy()
        row.update(status='COMPLETE',elapsed_seconds=elapsed,probability_sha256=rawsha(probability));save(path,row);rows.append(row)
        return probability
    def four_flips(pre,post,role,sid,tile_id):
        base=prepare_pair(pre,post);probabilities=[]
        for i,axes in enumerate(FLIPS):
            array=np.flip(base,axis=axes).copy() if axes else base.copy()
            probability=forward(array,role,sid,tile_id,i)
            probabilities.append(restore_probability(probability,axes))
        return np.stack(probabilities).mean(0)
    jobs=read(OUT/'rgb_jobs.json');first=jobs[0]
    pair=[np.asarray(Image.open(r['path'])).copy() for r in first['images']]
    mean=four_flips(*pair,'old_full1024_control',first['sample_id'],None)
    old=read(OUT/'old_control.json')
    with np.load(old['path'],allow_pickle=False) as d:
        error=float(np.max(np.abs(mean-d['mean_probabilities'])))
        pred=decode_probabilities(mean)[2];exact=np.array_equal(pred,d['prediction'])
    np.savez_compressed(OUT/'old_control_mean.npz',mean_probabilities=mean)
    if error>1e-5 or not exact:raise ValueError('Old full-scene regression failed')
    save(OUT/'old_control_report.json',{'status':'PASS','max_abs_mean_probability_difference':error,'prediction_pixel_identical':bool(exact),'calls':4})
    del mean,pred,pair
    for scene_index,job in enumerate(jobs,1):
        sid=job['sample_id'];pair=[]
        for r in job['images']:
            checked(r['path'],r['sha256']);pair.append(np.asarray(Image.open(r['path'])).copy())
        for unit in units():
            tile=unit['tile_id'];local=crop_pair(*pair,tile)
            mean=four_flips(*local,'native256',sid,tile);loc,damage,pred=decode_probabilities(mean)
            cache=OUT/'cache'/f'{sid}_{tile:02d}.npz'
            np.savez_compressed(cache,mean_probabilities=mean,prediction=pred)
            caches.append({'sample_id':sid,'tile_id':tile,'bounds':unit['bounds'],'cost':4,'path':str(cache),
                'sha256':digest(cache),'mean_probability_sha256':rawsha(mean),'prediction_pixel_sha256':rawsha(pred)})
            del local,mean,loc,damage,pred
        save(OUT/'cache_manifest.json',caches);save(OUT/'observations.json',rows)
        gc.collect();torch.mps.empty_cache()
        print(json.dumps({'scenes_completed':scene_index,'tiles_completed':len(caches),'model_calls':len(rows)}),flush=True)
    assert len(rows)==1540 and len(caches)==384
    save(OUT/'runtime.json',{'status':'COMPLETE','new_forward_calls':len(rows),'new_local_calls':1536,'old_control_calls':4,
         'forward_seconds':sum(r['elapsed_seconds'] for r in rows),'loop_seconds':time.monotonic()-start,'device':'mps','model_admission':info,
         'training':False,'gpu_rental':False,'new_RGB':0})


def evaluate():
    p=verify();refs={r['sample_id']:r for r in read(OUT/'references.json')}
    public={r['sample_id']:r for r in read(OUT/'public_static_orders.json')}
    cache=read(OUT/'cache_manifest.json');assert len(cache)==384
    tile_rows=[];scene_rows=[];policy_rows=[]
    (OUT/'assembled').mkdir()
    old_scores={r['sample_id']:r for r in read(PARENT/'scene_scores.json')}
    for sid in p['selected_ids']:
        truth=np.asarray(Image.open(refs[sid]['reference_file']));prediction=np.zeros((1024,1024),np.uint8);local_rows=[]
        items=sorted([r for r in cache if r['sample_id']==sid],key=lambda r:r['tile_id'])
        assert [r['tile_id'] for r in items]==list(range(16))
        for item in items:
            checked(item['path'],item['sha256']);x,y,xx,yy=item['bounds']
            with np.load(item['path'],allow_pickle=False) as d:pred=decode_probabilities(d['mean_probabilities'])[2]
            prediction[y:yy,x:xx]=pred;r=counts(truth[y:yy,x:xx],pred);local_rows.append(r)
            tile_rows.append({'sample_id':sid,'tile_id':item['tile_id'],**r})
        Image.fromarray(prediction).save(OUT/'assembled'/(sid+'.png'))
        total=counts(truth,prediction);score=outcome(total['tp'],total['fp'],total['reference'])
        scene_rows.append({'sample_id':sid,**score,'valid_pixels':total['valid_pixels'],
            'detectable_tiles':sum(r['tp']>0 for r in local_rows),'old_full1024_tp':old_scores[sid]['tp'],
            'all_grid_TP_fraction_of_old_full1024':score['tp']/old_scores[sid]['tp'] if old_scores[sid]['tp'] else None})
        oracle=oracle_order(local_rows)
        for budget in BUDGETS:
            for policy,order in [('static_coarse_change',public[sid]['order']),('oracle_true_gain',oracle)]:
                policy_rows.append({'sample_id':sid,'policy':policy,'budget':budget,**policy_result(local_rows,order,budget)})
            policy_rows.append({'sample_id':sid,'policy':'random_exact_expected_counts','budget':budget,**exact_random(local_rows,budget)})
    totals=[]
    for budget in BUDGETS:
        for policy in ['random_exact_expected_counts','static_coarse_change','oracle_true_gain']:
            rows=[r for r in policy_rows if r['budget']==budget and r['policy']==policy]
            tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);ref=sum(r['reference'] for r in rows)
            positive=[r for r in rows if r['reference']>0]
            totals.append({'budget':budget,'policy':policy,'scenes':24,**outcome(tp,fp,ref),
                'mean_TP_per_scene':tp/24,'mean_positive_scene_recall':sum(r['recall'] for r in positive)/len(positive),
                'total_acquisitions':sum(r['acquisitions'] for r in rows),'mean_area_spent':sum(r['spent'] for r in rows)/24,
                'mean_forward_calls_per_scene':sum(r['acquisitions']*4 for r in rows)/24})
    allgrid=outcome(sum(r['tp'] for r in scene_rows),sum(r['fp'] for r in scene_rows),sum(r['reference'] for r in scene_rows))
    old=read(PARENT/'summary.json');detectable=sum(r['tp']>0 for r in scene_rows)
    retention=allgrid['recall']/old['recall'];primary={r['policy']:r for r in totals if r['budget']==16}
    signal_ok=detectable>=6 and retention>=.5
    headroom=primary['oracle_true_gain']['tp']-primary['static_coarse_change']['tp']
    summary={'all_grid_native256':allgrid,'previous_full1024':old,'reference_positive_scenes':sum(r['reference']>0 for r in scene_rows),
        'local_detectable_scenes':detectable,'all_grid_recall_fraction_of_old_full1024':retention,
        'local_complete_miss_ids':[r['sample_id'] for r in scene_rows if r['reference']>0 and r['tp']==0],
        'primary_budget':16,'primary':primary,'primary_oracle_minus_static_TP':headroom,
        'engineering_signal_screen_passed':signal_ok,'diagnostic_oracle_static_headroom_positive':headroom>0,
        'decision':'PROCEED_TO_PROSPECTIVE_MULTIEVENT_MEASUREMENT_ONLY' if signal_ok and headroom>0 else 'STOP_PLANNER_UNDER_THIS_CONTRACT',
        'new_method_or_independent_efficacy_claim':False}
    for n,o in [('tile_scores.json',tile_rows),('scene_scores.json',scene_rows),('policy_scores.json',policy_rows),('totals.json',totals),('summary.json',summary)]:save(OUT/n,o)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','predict','evaluate'],required=True)
    args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'plan':plan,'predict':predict,'evaluate':evaluate}[args.stage]()
