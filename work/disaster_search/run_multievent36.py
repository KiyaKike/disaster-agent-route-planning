"""Apply the unchanged native256 budget diagnostic to the fixed three new events."""
import argparse
import gc
import time
import numpy as np
from PIL import Image
from prepare_multievent36 import OUT,PARENT,ROOT,HERE,read,digest,checked,rawsha,save,put,verify_selection
from run_strongbaseline24 import SOURCE
from strongbaseline24_core import prepare_pair,decode_probabilities,restore_probability,FLIPS,load_network
from strongbaseline_local24_core import units,crop_pair,coarse_pair,static_order,counts,outcome,policy_result,exact_random,oracle_order,BUDGETS
from multievent36_core import EVENTS


def verify():
    p=read(OUT/'inference_protocol.json');verify_selection()
    for n,h in p['source_code_sha256'].items():checked(HERE/n,h)
    for n,h in p['internal_inputs'].items():checked(OUT/n,h)
    for n,h in p['external_files'].items():checked(n,h)
    for n,h in p['upstream_files_sha256'].items():checked(SOURCE/n,h)
    return p


def plan():
    selection=verify_selection();assert read(OUT/'data_admission.json')['status']=='PASS' and read(OUT/'RGB_quality_report.json')['status']=='PASS'
    old=read(PARENT/'protocol.json');jobs=read(OUT/'rgb_jobs.json');assert len(jobs)==36
    jobs.sort(key=lambda r:r['sample_id']);save(OUT/'rgb_jobs.json',jobs)
    public=[];(OUT/'coarse').mkdir()
    for job in jobs:
        cp=coarse_pair(*[np.asarray(Image.open(r['path'])) for r in job['images']]);public.append({'sample_id':job['sample_id'],**static_order(*cp)})
        for mode,a in zip(['pre','post'],cp):Image.fromarray(a).save(OUT/'coarse'/(job['sample_id']+'_'+mode+'.png'))
    put('public_static_orders.json',public)
    control=read(PARENT/'cache_manifest.json')[0];controljob=next(j for j in read(PARENT/'rgb_jobs.json') if j['sample_id']==control['sample_id'])
    put('old_control.json',{'cache':control,'job':controljob})
    external={str(PARENT/'completion_receipt.json'):digest(PARENT/'completion_receipt.json'),control['path']:control['sha256']}
    for r in controljob['images']:external[r['path']]=r['sha256']
    inputs={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.png','.txt']}
    sources={**selection['source_code_sha256'],**{n:digest(HERE/n) for n in ['run_multievent36.py','audit_multievent36.py']}}
    p={'stage':'LOCKED_NEW36_THREE_EVENT_LOCAL_MEASUREMENT','client_date':'2026-10-28','source_code_sha256':sources,'internal_inputs':inputs,'external_files':external,
        'upstream_files_sha256':old['upstream_files_sha256'],'selected_ids':[r['sample_id'] for r in jobs],'events':list(EVENTS),
        'measurement_contract':'Unchanged parent native256/cost4/primary16,budgets8,16,32,64;free64BOXcoarse;BGR/127-1;fourfixedflips;loc>.38,4grade argmax;no additional native context',
        'call_cap':2308,'call_inventory':{'old_native256_control':4,'new36x16x4':2304},
        'order':'Four old first-local-tile control flips;sortednew36IDs,rowmajor16tiles,four flips;batch1MPSfloat32;no warmup',
        'aggregation':'Mandatory per-event results. Primary overall reward=equal-event meanTP/scene;also report per-event pooled recalls. Macro recall only defined when all3 have positive references;otherwise null,not an implicit zero or exclusion.',
        'random':'Exact additive-count expectation;precision/F1 are ratios of expected counts. Oracle is an unattainable diagnostic bound.',
        'screen_before_methods':'Report source-positive and local-detectable scenes per event;fewer6local-detectable scenes in any event means insufficient evidence for method work on this36,not permission to substitute or poolSantaRosa.',
        'new_data_interpretation':'Author-training-unseen events by published lists;new local RGB/results,not independently verified geographic/checkpoint-history isolation. These36 permanently development measurement,not final confirmation.',
        'training':False,'gpu_rental':False,'reserved_sets_enabled':False,'adaptive_method_enabled':False,'threshold_search':False}
    put('inference_protocol.json',p);print({'protocol':'FROZEN','calls':2308,'new_scenes':36},flush=True)


def predict():
    import torch
    p=verify()
    if (OUT/'calls').exists():raise ValueError('No unchecked model restart')
    torch.manual_seed(321);torch.set_num_threads(4);net,info=load_network(SOURCE/'utils/models.py',SOURCE/'strong_baseline_event.pt');net.to('mps')
    for name in ['calls','cache']:(OUT/name).mkdir()
    rows=[];caches=[];start=time.monotonic()
    def infer(pre,post,sid,tile,role):
        base=prepare_pair(pre,post);prob=[]
        for flip_id,axes in enumerate(FLIPS):
            array=np.flip(base,axis=axes).copy() if axes else base.copy();index=len(rows)+1
            if index>2308:raise ValueError('Forward cap exceeded')
            row={'index':index,'sample_id':sid,'tile_id':tile,'role':role,'flip_id':flip_id,'input_shape':list(array.shape),'input_sha256':rawsha(array),'protocol_sha256':digest(OUT/'inference_protocol.json'),'status':'STARTED'}
            path=OUT/'calls'/f'{index:04d}.json';save(path,row);x=torch.from_numpy(array).to('mps');torch.mps.synchronize();t=time.monotonic()
            with torch.inference_mode():logits=net(x)
            torch.mps.synchronize();elapsed=time.monotonic()-t
            assert tuple(logits.shape)==(1,5,256,256) and torch.isfinite(logits).all()
            a=torch.sigmoid(logits)[0].cpu().numpy().copy();prob.append(restore_probability(a,axes))
            row.update(status='COMPLETE',elapsed_seconds=elapsed,probability_sha256=rawsha(a));save(path,row);rows.append(row)
        return np.stack(prob).mean(0)
    control=read(OUT/'old_control.json');job=control['job'];cache=control['cache'];images=[np.asarray(Image.open(r['path'])) for r in job['images']]
    mean=infer(*crop_pair(*images,cache['tile_id']),job['sample_id'],cache['tile_id'],'old_control')
    with np.load(cache['path'],allow_pickle=False) as old:
        error=float(np.max(np.abs(mean-old['mean_probabilities'])));exact=np.array_equal(decode_probabilities(mean)[2],old['prediction'])
    np.savez_compressed(OUT/'old_control_mean.npz',mean_probabilities=mean)
    if error>1e-5 or not exact:raise ValueError('Old256 regression failed')
    save(OUT/'old_control_report.json',{'status':'PASS','max_abs_probability_difference':error,'prediction_identical':bool(exact),'calls':4})
    for j,job in enumerate(read(OUT/'rgb_jobs.json'),1):
        images=[np.asarray(Image.open(r['path'])).copy() for r in job['images']]
        for unit in units():
            tile=unit['tile_id'];mean=infer(*crop_pair(*images,tile),job['sample_id'],tile,'production');pred=decode_probabilities(mean)[2]
            path=OUT/'cache'/f"{job['sample_id']}_{tile:02d}.npz";np.savez_compressed(path,mean_probabilities=mean,prediction=pred)
            caches.append({'sample_id':job['sample_id'],'event':job['event'],'tile_id':tile,'path':str(path),'sha256':digest(path),'mean_probability_sha256':rawsha(mean),'prediction_sha256':rawsha(pred)})
        save(OUT/'cache_manifest.json',caches);save(OUT/'observations.json',rows);gc.collect();torch.mps.empty_cache()
        print({'scenes_completed':j,'new_tiles':len(caches),'calls':len(rows)},flush=True)
    assert len(rows)==2308 and len(caches)==576
    save(OUT/'runtime.json',{'status':'COMPLETE','new_forward_calls':2308,'control_calls':4,'production_calls':2304,'device':'mps','forward_seconds':sum(r['elapsed_seconds'] for r in rows),'loop_seconds':time.monotonic()-start,'model_admission':info,'training':False,'gpu_rental':False})


def evaluate():
    p=verify();refs={r['sample_id']:r for r in read(OUT/'reference_manifest.json')};ranks={r['sample_id']:r['order'] for r in read(OUT/'public_static_orders.json')};manifest=read(OUT/'cache_manifest.json')
    tiles=[];scenes=[];policies=[];(OUT/'assembled').mkdir()
    for sid in p['selected_ids']:
        ref=refs[sid];truth=np.asarray(Image.open(ref['reference_file']));prediction=np.zeros((1024,1024),np.uint8);local=[]
        rows=sorted([r for r in manifest if r['sample_id']==sid],key=lambda r:r['tile_id']);assert [r['tile_id'] for r in rows]==list(range(16))
        for row in rows:
            checked(row['path'],row['sha256']);i=row['tile_id'];x,y=i%4*256,i//4*256
            with np.load(row['path'],allow_pickle=False) as d:pred=decode_probabilities(d['mean_probabilities'])[2]
            c=counts(truth[y:y+256,x:x+256],pred);local.append(c);tiles.append({'sample_id':sid,'event':ref['event'],'tile_id':i,**c});prediction[y:y+256,x:x+256]=pred
        Image.fromarray(prediction).save(OUT/'assembled'/(sid+'.png'));c=counts(truth,prediction)
        scenes.append({'sample_id':sid,'event':ref['event'],**outcome(c['tp'],c['fp'],c['reference']),'valid_pixels':c['valid_pixels'],'detectable_tiles':sum(t['tp']>0 for t in local)})
        for budget in BUDGETS:
            for name,order in [('static_coarse_change',ranks[sid]),('oracle_true_gain',oracle_order(local))]:policies.append({'sample_id':sid,'event':ref['event'],'budget':budget,'policy':name,**policy_result(local,order,budget)})
            policies.append({'sample_id':sid,'event':ref['event'],'budget':budget,'policy':'random_exact_expected_counts',**exact_random(local,budget)})
    events=[]
    for event in EVENTS:
        ss=[r for r in scenes if r['event']==event];c=outcome(sum(r['tp'] for r in ss),sum(r['fp'] for r in ss),sum(r['reference'] for r in ss))
        events.append({'event':event,'scenes':12,'reference_positive_scenes':sum(r['reference']>0 for r in ss),'local_detectable_scenes':sum(r['tp']>0 for r in ss),'all_grid':c})
    totals=[];macro=[]
    for budget in BUDGETS:
        for policy in ['random_exact_expected_counts','static_coarse_change','oracle_true_gain']:
            group=[]
            for event in EVENTS:
                rs=[r for r in policies if r['event']==event and r['budget']==budget and r['policy']==policy]
                c=outcome(sum(r['tp'] for r in rs),sum(r['fp'] for r in rs),sum(r['reference'] for r in rs))
                row={'event':event,'budget':budget,'policy':policy,**c,'mean_TP_per_scene':c['tp']/12};totals.append(row);group.append(row)
            macro.append({'budget':budget,'policy':policy,'events':3,'equal_event_mean_TP_per_scene':sum(r['mean_TP_per_scene'] for r in group)/3,
                'macro_recall':sum(r['recall'] for r in group)/3 if all(r['recall'] is not None for r in group) else None,
                'defined_reference_positive_events':sum(r['recall'] is not None for r in group)})
    summary={'events':events,'primary_budget':16,'primary_per_event':[r for r in totals if r['budget']==16],'primary_macro':[r for r in macro if r['budget']==16],
        'events_with_at_least6_local_detectable_scenes':sum(r['local_detectable_scenes']>=6 for r in events),
        'screen':'SUFFICIENT_DENSITY_FOR_METHOD_DESIGN_REVIEW' if all(r['local_detectable_scenes']>=6 for r in events) else 'INSUFFICIENT_CROSS_EVENT_MEASUREMENT_SIGNAL_DO_NOT_TUNE_PLANNER',
        'old_SantaRosa_pooled_in':False,'new_method_run':False,'independent_efficacy_claim':False}
    for n,v in [('tile_scores.json',tiles),('scene_scores.json',scenes),('policy_scores.json',policies),('event_totals.json',totals),('macro_totals.json',macro),('summary.json',summary)]:save(OUT/n,v)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','predict','evaluate'],required=True);args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'plan':plan,'predict':predict,'evaluate':evaluate}[args.stage]()
