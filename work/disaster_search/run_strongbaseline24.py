"""Author-default full-scene development diagnostic on fixed24SantaRosa.

No training, detector selection, input sweep, reserved-set scores or policy.
Prediction reads RGB jobs only; evaluation separately reads frozen references.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from strongbaseline24_core import prepare_pair,decode_probabilities,restore_probability,FLIPS,load_network,severe_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output/disaster_search_strongbaseline24_v1'
PARENT=ROOT/'output/disaster_search_mexico_trace3_v1'
PAIR=ROOT/'output/disaster_search_perception_pair72_v1'
SOURCE=HERE/'attachments/external/disasteradaptivenet_fada4e7'
NEW_SOURCES=['acquire_strongbaseline24.py','strongbaseline24_core.py','test_strongbaseline24.py',
             'run_strongbaseline24.py','audit_strongbaseline24.py']


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rawsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def checked(p,h):
    if digest(p)!=h:raise ValueError('Changed file: '+str(p))
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def freeze(p,value):
    if Path(p).exists():raise ValueError('Already frozen: '+str(p))
    save(p,value)


def verify():
    protocol=read(OUT/'protocol.json')
    for n,h in protocol['source_code_sha256'].items():checked(HERE/n,h)
    for n,h in protocol['upstream_files_sha256'].items():checked(SOURCE/n,h)
    for n,h in protocol['external_files'].items():checked(n,h)
    checked(OUT/'rgb_jobs.json',protocol['rgb_jobs_sha256'])
    checked(OUT/'references.json',protocol['references_sha256'])
    checked(OUT/'test_log.txt',protocol['test_log_sha256'])
    return protocol


def plan():
    import torch
    parent=read(PARENT/'completion_receipt.json')
    assert digest(PARENT/'completion_receipt.json')=='f0e20b9b7783cecb73b7a36f8003213cf3ede6fb15b254f38fe5d6ba17205e2b'
    for n,h in parent['files'].items():checked(PARENT/n,h)
    for n,h in parent['external_files'].items():checked(n,h)
    for n,h in parent['source_code_sha256'].items():checked(HERE/n,h)
    assert 'Ran 349 tests' in (OUT/'test_log.txt').read_text() and (OUT/'test_log.txt').read_text().rstrip().endswith('OK')
    admission=read(OUT/'checkpoint_admission.json');assert admission['strict_state_dict'] and admission['new_forward_calls']==0
    assert admission['mps_available'] and torch.backends.mps.is_available()
    acquisition=read(OUT/'checkpoint_acquisition.json');checked(acquisition['path'],acquisition['sha256'])
    assert acquisition['sha256']=='fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6'
    provenance=read(PARENT/'prospective_event_baseline.json')
    members=[r for r in provenance['current72_membership'] if r['event']=='santa-rosa-wildfire']
    assert len(members)==24 and all(r['author_split']=='test' for r in members)
    selected=sorted(r['sample_id'] for r in members)
    author_lists={}
    external={str(PARENT/'completion_receipt.json'):digest(PARENT/'completion_receipt.json')}
    for row in provenance['split_files']:
        checked(row['file'],row['sha256']);external[row['file']]=row['sha256']
        author_lists[row['split']]=set(Path(row['file']).read_text().splitlines())
    for sid in selected:
        name='hold/images/'+sid
        assert name in author_lists['test'] and name not in author_lists['train']|author_lists['val']
    jobs=[r for r in read(PAIR/'rgb_jobs.json') if r['sample_id'] in selected]
    refs=[r for r in read(PAIR/'reference_manifest.json') if r['sample_id'] in selected]
    assert len(jobs)==len(refs)==24
    jobs.sort(key=lambda r:r['sample_id']);refs.sort(key=lambda r:r['sample_id'])
    for job in jobs:
        for im in job['images']:
            checked(im['path'],im['sha256']);external[im['path']]=im['sha256']
            with Image.open(im['path']) as image:assert image.mode=='RGB' and image.size==(1024,1024)
    for ref in refs:
        for p,h in [(ref['reference_file'],ref['reference_sha256']),(ref['label_path'],ref['label_sha256'])]:checked(p,h);external[p]=h
    for n in ['reference_manifest.json','rgb_jobs.json','completion_receipt.json']:
        external[str(PAIR/n)]=digest(PAIR/n)
    freeze(OUT/'rgb_jobs.json',jobs);freeze(OUT/'references.json',refs)
    upstream={str(p.relative_to(SOURCE)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
    synthetic=np.linspace(-1,255/127-1,6*64*64,dtype=np.float32).reshape(1,6,64,64)
    sources={**parent['source_code_sha256'],**{n:digest(HERE/n) for n in NEW_SOURCES}}
    protocol={'stage':'AUTHOR_DEFAULT_STRONG_BASELINE_FIXED24_SANTAROSA_DEVELOPMENT',
        'client_date':'2026-10-28','parent_receipt_sha256':digest(PARENT/'completion_receipt.json'),
        'source_commit':'fada4e791766f810dc815a4e82d52f2bb27116c6','source_code_sha256':sources,
        'upstream_files_sha256':upstream,'external_files':external,'selected_ids':selected,
        'rgb_jobs_sha256':digest(OUT/'rgb_jobs.json'),'references_sha256':digest(OUT/'references.json'),
        'test_log_sha256':digest(OUT/'test_log.txt'),'tests_passed_before_predictions':349,
        'checkpoint_sha256':acquisition['sha256'],'checkpoint_bytes':309003425,
        'input':'Original1024x1024 BGR pre+post concatenated;float32 /127 -1;no resize,crop,padding or training augmentation',
        'inference':'Author four flips: identity,height,width,both. Sequential batch1 instead of author batch4 to bound memory;eval and BatchNorm frozen;average inverse-flipped sigmoid probabilities in original order.',
        'flips_batched_tensor_axes':FLIPS,'device':'mps','float':'float32','autocast':False,
        'decoder':'channel0 sigmoid localization>.38;argmax channels1..4+1 for damage;damage*localization;USE_TRICKS=False',
        'call_cap':98,'call_inventory':{'synthetic64_cpu':1,'synthetic64_mps':1,'production_full1024':96},
        'controls':{'input_sha256':rawsha(synthetic),'shape':list(synthetic.shape),'max_abs_logit_cpu_mps_tolerance':0.005,
                    'no_ground_truth':'Synthetic deterministic64 input; no extra RGB/model warmup or score-based selection'},
        'forward_order':'syntheticCPU,syntheticMPS,then sorted24scene IDs with four fixed flips each;no unlogged warmup',
        'reference':'Existing frozen post-polygon raster,classes0..4;unknown and conflicting overlaps255 excluded;severe3or4',
        'primary_metrics':'Pooled severe pixel P/R/F1,per-scene P/R/F1,positive-scene complete misses;not official mixedF1 or instanceAP',
        'preview':'All24 scenes in four sheets,6 per sheet sortedID;no outcome-based selection',
        'interpretation':'Author-training-unseen according to lists;one already-open project development event. No fresh independent or multi-event generalization claim. Published median run only.',
        'comparison':'Prior512ChangeOS values may be cited as historical context only;different context,TTA and training exposure prevent fair checkpoint ranking.',
        'next_scope':'After results,prospectively specify cost-compatible local observation feasibility;full1024 context and flips not free in budget search;old stoppedEBD variants stay stopped.',
        'new_rgb':False,'training':False,'gpu_rental':False,'old_reserved_sets_enabled':False,
        'threshold_tuning':False,'policy_tuning':False,'checkpoint_sweep':False}
    freeze(OUT/'protocol.json',protocol)
    print(json.dumps({'status':'PLANNED','production_calls':96,'controls':2,'device':'mps'}),flush=True)


def predict():
    import torch
    protocol=verify()
    if (OUT/'runtime.json').exists() or (OUT/'calls').exists():raise ValueError('Prediction is not silently repeated')
    torch.manual_seed(321);np.random.seed(321);torch.set_num_threads(4)
    net,info=load_network(SOURCE/'utils/models.py',SOURCE/'strong_baseline_event.pt')
    (OUT/'calls').mkdir();(OUT/'cache').mkdir();(OUT/'predictions').mkdir()
    rows=[];start=time.monotonic()
    def forward(array,device,role,sid=None,flip_index=None):
        index=len(rows)+1
        if index>protocol['call_cap']:raise ValueError('Forward cap exceeded')
        row={'index':index,'role':role,'sample_id':sid,'flip_index':flip_index,'device':device,
             'input_shape':list(array.shape),'input_sha256':rawsha(array),'protocol_sha256':digest(OUT/'protocol.json'),
             'status':'STARTED','at_utc':datetime.now(timezone.utc).isoformat()}
        path=OUT/'calls'/f'{index:03d}.json';save(path,row)
        tensor=torch.from_numpy(array).to(device)
        if device=='mps':torch.mps.synchronize()
        t=time.monotonic()
        with torch.inference_mode():logits=net(tensor)
        if device=='mps':torch.mps.synchronize()
        elapsed=time.monotonic()-t
        if tuple(logits.shape)!=(1,5,*array.shape[-2:]) or not torch.isfinite(logits).all():raise ValueError('Invalid logits')
        raw=logits[0].detach().cpu().numpy().copy()
        probability=torch.sigmoid(logits)[0].detach().cpu().numpy().copy()
        row.update(status='COMPLETE',elapsed_seconds=elapsed,logit_sha256=rawsha(raw),probability_sha256=rawsha(probability))
        rows.append(row);save(path,row);save(OUT/'observations.json',rows)
        return raw,probability
    array=np.linspace(-1,255/127-1,6*64*64,dtype=np.float32).reshape(1,6,64,64)
    assert rawsha(array)==protocol['controls']['input_sha256']
    raw_cpu,prob_cpu=forward(array,'cpu','synthetic_control')
    net.to('mps')
    raw_mps,prob_mps=forward(array,'mps','synthetic_control')
    error=float(np.max(np.abs(raw_cpu-raw_mps)))
    np.savez_compressed(OUT/'control_outputs.npz',logits_cpu=raw_cpu,logits_mps=raw_mps,probabilities_cpu=prob_cpu,probabilities_mps=prob_mps)
    if error>protocol['controls']['max_abs_logit_cpu_mps_tolerance']:raise ValueError('CPU/MPS synthetic parity check failed')
    save(OUT/'control_report.json',{'status':'PASS','max_abs_logits_difference':error,'tolerance':0.005,'model_calls':2})
    caches=[]
    for job in read(OUT/'rgb_jobs.json'):
        images=[]
        for item in job['images']:
            checked(item['path'],item['sha256'])
            with Image.open(item['path']) as im:images.append(np.asarray(im).copy())
        base=prepare_pair(*images);probs=[]
        for i,flip in enumerate(FLIPS):
            array=np.flip(base,axis=flip).copy() if flip else base.copy()
            _,prob=forward(array,'mps','production',job['sample_id'],i)
            probs.append(restore_probability(prob,flip))
        average=np.stack(probs).mean(axis=0);loc,damage,pred=decode_probabilities(average)
        cache=OUT/'cache'/(job['sample_id']+'.npz')
        np.savez_compressed(cache,mean_probabilities=average,building=loc,ungated_damage=damage,prediction=pred)
        prediction=OUT/'predictions'/(job['sample_id']+'.png');Image.fromarray(pred).save(prediction)
        caches.append({'sample_id':job['sample_id'],'path':str(cache),'sha256':digest(cache),
            'mean_probabilities_sha256':rawsha(average),'prediction_path':str(prediction),'prediction_sha256':digest(prediction)})
        save(OUT/'cache_manifest.json',caches)
        print(json.dumps({'scenes_completed':len(caches),'model_calls':len(rows),'sample_id':job['sample_id']}),flush=True)
        del probs,average,base,array,prob,loc,damage,pred
        gc.collect();torch.mps.empty_cache()
    assert len(rows)==98 and len(caches)==24
    save(OUT/'runtime.json',{'status':'COMPLETE','loop_seconds':time.monotonic()-start,'new_forward_calls':len(rows),
        'production_calls':96,'synthetic_control_calls':2,'forward_seconds':sum(r['elapsed_seconds'] for r in rows),
        'device':'mps','model_admission':info,'training':False,'gpu_rental':False,'new_RGB':0,
        'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow']}})


def evaluate():
    verify();refs={r['sample_id']:r for r in read(OUT/'references.json')};scores=[]
    for item in read(OUT/'cache_manifest.json'):
        sid=item['sample_id'];ref=refs[sid];checked(item['path'],item['sha256']);checked(ref['reference_file'],ref['reference_sha256'])
        truth=np.asarray(Image.open(ref['reference_file']))
        with np.load(item['path'],allow_pickle=False) as cache:loc,damage,pred=decode_probabilities(cache['mean_probabilities'])
        scores.append({'sample_id':sid,**severe_score(truth,pred)})
    totals={k:sum(s[k] for s in scores) for k in ['tp','fp','fn','evaluated_pixels','reference_severe_pixels','predicted_severe_pixels']}
    tp,fp,fn=[totals[k] for k in ['tp','fp','fn']]
    totals.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    totals.update(scenes=len(scores),severe_positive_scenes=sum(s['severe_positive_scene'] for s in scores),
                  severe_positive_scenes_with_any_tp=sum(s['positive_scene_any_correct_pixel'] for s in scores),
                  complete_miss_ids=[s['sample_id'] for s in scores if s['severe_positive_scene'] and not s['positive_scene_any_correct_pixel']],
                  reference_negative_scene_ids=[s['sample_id'] for s in scores if not s['severe_positive_scene']])
    save(OUT/'scene_scores.json',scores);save(OUT/'summary.json',totals)
    print(json.dumps(totals),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','predict','evaluate'],required=True)
    args=parser.parse_args()
    if (OUT/'completion_receipt.json').exists():raise ValueError('Completed output frozen')
    {'plan':plan,'predict':predict,'evaluate':evaluate}[args.stage]()
