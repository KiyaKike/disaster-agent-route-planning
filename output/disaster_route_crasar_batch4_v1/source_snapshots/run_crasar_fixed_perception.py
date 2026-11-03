"""Frozen cross-sensor admission probe; independent window inference prevents peeking.

No training or parameter fitting. UAS caches are private paid-observation records.
"""
import sys,json,time,hashlib,datetime,os
from pathlib import Path
import numpy as np
from PIL import Image
from probe_crasar_region import ROOT,O,dump
sys.path.insert(0,str(ROOT/'work/disaster_search'))
from strongbaseline24_core import load_network,prepare_pair,FLIPS,restore_probability,decode_probabilities

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Output is sealed')
    q=json.loads((O/'qualification.json').read_text())
    if not q['geometry_eligible']:raise RuntimeError('No eligible building geometry')
    source=ROOT/'work/disaster_search/attachments/external/disasteradaptivenet_fada4e7'
    checkpoint=source/'strong_baseline_event.pt'
    assert sha(checkpoint)=='fe7ded8c5012dee3d39675226b0aa1fd671d8c19cef5c88080daa8b13645c6b6'
    source_info=json.loads((O/'protocol.json').read_text())['images']
    dates={m:datetime.datetime.strptime(v['Date (mm/dd/yyy)'],'%m/%d/%Y').date() for m,v in source_info.items()}
    lag=(dates['uas']-dates['post']).days
    secondary_scope='Secondary any-damage endpoint predeclared for this batch before label inspection; still training development,not independent confirmation' if os.environ.get('CRASAR_BATCH_PREDECLARED')=='1' else 'Secondary any-damage endpoint added after label prevalence inspection, before predictions; exploratory only'
    protocol={'status':'FROZEN_BEFORE_PREDICTIONS','model':'Existing pinned StrongBaselineNet/ResNet34, strict checkpoint, no fitting',
      'checkpoint_sha256':sha(checkpoint),'sources':['PRE/POST satellite for public prior','PRE/UAS for private paid observation'],
      'window_contract':'16 disjoint32m windows, independently resampled to128x128 RGB(.25m model grid); no UAS context outside paid window',
      'test_time_flips':[list(f) for f in FLIPS],'localization_threshold':.38,
      'damage_proxy':'Two predeclared score sets: mean localization score times normalized severe-class scores; and mean localization score times normalized all-damage scores. Separate results, no best-target selection.',
      'uncertainty_proxy':'Mean1−top-two normalized damage-score margin; not calibrated probability or expected information gain',
      'correct_check':'At least80% footprint accessed and relevant damage-class model pixels at least10% of accessed footprint pixels; primary truth=UAS major/destroyed; exploratory secondary=minor/major/destroyed.',
      'candidate':'Prior damage proxy × uncertainty proxy × newly accessed building fraction / incremental return-inclusive route cost',
      'budgets_proxy_s':[60,100,140],'main_budget_proxy_s':100,'assumed_speed_mps':5,'assumed_observation_s':10,'assumed_planning_s':1,
      'home':'South midpoint of ROI','static_controls':'Exact all-subset closed tours for damage proxy and for candidate public utility; never truth-informed',
      'limitations':['Cross-sensor model not calibrated','No measured flight footprint or energy','One development region','Reference labels not physical field ground truth',f'UAS minus prior POST image gap{lag}days',secondary_scope,'Independent registration accuracy not established;see qualification report'],
      'training':False,'gpu_rental':False}
    pp=O/'perception_route_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:dump(pp,protocol)
    import torch
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    net,info=load_network(source/'utils/models.py',checkpoint);net.to(device)
    imgs={m:Image.open(O/f'{m}_roi.png').convert('RGB') for m in ['pre','post','uas']}
    cache=O/'fixed_model_cache';cache.mkdir(exist_ok=True)
    logs=json.loads((O/'inference_log.json').read_text()) if (O/'inference_log.json').exists() else []
    calls_before=sum(r['forward_calls'] for r in logs);started=time.perf_counter()
    for j in range(16):
      row,col=divmod(j,4);box=(col*256,row*256,(col+1)*256,(row+1)*256)
      before=np.array(imgs['pre'].crop(box).resize((128,128),Image.Resampling.BILINEAR))
      for mode in ['post','uas']:
        target=cache/f'{mode}_{j:02d}.npz'
        if target.exists():
            matches=[r for r in logs if r['mode']==mode and r['window']==j]
            assert len(matches)==1 and sha(target)==matches[0]['cache_sha256'],'Unlogged or altered cache'
            continue
        after=np.array(imgs[mode].crop(box).resize((128,128),Image.Resampling.BILINEAR));base=prepare_pair(before,after)
        t=time.perf_counter();probs=[]
        with torch.inference_mode():
          for flip in FLIPS:
            a=np.flip(base,axis=flip).copy() if flip else base
            logits=net(torch.from_numpy(a).to(device))
            if isinstance(logits,(list,tuple)):raise RuntimeError('Unexpected model output')
            probability=torch.sigmoid(logits)[0].detach().to('cpu').numpy().astype(np.float32)
            probs.append(restore_probability(probability,flip))
        mean=np.stack(probs).mean(axis=0);_,_,pred=decode_probabilities(mean)
        np.savez_compressed(target,mean_scores=mean,prediction=pred)
        logs.append({'mode':mode,'window':j,'forward_calls':4,'seconds':time.perf_counter()-t,'cache_sha256':sha(target)})
        dump(O/'inference_log.json',logs)
        print(json.dumps(logs[-1]),flush=True)
    dump(O/'perception_runtime.json',{'device':device,'new_forward_calls_this_run':sum(r['forward_calls'] for r in logs)-calls_before,'total_logged_forward_calls':sum(r['forward_calls'] for r in logs),
      'cache_files':32,'wall_seconds':time.perf_counter()-started,'strict_model_load':info,'gpu_rental':False,'training':False})

if __name__=='__main__':run()
