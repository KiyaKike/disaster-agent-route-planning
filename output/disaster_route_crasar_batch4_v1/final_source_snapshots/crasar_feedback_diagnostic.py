"""Region-purged information diagnostic, not an online route updater.

Targets are future fixed-detector outputs, not human truth or rescue value.
All features use PRE/POST prior or the explicitly paid prefix only.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_crasar_batch import ROOT,B,dump

PRIOR=['window_prior_damage_score','window_prior_uncertainty','window_prior_localization','building_mask_fraction',
       'x_relative','y_relative','global_prior_damage','log_paid_count']
FEEDBACK=['mean_paid_model_damage','mean_paid_minus_prior','distance_weighted_paid_minus_prior','nearest_paid_minus_prior']

def assert_separate(target,prefix,window_ids):
    a=set(window_ids[target]);b=set().union(*(set(window_ids[j]) for j in prefix))
    if a&b:raise ValueError('Candidate shares a building with a paid window')

def feedback_features(j,prefix,prior,paid,xy):
    # paid is a dictionary restricted to the paid prefix: no candidate/future values.
    if set(paid)!=set(prefix) or j in paid:raise ValueError('Exactly paid-prefix values required')
    dist=np.linalg.norm(xy[prefix]-xy[j],axis=1);weights=1/(1+dist/32.)
    vals=np.array([paid[k] for k in prefix]);residual=vals-prior[prefix,0]
    return np.array([vals.mean(),residual.mean(),np.average(residual,weights=weights),residual[np.argmin(dist)]])

def ridge_fit_predict(train_x,train_y,test_x,weights,penalty=1.):
    # All scaling and fitting only on the training fold. Intercept is unpenalized.
    mu=np.average(train_x,axis=0,weights=weights)
    scale=np.sqrt(np.average((train_x-mu)**2,axis=0,weights=weights));scale[scale<1e-8]=1
    x=np.column_stack((np.ones(len(train_x)),(train_x-mu)/scale));z=np.column_stack((np.ones(len(test_x)),(test_x-mu)/scale))
    reg=np.eye(x.shape[1])*penalty;reg[0,0]=0
    coef=np.linalg.solve(x.T@(weights[:,None]*x)+reg,x.T@(weights*train_y))
    return np.clip(z@coef,0,1)

def load_case(folder):
    geo=[g for g in json.loads((folder/'public_geometry.json').read_text()) if g['eligible']]
    masks=[]
    for g in geo:
        im=Image.new('1',(512,512));ImageDraw.Draw(im).polygon([tuple(np.array(v)/2) for v in g['roi_polygon_px']],fill=1);masks.append(np.array(im))
    union=np.logical_or.reduce(masks) if masks else np.zeros((512,512),bool)
    prior=np.zeros((16,4));future=np.full(16,np.nan);ids=[]
    for j in range(16):
        rr,cc=divmod(j,4);m=union[rr*128:(rr+1)*128,cc*128:(cc+1)*128]
        ids.append([g['building_id'] for g,mask in zip(geo,masks) if mask[rr*128:(rr+1)*128,cc*128:(cc+1)*128].any()])
        prior[j,3]=m.mean()
        if not m.any():continue
        with np.load(folder/'fixed_model_cache'/f'post_{j:02d}.npz') as f:s=f['mean_scores']
        norm=s[1:]/np.maximum(s[1:].sum(axis=0),1e-8);ordered=np.sort(norm,axis=0)
        prior[j,:3]=[(s[0]*norm[1:].sum(axis=0))[m].mean(),(1-(ordered[-1]-ordered[-2]))[m].mean(),s[0][m].mean()]
        with np.load(folder/'fixed_model_cache'/f'uas_{j:02d}.npz') as f:future[j]=np.isin(f['prediction'],[2,3,4])[m].mean()
    route=next(r for r in json.loads((folder/'routes_any_damage.json').read_text()) if r['policy']=='distance' and r['budget_proxy_s']==100)['route']
    xy=np.array([[c*32+16,128-r*32-16] for r in range(4) for c in range(4)],float)
    roi=json.loads((folder/'roi_protocol.json').read_text())
    return {'folder':str(folder),'prior':prior,'future':future,'ids':ids,'all_ids':set(g['building_id'] for g in geo),'route':route,'xy':xy,'bounds':roi['bounds'],'crs':roi['crs']}

def make_rows(case):
    rows=[];excluded=0;prior=case['prior'];future=case['future'];route=case['route'];xy=case['xy']
    globalmean=float(np.mean(prior[prior[:,3]>0,0]))
    for k in range(1,len(route)+1):
        prefix=route[:k]
        # Empty-building windows supply no damage feedback; retain their paid count.
        useful=[j for j in prefix if np.isfinite(future[j])]
        if not useful:continue
        paid={j:future[j] for j in useful}
        for j in range(16):
            if j in prefix or not np.isfinite(future[j]):continue
            try:assert_separate(j,prefix,case['ids'])
            except ValueError:excluded+=1;continue
            x=np.r_[prior[j],xy[j]/128,globalmean,np.log1p(k)]
            f=feedback_features(j,useful,prior,paid,xy)
            rows.append({'candidate':j,'prefix':prefix,'prior':x,'feedback':f,'y':float(future[j])})
    return rows,excluded

def box_distance(a,b):
    dx=max(a[0]-b[2],b[0]-a[2],0);dy=max(a[1]-b[3],b[1]-a[3],0)
    return float(np.hypot(dx,dy))

def run():
    if (B/'completion_receipt.json').exists():raise RuntimeError('Batch sealed')
    protocol={'status':'FROZEN_BEFORE_BATCH_OUTCOMES','features_prior':PRIOR,'features_feedback':FEEDBACK,
      'target':'Fraction of eligible-building footprint pixels classified as any damage by the fixed model in a future UAS window. Model consistency,not human damage truth or marginal route gain.',
      'prefixes':'All nonempty prefixes of distance-priority route under100s proxy budget; never chosen by damage observations',
      'candidate_exclusion':'Visited windows,windows with no eligible footprint,and any window sharing a building ID with any paid window excluded',
      'folds':'Leave-one-new-region-out; older two inspected regions not used for fitting or selection',
      'purge':'Remove entire training region if shared eligible building IDs or ROI gap<=128m in same CRS with held-out region',
      'estimator':'Ridge regression,fixed lambda1,training-only weighted scaling,unpenalized intercept,clip predictions to[0,1];regions equal total training weight',
      'metrics':'Per-region MSE/MAE then macro average;no row-level significance or event generalization',
      'required_train_regions':2,'permutation':'None:no cross-fold feedback borrowing',
      'interpretation':'Incremental predictive information is necessary evidence,not sufficient proof of route or rescue benefit'}
    path=B/'feedback_diagnostic_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol)
    # This entry point may be called before acquisition just to freeze the specification.
    status=B/'execution_status.json'
    if not status.exists():print('Protocol frozen;waiting for batch');return
    folders=[Path(r['folder']) for r in json.loads(status.read_text()) if r['status']=='PROVISIONAL_REPLAY_COMPLETE']
    cases={f.name:load_case(f) for f in folders};rows={};excluded={}
    for name,c in cases.items():rows[name],excluded[name]=make_rows(c)
    scores=[];predictions=[]
    for held,test in cases.items():
        train=[];purged=[]
        for name,c in cases.items():
            if name==held:continue
            shared=bool(c['all_ids']&test['all_ids'])
            gap=box_distance(c['bounds'],test['bounds']) if c['crs']==test['crs'] else None
            if shared or (gap is not None and gap<=128):purged.append({'case':name,'shared_ids':shared,'roi_gap_m':gap});continue
            if rows[name]:train.append(name)
        if len(train)<2 or not rows[held]:
            scores.append({'held_out':held,'status':'INSUFFICIENT_SEPARATED_REGIONS_OR_ROWS','training_regions':train,'purged':purged,'test_rows':len(rows[held])});continue
        training=[(name,r) for name in train for r in rows[name]];testing=rows[held]
        w=np.array([1/len(rows[name]) for name,r in training]);w=w/w.sum()*len(training)
        y=np.array([r['y'] for _,r in training]);yt=np.array([r['y'] for r in testing])
        result={}
        for mode in ['prior','prior_plus_paid']:
            def feature(r):return r['prior'] if mode=='prior' else np.r_[r['prior'],r['feedback']]
            x=np.array([feature(r) for _,r in training]);z=np.array([feature(r) for r in testing])
            pred=ridge_fit_predict(x,y,z,w)
            result[mode]={'mse':float(np.mean((pred-yt)**2)),'mae':float(np.mean(abs(pred-yt)))}
            for r,v in zip(testing,pred):predictions.append({'held_out':held,'mode':mode,'prefix':r['prefix'],'candidate':r['candidate'],'target_future_model_fraction':r['y'],'prediction':float(v)})
        scores.append({'held_out':held,'status':'EVALUATED','training_regions':train,'purged':purged,'test_rows':len(testing),**result,
            'delta_mse_paid_minus_prior':result['prior_plus_paid']['mse']-result['prior']['mse']})
    valid=[s for s in scores if s['status']=='EVALUATED']
    aggregate={'status':'EVALUATED_EXPLORATORY' if valid else 'INSUFFICIENT_SEPARATED_DATA','regions_available':len(cases),'regions_evaluated':len(valid),
      'excluded_shared_building_candidate_rows':excluded,'per_region':scores,
      'macro_mse_prior':float(np.mean([s['prior']['mse'] for s in valid])) if valid else None,
      'macro_mse_prior_plus_paid':float(np.mean([s['prior_plus_paid']['mse'] for s in valid])) if valid else None,
      'regions_improved':sum(s['delta_mse_paid_minus_prior']<0 for s in valid),
      'method_gain_established':False,'event_generalization':False,'target_is_human_truth':False}
    dump(B/'feedback_increment_summary.json',aggregate);dump(B/'feedback_predictions.json',predictions)
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
