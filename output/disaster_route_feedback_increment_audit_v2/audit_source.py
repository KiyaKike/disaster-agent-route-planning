"""Post-run audit of Claude's v1. Preserve v1, correct primary-cell metrics.

No new method or parameter selection. This is explicitly a post-hoc metric audit,
not a new preregistered experiment. The cross-fold donor control is excluded.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import feedback_increment_core as core

ROOT=Path(__file__).resolve().parents[2]
V1=ROOT/'output/disaster_route_feedback_increment_v1'
OUT=ROOT/'output/disaster_route_feedback_increment_audit_v2'
SRC=ROOT/'output/disaster_route_real_xbd36_v1'


def average_precision_tied(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    positives=y.sum()
    if not positives:return None
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(yy)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])/positives * tp/(ends+1)))


def checked_fit(X,y,l2=core.L2):
    X=np.asarray(X,float);y=np.asarray(y,float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1
    z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    def loss(w):
        s=z@w;g=z.T@(expit(s)-y);g[1:]+=l2*w[1:]
        return np.sum(np.logaddexp(0,s)-y*s)+.5*l2*np.sum(w[1:]**2),g
    r=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B')
    FITS.append({'n':len(y),'positives':int(y.sum()),'success':bool(r.success),'message':str(r.message)})
    return {'mu':mu,'sd':sd,'w':r.x}


FITS=[]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    OUT.mkdir(exist_ok=False)
    source_hashes={str(p.relative_to(V1)):sha(p) for p in V1.rglob('*') if p.is_file()}
    old=json.loads((V1/'summary.json').read_text())
    protocol=json.loads((V1/'protocol.json').read_text())
    assert all(sha(SRC/n)==h for n,h in protocol['input_sha256'].items())
    # Snapshot the audit before calculating corrected metrics.
    (OUT/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    dump(OUT/'scope.json',{'date':'2026-11-03','post_hoc_audit':True,'primary_cell_only':True,
         'changes':['threshold-grouped AP, invariant to tied-score ordering',
                    'exclude cross-fold donor control from strict transfer conclusions',
                    'raw ranking utilities get no probabilistic log-loss or Brier interpretation'],
         'original_model_features_regularization_splits_unchanged':True,'original_files':source_hashes})
    public=json.loads((SRC/'public_features.json').read_text());private=json.loads((SRC/'private_tile_scores.json').read_text())
    states=core.route_prefixes(json.loads((SRC/'routes.json').read_text()))
    rows=core.build_rows(states,public,private);y=np.array([r['y_tp'] for r in rows])
    donor=core.permuted_donors(states)
    leakage={e:sum(s['event']!=e and d['event']==e for s,d in zip(states,donor)) for e in core.EVENTS}
    core.average_precision=average_precision_tied
    core.fit_logistic=checked_fit
    scores={}
    for name in ('prior','prior_feedback'):
        fn=(lambda r:r['prior']) if name=='prior' else (lambda r:r['prior']+r['feedback'])
        scores[name]=core.out_of_fold(rows,'y_tp',fn,'event')
    metrics={n:core.metric_block(y,s) for n,s in scores.items()}
    # Independently reproduce the previously stored non-AP metrics.
    for n in scores:
        for k in ('auroc','log_loss','brier'):
            assert abs(metrics[n][k]-old['primary_pooled'][n][k])<1e-7,(n,k)
    per_event={e:{n:core.metric_block(y[mask],s[mask]) for n,s in scores.items()}
               for e in core.EVENTS for mask in [np.array([r['event']==e for r in rows])]}
    for n,key in [('post_utility_unfitted','post_utility'),('neighbor_rule_unfitted','neighbor_rule')]:
        s=np.array([r[key] for r in rows]);scores[n]=s
        metrics[n]={'n':len(y),'positives':int(y.sum()),'auroc':core.auroc(y,s),
                    'average_precision':average_precision_tied(y,s),'log_loss':None,'brier':None}
    started=time.perf_counter()
    ci=core.chip_bootstrap_delta(rows,'y_tp',scores['prior'],scores['prior_feedback'])
    point={k:metrics['prior_feedback'][k]-metrics['prior'][k] for k in ('auroc','average_precision','log_loss','brier')}
    next_pick={n:core.next_pick_hits(rows,s,'y_tp') for n,s in scores.items()}
    result={'primary_cell':'route|y_tp|event','states':len(states),'rows':len(rows),'unique_chips':36,
       'unique_positive_tp_tiles':sum(c['tp']>0 for cs in private.values() for c in cs),
       'metrics':metrics,'per_event':per_event,'feedback_delta':point,'conditional_fixed_prediction_bootstrap':ci,
       'cross_fold_donor_training_states':leakage,'optimizer_checks':FITS,
       'next_pick':{n:{'opportunity_states':len(v),'hits':sum(x[1] for x in v),'rate':np.mean([x[1] for x in v])} for n,v in next_pick.items()},
       'decision':'NOT_ESTABLISHED','interpretation':'No robust incremental benefit established for this model/features on reused36;not proof that feedback has no value.',
       'v1_decision_unchanged':old['decision']=='NOT_ESTABLISHED',
       'scope':'Development cross-validation; repeated prefix/tile rows are not independent samples. Bootstrap conditions on fitted predictions, does not refit or quantify transfer to new events.',
       'bootstrap_seconds':time.perf_counter()-started,'gpu_rental':False,'new_perception_inference':0}
    assert point['average_precision']<=0 or ci['log_loss']['ci95'][1]>=0
    dump(OUT/'summary.json',result)
    dump(OUT/'primary_scores.json',[{**{k:r[k] for k in ('state','sample_id','event','k','tile','y_tp','y_ref')},
                                   **{n:float(s[i]) for n,s in scores.items()}} for i,r in enumerate(rows)])
    assert all(sha(V1/n)==h for n,h in source_hashes.items())
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':run()
