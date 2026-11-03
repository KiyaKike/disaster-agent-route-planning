"""Does paid local feedback add predictive information about unvisited tiles?

Development diagnostic on the already-used xBD36 cache. Free inputs are only the
public 64px pre-texture / post-change cues. Feedback inputs are only the fixed
detector's predicted severe-pixel counts on tiles inside a paid prefix. Labels
(tp / reference) never enter features. Not a routing method, not validation.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from real_xbd_route_core import midrank, utility_from_evidence

EVENTS=('nepal-flooding','sunda-tsunami','tuscaloosa-tornado')
PRIOR_NAMES=('pre_rank','change_rank','log_pre_texture','log_post_change',
             'chip_mean_log_pre_texture','chip_mean_log_post_change','log_prefix_k')
FEEDBACK_NAMES=('any_observed_positive','fraction_observed_positive','log1p_mean_observed_pixels',
                'has_observed_neighbor','fraction_neighbor_positive','distance_to_observed_positive')
L2=1.0


def manhattan(a,b):
    return abs(a//4-b//4)+abs(a%4-b%4)


def route_prefixes(routes,policy='post_static',budget=240):
    """Prefixes of an already executed feedback-free route; not chosen by outcomes."""
    states=[]
    for r in routes:
        if r['policy']!=policy or r['budget_s_proxy']!=budget:continue
        visited=[h['tile_id'] for h in r['state']['history']]
        for k in range(1,len(visited)+1):
            states.append({'scheme':'route','sample_id':r['sample_id'],'event':r['event'],'prefix':visited[:k]})
    return states


def random_prefixes(sample_ids,ks=(1,2,4),draws=30,seed=20261102):
    rng=np.random.default_rng(seed);states=[]
    for sid in sorted(sample_ids):
        for k in ks:
            for _ in range(draws):
                states.append({'scheme':'random','sample_id':sid,'event':event_of(sid),
                               'prefix':[int(x) for x in rng.choice(16,size=k,replace=False)]})
    return states


def event_of(sample_id):
    for e in EVENTS:
        if sample_id.startswith(e):return e
    raise ValueError(sample_id)


def prior_matrix(pub,k):
    pre=np.asarray(pub['pre_texture'],float);chg=np.asarray(pub['post_change'],float)
    lp=np.log(pre+1e-6);lc=np.log(chg+1e-6)
    return np.column_stack([midrank(pre),midrank(chg),lp,lc,np.full(16,lp.mean()),np.full(16,lc.mean()),
                            np.full(16,math.log(k))])


def feedback_vector(tile,observed):
    """observed: {tile_id: predicted_severe_pixels} for paid prefix only."""
    vals=np.array(list(observed.values()),float);pos=[t for t,v in observed.items() if v>0]
    nbrs=[t for t in observed if manhattan(t,tile)==1]
    return [float(len(pos)>0),float((vals>0).mean()),math.log1p(vals.mean()),float(bool(nbrs)),
            float(np.mean([observed[t]>0 for t in nbrs])) if nbrs else 0.,
            float(min(manhattan(t,tile) for t in pos)) if pos else 7.]


def observed_from(prefix,private_chip):
    return {t:private_chip[t]['predicted_severe_pixels'] for t in prefix}


def build_rows(states,public,private,donors=None):
    """Return list of row dicts. donors[i] (optional) supplies another state's observed values
    paired by prefix order, used only for the permuted-feedback control."""
    pub={p['sample_id']:p for p in public};rows=[]
    for i,s in enumerate(states):
        sid=s['sample_id'];prefix=s['prefix'];k=len(prefix)
        if donors is None:observed=observed_from(prefix,private[sid])
        else:
            d=donors[i];donor_vals=[private[d['sample_id']][t]['predicted_severe_pixels'] for t in d['prefix']]
            observed=dict(zip(prefix,donor_vals))
        prior=prior_matrix(pub[sid],k)
        history=[{'tile_id':t,'predicted_severe_pixels':observed[t]} for t in prefix]
        neighbor_rule=utility_from_evidence(pub[sid]['post_utility'],history)
        for tile in range(16):
            if tile in prefix:continue
            cell=private[sid][tile]
            rows.append({'state':i,'sample_id':sid,'event':s['event'],'k':k,'tile':tile,
                         'prior':prior[tile].tolist(),'feedback':feedback_vector(tile,observed),
                         'post_utility':pub[sid]['post_utility'][tile],'neighbor_rule':float(neighbor_rule[tile]),
                         'y_tp':int(cell['tp']>0),'y_ref':int(cell['reference']>0)})
    return rows


def permuted_donors(states):
    """Cyclic shift by half the group among states with same scheme and k (other chips)."""
    groups={}
    for i,s in enumerate(states):groups.setdefault((s['scheme'],len(s['prefix'])),[]).append(i)
    donors=[None]*len(states)
    for idx in groups.values():
        idx=sorted(idx,key=lambda i:(states[i]['sample_id'],i));n=len(idx);shift=max(1,n//2)
        for j,i in enumerate(idx):
            d=idx[(j+shift)%n]
            # never borrow from the same chip
            step=0
            while states[d]['sample_id']==states[i]['sample_id'] and step<n:
                step+=1;d=idx[(j+shift+step)%n]
            donors[i]=states[d]
    return donors


def fit_logistic(X,y,l2=L2):
    X=np.asarray(X,float);y=np.asarray(y,float)
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-12]=1.
    Z=(X-mu)/sd;Z1=np.column_stack([np.ones(len(Z)),Z])
    def loss(w):
        s=Z1@w;ll=np.sum(np.logaddexp(0,s)-y*s)+.5*l2*np.sum(w[1:]**2)
        p=1/(1+np.exp(-s));g=Z1.T@(p-y);g[1:]+=l2*w[1:]
        return ll,g
    w=minimize(loss,np.zeros(Z1.shape[1]),jac=True,method='L-BFGS-B').x
    return {'mu':mu,'sd':sd,'w':w}


def predict(model,X):
    Z=(np.asarray(X,float)-model['mu'])/model['sd']
    return 1/(1+np.exp(-(model['w'][0]+Z@model['w'][1:])))


def auroc(y,s):
    y=np.asarray(y);s=np.asarray(s,float);n1=int(y.sum());n0=len(y)-n1
    if n1==0 or n0==0:return None
    ranks=rankdata(s)
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def average_precision(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    if y.sum()==0:return None
    order=np.lexsort((np.arange(len(s)),-s));yy=y[order];tp=np.cumsum(yy)
    return float(np.sum((tp/np.arange(1,len(yy)+1))[yy==1])/yy.sum())


def log_loss(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6);y=np.asarray(y)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def brier(y,p):
    return float(np.mean((np.asarray(p,float)-np.asarray(y))**2))


def next_pick_hits(rows,scores,label):
    """Per state with any positive unvisited tile: is the top-scored unvisited tile positive?"""
    by={}
    for r,s in zip(rows,scores):by.setdefault(r['state'],[]).append((-s,r['tile'],r[label],r['sample_id']))
    hits=[]
    for st,items in by.items():
        if not any(x[2] for x in items):continue
        hits.append((items[0][3],int(min(items)[2])))
    return hits


def metric_block(y,p):
    return {'n':len(y),'positives':int(np.sum(y)),'auroc':auroc(y,p),'average_precision':average_precision(y,p),
            'log_loss':log_loss(y,p),'brier':brier(y,p)}


def out_of_fold(rows,label,feature_fn,split):
    """split: 'event' (leave-one-event-out) or 'chip' (leave-one-chip-out)."""
    key='event' if split=='event' else 'sample_id'
    X=np.array([feature_fn(r) for r in rows]);y=np.array([r[label] for r in rows]);pred=np.full(len(rows),np.nan)
    for g in sorted({r[key] for r in rows}):
        test=np.array([r[key]==g for r in rows])
        if y[~test].sum()==0:pred[test]=1e-3;continue
        pred[test]=predict(fit_logistic(X[~test],y[~test]),X[test])
    assert np.isfinite(pred).all()
    return pred


def chip_bootstrap_delta(rows,label,base,new,reps=2000,seed=7):
    """Resample chips with replacement; predictions held fixed (no refit)."""
    rng=np.random.default_rng(seed);chips=sorted({r['sample_id'] for r in rows})
    idx={c:[] for c in chips}
    for i,r in enumerate(rows):idx[r['sample_id']].append(i)
    y=np.array([r[label] for r in rows]);base=np.asarray(base);new=np.asarray(new);out={'auroc':[],'average_precision':[],'log_loss':[]}
    for _ in range(reps):
        sel=np.concatenate([idx[c] for c in rng.choice(chips,size=len(chips))])
        yy=y[sel]
        if yy.sum()==0 or yy.sum()==len(yy):continue
        out['auroc'].append(auroc(yy,new[sel])-auroc(yy,base[sel]))
        out['average_precision'].append(average_precision(yy,new[sel])-average_precision(yy,base[sel]))
        out['log_loss'].append(log_loss(yy,new[sel])-log_loss(yy,base[sel]))
    return {k:{'ci95':[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))],'valid_reps':len(v)} if v else None
            for k,v in out.items()}
