"""Fixed disjoint256 observations, coarse-only public ranking and area budgets."""
from math import comb
import numpy as np
from PIL import Image

TILE_COST=4
BUDGETS=(8,16,32,64)
PRIMARY_BUDGET=16


def units():
    return [{'tile_id':i,'bounds':[i%4*256,i//4*256,(i%4+1)*256,(i//4+1)*256],
             'area_cost':4,'forward_calls':4} for i in range(16)]


def crop_pair(pre,post,tile_id):
    arrays=[np.asarray(im) for im in (pre,post)]
    if type(tile_id) is not int or not 0<=tile_id<16:raise ValueError('Unknown tile')
    if any(a.shape!=(1024,1024,3) or a.dtype!=np.uint8 for a in arrays):raise ValueError('Native uint8 RGB1024 pair required')
    x,y,xx,yy=units()[tile_id]['bounds']
    return tuple(a[y:yy,x:xx].copy() for a in arrays)


def coarse_pair(pre,post):
    if any(np.asarray(im).shape!=(1024,1024,3) or np.asarray(im).dtype!=np.uint8 for im in (pre,post)):raise ValueError('RGB1024 required')
    return tuple(np.asarray(Image.fromarray(np.asarray(im)).resize((64,64),Image.Resampling.BOX)).copy() for im in (pre,post))


def static_order(coarse_pre,coarse_post):
    if any(a.shape!=(64,64,3) or a.dtype!=np.uint8 for a in (coarse_pre,coarse_post)):raise ValueError('Public64RGB only; no native scene or prediction access')
    difference=np.abs(coarse_pre.astype(np.int16)-coarse_post.astype(np.int16)).astype(np.float64)/255
    scores=[float(difference[i//4*16:(i//4+1)*16,i%4*16:(i%4+1)*16].mean()) for i in range(16)]
    order=sorted(range(16),key=lambda i:(-scores[i],i))
    return {'order':order,'scores_row_major':scores}


def acquisitions(order,budget):
    if len(order)!=16 or sorted(order)!=list(range(16)) or any(type(i) is not int for i in order):raise ValueError('One complete unique tile permutation required')
    if type(budget) is not int or not 0<=budget<=64:raise ValueError('Invalid area budget')
    return [{'tile_id':i,'cost':4,'cumulative_cost':4*(k+1)} for k,i in enumerate(order[:budget//4])]


def counts(truth,pred):
    if truth.shape!=pred.shape or truth.dtype!=np.uint8 or pred.dtype!=np.uint8:raise ValueError('Matched uint8 labels required')
    if not set(np.unique(truth))<={0,1,2,3,4,255} or not set(np.unique(pred))<={0,1,2,3,4}:raise ValueError('Bad labels')
    valid=truth!=255;target=np.isin(truth,[3,4]);found=np.isin(pred,[3,4])
    return {'tp':int((valid & target & found).sum()),'fp':int((valid & ~target & found).sum()),
            'reference':int((valid & target).sum()),'valid_pixels':int(valid.sum())}


def outcome(tp,fp,reference):
    if min(tp,fp,reference)<0 or tp>reference:raise ValueError('Invalid coverage')
    return {'tp':tp,'fp':fp,'fn':reference-tp,'reference':reference,
        'precision_ratio_of_counts':tp/(tp+fp) if tp+fp else None,
        'recall':tp/reference if reference else None,
        'f1_ratio_of_counts':2*tp/(tp+fp+reference) if tp+fp+reference else None}


def policy_result(tile_counts,order,budget):
    if len(tile_counts)!=16:raise ValueError('All16 fixed action outcomes required')
    trace=acquisitions(order,budget);chosen=[r['tile_id'] for r in trace]
    return {**outcome(sum(tile_counts[i]['tp'] for i in chosen),sum(tile_counts[i]['fp'] for i in chosen),sum(r['reference'] for r in tile_counts)),
            'acquisitions':len(chosen),'spent':len(chosen)*4,'trace':trace}


def exact_random(tile_counts,budget):
    # Exact expected additive counts for uniform sampling without replacement.
    trace=acquisitions(list(range(16)),budget);k=len(trace);prob=k/16
    reference=sum(r['reference'] for r in tile_counts);d=sum(r['tp']>0 for r in tile_counts)
    hit=1-comb(16-d,k)/comb(16,k) if k<=16-d else 1.0
    return {**outcome(prob*sum(r['tp'] for r in tile_counts),prob*sum(r['fp'] for r in tile_counts),reference),
            'acquisitions':k,'spent':k*4,'scene_any_tp_probability':hit,
            'interpretation':'Exact expected TP/FP and recall;precision/F1 are ratios of expected counts,not expectations of nonlinear metrics.'}


def oracle_order(tile_counts):
    if len(tile_counts)!=16:raise ValueError('All16 action outcomes required')
    return sorted(range(16),key=lambda i:(-tile_counts[i]['tp'],i))
