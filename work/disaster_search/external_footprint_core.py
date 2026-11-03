"""Native-area cost, information-bounded block inputs and disjoint-unit evaluation."""
import random

import numpy as np
from PIL import Image

from external_area_core import CELLS,box
from external_information_core import require_pair,coarse_canvas

NATIVE128="cached_native128"
NATIVE256="native256_contiguous"
CONTEXT256="free_coarse512_plus_native256"
FULL512="cached_native512"
MODES=(NATIVE128,NATIVE256,CONTEXT256,FULL512)
NEW_MODES=(NATIVE256,CONTEXT256)
BLOCKS=tuple(f"b{r}{c}" for r in range(2) for c in range(2))
BUDGETS=(4,8,16)
ORDER_SEEDS=tuple(range(20261030,20261040))


def units(mode):
    if mode==NATIVE128:return [{"unit_id":c,"bounds":box(c),"cost":1,"cells":[c]} for c in CELLS]
    if mode in NEW_MODES:
        return [{"unit_id":f"b{r}{c}","bounds":(c*256,r*256,(c+1)*256,(r+1)*256),"cost":4,
            "cells":[f"r{rr:02d}c{cc:02d}" for rr in range(2*r,2*r+2) for cc in range(2*c,2*c+2)]} for r in range(2) for c in range(2)]
    if mode==FULL512:return [{"unit_id":"scene","bounds":(0,0,512,512),"cost":16,"cells":list(CELLS)}]
    raise ValueError("Unplanned footprint mode")


def block_bounds(block):
    if block not in BLOCKS:raise ValueError("Not one of four fixed blocks")
    return next(u["bounds"] for u in units(NATIVE256) if u["unit_id"]==block)


def compose_block(mode,block_pair,block,coarse_pair=None):
    bounds=block_bounds(block);require_pair(block_pair,(256,256))
    if mode==NATIVE256:
        if coarse_pair is not None:raise ValueError("Native256 must not receive other context")
        return tuple(im.copy() for im in block_pair)
    if mode!=CONTEXT256:raise ValueError("Unplanned block input")
    canvases=coarse_canvas(coarse_pair)
    for canvas,image in zip(canvases,block_pair):canvas.paste(image,bounds[:2])
    return canvases


def canonical_block(mask,mode,block):
    bounds=block_bounds(block);size=256 if mode==NATIVE256 else 512 if mode==CONTEXT256 else None
    if mask.dtype!=np.bool_ or mask.shape!=(size,size):raise ValueError("Wrong raw block prediction")
    if mode==NATIVE256:return mask.copy()
    x,y,xx,yy=bounds;return mask[y:yy,x:xx].copy()


def ranked_acquisitions(features,mode,budget,order_seed):
    """Accepts public coarse change features only; no masks, reference or feedback."""
    if set(features)!=set(CELLS) or type(budget) is not int or budget<0:raise ValueError("Bad public features/budget")
    if any(not np.isfinite(r["change"]) or not 0<=r["change"]<=1 for r in features.values()):raise ValueError("Invalid public change cue")
    order=list(CELLS);random.Random(order_seed).shuffle(order);rank={c:i for i,c in enumerate(order)}
    rows=[]
    for unit in units(mode):
        score=sum(features[c]["change"] for c in unit["cells"])/len(unit["cells"])
        rows.append({**unit,"public_score":score,"tie_rank":min(rank[c] for c in unit["cells"])})
    rows.sort(key=lambda r:(-r["public_score"],r["tie_rank"]))
    acquired=[];spent=0
    for row in rows:
        if spent+row["cost"]>budget:continue
        spent+=row["cost"];acquired.append({"unit_id":row["unit_id"],"cost":row["cost"],"cumulative_cost":spent,"public_score":row["public_score"],"tie_rank":row["tie_rank"]})
    return acquired


def validate_arrays(reference,free,domain):
    if any(a.shape!=(512,512) or a.dtype!=np.bool_ for a in [reference,free,domain]):raise ValueError("Boolean512 reference/free/domain required")


def score_union(reference,free,paid,domain):
    validate_arrays(reference,free,domain)
    if paid.shape!=(512,512) or paid.dtype!=np.bool_:raise ValueError("Boolean512 paid union required")
    baseline=free&domain;extra=paid&~free&domain;total=baseline|extra;target=reference&domain
    ft=int((baseline&target).sum());ff=int((baseline&~target).sum());it=int((extra&target).sum());iff=int((extra&~target).sum());n=int(target.sum())
    return {"reference_pixels":n,"free_covered_reference_pixels":ft,"free_false_positive_pixels":ff,
        "incremental_covered_reference_pixels":it,"incremental_false_positive_pixels":iff,
        "covered_reference_pixels":ft+it,"reference_false_positive_pixels":ff+iff,"predicted_pixels":int(total.sum()),
        "area_recall":(ft+it)/n if n else None,"reference_precision":(ft+it)/int(total.sum()) if total.any() else None}


def evaluate_acquisitions(trace,masks,mode,budget,reference,free,domain):
    geometry={u["unit_id"]:u for u in units(mode)};paid=np.zeros((512,512),dtype=np.bool_);seen=set();spent=0
    for action in trace:
        uid=action["unit_id"]
        if uid not in geometry or uid in seen:raise ValueError("Unknown/duplicate paid unit")
        unit=geometry[uid];spent+=unit["cost"]
        if action["cost"]!=unit["cost"] or action["cumulative_cost"]!=spent or spent>budget:raise ValueError("Invalid native-area payment")
        mask=masks[uid];x,y,xx,yy=unit["bounds"]
        if mask.dtype!=np.bool_ or mask.shape!=(yy-y,xx-x):raise ValueError("Prediction outside purchased footprint")
        paid[y:yy,x:xx]|=mask;seen.add(uid)
    return {**score_union(reference,free,paid,domain),"spent":spent,"acquisitions":len(trace),"affordable":budget>=next(iter(geometry.values()))["cost"]}


def exact_random(masks,mode,budget,reference,free,domain):
    geometry=units(mode);cost=geometry[0]["cost"]
    if set(masks)!={u["unit_id"] for u in geometry} or type(budget) is not int or budget<0:raise ValueError("Wrong fixed random population/budget")
    all_trace=[{"unit_id":u["unit_id"],"cost":cost,"cumulative_cost":(i+1)*cost} for i,u in enumerate(geometry)]
    total=evaluate_acquisitions(all_trace,masks,mode,16,reference,free,domain);k=min(len(geometry),budget//cost);prob=k/len(geometry)
    it=prob*total["incremental_covered_reference_pixels"];iff=prob*total["incremental_false_positive_pixels"]
    ft=total["free_covered_reference_pixels"];ff=total["free_false_positive_pixels"];pred=ft+ff+it+iff;n=total["reference_pixels"]
    return {"reference_pixels":n,"free_covered_reference_pixels":ft,"free_false_positive_pixels":ff,
        "incremental_covered_reference_pixels":it,"incremental_false_positive_pixels":iff,
        "covered_reference_pixels":ft+it,"reference_false_positive_pixels":ff+iff,"predicted_pixels":pred,
        "area_recall":(ft+it)/n if n else None,"reference_precision":(ft+it)/pred if pred else None,
        "spent":k*cost,"acquisitions":k,"affordable":budget>=cost}


def aggregate(scene_rows):
    from collections import defaultdict
    groups=defaultdict(list)
    for row in scene_rows:groups[row["domain"],row["input_mode"],row["budget"],row["policy"],row["event"]].append(row)
    additive=["reference_pixels","free_covered_reference_pixels","free_false_positive_pixels","incremental_covered_reference_pixels","incremental_false_positive_pixels","covered_reference_pixels","reference_false_positive_pixels","predicted_pixels"]
    events=[]
    for (domain,mode,budget,policy,event),rows in sorted(groups.items()):
        values={key:sum(r[key] for r in rows) for key in additive};pred=values["predicted_pixels"];n=values["reference_pixels"]
        events.append({"domain":domain,"input_mode":mode,"budget":budget,"policy":policy,"event":event,"scenes":len(rows),"affordable":all(r["affordable"] for r in rows),**values,
            "area_recall":values["covered_reference_pixels"]/n if n else None,"reference_precision":values["covered_reference_pixels"]/pred if pred else None,
            "mean_spent":sum(r["spent"] for r in rows)/len(rows),"detectable_reference_scenes":sum(r["incremental_covered_reference_pixels"]>0 for r in rows)})
    groups=defaultdict(list)
    for row in events:groups[row["domain"],row["input_mode"],row["budget"],row["policy"]].append(row)
    totals=[]
    for (domain,mode,budget,policy),rows in sorted(groups.items()):
        values={key:sum(r[key] for r in rows) for key in additive};pred=values["predicted_pixels"];n=values["reference_pixels"]
        totals.append({"domain":domain,"input_mode":mode,"budget":budget,"policy":policy,"events":len(rows),"scenes":sum(r["scenes"] for r in rows),"affordable":all(r["affordable"] for r in rows),**values,
            "area_recall":values["covered_reference_pixels"]/n if n else None,"reference_precision":values["covered_reference_pixels"]/pred if pred else None,
            "event_macro_incremental_covered_pixels_per_scene":sum(r["incremental_covered_reference_pixels"]/r["scenes"] for r in rows)/len(rows),
            "event_macro_false_pixels_per_scene":sum(r["reference_false_positive_pixels"]/r["scenes"] for r in rows)/len(rows)})
    return {"event_rows":events,"totals":totals}
