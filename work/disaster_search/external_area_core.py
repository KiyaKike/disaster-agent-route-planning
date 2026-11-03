"""4x4 geometry, public-only policies and separate pixel-reference evaluator."""
import math
import random

import numpy as np
from PIL import Image

from cross_cell_count_smoke import quality

CELLS=tuple(f"r{r:02d}c{c:02d}" for r in range(4) for c in range(4))
POLICIES=("random_survey","prior_survey","prior_gate")
QUALITY_SEEDS=tuple(range(20261025,20261030))
ORDER_SEEDS=tuple(range(20261030,20261040))
BUDGETS=(4,8,16)


def box(cell):
    if cell not in CELLS:raise ValueError("Not a4x4 cell")
    r,c=int(cell[1:3]),int(cell[4:6])
    return (c*128,r*128,(c+1)*128,(r+1)*128)


def crop_cell(pre,post,cell):
    if pre.size!=(512,512) or post.size!=(512,512):raise ValueError("512px RGB required")
    bounds=box(cell)
    return pre.crop(bounds),post.crop(bounds)


def public_features(pre,post):
    if pre.size!=(512,512) or post.size!=(512,512):raise ValueError("Wrong public image size")
    a,b=[np.asarray(im.resize((64,64),Image.Resampling.BOX),dtype=float) for im in [pre,post]]
    delta=np.abs(a-b).mean(axis=2)/255
    usable=np.any(a>0,axis=2)&np.any(b>0,axis=2)
    result={}
    for cell in CELLS:
        r,c=int(cell[1:3]),int(cell[4:6]);window=(slice(r*16,(r+1)*16),slice(c*16,(c+1)*16))
        result[cell]={"change":float(np.percentile(delta[window],75)),"coarse_nonzero_fraction":float(usable[window].mean())}
    return result


class BinaryEnvironment:
    """Only paid booleans returned; masks/reference/target counts are not stored."""
    def __init__(self,booleans,sid,quality_seed,upgrade_cost=2):
        if set(booleans)!={(c,m) for c in CELLS for m in ["native","blurred"]} or any(type(v) is not bool for v in booleans.values()):
            raise ValueError("Exactly32 sensor booleans required")
        self._observations,self._sid,self._seed,self._cost=booleans,sid,quality_seed,upgrade_cost
        self._latest={}

    def observe(self,cell,kind):
        if cell not in CELLS:raise ValueError("Invalid cell")
        old=self._latest.get(cell)
        if old is None and kind=="survey":mode=quality("mixed_survey",self._seed,self._sid,cell);cost=1
        elif old=="blurred" and kind=="upgrade":mode="native";cost=self._cost
        else:raise ValueError("Ineligible paid action")
        self._latest[cell]=mode
        return {"cell_id":cell,"kind":kind,"mode":mode,"declared_target":self._observations[cell,mode],"cost":cost}


def run_policy(features,environment,policy,budget,order_seed,upgrade_cost=2):
    if policy not in POLICIES or set(features)!=set(CELLS) or type(budget) is not int or budget<0:raise ValueError("Unplanned policy or inputs")
    if any(not math.isfinite(f["change"]) or not 0<=f["change"]<=1 for f in features.values()):raise ValueError("Invalid public change")
    order=list(CELLS);random.Random(order_seed).shuffle(order);ranks={c:i for i,c in enumerate(order)}
    # Zero unseen, one blurred-negative, two blurred-positive, three native terminal.
    state={c:0 for c in CELLS};spent=0;trace=[]
    while spent<budget:
        candidates=[]
        for cell,status in state.items():
            if status==0:
                cost,kind=1,"survey";score=1. if policy=="random_survey" else features[cell]["change"]
            elif policy=="prior_gate" and status==1:
                cost,kind=upgrade_cost,"upgrade";score=features[cell]["change"]/cost
            else:continue
            if cost<=budget-spent:candidates.append((score,-ranks[cell],cell,kind,cost))
        if not candidates:break
        score,_,cell,kind,cost=max(candidates)
        result=environment.observe(cell,kind)
        if result["cost"]!=cost:raise ValueError("Paid cost differs")
        spent+=cost;trace.append({**result,"cumulative_cost":spent,"public_utility":score})
        state[cell]=3 if result["mode"]=="native" else 2 if result["declared_target"] else 1
    return trace


def evaluate_trace(trace,reference,masks,domain=None):
    """Pixel union includes even1..3 severe pixels below the boolean declaration."""
    if reference.shape!=(512,512) or reference.dtype!=np.bool_:raise ValueError("Boolean severe reference required")
    if domain is None:domain=np.ones_like(reference)
    if domain.shape!=reference.shape or domain.dtype!=np.bool_:raise ValueError("Boolean reference domain required")
    union=np.zeros_like(reference);seen=set();curve=[]
    for action in trace:
        identity=action["cell_id"],action["mode"]
        if identity in seen:raise ValueError("Duplicate acquired observation")
        seen.add(identity);mask=masks[identity]
        if mask.dtype!=np.bool_ or mask.shape!=(128,128):raise ValueError("Canonical local mask required")
        x0,y0,x1,y1=box(action["cell_id"]);union[y0:y1,x0:x1]|=mask
        curve.append({"cost":action["cumulative_cost"],"covered_reference_pixels":int((union&reference&domain).sum()),"reference_false_positive_pixels":int((union&~reference&domain).sum())})
    target=int((reference&domain).sum());found=int((union&reference&domain).sum());fp=int((union&~reference&domain).sum())
    return {"reference_pixels":target,"covered_reference_pixels":found,"reference_false_positive_pixels":fp,
        "area_recall":found/target if target else None,"reference_precision":found/(found+fp) if found+fp else None,
        "surveys":sum(a["kind"]=="survey" for a in trace),"upgrades":sum(a["kind"]=="upgrade" for a in trace),
        "spent":trace[-1]["cumulative_cost"] if trace else 0,"curve":curve}


def exact_random_survey(reference,masks,sid,quality_seed,budget,domain=None):
    """Evaluator-only expected area for sampling disjoint cells without replacement."""
    if not 0<=budget<=len(CELLS):raise ValueError("Budget exceeds disjoint survey population")
    trace=[]
    for i,cell in enumerate(CELLS,1):
        trace.append({"cell_id":cell,"mode":quality("mixed_survey",quality_seed,sid,cell),"kind":"survey","cumulative_cost":i})
    total=evaluate_trace(trace,reference,masks,domain)
    ratio=budget/len(CELLS)
    return {"expected_covered_reference_pixels":ratio*total["covered_reference_pixels"],
        "expected_reference_false_positive_pixels":ratio*total["reference_false_positive_pixels"],
        "expected_area_recall":ratio*total["covered_reference_pixels"]/total["reference_pixels"] if total["reference_pixels"] else None}
