"""Pure geometry/metrics for the offline full-context diagnostic."""
import hashlib
import numpy as np

from external_area_core import CELLS,box

CONTROL_EVENTS=("hurricane-harvey","santa-rosa-wildfire")


def select_controls(observations):
    result=[]
    for event in CONTROL_EVENTS:
        eligible=[r for r in observations if r["event"]==event and r["mode"]=="native" and r["declared_target"] is True]
        if not eligible:raise ValueError("Missing already-used positive development control")
        result.append(min(eligible,key=lambda r:hashlib.sha256(("context-control-v1|20261026|"+r["observation_id"]).encode()).hexdigest()))
    return result


def stitch_native(masks):
    if set(masks)!=set(CELLS):raise ValueError("Exactly16 native cells required")
    canvas=np.zeros((512,512),dtype=np.bool_)
    for cell,mask in masks.items():
        if mask.dtype!=np.bool_ or mask.shape!=(128,128):raise ValueError("Canonical128px boolean mask required")
        x0,y0,x1,y1=box(cell);canvas[y0:y1,x0:x1]=mask
    return canvas


def overlap_metrics(predicted,reference,domain):
    if any(a.shape!=(512,512) or a.dtype!=np.bool_ for a in [predicted,reference,domain]):raise ValueError("Matched512px boolean domains required")
    truth=reference&domain;prediction=predicted&domain
    target=int(truth.sum());found=int((truth&prediction).sum());fp=int((prediction&~truth).sum())
    return {"reference_pixels":target,"predicted_pixels":int(prediction.sum()),"covered_reference_pixels":found,
        "reference_false_positive_pixels":fp,"area_recall":found/target if target else None,
        "reference_precision":found/(found+fp) if found+fp else None,
        "reference_iou":found/(target+fp) if target+fp else None}


def paired_changes(mosaic,whole,reference,domain):
    for a in [mosaic,whole,reference,domain]:
        if a.shape!=(512,512) or a.dtype!=np.bool_:raise ValueError("Matched boolean masks required")
    ref=reference&domain
    return {"gained_reference_pixels":int((ref&whole&~mosaic).sum()),
        "lost_reference_pixels":int((ref&mosaic&~whole).sum()),
        "preserved_reference_pixels":int((ref&mosaic&whole).sum()),
        "prediction_disagreement_pixels":int(((mosaic^whole)&domain).sum())}


def summarize(rows):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[r["domain"],r["input_mode"],r["event"]].append(r)
    events=[]
    for (domain,mode,event),part in sorted(groups.items()):
        target=sum(r["reference_pixels"] for r in part);found=sum(r["covered_reference_pixels"] for r in part);fp=sum(r["reference_false_positive_pixels"] for r in part)
        events.append({"domain":domain,"input_mode":mode,"event":event,"scenes":len(part),
            "reference_positive_scenes":sum(r["reference_pixels"]>0 for r in part),"detectable_reference_scenes":sum(r["covered_reference_pixels"]>0 for r in part),
            "reference_pixels":target,"covered_reference_pixels":found,"reference_false_positive_pixels":fp,
            "pooled_area_recall":found/target if target else None,"pooled_reference_precision":found/(found+fp) if found+fp else None})
    groups=defaultdict(list)
    for r in events:groups[r["domain"],r["input_mode"]].append(r)
    totals=[]
    for (domain,mode),part in sorted(groups.items()):
        target=sum(r["reference_pixels"] for r in part);found=sum(r["covered_reference_pixels"] for r in part);fp=sum(r["reference_false_positive_pixels"] for r in part)
        positive=[r for r in part if r["reference_pixels"]]
        totals.append({"domain":domain,"input_mode":mode,"events":len(part),"scenes":sum(r["scenes"] for r in part),
            "reference_positive_events":len(positive),"detectable_reference_events":sum(r["covered_reference_pixels"]>0 for r in part),
            "detectable_reference_scenes":sum(r["detectable_reference_scenes"] for r in part),"reference_pixels":target,"covered_reference_pixels":found,"reference_false_positive_pixels":fp,
            "pooled_area_recall":found/target if target else None,"pooled_reference_precision":found/(found+fp) if found+fp else None,
            "event_macro_covered_pixels_per_scene":sum(r["covered_reference_pixels"]/r["scenes"] for r in part)/len(part),
            "event_macro_recall_target_events":sum(r["pooled_area_recall"] for r in positive)/len(positive) if positive else None})
    return {"event_rows":events,"totals":totals}
