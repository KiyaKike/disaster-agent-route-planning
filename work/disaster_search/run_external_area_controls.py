"""Descriptive area controls, with evaluator and paid-boolean policy isolated."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import (CELLS,POLICIES,BUDGETS,QUALITY_SEEDS,ORDER_SEEDS,BinaryEnvironment,
    run_policy,evaluate_trace,exact_random_survey,public_features,box)
from extract_external_area_masks import decode_reference
from plan_search_v2 import freeze

DOMAINS=("all_published_pixels","both_dates_nonzero")
AVERAGE_FIELDS=("covered_reference_pixels","reference_false_positive_pixels","surveys","upgrades","spent")


def average(records):
    result={k:sum(r[k] for r in records)/len(records) for k in AVERAGE_FIELDS}
    targets={r["reference_pixels"] for r in records}
    if len(targets)!=1:raise ValueError("Reference domain changed across trajectories")
    result["reference_pixels"]=next(iter(targets));result["area_recall"]=result["covered_reference_pixels"]/result["reference_pixels"] if result["reference_pixels"] else None
    result["trajectories"]=len(records)
    return result


def aggregate(scene_rows):
    groups=defaultdict(list)
    for r in scene_rows:groups[r["domain"],r["budget"],r["policy"],r["event"]].append(r)
    events=[]
    for (domain,budget,policy,event),rows in sorted(groups.items()):
        ref=sum(r["reference_pixels"] for r in rows);found=sum(r["covered_reference_pixels"] for r in rows)
        positive=[r for r in rows if r["reference_pixels"]>0]
        events.append({"domain":domain,"budget":budget,"policy":policy,"event":event,"scenes":len(rows),
            "reference_positive_scenes":len(positive),"reference_pixels":ref,
            **{k:sum(r[k] for r in rows)/len(rows) for k in AVERAGE_FIELDS},
            "pooled_area_recall":found/ref if ref else None,
            "mean_recall_target_scenes":sum(r["area_recall"] for r in positive)/len(positive) if positive else None})
    groups=defaultdict(list)
    for r in events:groups[r["domain"],r["budget"],r["policy"]].append(r)
    macros=[]
    for (domain,budget,policy),rows in sorted(groups.items()):
        target_events=[r for r in rows if r["reference_pixels"]>0]
        macros.append({"domain":domain,"budget":budget,"policy":policy,"events":len(rows),"target_events":len(target_events),
            **{k:sum(r[k] for r in rows)/len(rows) for k in AVERAGE_FIELDS},
            "mean_pooled_recall_target_events":sum(r["pooled_area_recall"] for r in target_events)/len(target_events) if target_events else None})
    return {"event_rows":events,"event_macro_rows":macros}


def load_scene(sid,rgb,refs,observations,features):
    ref=refs[sid]
    if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
    label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3
    pair=[]
    for mode in ["pre","post"]:
        r=rgb[sid,mode]
        if sha(Path(r["path"]).read_bytes())!=r["sha256"]:raise ValueError("RGB changed")
        with Image.open(r["path"]) as image:pair.append(image.copy())
    if public_features(*pair)!=features[sid]:raise ValueError("Public coarse features replay differs")
    valid=np.any(np.asarray(pair[0])>0,axis=2)&np.any(np.asarray(pair[1])>0,axis=2)
    masks,booleans={},{}
    for r in observations:
        if r["sample_id"]!=sid:continue
        path=Path(r["cache_file"])
        if sha(path.read_bytes())!=r["cache_sha256"]:raise ValueError("Prediction mask changed")
        with np.load(path,allow_pickle=False) as cache:mask=cache["severe_mask"]
        if mask.dtype!=np.bool_ or mask.shape!=(128,128) or int(mask.sum())!=r["predicted_severe_pixels"] or bool(mask.sum()>=4)!=r["declared_target"]:raise ValueError("Canonical prediction/declaration differs")
        identity=r["cell_id"],r["mode"]
        if identity in masks:raise ValueError("Duplicate observation")
        masks[identity]=mask;booleans[identity]=r["declared_target"]
    if len(masks)!=32:raise ValueError("Incomplete scene sensor cache")
    return target,{"all_published_pixels":np.ones_like(target),"both_dates_nonzero":valid},masks,booleans


def opportunity(target,domain,masks):
    union=np.zeros((512,512),dtype=np.bool_);result={"reference_pixels":int((target&domain).sum())}
    for mode in ["native","blurred"]:
        full=np.zeros_like(union)
        for cell in CELLS:
            x0,y0,x1,y1=box(cell);full[y0:y1,x0:x1]=masks[cell,mode]
        union|=full
        found=int((full&target&domain).sum());fp=int((full&~target&domain).sum())
        result[mode]={"covered_reference_pixels":found,"reference_false_positive_pixels":fp,"area_recall":found/result["reference_pixels"] if result["reference_pixels"] else None}
    result["either_mode"]={"covered_reference_pixels":int((union&target&domain).sum()),"reference_false_positive_pixels":int((union&~target&domain).sum())}
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb",type=Path,required=True);p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    plan=json.loads((root/"control_protocol.json").read_text())
    if plan["core_sha256"]!=sha(Path(__file__).with_name("external_area_core.py").read_bytes()):raise ValueError("Prospective core changed")
    if plan["budgets"]!=list(BUDGETS) or plan["policies"]!=list(POLICIES):raise ValueError("Prospective controls differ")
    paths={"rgb_manifest":args.rgb/"rgb_manifest.json","references":root/"reference_manifest.json","observations":root/"observations.json","features":root/"public_features.json"}
    freeze(root/"controls_execution_protocol.json",{"control_plan_sha256":sha((root/"control_protocol.json").read_bytes()),
        "source_sha256":{name:sha(path.read_bytes()) for name,path in paths.items()},"runner_sha256":sha(Path(__file__).read_bytes()),
        "pixel_domains":list(DOMAINS),"trajectories":8100,"scores":16200,"new_fitting":False,"budget_primary":8})
    rgb={(r["sample_id"],r["mode"]):r for r in json.loads(paths["rgb_manifest"].read_text())}
    refs={r["sample_id"]:r for r in json.loads(paths["references"].read_text())}
    observations=json.loads(paths["observations"].read_text());features=json.loads(paths["features"].read_text())
    scene_rows=[];opportunities=[];traces=0;actions=0
    with gzip.open(root/"area_trajectories.jsonl.gz","wt") as stream:
        for sid in sorted(refs):
            event=refs[sid]["event"];target,domains,masks,booleans=load_scene(sid,rgb,refs,observations,features)
            for name,domain in domains.items():opportunities.append({"sample_id":sid,"event":event,"domain":name,**opportunity(target,domain,masks)})
            grouped=defaultdict(list)
            for budget in BUDGETS:
                for seed in QUALITY_SEEDS:
                    for order_seed in ORDER_SEEDS:
                        for policy in POLICIES:
                            environment=BinaryEnvironment(booleans,sid,seed,2)
                            trace=run_policy(features[sid],environment,policy,budget,order_seed,2)
                            row={"sample_id":sid,"event":event,"budget":budget,"quality_seed":seed,"order_seed":order_seed,"policy":policy,"trace":trace,"scores":{}}
                            for name,domain in domains.items():
                                metrics=evaluate_trace(trace,target,masks,domain);row["scores"][name]=metrics;grouped[name,budget,policy].append(metrics)
                            stream.write(json.dumps(row,separators=(",",":"))+"\n");traces+=1;actions+=len(trace)
                    for name,domain in domains.items():
                        exact=exact_random_survey(target,masks,sid,seed,budget,domain)
                        grouped[name,budget,"random_exact_expectation"].append({"reference_pixels":int((target&domain).sum()),
                            "covered_reference_pixels":exact["expected_covered_reference_pixels"],"reference_false_positive_pixels":exact["expected_reference_false_positive_pixels"],"surveys":budget,"upgrades":0,"spent":budget})
            for (domain,budget,policy),records in sorted(grouped.items()):
                scene_rows.append({"sample_id":sid,"event":event,"domain":domain,"budget":budget,"policy":policy,**average(records)})
            print(json.dumps({"sample_id":sid,"complete_trajectories":traces}),flush=True)
    if traces!=8100:raise ValueError("Unplanned trajectory count")
    atomic_json(root/"area_scene_means.json",scene_rows);atomic_json(root/"sensor_area_opportunity.json",opportunities)
    summary=aggregate(scene_rows);atomic_json(root/"area_summary.json",summary)
    atomic_json(root/"controls_report.json",{"trajectories":traces,"domain_scores":traces*2,"actions":actions,
        "trajectory_sha256":sha((root/"area_trajectories.jsonl.gz").read_bytes()),"new_inference_in_controls":0,
        "primary":[r for r in summary["event_macro_rows"] if r["budget"]==8 and r["domain"]=="all_published_pixels"],
        "independent_efficacy_or_building_count_claim":False})


if __name__=="__main__":main()
