"""New36-only native-area comparison; fixed exact-random and public ranks."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_footprint_core import NATIVE128,NATIVE256,BUDGETS,ORDER_SEEDS,units,ranked_acquisitions,evaluate_acquisitions,exact_random,aggregate
from evaluate_external_footprint import mean_scores
from extract_external_area_masks import decode_reference
from native_replication36_core import decide_replication
from run_external_context_diagnostic import checked_mask

MODES=(NATIVE128,NATIVE256)
COARSE="free_coarse512_only"


def read(path):return json.loads(Path(path).read_text())


def load_scene(root,sid,protocol):
    ref=next(r for r in read(root/"reference_manifest.json") if r["sample_id"]==sid)
    if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
    labels,_=decode_reference(Path(ref["path"]).read_bytes());target=labels>=3
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};rgb=[]
    for date in ["pre","post"]:
        r=images[sid,date]
        if sha(Path(r["path"]).read_bytes())!=r["sha256"]:raise ValueError("RGB changed")
        with Image.open(r["path"]) as im:rgb.append(np.asarray(im.convert("RGB")))
    domains={"all_published_pixels":np.ones_like(target),"both_dates_nonzero":np.any(rgb[0]>0,axis=2)&np.any(rgb[1]>0,axis=2)}
    rows=read(root/"scene_records"/(sid+".json"));masks={mode:{} for mode in MODES};free=None
    for r in rows:
        size=512 if r["input_mode"]==COARSE else 128 if r["input_mode"]==NATIVE128 else 256
        mask=checked_mask(Path(r["cache_file"]),r["cache_sha256"],(size,size))
        if r["input_mode"]==COARSE:free=mask
        else:masks[r["input_mode"]][r["unit_id"]]=mask
    if free is None:raise ValueError("Free baseline missing")
    return ref["event"],target,free,domains,masks


def evaluate(root,protocol):
    for name,key in [("reference_manifest.json","reference_manifest_sha256"),("public_features.json","public_features_sha256")]:
        if sha((root/name).read_bytes())!=protocol[key]:raise ValueError("Evaluation source changed")
    controls=read(root/"positive_control_results.json")
    if len(controls)!=2 or not all(r["exact_mask_equal"] for r in controls):raise ValueError("Controls not passed")
    features=read(root/"public_features.json");scene_rows=[];opportunities=[];grouped=defaultdict(list);traces=0;actions=0
    with gzip.open(root/"ranked_acquisitions.jsonl.gz","wt") as stream:
        for sid in protocol["selected_ids"]:
            event,ref,free,domains,masks=load_scene(root,sid,protocol)
            for mode in MODES:
                all_trace=[{"unit_id":u["unit_id"],"cost":u["cost"],"cumulative_cost":sum(v["cost"] for v in units(mode)[:i+1])} for i,u in enumerate(units(mode))]
                for domain,mask in domains.items():opportunities.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,**evaluate_acquisitions(all_trace,masks[mode],mode,16,ref,free,mask)})
                for budget in BUDGETS:
                    for domain,mask in domains.items():scene_rows.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"random_exact",**exact_random(masks[mode],mode,budget,ref,free,mask)})
                    for seed in ORDER_SEEDS:
                        trace=ranked_acquisitions(features[sid],mode,budget,seed)
                        for a in trace:a["declared_target"]=int(masks[mode][a["unit_id"]].sum())>=4
                        scores={domain:evaluate_acquisitions(trace,masks[mode],mode,budget,ref,free,mask) for domain,mask in domains.items()}
                        stream.write(json.dumps({"sample_id":sid,"event":event,"input_mode":mode,"budget":budget,"order_seed":seed,"trace":trace,"scores":scores},sort_keys=True)+"\n")
                        traces+=1;actions+=len(trace)
                        for domain,result in scores.items():grouped[sid,event,domain,mode,budget].append(result)
    for (sid,event,domain,mode,budget),rows in grouped.items():scene_rows.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"raw_change_rank",**mean_scores(rows)})
    atomic_json(root/"full_opportunities.json",opportunities);atomic_json(root/"scene_scores.json",scene_rows)
    summary=aggregate(scene_rows);atomic_json(root/"summary.json",summary);atomic_json(root/"next_step_decision.json",decide_replication(summary,opportunities))
    atomic_json(root/"evaluation_report.json",{"ranked_records":traces,"ranked_domain_scores":traces*2,"paid_acquisitions":actions,
        "exact_random_scene_domain_scores":432,"scene_domain_policy_means":len(scene_rows),"full_coverage_scores":len(opportunities),
        "trajectory_sha256":sha((root/"ranked_acquisitions.jsonl.gz").read_bytes()),"new36_only":True,"old18_pooled":False,"controller_fitting":False})
    print(json.dumps({"ranked_records":traces,"paid_acquisitions":actions,"decision":read(root/"next_step_decision.json")["decision"]}))
