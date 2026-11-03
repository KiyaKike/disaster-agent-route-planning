"""Native-area matched observations: exact random and fixed public-change rank."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_footprint_core import (MODES,NATIVE128,NATIVE256,CONTEXT256,FULL512,BUDGETS,ORDER_SEEDS,units,
    ranked_acquisitions,evaluate_acquisitions,exact_random,aggregate)
from extract_external_area_masks import decode_reference
from run_external_context_diagnostic import checked_mask


def read(path):return json.loads(Path(path).read_text())


def load_scene(sid,root,area,protocol):
    references={r["sample_id"]:r for r in read(area/"reference_manifest.json")};ref=references[sid]
    if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
    label,_=decode_reference(Path(ref["path"]).read_bytes());reference=label>=3
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};pair=[]
    for date in ["pre","post"]:
        row=images[sid,date]
        if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("RGB changed")
        with Image.open(row["path"]) as im:pair.append(np.asarray(im.convert("RGB")))
    domains={"all_published_pixels":np.ones_like(reference),"both_dates_nonzero":np.any(pair[0]>0,axis=2)&np.any(pair[1]>0,axis=2)}
    masks={mode:{} for mode in MODES}
    for row in protocol["previous_native_observations"]:
        if row["sample_id"]==sid:masks[NATIVE128][row["cell_id"]]=checked_mask(Path(row["cache_file"]),row["cache_sha256"],(128,128))
    for row in read(root/"scene_records"/(sid+".json")):
        masks[row["input_mode"]][row["block_id"]]=checked_mask(Path(row["cache_file"]),row["cache_sha256"],(256,256))
    whole=next(r for r in protocol["previous_whole_observations"] if r["sample_id"]==sid)
    masks[FULL512]["scene"]=checked_mask(Path(whole["cache_file"]),whole["cache_sha256"],(512,512))
    coarse=next(r for r in protocol["previous_coarse_observations"] if r["sample_id"]==sid)
    free=checked_mask(Path(coarse["cache_file"]),coarse["cache_sha256"],(512,512))
    return ref["event"],reference,free,domains,masks


def mean_scores(rows):
    keys=[k for k in rows[0] if k not in ["area_recall","reference_precision","affordable"]]
    result={k:sum(r[k] for r in rows)/len(rows) for k in keys}
    n=result["reference_pixels"];pred=result["predicted_pixels"]
    result.update(affordable=all(r["affordable"] for r in rows),area_recall=result["covered_reference_pixels"]/n if n else None,
        reference_precision=result["covered_reference_pixels"]/pred if pred else None)
    return result


def decide(summary):
    screens=[]
    for domain in ["all_published_pixels","both_dates_nonzero"]:
        rows=[r for r in summary["event_rows"] if r["domain"]==domain and r["policy"]=="random_exact" and r["budget"]==8]
        totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]==domain and r["policy"]=="random_exact" and r["budget"]==8}
        positive=sorted({r["event"] for r in rows if r["reference_pixels"]>0});main=[r for r in rows if r["input_mode"]==NATIVE256]
        a=totals[NATIVE256];b=totals[NATIVE128]
        checks={"both_reference_positive_events_have_incremental_opportunity":len(positive)==2 and all(next(r for r in main if r["event"]==e)["incremental_covered_reference_pixels"]>0 for e in positive),
            "budget8_exact_random_macro_incremental_coverage_exceeds128":a["event_macro_incremental_covered_pixels_per_scene"]>b["event_macro_incremental_covered_pixels_per_scene"],
            "budget8_pooled_reference_precision_not_below128":a["reference_precision"] is not None and b["reference_precision"] is not None and a["reference_precision"]>=b["reference_precision"]}
        screens.append({"domain":domain,"checks":checks,"passed":all(checks.values()),"reference_positive_events":positive})
    passed=screens[0]["passed"]
    return {"primary_condition":NATIVE256,"primary_budget":8,"primary_policy":"random_exact","primary_domain":"all_published_pixels",
        "domain_screens":screens,"primary_screen_passed":passed,
        "decision":"PRIMARY_NATIVE256_MEASUREMENT_GATE_PASSED" if passed else "STOP_MEASUREMENT_INTERFACE_VARIANTS_ON_THESE18",
        "secondary_cannot_replace_failed_primary":True,"new_method_selected":False,"new_fitting_enabled":False,"independent_efficacy_claim":False,
        "next_scope":"Separately register wider data/reference or fixed-detector assessment before planner development." if not passed else "Separately register mixed-quality measurement checks before search-method development."}


def evaluate(root,area,protocol):
    if sha((area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest changed")
    controls=read(root/"positive_control_results.json")
    if len(controls)!=2 or not all(r["exact_mask_equal"] for r in controls):raise ValueError("Controls incomplete")
    features=read(area/"public_features.json")
    if sha((area/"public_features.json").read_bytes())!=protocol["public_features_sha256"]:raise ValueError("Public cues changed")
    grouped=defaultdict(list);scene_rows=[];opportunities=[];trace_count=0;action_count=0
    with gzip.open(root/"ranked_acquisitions.jsonl.gz","wt") as stream:
        for sid in protocol["selected_ids"]:
            event,ref,free,domains,masks=load_scene(sid,root,area,protocol)
            for mode in MODES:
                for domain,mask in domains.items():
                    full_trace=[{"unit_id":u["unit_id"],"cost":u["cost"],"cumulative_cost":sum(v["cost"] for v in units(mode)[:i+1])} for i,u in enumerate(units(mode))]
                    opportunities.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,
                        **evaluate_acquisitions(full_trace,masks[mode],mode,16,ref,free,mask)})
                for budget in BUDGETS:
                    for domain,mask in domains.items():
                        scene_rows.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"random_exact",
                            **exact_random(masks[mode],mode,budget,ref,free,mask)})
                    for seed in ORDER_SEEDS:
                        trace=ranked_acquisitions(features[sid],mode,budget,seed)
                        for action in trace:action["declared_target"]=int(masks[mode][action["unit_id"]].sum())>=4
                        scores={name:evaluate_acquisitions(trace,masks[mode],mode,budget,ref,free,mask) for name,mask in domains.items()}
                        row={"sample_id":sid,"event":event,"input_mode":mode,"budget":budget,"order_seed":seed,"trace":trace,"scores":scores}
                        stream.write(json.dumps(row,sort_keys=True)+"\n");trace_count+=1;action_count+=len(trace)
                        for domain,result in scores.items():grouped[sid,event,domain,mode,budget].append(result)
    for (sid,event,domain,mode,budget),rows in grouped.items():
        scene_rows.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"raw_change_rank",**mean_scores(rows)})
    atomic_json(root/"full_opportunities.json",opportunities);atomic_json(root/"scene_scores.json",scene_rows)
    summary=aggregate(scene_rows);atomic_json(root/"summary.json",summary);atomic_json(root/"next_step_decision.json",decide(summary))
    atomic_json(root/"evaluation_report.json",{"ranked_acquisition_records":trace_count,"ranked_domain_scores":trace_count*2,"paid_acquisitions":action_count,
        "exact_random_scene_domain_scores":432,"scene_domain_policy_means":len(scene_rows),"full_coverage_scene_domain_scores":len(opportunities),
        "old_mixed_quality_scores_reused_as_comparison":False,"controller_fitting":False,"full512_calls_at_budget8":0,
        "trajectory_sha256":sha((root/"ranked_acquisitions.jsonl.gz").read_bytes())})
    print(json.dumps({"records":trace_count,"acquisitions":action_count,"decision":read(root/"next_step_decision.json")["decision"]}))
