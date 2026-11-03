"""Independent native-area geometry, tensors, payments and random expectation replay."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS
from external_footprint_core import MODES,NATIVE128,NATIVE256,CONTEXT256,FULL512,BUDGETS,ORDER_SEEDS,aggregate
from extract_external_area_masks import verify_parent,decode_reference
from run_changeos_pilot import crop_pair,prepare_tensor
from run_external_context_diagnostic import checked_mask
from evaluate_external_footprint import mean_scores


def read(path):return json.loads(Path(path).read_text())


def geometry(mode):
    if mode==NATIVE128:return [(cell,(int(cell[4:6])*128,int(cell[1:3])*128,128),1) for cell in CELLS]
    if mode in [NATIVE256,CONTEXT256]:return [(f"b{r}{c}",(c*256,r*256,256),4) for r in range(2) for c in range(2)]
    if mode==FULL512:return [("scene",(0,0,512),16)]
    raise ValueError("Unknown geometry")


def independent_score(paid,ref,free,domain,spent,acquisitions,affordable):
    target=ref&domain;baseline=free&domain;new=paid&~free&domain
    ft=int(np.count_nonzero(baseline&target));ff=int(np.count_nonzero(baseline&~target))
    it=int(np.count_nonzero(new&target));iff=int(np.count_nonzero(new&~target));n=int(target.sum());pred=ft+ff+it+iff
    return {"reference_pixels":n,"free_covered_reference_pixels":ft,"free_false_positive_pixels":ff,
        "incremental_covered_reference_pixels":it,"incremental_false_positive_pixels":iff,"covered_reference_pixels":ft+it,
        "reference_false_positive_pixels":ff+iff,"predicted_pixels":pred,"area_recall":(ft+it)/n if n else None,
        "reference_precision":(ft+it)/pred if pred else None,"spent":spent,"acquisitions":acquisitions,"affordable":affordable}


def random_by_combinatorics(masks,mode,budget,ref,free,domain):
    population=geometry(mode);n=len(population);cost=population[0][2];k=min(n,budget//cost)
    probability=math.comb(n-1,k-1)/math.comb(n,k) if k else 0
    baseline=independent_score(np.zeros_like(ref),ref,free,domain,0,0,budget>=cost);parts=[]
    for uid,(x,y,width),_ in population:
        paid=np.zeros_like(ref);paid[y:y+width,x:x+width]=masks[uid]
        parts.append(independent_score(paid,ref,free,domain,cost,1,True))
    it=probability*sum(r["incremental_covered_reference_pixels"] for r in parts);iff=probability*sum(r["incremental_false_positive_pixels"] for r in parts)
    ft=baseline["free_covered_reference_pixels"];ff=baseline["free_false_positive_pixels"];pred=ft+ff+it+iff;target=baseline["reference_pixels"]
    baseline.update(incremental_covered_reference_pixels=it,incremental_false_positive_pixels=iff,covered_reference_pixels=ft+it,
        reference_false_positive_pixels=ff+iff,predicted_pixels=pred,area_recall=(ft+it)/target if target else None,
        reference_precision=(ft+it)/pred if pred else None,spent=k*cost,acquisitions=k)
    return baseline


def independent_rank(features,mode,budget,seed):
    import random
    order=list(CELLS);random.Random(seed).shuffle(order);ranks={c:i for i,c in enumerate(order)};rows=[]
    for uid,(x,y,width),cost in geometry(mode):
        cells=[f"r{r:02d}c{c:02d}" for r in range(y//128,(y+width)//128) for c in range(x//128,(x+width)//128)]
        value=sum(features[c]["change"] for c in cells)/len(cells);tie=min(ranks[c] for c in cells)
        rows.append((value,tie,uid,cost))
    rows.sort(key=lambda r:(-r[0],r[1]));result=[];spent=0
    for score,tie,uid,cost in rows:
        if spent+cost<=budget:
            spent+=cost;result.append({"unit_id":uid,"cost":cost,"cumulative_cost":spent,"public_score":score,"tie_rank":tie})
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["root","information","context","area","rgb","metadata"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();root=args.root;protocol=read(root/"protocol.json")
    serial=read("output/disaster_search_serial_multifidelity_v1/completion_receipt.json")
    sources={Path(n).name:d for n,d in serial["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    for name in ["metadata","rgb","area","context","information"]:
        parent=getattr(args,name);digest=verify_parent(parent)
        if name in protocol["parent_receipt_sha256"] and digest!=protocol["parent_receipt_sha256"][name]:raise ValueError("Parent binding differs")
        sources.update(read(parent/"completion_receipt.json")["source_code_sha256"])
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Old source changed: "+name)
    for name,digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Preregistered code changed")
    if sha((args.information/"prospective_footprint_diagnostic.json").read_bytes())!=protocol["prospective_protocol_sha256"]:raise ValueError("Plan differs")
    log=(root/"test_log.txt").read_bytes()
    if sha(log)!=protocol["test_log_sha256"] or b"Ran 302 tests" not in log or not log.rstrip().endswith(b"OK"):raise ValueError("Test receipt differs")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((Path(protocol["model_directory"])/name).read_bytes())!=protocol["checkpoint"][key]:raise ValueError("Model changed")
    calls=[read(p) for p in sorted((root/"calls").glob("call_*.json"))]
    if len(calls)!=146 or len(list((root/"cache").glob("*.npz")))!=146:raise ValueError("Wrong cache/call count")
    observations=read(root/"observations.json");by_job={r["job"]:r for r in observations}
    if len(observations)!=144 or len(by_job)!=144:raise ValueError("Observation count differs")
    tensor_count=0;roi_count=0;index=0
    def verify_call(job,pair,mode,block=None):
        nonlocal index,tensor_count,roi_count
        row=calls[index];index+=1;tensor=prepare_tensor(*pair,"native")
        if row["index"]!=index or row["status"]!="COMPLETE" or row["job"]!=job or row["input_mode"]!=mode or row["block_id"]!=block:raise ValueError("Call identity differs")
        if list(tensor.shape)!=row["input_tensor_shape"] or sha(tensor.numpy().tobytes())!=row["input_tensor_sha256"]:raise ValueError("Independent tensor reconstruction differs")
        tensor_count+=1;size=128 if mode=="positive_control" else 256
        mask=checked_mask(Path(row["cache_file"]),row["cache_sha256"],(size,size))
        with np.load(row["cache_file"],allow_pickle=False) as z:raw=z["raw_severe_mask"]
        expected_shape=(512,512) if mode==CONTEXT256 else (size,size)
        if raw.dtype!=np.bool_ or raw.shape!=expected_shape:raise ValueError("Raw mask differs")
        if mode==CONTEXT256:
            r,c=int(block[1]),int(block[2]);expected=raw[r*256:(r+1)*256,c*256:(c+1)*256]
        else:expected=raw
        if not np.array_equal(mask,expected) or row["predicted_severe_pixels"]!=int(mask.sum()) or row["declared_target"]!=(int(mask.sum())>=4):raise ValueError("Canonical ROI/summary differs")
        if int(raw.sum())!=row["raw_summary"]["predicted_severe_pixels"] or (int(raw.sum())>=4)!=row["raw_summary"]["declared_target"]:raise ValueError("Raw summary differs")
        if mode!="positive_control":
            if any(by_job[job].get(k)!=v for k,v in row.items()):raise ValueError("Observation receipt differs")
            roi_count+=1
        return row,mask
    controls=read(root/"positive_control_results.json")
    if len(controls)!=2:raise ValueError("Missing controls")
    for old,current in zip(protocol["positive_controls"],controls):
        pair=[]
        for image in old["images"]:
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("Control source differs")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        row,mask=verify_call(old["sample_id"]+"_positive_control",crop_pair(*pair,old["cell_id"]),"positive_control")
        previous=checked_mask(Path(old["legacy_cache_file"]),old["legacy_cache_sha256"],(128,128))
        if not np.array_equal(mask,previous) or not current["exact_mask_equal"] or current["different_pixels_from_legacy"]!=0 or any(current.get(k)!=v for k,v in row.items()):raise ValueError("Positive control differs")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};scenes={}
    refs={r["sample_id"]:r for r in read(args.area/"reference_manifest.json")}
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest differs")
    for sid in protocol["selected_ids"]:
        pair=[]
        for date in ["pre","post"]:
            image=images[sid,date]
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("RGB differs")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX) for im in pair);base=tuple(im.resize((512,512),Image.Resampling.BILINEAR) for im in coarse);rows=[]
        masks={mode:{} for mode in MODES}
        for mode in [NATIVE256,CONTEXT256]:
            for block,(x,y,width),cost in geometry(mode):
                patches=tuple(im.crop((x,y,x+width,y+width)) for im in pair)
                if mode==NATIVE256:inputs=patches
                else:
                    inputs=tuple(im.copy() for im in base)
                    for im,patch in zip(inputs,patches):im.paste(patch,(x,y))
                job=sid+"_"+block+"_"+mode;_,mask=verify_call(job,inputs,mode,block);obs=by_job[job];rows.append(obs);masks[mode][block]=mask
                if obs["purchased_rgb_sha256"]!=[sha(im.tobytes()) for im in patches] or obs["coarse_rgb_sha256"]!=([sha(im.tobytes()) for im in coarse] if mode==CONTEXT256 else None):raise ValueError("Allowed information binding differs")
        if rows!=read(root/"scene_records"/(sid+".json")):raise ValueError("Scene record differs")
        for obs in protocol["previous_native_observations"]:
            if obs["sample_id"]==sid:masks[NATIVE128][obs["cell_id"]]=checked_mask(Path(obs["cache_file"]),obs["cache_sha256"],(128,128))
        full=next(r for r in protocol["previous_whole_observations"] if r["sample_id"]==sid)
        masks[FULL512]["scene"]=checked_mask(Path(full["cache_file"]),full["cache_sha256"],(512,512))
        free_row=next(r for r in protocol["previous_coarse_observations"] if r["sample_id"]==sid)
        free=checked_mask(Path(free_row["cache_file"]),free_row["cache_sha256"],(512,512))
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference differs")
        labels,_=decode_reference(Path(ref["path"]).read_bytes())
        scenes[sid]=(ref["event"],labels>=3,free,{"all_published_pixels":np.ones((512,512),dtype=np.bool_),"both_dates_nonzero":(np.asarray(pair[0]).max(2)>0)&(np.asarray(pair[1]).max(2)>0)},masks)
    if index!=146:raise ValueError("Call accounting differs")
    features=read(args.area/"public_features.json")
    if sha((args.area/"public_features.json").read_bytes())!=protocol["public_features_sha256"]:raise ValueError("Public cues differ")
    report=read(root/"inference_report.json")
    if report["status"]!="COMPLETE" or report["total_forward_calls"]!=146 or sha((root/"observations.json").read_bytes())!=report["observations_sha256"]:raise ValueError("Inference report differs")
    ranked=defaultdict(list);seen=set();trace_count=0;actions=0
    if sha((root/"ranked_acquisitions.jsonl.gz").read_bytes())!=read(root/"evaluation_report.json")["trajectory_sha256"]:raise ValueError("Ranked file changed")
    with gzip.open(root/"ranked_acquisitions.jsonl.gz","rt") as stream:
        for line in stream:
            row=json.loads(line);sid=row["sample_id"];mode=row["input_mode"];budget=row["budget"];seed=row["order_seed"];event,ref,free,domains,masks=scenes[sid]
            key=(sid,mode,budget,seed)
            if key in seen or mode not in MODES or budget not in BUDGETS or seed not in ORDER_SEEDS or row["event"]!=event:raise ValueError("Unplanned/duplicate ranked record")
            seen.add(key);trace=independent_rank(features[sid],mode,budget,seed);geometry_map={uid:(bounds,cost) for uid,bounds,cost in geometry(mode)}
            paid=np.zeros_like(ref);spent=0
            for action in trace:
                uid=action["unit_id"];(x,y,width),cost=geometry_map[uid];spent+=cost
                action["declared_target"]=int(masks[mode][uid].sum())>=4;paid[y:y+width,x:x+width]|=masks[mode][uid]
                if action["cost"]!=cost or action["cumulative_cost"]!=spent or spent>budget:raise ValueError("Payment mismatch")
            if trace!=row["trace"] or mode==FULL512 and budget<16 and trace:raise ValueError("Rank replay/affordability differs")
            for domain,mask in domains.items():
                score=independent_score(paid,ref,free,mask,spent,len(trace),budget>=geometry(mode)[0][2])
                if score!=row["scores"][domain]:raise ValueError("Independent paid union differs")
                ranked[sid,event,domain,mode,budget].append(score)
            trace_count+=1;actions+=len(trace)
    if trace_count!=2160 or actions!=7740:raise ValueError("Acquisition inventory differs")
    expected=[];opportunities=[]
    for sid,(event,ref,free,domains,masks) in scenes.items():
        for mode in MODES:
            paid=np.zeros_like(ref)
            for uid,(x,y,width),cost in geometry(mode):paid[y:y+width,x:x+width]|=masks[mode][uid]
            for domain,mask in domains.items():
                opportunities.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,**independent_score(paid,ref,free,mask,16,len(geometry(mode)),True)})
            for budget in BUDGETS:
                for domain,mask in domains.items():expected.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"random_exact",**random_by_combinatorics(masks[mode],mode,budget,ref,free,mask)})
    for (sid,event,domain,mode,budget),rows in ranked.items():expected.append({"sample_id":sid,"event":event,"domain":domain,"input_mode":mode,"budget":budget,"policy":"raw_change_rank",**mean_scores(rows)})
    if expected!=read(root/"scene_scores.json") or opportunities!=read(root/"full_opportunities.json") or aggregate(expected)!=read(root/"summary.json"):raise ValueError("Random/aggregate replay differs")
    # Main gate is checked directly against reconstructed counts; secondary is never promoted.
    summary=read(root/"summary.json");decision=read(root/"next_step_decision.json")
    for screen in decision["domain_screens"]:
        rows=[r for r in summary["event_rows"] if r["domain"]==screen["domain"] and r["budget"]==8 and r["policy"]=="random_exact"]
        totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]==screen["domain"] and r["budget"]==8 and r["policy"]=="random_exact"}
        pos=screen["reference_positive_events"];a=totals[NATIVE256];b=totals[NATIVE128]
        checks={"both_reference_positive_events_have_incremental_opportunity":len(pos)==2 and all(next(r for r in rows if r["event"]==e and r["input_mode"]==NATIVE256)["incremental_covered_reference_pixels"]>0 for e in pos),
            "budget8_exact_random_macro_incremental_coverage_exceeds128":a["event_macro_incremental_covered_pixels_per_scene"]>b["event_macro_incremental_covered_pixels_per_scene"],
            "budget8_pooled_reference_precision_not_below128":a["reference_precision"] is not None and b["reference_precision"] is not None and a["reference_precision"]>=b["reference_precision"]}
        if screen["checks"]!=checks or screen["passed"]!=all(checks.values()):raise ValueError("Gate replay differs")
    atomic_json(root/"audit.json",{"status":"INPUTS_AREA_PAYMENTS_UNIONS_AND_RANDOM_EXPECTATIONS_REPLAYED","old_frozen_sources_unchanged":len(sources),
        "positive_controls_pixel_identical":2,"new_forward_calls":146,"input_tensors_reconstructed":tensor_count,"paid256_rois_reconstructed":roi_count,
        "ranked_records_replayed":trace_count,"ranked_domain_scores_replayed":trace_count*2,"paid_area_acquisitions_verified":actions,
        "exact_random_scene_domain_scores_replayed":432,"scene_domain_policy_means_replayed":len(expected),"full_coverage_scores_replayed":len(opportunities),
        "tests_passed":302,"new_tests":18,"primary_screen_passed":decision["primary_screen_passed"],"full512_acquisitions_at_budget8":0,
        "new_network_bytes":0,"new_fitting":False,"paid_gpu":False,"new_neural_calls_by_audit":0,"human_annotations_created":0,
        "building_instances_created":0,"old_reserved_sets_scored":False,"independent_efficacy_claim":False})
    print(json.dumps(read(root/"audit.json")))


if __name__=="__main__":main()
