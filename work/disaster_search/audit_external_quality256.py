"""Independently reconstruct blur bytes/tensors and all fixed quality measurements."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageFilter

from build_xbd_mini import atomic_json,sha
from external_quality256_core import QUALITIES,quality_summary
from extract_external_area_masks import verify_parent,decode_reference
from run_changeos_pilot import crop_pair,prepare_tensor
from run_external_context_diagnostic import checked_mask


def read(path):return json.loads(Path(path).read_text())


def independent_counts(paid,reference,free,domain):
    target=reference&domain;baseline=free&domain;extra=paid&~free&domain
    ft=int(np.count_nonzero(target&baseline));ff=int(np.count_nonzero(~target&baseline))
    it=int(np.count_nonzero(target&extra));iff=int(np.count_nonzero(~target&extra));n=int(target.sum());pred=ft+ff+it+iff
    return {"reference_pixels":n,"free_covered_reference_pixels":ft,"free_false_positive_pixels":ff,
        "incremental_covered_reference_pixels":it,"incremental_false_positive_pixels":iff,"covered_reference_pixels":ft+it,
        "reference_false_positive_pixels":ff+iff,"predicted_pixels":pred,"area_recall":(ft+it)/n if n else None,"reference_precision":(ft+it)/pred if pred else None}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["root","footprint","information","context","area","rgb","metadata"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();root=args.root;protocol=read(root/"protocol.json")
    serial=read("output/disaster_search_serial_multifidelity_v1/completion_receipt.json")
    sources={Path(n).name:d for n,d in serial["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    for name in ["metadata","rgb","area","context","information","footprint"]:
        path=getattr(args,name);digest=verify_parent(path)
        if name in protocol["parent_receipt_sha256"] and digest!=protocol["parent_receipt_sha256"][name]:raise ValueError("Parent binding differs")
        sources.update(read(path/"completion_receipt.json")["source_code_sha256"])
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Old source changed: "+name)
    for name,digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Locked current source changed")
    if sha((args.footprint/"prospective_native256_quality_check.json").read_bytes())!=protocol["prospective_protocol_sha256"]:raise ValueError("Prospective source differs")
    log=(root/"test_log.txt").read_bytes()
    if sha(log)!=protocol["preflight_test_log_sha256"] or b"Ran 314 tests" not in log or not log.rstrip().endswith(b"OK"):raise ValueError("Test evidence differs")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((Path(protocol["model_directory"])/name).read_bytes())!=protocol["checkpoint"][key]:raise ValueError("Checkpoint differs")
    calls=[read(path) for path in sorted((root/"calls").glob("call_*.json"))]
    if len(calls)!=74 or len(list((root/"cache").glob("*.npz")))!=74:raise ValueError("Forward accounting differs")
    observations=read(root/"observations.json");by_job={r["job"]:r for r in observations}
    if len(observations)!=72 or len(by_job)!=72:raise ValueError("Blur observation count differs")
    index=0;tensors=0
    def check_call(job,pair,mode,block=None):
        nonlocal index,tensors
        row=calls[index];index+=1;tensor=prepare_tensor(*pair,"native")
        if row["index"]!=index or row["status"]!="COMPLETE" or row["job"]!=job or row["input_mode"]!=mode or row["block_id"]!=block:raise ValueError("Call identity differs")
        if sha(tensor.numpy().tobytes())!=row["input_tensor_sha256"] or list(tensor.shape)!=row["input_tensor_shape"]:raise ValueError("Independent tensor differs")
        tensors+=1;size=128 if mode=="positive_control" else 256;mask=checked_mask(Path(row["cache_file"]),row["cache_sha256"],(size,size))
        if int(mask.sum())!=row["predicted_severe_pixels"] or (int(mask.sum())>=4)!=row["declared_target"]:raise ValueError("Canonical summary differs")
        if mode!="positive_control" and any(by_job[job].get(k)!=v for k,v in row.items()):raise ValueError("Call/observation receipt differs")
        return row,mask
    controls=read(root/"positive_control_results.json")
    if len(controls)!=2:raise ValueError("Missing controls")
    for previous,current in zip(protocol["positive_controls"],controls):
        pair=[]
        for image in previous["images"]:
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("Control image differs")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        row,mask=check_call(previous["sample_id"]+"_positive_control",crop_pair(*pair,previous["cell_id"]),"positive_control")
        old=checked_mask(Path(previous["legacy_cache_file"]),previous["legacy_cache_sha256"],(128,128))
        if not np.array_equal(old,mask) or not current["exact_mask_equal"] or current["different_pixels_from_legacy"]!=0 or any(current.get(k)!=v for k,v in row.items()):raise ValueError("Positive control differs")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};prepared={(r["sample_id"],r["block_id"]):r for r in protocol["prepared_inputs"]}
    native={(r["sample_id"],r["block_id"]):r for r in protocol["previous_native256_observations"]};free_rows={r["sample_id"]:r for r in protocol["previous_coarse_observations"]}
    refs={r["sample_id"]:r for r in read(args.area/"reference_manifest.json")}
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference binding differs")
    scores=[];pairs=[];blocks=[];feedback=[];blur_bytes_verified=0
    for sid in protocol["selected_ids"]:
        rgb=[]
        for date in ["pre","post"]:
            row=images[sid,date]
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("RGB differs")
            with Image.open(row["path"]) as im:rgb.append(im.convert("RGB"))
        mosaics={q:np.zeros((512,512),dtype=np.bool_) for q in QUALITIES};local={q:{} for q in QUALITIES};scene_observations=[]
        for r in range(2):
            for c in range(2):
                block=f"b{r}{c}";bounds=(c*256,r*256,(c+1)*256,(r+1)*256);raw=tuple(im.crop(bounds) for im in rgb)
                degraded=tuple(im.filter(ImageFilter.GaussianBlur(1.5)).resize((128,128),Image.Resampling.BOX).resize((256,256),Image.Resampling.BILINEAR) for im in raw)
                pre=prepared[sid,block]
                if pre["original_rgb_sha256"]!=[sha(im.tobytes()) for im in raw] or pre["degraded_rgb_sha256"]!=[sha(im.tobytes()) for im in degraded]:raise ValueError("Independent blur bytes differ")
                tensor=prepare_tensor(*raw,"blurred").numpy()
                if sha(tensor.tobytes())!=pre["input_tensor_sha256"] or list(tensor.shape)!=pre["tensor_shape"] or not pre["frozen_helper_tensor_equal"]:raise ValueError("Old blur helper differs")
                blur_bytes_verified+=1;job=sid+"_"+block+"_blurred256";_,mask=check_call(job,degraded,QUALITIES[1],block);scene_observations.append(by_job[job])
                old=native[sid,block];native_mask=checked_mask(Path(old["cache_file"]),old["cache_sha256"],(256,256))
                if int(native_mask.sum())!=old["predicted_severe_pixels"] or (int(native_mask.sum())>=4)!=old["declared_target"]:raise ValueError("Native cache summary differs")
                for quality,pred in [(QUALITIES[0],native_mask),(QUALITIES[1],mask)]:
                    local[quality][block]=pred;mosaics[quality][r*256:(r+1)*256,c*256:(c+1)*256]=pred
                feedback.append({"sample_id":sid,"event":refs[sid]["event"],"block_id":block,"native_declared_target":int(native_mask.sum())>=4,"blurred_declared_target":int(mask.sum())>=4})
        if scene_observations!=read(root/"scene_records"/(sid+".json")):raise ValueError("Scene receipt differs")
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference PNG differs")
        labels,_=decode_reference(Path(ref["path"]).read_bytes());target=labels>=3;free_row=free_rows[sid]
        free=checked_mask(Path(free_row["cache_file"]),free_row["cache_sha256"],(512,512))
        domains={"all_published_pixels":np.ones((512,512),dtype=np.bool_),"both_dates_nonzero":(np.asarray(rgb[0]).max(2)>0)&(np.asarray(rgb[1]).max(2)>0)}
        for domain_name,domain in domains.items():
            for quality in QUALITIES:scores.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"input_mode":quality,**independent_counts(mosaics[quality],target,free,domain)})
            a=mosaics[QUALITIES[0]]|free;b=mosaics[QUALITIES[1]]|free;truth=target&domain
            gain=int(np.count_nonzero(truth&b&~a));loss=int(np.count_nonzero(truth&a&~b));keep=int(np.count_nonzero(truth&a&b))
            pairs.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"gained_reference_pixels":gain,"lost_reference_pixels":loss,"preserved_reference_pixels":keep,"prediction_disagreement_pixels":int(np.count_nonzero((a!=b)&domain))})
            if gain-loss!=scores[-1]["covered_reference_pixels"]-scores[-2]["covered_reference_pixels"]:raise ValueError("Gain/loss conservation differs")
            for r in range(2):
                for c in range(2):
                    block=f"b{r}{c}";y=slice(r*256,(r+1)*256);x=slice(c*256,(c+1)*256)
                    for quality in QUALITIES:
                        prediction=(local[quality][block]|free[y,x])&domain[y,x];n=int(np.count_nonzero(target[y,x]&domain[y,x]));tp=int(np.count_nonzero(prediction&target[y,x]));fp=int(prediction.sum())-tp
                        blocks.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"block_id":block,"input_mode":quality,"reference_pixels":n,"covered_reference_pixels":tp,
                            "reference_false_positive_pixels":fp,"predicted_pixels":tp+fp,"area_recall":tp/n if n else None,"reference_precision":tp/(tp+fp) if tp+fp else None})
    if scores!=read(root/"scene_scores.json") or pairs!=read(root/"paired_changes.json") or blocks!=read(root/"block_scores.json") or feedback!=read(root/"block_feedback.json"):raise ValueError("Independent quality measurement differs")
    summary=quality_summary(scores)
    if summary!=read(root/"summary.json"):raise ValueError("Event aggregation differs")
    previous={(r["sample_id"],r["domain"]):r for r in read(args.footprint/"full_opportunities.json") if r["input_mode"]=="native256_contiguous"}
    for row in scores:
        if row["input_mode"]==QUALITIES[0] and any(row[k]!=previous[row["sample_id"],row["domain"]][k] for k in ["reference_pixels","free_covered_reference_pixels","free_false_positive_pixels","incremental_covered_reference_pixels","incremental_false_positive_pixels","covered_reference_pixels","reference_false_positive_pixels","predicted_pixels","area_recall","reference_precision"]):raise ValueError("Previous native result changed")
    decision=read(root/"next_step_decision.json")
    for screen in decision["domain_screens"]:
        positive=[r for r in summary["event_rows"] if r["domain"]==screen["domain"] and r["input_mode"]==QUALITIES[1] and r["reference_pixels"]>0]
        expected=len(positive)==2 and all(r["incremental_covered_reference_pixels"]>0 for r in positive)
        if screen["passed"]!=expected or screen["checks"]["two_reference_positive_events_retained"]!=(len(positive)==2) or screen["checks"]["both_positive_events_have_blurred_reference_overlap"]!=expected:raise ValueError("Quality screen replay differs")
    report=read(root/"inference_report.json")
    if index!=74 or report["status"]!="COMPLETE" or report["total_forward_calls"]!=74 or sha((root/"observations.json").read_bytes())!=report["observations_sha256"]:raise ValueError("Final forward accounting differs")
    atomic_json(root/"audit.json",{"status":"BLUR_BYTES_TENSORS_MASKS_AND_REFERENCE_MEASUREMENTS_REPLAYED","old_frozen_sources_unchanged":len(sources),
        "new_forward_calls":74,"positive_controls_pixel_identical":2,"real_blur_inputs_matching_frozen_helper":blur_bytes_verified,"input_tensors_reconstructed":tensors,
        "blurred256_masks_verified":72,"cached_native256_masks_verified":72,"coarse_baseline_masks_verified":18,
        "scene_domain_scores_replayed":len(scores),"paired_domain_changes_replayed":len(pairs),"block_domain_scores_replayed":len(blocks),"block_feedback_pairs_replayed":len(feedback),
        "previous_native_scene_domain_scores_preserved":36,"tests_passed":314,"new_tests":12,"primary_screen_passed":decision["primary_screen_passed"],
        "new_neural_calls_by_audit":0,"new_network_bytes":0,"new_fitting":False,"new_search_runs":0,"paid_gpu":False,"human_annotations_created":0,
        "building_instances_created":0,"old_reserved_sets_scored":False,"independent_efficacy_claim":False})
    print(json.dumps(read(root/"audit.json")))


if __name__=="__main__":main()
