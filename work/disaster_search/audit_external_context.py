"""Replay input tensors, cache lineage and reference counts without model calls."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_context_core import select_controls,summarize
from extract_external_area_masks import verify_parent,decode_reference
from run_changeos_pilot import crop_pair,prepare_tensor


def checked(path,digest,shape):
    path=Path(path)
    if sha(path.read_bytes())!=digest:raise ValueError("Cache changed: "+str(path))
    with np.load(path,allow_pickle=False) as cache:mask=cache["severe_mask"]
    if mask.shape!=shape or mask.dtype!=np.bool_:raise ValueError("Wrong canonical mask")
    return mask


def counts(pred,ref,domain):
    # Direct integer identities, separately implemented from the experiment helper.
    n=int(np.count_nonzero(ref[domain]));p=int(np.count_nonzero(pred[domain]))
    tp=int(np.count_nonzero(np.logical_and(pred,ref)[domain]));fp=p-tp
    return {"reference_pixels":n,"predicted_pixels":p,"covered_reference_pixels":tp,
        "reference_false_positive_pixels":fp,"area_recall":tp/n if n else None,
        "reference_precision":tp/p if p else None,"reference_iou":tp/(n+fp) if n+fp else None}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["root","area","rgb","metadata","legacy"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();root=args.root;get=lambda n:json.loads((root/n).read_text())
    protocol=get("protocol.json")
    for parent,key in [(args.area,"area_completion_sha256"),(args.rgb,"rgb_completion_sha256")]:
        if verify_parent(parent)!=protocol[key]:raise ValueError("Parent receipt changed")
    verify_parent(args.metadata)
    serial=json.loads(Path("output/disaster_search_serial_multifidelity_v1/completion_receipt.json").read_text())
    sources={Path(n).name:d for n,d in serial["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    for parent in [args.metadata,args.rgb,args.area]:sources.update(json.loads((parent/"completion_receipt.json").read_text())["source_code_sha256"])
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Frozen source changed: "+name)
    for name,digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Preregistered source changed")
    for name,key in [("perception_protocol.json","legacy_protocol_sha256"),("observations.json","legacy_observations_sha256")]:
        if sha((args.legacy/name).read_bytes())!=protocol[key]:raise ValueError("Legacy binding changed")
    selected=select_controls(json.loads((args.legacy/"observations.json").read_text()))
    if [r["observation_id"] for r in selected]!=[r["observation_id"] for r in protocol["positive_controls"]]:raise ValueError("Control selection changed")
    controls={r["observation_id"]:r for r in get("positive_control_results.json")}
    if len(controls)!=2:raise ValueError("Two controls required")
    for old in protocol["positive_controls"]:
        pair=[]
        for image in old["images"]:
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("Control image changed")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        tensor=prepare_tensor(*crop_pair(*pair,old["cell_id"]),"native")
        if list(tensor.shape)!=old["prepared_tensor_shape"] or sha(tensor.numpy().tobytes())!=old["prepared_tensor_sha256"]:raise ValueError("Control tensor differs")
        before=checked(old["legacy_cache_file"],old["legacy_cache_sha256"],(128,128));now=controls[old["observation_id"]]
        after=checked(now["cache_file"],now["cache_sha256"],(128,128))
        if not np.array_equal(before,after) or not now["exact_mask_equal"] or now["different_pixels_from_legacy"]!=0:raise ValueError("Control replay differs")
        if int(after.sum())!=old["legacy_severe_pixels"] or int(after.sum())!=now["predicted_severe_pixels"]:raise ValueError("Control count differs")
    refs={r["sample_id"]:r for r in json.loads((args.area/"reference_manifest.json").read_text())}
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest changed")
    whole={r["sample_id"]:r for r in get("whole_observations.json")}
    if set(whole)!=set(protocol["selected_ids"]) or len(whole)!=18:raise ValueError("Cohort differs")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]}
    scores=[];paired=[];native_count=0
    for sid in protocol["selected_ids"]:
        pair=[]
        for mode in ["pre","post"]:
            image=images[sid,mode]
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("RGB changed")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        tensor=prepare_tensor(*pair,"native")
        if list(tensor.shape)!=[1,6,512,512] or sha(tensor.numpy().tobytes())!=whole[sid]["input_tensor_sha256"]:raise ValueError("Whole tensor differs")
        if json.loads((root/"scene_records"/(sid+".json")).read_text())!=whole[sid]:raise ValueError("Scene record differs")
        full=checked(whole[sid]["cache_file"],whole[sid]["cache_sha256"],(512,512))
        if int(full.sum())!=whole[sid]["predicted_severe_pixels"] or (int(full.sum())>=4)!=whole[sid]["declared_target"]:raise ValueError("Whole summary differs")
        native=np.zeros((512,512),dtype=np.bool_);cells=set()
        for obs in protocol["previous_native_observations"]:
            if obs["sample_id"]!=sid:continue
            cell=obs["cell_id"];row=int(cell[1:3]);col=int(cell[4:6])
            if cell in cells or not 0<=row<4 or not 0<=col<4:raise ValueError("Duplicate/invalid cell")
            cells.add(cell);mask=checked(obs["cache_file"],obs["cache_sha256"],(128,128))
            if int(mask.sum())!=obs["predicted_severe_pixels"]:raise ValueError("Native summary differs")
            native[row*128:(row+1)*128,col*128:(col+1)*128]=mask;native_count+=1
        if len(cells)!=16:raise ValueError("Incomplete mosaic")
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
        label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3
        valid=(np.asarray(pair[0]).max(axis=2)>0)&(np.asarray(pair[1]).max(axis=2)>0)
        for name,domain in [("all_published_pixels",np.ones((512,512),dtype=np.bool_)),("both_dates_nonzero",valid)]:
            for mode,mask in [("crop128_mosaic_native",native),("whole512_native",full)]:
                scores.append({"sample_id":sid,"event":ref["event"],"domain":name,"input_mode":mode,**counts(mask,target,domain)})
            gain=int(np.count_nonzero(full&~native&target&domain));loss=int(np.count_nonzero(native&~full&target&domain));keep=int(np.count_nonzero(native&full&target&domain))
            if gain-loss!=scores[-1]["covered_reference_pixels"]-scores[-2]["covered_reference_pixels"]:raise ValueError("Gain/loss conservation failed")
            paired.append({"sample_id":sid,"event":ref["event"],"domain":name,"gained_reference_pixels":gain,"lost_reference_pixels":loss,
                "preserved_reference_pixels":keep,"prediction_disagreement_pixels":int(np.count_nonzero((native!=full)&domain))})
    if scores!=get("scene_scores.json") or paired!=get("paired_changes.json"):raise ValueError("Independent pixel replay differs")
    if summarize(scores)!=get("summary.json"):raise ValueError("Aggregation replay differs")
    # Reconcile the native comparison with the preceding stage's full-grid opportunity.
    previous=json.loads((args.area/"sensor_area_opportunity.json").read_text())
    for row in scores:
        if row["input_mode"]!="crop128_mosaic_native":continue
        prior=next(r for r in previous if r["sample_id"]==row["sample_id"] and r["domain"]==row["domain"])
        for key in ["reference_pixels","covered_reference_pixels","reference_false_positive_pixels"]:
            expected=prior[key] if key=="reference_pixels" else prior["native"][key]
            if row[key]!=expected:raise ValueError("Previous opportunity mismatch")
    report=get("inference_report.json")
    if report["status"]!="COMPLETE" or report["positive_control_calls"]!=2 or report["whole_calls_this_invocation"]!=18 or report["new_forward_calls_this_invocation"]!=20:raise ValueError("Forward accounting differs")
    if sha((root/"whole_observations.json").read_bytes())!=report["observations_sha256"] or len(list((root/"cache").glob("*.npz")))!=20:raise ValueError("Output accounting differs")
    log=(root/"test_log.txt").read_text()
    if "Ran 270 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("270-test evidence missing")
    atomic_json(root/"audit.json",{"status":"TENSORS_MASKS_AND_REFERENCE_COUNTS_REPLAYED","old_frozen_sources_unchanged":len(sources),"preregistered_sources_verified":len(protocol["source_code_sha256"]),
        "positive_controls_pixel_identical":2,"whole_tensors_verified":18,"whole_cache_masks_verified":18,"native_cache_masks_verified":native_count,
        "scene_domain_scores_replayed":len(scores),"paired_domain_changes_replayed":len(paired),"tests_passed":270,"new_tests":8,"new_forward_calls":20,
        "new_inference_by_audit":0,"new_network_bytes":0,"new_search_trajectories":0,"new_fitting":False,"paid_gpu":False,
        "human_annotations_created":0,"building_instances_created":0,"old_reserved_sets_scored":False,"independent_efficacy_claim":False})
    print(json.dumps(json.loads((root/"audit.json").read_text())))


if __name__=="__main__":main()
