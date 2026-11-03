"""Reconstruct every allowed input and replay pixel scores without neural calls."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_external_context import counts
from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS
from external_context_core import summarize
from external_information_core import PATCH_MODES,COARSE_MODE
from extract_external_area_masks import verify_parent,decode_reference
from run_changeos_pilot import crop_pair,prepare_tensor
from run_external_context_diagnostic import checked_mask


def read(path):return json.loads(Path(path).read_text())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["root","context","area","rgb","metadata"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();root=args.root;protocol=read(root/"protocol.json")
    sources={Path(n).name:d for n,d in read("output/disaster_search_serial_multifidelity_v1/completion_receipt.json")["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    for name in ["metadata","rgb","area","context"]:
        path=getattr(args,name);digest=verify_parent(path)
        if name in protocol["parent_receipt_sha256"] and digest!=protocol["parent_receipt_sha256"][name]:raise ValueError("Parent binding changed")
        sources.update(read(path/"completion_receipt.json")["source_code_sha256"])
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Old source changed: "+name)
    for name,digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Preregistered source changed")
    if sha((args.context/"prospective_information_diagnostic.json").read_bytes())!=protocol["prospective_protocol_sha256"]:raise ValueError("Prospective source changed")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((Path(protocol["model_directory"])/name).read_bytes())!=protocol["checkpoint"][key]:raise ValueError("Model/config changed")
    test_log=(root/"test_log.txt").read_bytes()
    if sha(test_log)!=protocol["preflight_test_log_sha256"] or b"Ran 284 tests" not in test_log or not test_log.rstrip().endswith(b"OK"):raise ValueError("Frozen284-test evidence differs")
    files=sorted((root/"calls").glob("call_*.json"));calls=[read(p) for p in files]
    if len(calls)!=884 or len(list((root/"cache").glob("*.npz")))!=884:raise ValueError("Call/cache inventory differs")
    if any(r["index"]!=i or r["status"]!="COMPLETE" for i,r in enumerate(calls,1)):raise ValueError("Noncomplete/duplicate call")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]}
    observations=read(root/"observations.json");by_job={r["job"]:r for r in observations}
    if len(observations)!=882 or len(by_job)!=882:raise ValueError("Observation inventory differs")
    controls=read(root/"positive_control_results.json")
    if len(controls)!=2:raise ValueError("Two positive controls required")
    index=0;canonical={};tensor_count=0;roi_count=0

    def check_call(job,pair,mode,cell=None):
        nonlocal index,tensor_count,roi_count
        row=calls[index];index+=1
        if row["job"]!=job or row["input_mode"]!=mode or row["cell_id"]!=cell:raise ValueError("Call order differs")
        tensor=prepare_tensor(*pair,"native")
        if list(tensor.shape)!=row["input_tensor_shape"] or sha(tensor.numpy().tobytes())!=row["input_tensor_sha256"]:raise ValueError("Independent input reconstruction differs: "+job)
        tensor_count+=1;shape=(128,128) if mode in PATCH_MODES or mode=="positive_control" else (512,512)
        mask=checked_mask(Path(row["cache_file"]),row["cache_sha256"],shape)
        if int(mask.sum())!=row["predicted_severe_pixels"] or (int(mask.sum())>=4)!=row["declared_target"]:raise ValueError("Canonical summary differs")
        if mode in PATCH_MODES:
            with np.load(row["cache_file"],allow_pickle=False) as cache:raw=cache["raw_severe_mask"]
            if raw.dtype!=np.bool_ or raw.shape!=(512,512):raise ValueError("Raw audit prediction differs")
            # Independent nearest-centre mapping or direct row/column slicing.
            if mode==PATCH_MODES[0]:expected=raw[2::4,2::4]
            else:
                r,c=int(cell[1:3]),int(cell[4:6]);expected=raw[r*128:(r+1)*128,c*128:(c+1)*128]
            if not np.array_equal(mask,expected):raise ValueError("Noncanonical paid ROI")
            roi_count+=1
        else:raw=mask
        if int(raw.sum())!=row["full_output_summary"]["predicted_severe_pixels"] or (int(raw.sum())>=4)!=row["full_output_summary"]["declared_target"]:raise ValueError("Raw summary differs")
        if mode!="positive_control":
            obs=by_job[job]
            if any(obs.get(k)!=v for k,v in row.items()):raise ValueError("Observation/call receipt differs")
            canonical[obs["sample_id"],mode,cell]=mask
        return row,mask

    for old,saved in zip(protocol["positive_controls"],controls):
        pair=[]
        for im in old["images"]:
            if sha(Path(im["path"]).read_bytes())!=im["sha256"]:raise ValueError("Control image differs")
            with Image.open(im["path"]) as image:pair.append(image.convert("RGB"))
        row,mask=check_call(old["sample_id"]+"_positive_control",crop_pair(*pair,old["cell_id"]),"positive_control")
        before=checked_mask(Path(old["legacy_cache_file"]),old["legacy_cache_sha256"],(128,128))
        if not np.array_equal(before,mask) or not saved["exact_mask_equal"] or saved["different_pixels_from_legacy"]!=0:raise ValueError("Control mask changed")
        if any(saved.get(k)!=v for k,v in row.items()) or saved["observation_id"]!=old["observation_id"]:raise ValueError("Control receipt differs")
    domains_by_scene={}
    for sid in protocol["selected_ids"]:
        pair=[]
        for mode in ["pre","post"]:
            image=images[sid,mode]
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("RGB changed")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX) for im in pair)
        base=tuple(im.resize((512,512),Image.Resampling.BILINEAR) for im in coarse)
        hashes=[sha(im.tobytes()) for im in coarse]
        job=sid+"_"+COARSE_MODE;check_call(job,base,COARSE_MODE)
        if by_job[job]["coarse_rgb_sha256"]!=hashes:raise ValueError("Coarse-only source differs")
        scene_rows=[by_job[job]]
        for mode in PATCH_MODES:
            for cell in CELLS:
                r,c=int(cell[1:3]),int(cell[4:6]);bounds=(c*128,r*128,(c+1)*128,(r+1)*128)
                patches=tuple(im.crop(bounds) for im in pair)
                if mode==PATCH_MODES[0]:inputs=tuple(im.resize((512,512),Image.Resampling.BILINEAR) for im in patches)
                else:
                    inputs=tuple(Image.new("RGB",(512,512),(124,116,104)) for _ in range(2)) if mode==PATCH_MODES[1] else tuple(im.copy() for im in base)
                    for im,patch in zip(inputs,patches):im.paste(patch,bounds[:2])
                job=sid+"_"+cell+"_"+mode;check_call(job,inputs,mode,cell)
                obs=by_job[job]
                if obs["sample_id"]!=sid or obs["event"]!=images[sid,"pre"]["event"] or obs["patch_rgb_sha256"]!=[sha(im.tobytes()) for im in patches]:raise ValueError("Patch binding differs")
                if obs["coarse_rgb_sha256"]!=(hashes if mode==PATCH_MODES[2] else None):raise ValueError("Unexpected coarse information")
                scene_rows.append(obs)
        if scene_rows!=read(root/"scene_records"/(sid+".json")):raise ValueError("Scene receipt differs")
        valid=(np.asarray(pair[0]).max(axis=2)>0)&(np.asarray(pair[1]).max(axis=2)>0)
        domains_by_scene[sid]={"all_published_pixels":np.ones((512,512),dtype=np.bool_),"both_dates_nonzero":valid}
    if index!=884:raise ValueError("Call accounting incomplete")
    refs={r["sample_id"]:r for r in read(args.area/"reference_manifest.json")}
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference binding differs")
    whole={r["sample_id"]:r for r in protocol["offline_whole_observations"]};scores=[];changes=[]
    comparisons=[("crop128_mosaic_native",m) for m in PATCH_MODES]+[(PATCH_MODES[1],PATCH_MODES[2]),(COARSE_MODE,PATCH_MODES[2])]
    for sid in protocol["selected_ids"]:
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference PNG differs")
        label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3
        masks={"crop128_mosaic_native":np.zeros((512,512),dtype=np.bool_),"whole512_native":checked_mask(Path(whole[sid]["cache_file"]),whole[sid]["cache_sha256"],(512,512))}
        for obs in protocol["previous_native_observations"]:
            if obs["sample_id"]!=sid:continue
            cell=obs["cell_id"];r,c=int(cell[1:3]),int(cell[4:6])
            masks["crop128_mosaic_native"][r*128:(r+1)*128,c*128:(c+1)*128]=checked_mask(Path(obs["cache_file"]),obs["cache_sha256"],(128,128))
        for mode in PATCH_MODES:
            masks[mode]=np.zeros((512,512),dtype=np.bool_)
            for cell in CELLS:
                r,c=int(cell[1:3]),int(cell[4:6]);masks[mode][r*128:(r+1)*128,c*128:(c+1)*128]=canonical[sid,mode,cell]
        masks[COARSE_MODE]=canonical[sid,COARSE_MODE,None]
        for name,domain in domains_by_scene[sid].items():
            for mode,mask in masks.items():scores.append({"sample_id":sid,"event":ref["event"],"domain":name,"input_mode":mode,**counts(mask,target,domain)})
            for before,after in comparisons:
                a,b=masks[before],masks[after];gain=int(np.count_nonzero(target&domain&b&~a));loss=int(np.count_nonzero(target&domain&a&~b))
                changes.append({"sample_id":sid,"event":ref["event"],"domain":name,"before":before,"after":after,
                    "gained_reference_pixels":gain,"lost_reference_pixels":loss,"preserved_reference_pixels":int(np.count_nonzero(target&domain&a&b)),
                    "prediction_disagreement_pixels":int(np.count_nonzero(domain&(a!=b)))})
                if gain-loss!=counts(b,target,domain)["covered_reference_pixels"]-counts(a,target,domain)["covered_reference_pixels"]:raise ValueError("Pair conservation differs")
    if scores!=read(root/"scene_scores.json") or changes!=read(root/"paired_changes.json") or summarize(scores)!=read(root/"summary.json"):raise ValueError("Independent score/aggregation replay differs")
    # Previous-stage comparisons must remain byte-for-value identical.
    previous={(r["sample_id"],r["domain"],r["input_mode"]):r for r in read(args.context/"scene_scores.json")}
    for row in scores:
        if row["input_mode"] in ["crop128_mosaic_native","whole512_native"] and row!=previous[row["sample_id"],row["domain"],row["input_mode"]]:raise ValueError("Old context comparison changed")
    report=read(root/"inference_report.json")
    if report["total_forward_calls"]!=884 or report["status"]!="COMPLETE" or report["observations_sha256"]!=sha((root/"observations.json").read_bytes()):raise ValueError("Inference report differs")
    decision=read(root/"next_step_decision.json");summary=read(root/"summary.json")
    for gate in decision["domain_screens"]:
        rows=[r for r in summary["event_rows"] if r["domain"]==gate["domain"]];totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]==gate["domain"]}
        event={(r["event"],r["input_mode"]):r for r in rows};pos=gate["reference_positive_events"];main=totals[PATCH_MODES[2]];baseline=totals["crop128_mosaic_native"]
        expected=[len(pos)==2 and all(event[e,PATCH_MODES[2]]["covered_reference_pixels"]>0 for e in pos),main["covered_reference_pixels"]>baseline["covered_reference_pixels"],
            main["pooled_reference_precision"] is not None and main["pooled_reference_precision"]>=baseline["pooled_reference_precision"],
            main["covered_reference_pixels"]>totals[PATCH_MODES[1]]["covered_reference_pixels"],len(pos)==2 and all(event[e,PATCH_MODES[2]]["covered_reference_pixels"]>event[e,COARSE_MODE]["covered_reference_pixels"] for e in pos)]
        names=["both_reference_positive_events_have_overlap","pooled_coverage_exceeds_cached_native128","pooled_reference_precision_not_lower","pooled_coverage_exceeds_blank_frame","positive_net_coverage_beyond_free_coarse_in_each_positive_event"]
        if [gate["checks"][name] for name in names]!=expected or gate["passed"]!=all(expected):raise ValueError("Prospective screen replay differs")
    atomic_json(root/"audit.json",{"status":"ALLOWED_INPUTS_ROIS_AND_REFERENCE_SCORES_REPLAYED","old_frozen_sources_unchanged":len(sources),
        "new_forward_calls":884,"reserved_call_records":884,"complete_call_records":884,"positive_controls_pixel_identical":2,
        "input_tensors_reconstructed":tensor_count,"canonical_paid_rois_reconstructed":roi_count,"coarse_only_inputs":18,
        "scene_domain_scores_replayed":len(scores),"paired_domain_changes_replayed":len(changes),"old_context_scores_preserved":72,
        "tests_passed":284,"new_tests":14,"primary_screen_passed":decision["primary_screen_passed"],"new_neural_calls_by_audit":0,
        "new_network_bytes":0,"new_search_trajectories":0,"new_fitting":False,"paid_gpu":False,"human_annotations_created":0,
        "building_instances_created":0,"old_reserved_sets_scored":False,"independent_efficacy_claim":False})
    print(json.dumps(read(root/"audit.json")))


if __name__=="__main__":main()
