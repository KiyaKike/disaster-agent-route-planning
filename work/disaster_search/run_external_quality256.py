"""74-call fixed blur quality check after native256 footprint feasibility."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_footprint_core import BLOCKS,block_bounds
from external_quality256_core import (QUALITIES,blurred_inputs,canonical_summary,mosaic,measure_pair,quality_summary,measurement_decision)
from extract_external_area_masks import verify_parent,decode_reference
from plan_search_v2 import freeze
from run_changeos_pilot import crop_pair,prepare_tensor,predict_summary,sync,CODE_COMMIT
from run_external_context_diagnostic import checked_mask


def read(path):return json.loads(Path(path).read_text())


def read_pair(sid,images):
    pair=[]
    for date in ["pre","post"]:
        row=images[sid,date]
        if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("RGB changed")
        with Image.open(row["path"]) as im:pair.append(im.convert("RGB"))
    return pair


def plan(args):
    hashes={name:verify_parent(getattr(args,name)) for name in ["footprint","area","rgb"]}
    previous=read(args.footprint/"protocol.json");prospect=read(args.footprint/"prospective_native256_quality_check.json")
    if prospect["selected_ids"]!=previous["selected_ids"] or prospect["max_new_forward_calls"]!=74:raise ValueError("Prospective inventory differs")
    for name,file in [("protocol","protocol.json"),("evidence","evidence_summary.json"),("decision","next_step_decision.json")]:
        if sha((args.footprint/file).read_bytes())!=prospect["basis_"+name+"_sha256"]:raise ValueError("Prospective basis changed")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((args.model/name).read_bytes())!=prospect["checkpoint"][key]:raise ValueError("Fixed model changed")
    log=(args.out/"test_log.txt").read_text()
    if "Ran 314 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("314-test preflight required")
    images={(r["sample_id"],r["mode"]):r for r in previous["rgb_inputs"]};prepared=[]
    for sid in prospect["selected_ids"]:
        pair=read_pair(sid,images)
        for block in BLOCKS:
            patches=tuple(im.crop(block_bounds(block)) for im in pair);degraded=blurred_inputs(patches)
            new=prepare_tensor(*degraded,"native").numpy();old=prepare_tensor(*patches,"blurred").numpy()
            if not np.array_equal(new,old):raise ValueError("Real blur tensor differs from frozen helper")
            prepared.append({"sample_id":sid,"block_id":block,"original_rgb_sha256":[sha(im.tobytes()) for im in patches],
                "degraded_rgb_sha256":[sha(im.tobytes()) for im in degraded],"input_tensor_sha256":sha(new.tobytes()),"tensor_shape":list(new.shape),"frozen_helper_tensor_equal":True})
    if len(prepared)!=72:raise ValueError("Incomplete blur preflight")
    names=["run_external_quality256.py","external_quality256_core.py","test_external_quality256.py","run_changeos_pilot.py",
        "external_footprint_core.py","external_context_core.py","external_information_core.py","external_area_core.py",
        "run_external_context_diagnostic.py","extract_external_area_masks.py","build_xbd_mini.py","plan_search_v2.py"]
    native=[r for r in read(args.footprint/"observations.json") if r["input_mode"]=="native256_contiguous"]
    if len(native)!=72:raise ValueError("Native cache incomplete")
    protocol={"stage":"FIXED_NATIVE256_SYNTHETIC_BLUR_MEASUREMENT_ONLY","client_report_date":"2026-10-26","parent_receipt_sha256":hashes,
        "prospective_protocol_sha256":sha((args.footprint/"prospective_native256_quality_check.json").read_bytes()),"prospective_protocol":prospect,
        "selected_ids":prospect["selected_ids"],"checkpoint":prospect["checkpoint"],"model_directory":str(args.model.resolve()),
        "rgb_inputs":previous["rgb_inputs"],"positive_controls":previous["positive_controls"],"previous_native256_observations":native,
        "previous_coarse_observations":previous["previous_coarse_observations"],"prepared_inputs":prepared,
        "reference_manifest_sha256":sha((args.area/"reference_manifest.json").read_bytes()),"preflight_test_log_sha256":sha((args.out/"test_log.txt").read_bytes()),
        "source_code_sha256":{name:sha(Path(__file__).with_name(name).read_bytes()) for name in names},"preflight_tests":314,
        "max_new_forward_calls":74,"call_inventory":{"native128_controls":2,"blurred256_blocks":72},"no_hidden_warmup":True,
        "forward_order":"2 native128 locked controls, then18 sorted scene IDs x4 row-major256 blocks, blurred inputs prepared exactly once.",
        "reference_pixels_decoded_by_prediction":False,"quality_fitting":False,"new_search_runs":0,"new_download":False,"paid_gpu":False,
        "evaluation_scope":"Full16-area native/blur measurements; no budget8 policy results and no secondary context input extension."}
    freeze(args.out/"protocol.json",protocol);return protocol


def predict(args,protocol):
    for key in ["HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"]:os.environ[key]="1"
    os.environ["WANDB_MODE"]="disabled"
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model
    direct=importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if not direct or CODE_COMMIT not in direct:raise ValueError("Unpinned model code")
    torch.set_num_threads(4);torch.manual_seed(20261025)
    config=read(args.model/"config.json")["config"];config["encoder"]["params"]["weights"]=None
    model=ChangeOS(config).eval();missing,unexpected=load_model(model,str(args.model/"model.safetensors"),strict=True)
    if missing or unexpected:raise ValueError("Strict load failed")
    model.requires_grad_(False);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device);sync(device)
    for name in ["cache","calls","scene_records"]:(args.out/name).mkdir(exist_ok=True)
    index=0;new_calls=0;start=time.perf_counter()
    def forward(job,pair,mode,block=None,expected_tensor=None):
        nonlocal index,new_calls
        index+=1
        if index>74:raise ValueError("Call cap exceeded")
        tensor=prepare_tensor(*pair,"native");digest=sha(tensor.numpy().tobytes())
        if expected_tensor is not None and expected_tensor!=digest:raise ValueError("Input differs from preflight")
        identity={"index":index,"job":job,"input_mode":mode,"block_id":block,"input_tensor_sha256":digest,"input_tensor_shape":list(tensor.shape)}
        file=args.out/"calls"/f"call_{index:03d}.json";cache=args.out/"cache"/(job+".npz")
        if file.exists():
            row=read(file)
            if row["status"]!="COMPLETE" or any(row.get(k)!=v for k,v in identity.items()):raise ValueError("Unfinished or changed reservation")
            return row,checked_mask(Path(row["cache_file"]),row["cache_sha256"],(128,128) if mode=="positive_control" else (256,256))
        freeze(file,{**identity,"status":"RESERVED"});t=time.perf_counter()
        with torch.inference_mode():output=model(tensor.to(device));sync(device);summary,mask=predict_summary(output)
        new_calls+=1
        if mode!="positive_control":
            if canonical_summary(mask)!={k:summary[k] for k in ["predicted_severe_pixels","declared_target"]}:raise ValueError("Canonical summary differs")
        np.savez_compressed(cache,severe_mask=mask)
        row={**identity,"status":"COMPLETE",**summary,"elapsed_seconds":time.perf_counter()-t,"cache_file":str(cache.resolve()),"cache_sha256":sha(cache.read_bytes())}
        atomic_json(file,row);del tensor,output;return row,mask
    controls=[]
    for old in protocol["positive_controls"]:
        pair=[]
        for image in old["images"]:
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("Control image changed")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        record,mask=forward(old["sample_id"]+"_positive_control",crop_pair(*pair,old["cell_id"]),"positive_control",expected_tensor=old["prepared_tensor_sha256"])
        previous=checked_mask(Path(old["legacy_cache_file"]),old["legacy_cache_sha256"],(128,128));different=int((mask!=previous).sum())
        controls.append({"observation_id":old["observation_id"],"exact_mask_equal":different==0,"different_pixels_from_legacy":different,**record})
        atomic_json(args.out/"positive_control_results.json",controls)
        if different:raise ValueError("Control mismatch; stop before blurred inference")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};prepared={(r["sample_id"],r["block_id"]):r for r in protocol["prepared_inputs"]};observations=[]
    for sid in protocol["selected_ids"]:
        pair=read_pair(sid,images);rows=[]
        for block in BLOCKS:
            raw=tuple(im.crop(block_bounds(block)) for im in pair);degraded=blurred_inputs(raw);expected=prepared[sid,block]
            if [sha(im.tobytes()) for im in raw]!=expected["original_rgb_sha256"] or [sha(im.tobytes()) for im in degraded]!=expected["degraded_rgb_sha256"]:raise ValueError("Selected RGB differs")
            record,_=forward(sid+"_"+block+"_blurred256",degraded,QUALITIES[1],block,expected["input_tensor_sha256"])
            rows.append({"sample_id":sid,"event":images[sid,"pre"]["event"],**record})
        atomic_json(args.out/"scene_records"/(sid+".json"),rows);observations.extend(rows)
        print(json.dumps({"scene":sid,"scenes_complete":len(observations)//4,"completed_or_reused_calls":index,"device":str(device)}),flush=True)
    if index!=74 or len(observations)!=72:raise ValueError("Incomplete forward inventory")
    atomic_json(args.out/"observations.json",observations)
    atomic_json(args.out/"inference_report.json",{"status":"COMPLETE","total_forward_calls":74,"new_calls_this_invocation":new_calls,"native128_control_calls":2,"blurred256_calls":72,
        "device":str(device),"loop_seconds":time.perf_counter()-start,"new_network_bytes":0,"new_fitting":False,"paid_gpu":False,
        "reference_pixels_decoded_by_prediction":False,"new_search_runs":0,"observations_sha256":sha((args.out/"observations.json").read_bytes())})


def evaluate(args,protocol):
    controls=read(args.out/"positive_control_results.json")
    if len(controls)!=2 or not all(r["exact_mask_equal"] for r in controls):raise ValueError("Controls incomplete")
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest changed")
    refs={r["sample_id"]:r for r in read(args.area/"reference_manifest.json")};observations=read(args.out/"observations.json")
    if len(observations)!=72 or len({(r["sample_id"],r["block_id"]) for r in observations})!=72:raise ValueError("Blur cache incomplete")
    native={(r["sample_id"],r["block_id"]):r for r in protocol["previous_native256_observations"]};blur={(r["sample_id"],r["block_id"]):r for r in observations}
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};free_rows={r["sample_id"]:r for r in protocol["previous_coarse_observations"]}
    scores=[];pairs=[];block_scores=[];feedback=[]
    for sid in protocol["selected_ids"]:
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference changed")
        label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3;rgb=read_pair(sid,images)
        domains={"all_published_pixels":np.ones_like(target),"both_dates_nonzero":np.any(np.asarray(rgb[0])>0,axis=2)&np.any(np.asarray(rgb[1])>0,axis=2)}
        masks={quality:{} for quality in QUALITIES};free=checked_mask(Path(free_rows[sid]["cache_file"]),free_rows[sid]["cache_sha256"],(512,512))
        for block in BLOCKS:
            for quality,lookup in [(QUALITIES[0],native),(QUALITIES[1],blur)]:
                row=lookup[sid,block];masks[quality][block]=checked_mask(Path(row["cache_file"]),row["cache_sha256"],(256,256))
            feedback.append({"sample_id":sid,"event":ref["event"],"block_id":block,"native_declared_target":canonical_summary(masks[QUALITIES[0]][block])["declared_target"],"blurred_declared_target":canonical_summary(masks[QUALITIES[1]][block])["declared_target"]})
        native_full=mosaic(masks[QUALITIES[0]]);blur_full=mosaic(masks[QUALITIES[1]])
        for domain_name,domain in domains.items():
            rows,changes=measure_pair(native_full,blur_full,target,free,domain)
            for quality,metrics in rows.items():scores.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"input_mode":quality,**metrics})
            pairs.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,**changes})
            for block in BLOCKS:
                x,y,xx,yy=block_bounds(block);truth=target[y:yy,x:xx];allowed=domain[y:yy,x:xx];baseline=free[y:yy,x:xx]
                for quality in QUALITIES:
                    pred=masks[quality][block];union=(pred|baseline)&allowed;tp=int((union&truth).sum());fp=int((union&~truth).sum());n=int((truth&allowed).sum())
                    block_scores.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"block_id":block,"input_mode":quality,
                        "reference_pixels":n,"covered_reference_pixels":tp,"reference_false_positive_pixels":fp,"predicted_pixels":int(union.sum()),
                        "area_recall":tp/n if n else None,"reference_precision":tp/(tp+fp) if tp+fp else None})
    for name,rows in [("scene_scores.json",scores),("paired_changes.json",pairs),("block_scores.json",block_scores),("block_feedback.json",feedback)]:atomic_json(args.out/name,rows)
    summary=quality_summary(scores);atomic_json(args.out/"summary.json",summary);atomic_json(args.out/"next_step_decision.json",measurement_decision(summary))
    print(json.dumps({"scene_domain_scores":len(scores),"paired_changes":len(pairs),"block_domain_scores":len(block_scores),"decision":read(args.out/"next_step_decision.json")["decision"]}))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["footprint","area","rgb","model","out"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=["plan","predict","evaluate"],required=True);args=p.parse_args();args.out.mkdir(exist_ok=True)
    if (args.out/"completion_receipt.json").exists():raise ValueError("Completed output frozen; use a new audit output")
    protocol=plan(args)
    if args.stage=="predict":predict(args,protocol)
    elif args.stage=="evaluate":evaluate(args,protocol)
    else:print(json.dumps({"planned_forward_calls":74,"real_blur_tensors_matching_frozen_helper":72,"preflight_tests":314,"new_inference":0}))


if __name__=="__main__":main()
