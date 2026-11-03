"""Two old positive controls and18 offline full512px native predictions."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS
from external_context_core import select_controls,stitch_native,overlap_metrics,paired_changes,summarize
from extract_external_area_masks import verify_parent,decode_reference
from plan_search_v2 import freeze
from run_changeos_pilot import (MODEL_ID,MODEL_REVISION,WEIGHTS_SHA256,CONFIG_SHA256,CODE_COMMIT,
    crop_pair,prepare_tensor,predict_summary,sync)


def checked_mask(path,digest,shape):
    if sha(path.read_bytes())!=digest:raise ValueError("Prediction cache changed")
    with np.load(path,allow_pickle=False) as cache:mask=cache["severe_mask"]
    if mask.dtype!=np.bool_ or mask.shape!=shape:raise ValueError("Canonical boolean mask shape differs")
    return mask


def plan(args):
    area_hash=verify_parent(args.area);rgb_hash=verify_parent(args.rgb)
    old_protocol=json.loads((args.legacy/"perception_protocol.json").read_text())
    old_observations=json.loads((args.legacy/"observations.json").read_text())
    by_old={r["sample_id"]:r for r in old_protocol["samples"]};controls=[]
    for old in select_controls(old_observations):
        sample=by_old[old["sample_id"]];pair=[];source_images=[]
        raw_root=Path(sample["pre_image"]).parents[2];manifest_path=raw_root/"manifest.json"
        if sha(manifest_path.read_bytes())!=old_protocol["manifest_sha256"]:raise ValueError("Legacy image manifest changed")
        expected={str((raw_root/r["path"]).resolve()):r["sha256"] for r in json.loads(manifest_path.read_text())["files"]}
        for mode in ["pre","post"]:
            path=Path(sample[mode+"_image"])
            if sha(path.read_bytes())!=expected[str(path.resolve())]:raise ValueError("Legacy RGB changed")
            with Image.open(path) as image:pair.append(image.convert("RGB"))
            source_images.append({"mode":mode,"path":str(path.resolve()),"sha256":sha(path.read_bytes())})
        patch=crop_pair(*pair,old["cell_id"])
        tensor=prepare_tensor(*patch,"native")
        reference=checked_mask(Path(old["cache_file"]),old["cache_sha256"],(128,128))
        if int(reference.sum())!=old["predicted_severe_pixels"] or int(reference.sum())<4:raise ValueError("Invalid legacy positive control")
        controls.append({"observation_id":old["observation_id"],"sample_id":old["sample_id"],"event":old["event"],"cell_id":old["cell_id"],
            "images":source_images,"legacy_cache_file":old["cache_file"],"legacy_cache_sha256":old["cache_sha256"],
            "legacy_severe_pixel_sha256":sha(reference.tobytes()),"legacy_severe_pixels":int(reference.sum()),
            "prepared_tensor_sha256":sha(tensor.numpy().tobytes()),"prepared_tensor_shape":list(tensor.shape)})
    rgb_rows=json.loads((args.rgb/"rgb_manifest.json").read_text());images={(r["sample_id"],r["mode"]):r for r in rgb_rows}
    native=[r for r in json.loads((args.area/"observations.json").read_text()) if r["mode"]=="native"]
    selected=sorted({r["sample_id"] for r in native})
    if len(selected)!=18 or len(native)!=288:raise ValueError("Prescribed18 native mosaics required")
    for sid in selected:
        if {r["cell_id"] for r in native if r["sample_id"]==sid}!=set(CELLS):raise ValueError("Incomplete16cell mosaic")
        for mode in ["pre","post"]:
            row=images[sid,mode]
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("Fixed external RGB changed")
    for name,expected in [("model.safetensors",WEIGHTS_SHA256),("config.json",CONFIG_SHA256)]:
        if sha((args.model/name).read_bytes())!=expected:raise ValueError("Model/config changed")
    source_names=["run_external_context_diagnostic.py","external_context_core.py","run_changeos_pilot.py","external_area_core.py","extract_external_area_masks.py"]
    protocol={"stage":"OFFLINE_NATIVE_INPUT_CONTEXT_DIAGNOSTIC_ONLY","area_completion_sha256":area_hash,"rgb_completion_sha256":rgb_hash,
        "legacy_protocol_sha256":sha((args.legacy/"perception_protocol.json").read_bytes()),"legacy_observations_sha256":sha((args.legacy/"observations.json").read_bytes()),
        "control_selection":"For each Harvey/SantaRosa, minimalSHA256(context-control-v1|20261026|observation_id) among previously cached native declared positives. Pipeline regression controls, not held-out outcomes.",
        "positive_controls":controls,"selected_ids":selected,"rgb_inputs":rgb_rows,"previous_native_observations":native,
        "reference_manifest_sha256":sha((args.area/"reference_manifest.json").read_bytes()),
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,"weights_sha256":WEIGHTS_SHA256,"config_sha256":CONFIG_SHA256,"code_commit":CODE_COMMIT,
        "source_code_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in source_names},
        "max_new_forward_calls":20,"positive_controls_exact_mask_equality_required":True,"whole_scene_shape":[512,512],
        "comparison":"Same native512px RGB content presented in one512px call vs16independent cached128px calls, all18fixed pairs retained.",
        "cost_scope":"Whole512px input includes unpurchased high-resolution context at budget8; offline full-information diagnostic, no search-policy or cost-saving claim.",
        "reference_scope":"Published EBD semantic pixels only; not human or geographic validation; old per-instance matching not used.",
        "domains":["all_published_pixels","both_dates_nonzero"],"reference_contents_read_by_prediction":False,
        "no_parameter_fitting_or_paid_gpu_or_new_download":True,
        "followup_rule":"Recommend a bounded input-scaling/free-coarse-context diagnostic only if whole-scene overlap recovers a previously zero-overlap event, totalcovered reference pixels increase, and pooled reference precision does not decrease. This is an engineering screening rule, not statistical validation or a method contribution."}
    freeze(args.out/"protocol.json",protocol);return protocol


def predict(args,protocol):
    for key in ["HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"]:os.environ[key]="1"
    os.environ["WANDB_MODE"]="disabled"
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model
    direct=importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if not direct or CODE_COMMIT not in direct:raise ValueError("Model code not pinned")
    torch.set_num_threads(4);torch.manual_seed(20261025)
    config=json.loads((args.model/"config.json").read_text())["config"];config["encoder"]["params"]["weights"]=None
    model=ChangeOS(config).eval();missing,unexpected=load_model(model,str(args.model/"model.safetensors"),strict=True)
    if missing or unexpected:raise ValueError("Strict checkpoint load failed")
    model.requires_grad_(False);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device);sync(device)
    (args.out/"cache").mkdir(exist_ok=True);(args.out/"scene_records").mkdir(exist_ok=True)
    call_count=0;control_rows=[]
    for control in protocol["positive_controls"]:
        pair=[]
        for row in control["images"]:
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("Control RGB changed")
            with Image.open(row["path"]) as im:pair.append(im.convert("RGB"))
        tensor=prepare_tensor(*crop_pair(*pair,control["cell_id"]),"native")
        if sha(tensor.numpy().tobytes())!=control["prepared_tensor_sha256"]:raise ValueError("Control tensor changed")
        t=time.perf_counter()
        with torch.inference_mode():output=model(tensor.to(device));sync(device);summary,mask=predict_summary(output)
        call_count+=1;old=checked_mask(Path(control["legacy_cache_file"]),control["legacy_cache_sha256"],(128,128))
        different=int((mask!=old).sum());path=args.out/"cache"/(control["sample_id"]+"_positive_control.npz");np.savez_compressed(path,severe_mask=mask)
        control_rows.append({"observation_id":control["observation_id"],"sample_id":control["sample_id"],"event":control["event"],"different_pixels_from_legacy":different,
            "exact_mask_equal":different==0,"elapsed_seconds":time.perf_counter()-t,**summary,"cache_file":str(path.resolve()),"cache_sha256":sha(path.read_bytes())})
        atomic_json(args.out/"positive_control_results.json",control_rows)
        if different:
            atomic_json(args.out/"inference_report.json",{"status":"STOPPED_LEGACY_CONTROL_MISMATCH","new_forward_calls":call_count,"external_whole_scenes":0,"paid_gpu":False})
            raise ValueError("Legacy positive mask differs; whole-scene diagnostic stopped")
        del output,tensor
    observations=[];inputs={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};start=time.perf_counter();new_whole=0
    for sid in protocol["selected_ids"]:
        record_path=args.out/"scene_records"/(sid+".json")
        if record_path.exists():
            record=json.loads(record_path.read_text());checked_mask(Path(record["cache_file"]),record["cache_sha256"],(512,512))
        else:
            pair=[]
            for mode in ["pre","post"]:
                row=inputs[sid,mode]
                if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("External RGB changed")
                with Image.open(row["path"]) as im:pair.append(im.convert("RGB"))
            tensor=prepare_tensor(*pair,"native")
            if tuple(tensor.shape)!=(1,6,512,512):raise ValueError("Wrong whole-scene tensor")
            tensor_sha=sha(tensor.numpy().tobytes());t=time.perf_counter()
            with torch.inference_mode():output=model(tensor.to(device));sync(device);summary,mask=predict_summary(output)
            call_count+=1;new_whole+=1
            if mask.dtype!=np.bool_ or mask.shape!=(512,512):raise ValueError("Wrong full-scene mask shape")
            path=args.out/"cache"/(sid+"_whole_native.npz");np.savez_compressed(path,severe_mask=mask)
            record={"sample_id":sid,"event":inputs[sid,"pre"]["event"],"input_mode":"whole512_native",**summary,
                "input_tensor_sha256":tensor_sha,"elapsed_seconds":time.perf_counter()-t,"cache_file":str(path.resolve()),"cache_sha256":sha(path.read_bytes())}
            atomic_json(record_path,record);del output,tensor
        observations.append(record);print(json.dumps({"sample_id":sid,"whole_scenes_complete":len(observations),"device":str(device)}),flush=True)
    atomic_json(args.out/"whole_observations.json",observations)
    atomic_json(args.out/"inference_report.json",{"status":"COMPLETE","positive_control_calls":2,"whole_calls_this_invocation":new_whole,"new_forward_calls_this_invocation":call_count,
        "whole_scenes":18,"device":str(device),"whole_loop_seconds_this_invocation":time.perf_counter()-start,"reference_contents_read_by_prediction":False,
        "paid_gpu":False,"new_fitting":False,"observations_sha256":sha((args.out/"whole_observations.json").read_bytes())})


def evaluate(args,protocol):
    controls=json.loads((args.out/"positive_control_results.json").read_text())
    if len(controls)!=2 or not all(r["exact_mask_equal"] for r in controls):raise ValueError("Controls not passed")
    refs=json.loads((args.area/"reference_manifest.json").read_text())
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest changed")
    references={r["sample_id"]:r for r in refs};whole={r["sample_id"]:r for r in json.loads((args.out/"whole_observations.json").read_text())}
    if set(whole)!=set(protocol["selected_ids"]):raise ValueError("Incomplete whole-scene predictions")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};scores=[];pairs=[]
    for sid in protocol["selected_ids"]:
        ref=references[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference PNG changed")
        label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3
        masks={r["cell_id"]:checked_mask(Path(r["cache_file"]),r["cache_sha256"],(128,128)) for r in protocol["previous_native_observations"] if r["sample_id"]==sid}
        mosaic=stitch_native(masks);full=checked_mask(Path(whole[sid]["cache_file"]),whole[sid]["cache_sha256"],(512,512))
        pair=[]
        for mode in ["pre","post"]:
            with Image.open(images[sid,mode]["path"]) as im:pair.append(np.asarray(im.convert("RGB")))
        valid=np.any(pair[0]>0,axis=2)&np.any(pair[1]>0,axis=2)
        for name,domain in [("all_published_pixels",np.ones_like(target)),("both_dates_nonzero",valid)]:
            for input_mode,mask in [("crop128_mosaic_native",mosaic),("whole512_native",full)]:
                scores.append({"sample_id":sid,"event":ref["event"],"domain":name,"input_mode":input_mode,**overlap_metrics(mask,target,domain)})
            pairs.append({"sample_id":sid,"event":ref["event"],"domain":name,**paired_changes(mosaic,full,target,domain)})
    atomic_json(args.out/"scene_scores.json",scores);atomic_json(args.out/"paired_changes.json",pairs);atomic_json(args.out/"summary.json",summarize(scores))
    print(json.dumps({"scene_domain_scores":len(scores),"new_search_trajectories":0}))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["area","rgb","legacy","model","out"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=["plan","predict","evaluate"],required=True);args=p.parse_args()
    protocol=plan(args)
    if args.stage=="predict":predict(args,protocol)
    elif args.stage=="evaluate":evaluate(args,protocol)
    else:print(json.dumps({"planned_controls":2,"planned_whole_native":18,"new_inference":0}))


if __name__=="__main__":main()
