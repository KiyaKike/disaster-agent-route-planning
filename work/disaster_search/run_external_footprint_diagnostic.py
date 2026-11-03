"""Fixed146-call native acquisition-area comparison with immutable code plan."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_footprint_core import NEW_MODES,NATIVE256,CONTEXT256,BLOCKS,block_bounds,compose_block,canonical_block
from extract_external_area_masks import verify_parent
from plan_search_v2 import freeze
from run_changeos_pilot import crop_pair,prepare_tensor,predict_summary,sync,CODE_COMMIT
from run_external_context_diagnostic import checked_mask
from evaluate_external_footprint import evaluate


def read(path):return json.loads(Path(path).read_text())


def plan(args):
    hashes={name:verify_parent(getattr(args,name)) for name in ["information","area","rgb"]}
    prospect=read(args.information/"prospective_footprint_diagnostic.json");previous=read(args.information/"protocol.json")
    if prospect["selected_ids"]!=previous["selected_ids"] or prospect["max_new_forward_calls"]!=146:raise ValueError("Prospective inventory differs")
    for name in ["protocol","evidence","decision"]:
        file={"protocol":"protocol.json","evidence":"evidence_summary.json","decision":"next_step_decision.json"}[name]
        if sha((args.information/file).read_bytes())!=prospect["basis_"+name+"_sha256"]:raise ValueError("Prospective basis changed")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((args.model/name).read_bytes())!=prospect["checkpoint"][key]:raise ValueError("Checkpoint changed")
    log=(args.out/"test_log.txt").read_text()
    if "Ran 302 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("302-test preflight required")
    names=["run_external_footprint_diagnostic.py","evaluate_external_footprint.py","external_footprint_core.py","test_external_footprint.py",
        "run_changeos_pilot.py","run_external_context_diagnostic.py","external_area_core.py","external_information_core.py","extract_external_area_masks.py","plan_search_v2.py","build_xbd_mini.py"]
    coarse=[r for r in read(args.information/"observations.json") if r["input_mode"]=="free_coarse512_only"]
    if len(coarse)!=18:raise ValueError("Free baseline incomplete")
    protocol={"stage":"NATIVE_AREA_COST_FOOTPRINT_DIAGNOSTIC","client_report_date":"2026-10-26","parent_receipt_sha256":hashes,
        "prospective_protocol":prospect,"prospective_protocol_sha256":sha((args.information/"prospective_footprint_diagnostic.json").read_bytes()),
        "selected_ids":prospect["selected_ids"],"checkpoint":prospect["checkpoint"],"model_directory":str(args.model.resolve()),
        "positive_controls":previous["positive_controls"],"rgb_inputs":previous["rgb_inputs"],"previous_native_observations":previous["previous_native_observations"],
        "previous_whole_observations":previous["offline_whole_observations"],"previous_coarse_observations":coarse,
        "reference_manifest_sha256":sha((args.area/"reference_manifest.json").read_bytes()),"public_features_sha256":sha((args.area/"public_features.json").read_bytes()),
        "max_new_forward_calls":146,"call_inventory":{"locked_controls":2,"native256_contiguous":72,"free_coarse512_plus_native256":72},
        "forward_order":"2 locked128 controls, then18 sorted scenes, native256 then coarse+256, each4 row-major nonoverlapping blocks; no warmup.",
        "canonical_boolean":"At least4 severe pixels within canonical256 paid footprint; no outside-canvas output enters decisions or rewards.",
        "cost_precision":"Costs count purchased native128-pixel-equivalent area; same native-only quality. Random precision = ratio of expected pooled counts, not expected trajectory precision.",
        "primary_screen":"native256 only, random exact at budget8, all published pixels; keep secondary separate and retain every budget/event/domain.",
        "source_code_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in names},"preflight_tests":302,"test_log_sha256":sha((args.out/"test_log.txt").read_bytes()),
        "reference_pixels_decoded_by_prediction":False,"new_fitting":False,"paid_gpu":False,"new_download":False}
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
    if missing or unexpected:raise ValueError("Strict checkpoint load failed")
    model.requires_grad_(False);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device);sync(device)
    for folder in ["cache","calls","scene_records"]:(args.out/folder).mkdir(exist_ok=True)
    index=0;new_calls=0;start=time.perf_counter()
    def forward(job,pair,mode,block=None):
        nonlocal index,new_calls
        index+=1
        if index>146:raise ValueError("Forward cap exceeded")
        tensor=prepare_tensor(*pair,"native");path=args.out/"cache"/(job+".npz");receipt=args.out/"calls"/f"call_{index:03d}.json"
        identity={"index":index,"job":job,"input_mode":mode,"block_id":block,"input_tensor_shape":list(tensor.shape),"input_tensor_sha256":sha(tensor.numpy().tobytes())}
        if receipt.exists():
            record=read(receipt)
            if record["status"]!="COMPLETE" or any(record.get(k)!=v for k,v in identity.items()):raise ValueError("Unfinished or changed call reservation")
            mask=checked_mask(Path(record["cache_file"]),record["cache_sha256"],(128,128) if mode=="positive_control" else (256,256))
            return record,mask
        freeze(receipt,{**identity,"status":"RESERVED"});t=time.perf_counter()
        with torch.inference_mode():output=model(tensor.to(device));sync(device);summary,raw=predict_summary(output)
        new_calls+=1;mask=raw if mode=="positive_control" else canonical_block(raw,mode,block)
        np.savez_compressed(path,severe_mask=mask,raw_severe_mask=raw)
        record={**identity,"status":"COMPLETE","predicted_severe_pixels":int(mask.sum()),"declared_target":int(mask.sum())>=4,"raw_summary":summary,
            "cache_file":str(path.resolve()),"cache_sha256":sha(path.read_bytes()),"elapsed_seconds":time.perf_counter()-t}
        atomic_json(receipt,record);del tensor,output;return record,mask
    controls=[]
    for control in protocol["positive_controls"]:
        pair=[]
        for row in control["images"]:
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("Control RGB changed")
            with Image.open(row["path"]) as image:pair.append(image.convert("RGB"))
        patches=crop_pair(*pair,control["cell_id"])
        if sha(prepare_tensor(*patches,"native").numpy().tobytes())!=control["prepared_tensor_sha256"]:raise ValueError("Control tensor changed")
        record,mask=forward(control["sample_id"]+"_positive_control",patches,"positive_control")
        old=checked_mask(Path(control["legacy_cache_file"]),control["legacy_cache_sha256"],(128,128));different=int((old!=mask).sum())
        controls.append({"observation_id":control["observation_id"],"exact_mask_equal":different==0,"different_pixels_from_legacy":different,**record});atomic_json(args.out/"positive_control_results.json",controls)
        if different:raise ValueError("Control differs; stop before new predictions")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};observations=[]
    for sid in protocol["selected_ids"]:
        pair=[]
        for date in ["pre","post"]:
            row=images[sid,date]
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("RGB changed")
            with Image.open(row["path"]) as im:pair.append(im.convert("RGB"))
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX) for im in pair);rows=[]
        for mode in NEW_MODES:
            for block in BLOCKS:
                patches=tuple(im.crop(block_bounds(block)) for im in pair)
                inputs=compose_block(mode,patches,block,coarse if mode==CONTEXT256 else None)
                record,_=forward(sid+"_"+block+"_"+mode,inputs,mode,block)
                rows.append({"sample_id":sid,"event":images[sid,"pre"]["event"],"purchased_rgb_sha256":[sha(im.tobytes()) for im in patches],
                    "coarse_rgb_sha256":[sha(im.tobytes()) for im in coarse] if mode==CONTEXT256 else None,**record})
        atomic_json(args.out/"scene_records"/(sid+".json"),rows);observations.extend(rows)
        print(json.dumps({"scene":sid,"scenes_complete":len(observations)//8,"completed_or_reused_calls":index,"device":str(device)}),flush=True)
    if index!=146 or len(observations)!=144:raise ValueError("Inventory incomplete")
    atomic_json(args.out/"observations.json",observations)
    atomic_json(args.out/"inference_report.json",{"status":"COMPLETE","total_forward_calls":146,"new_calls_this_invocation":new_calls,"positive_control_calls":2,
        "native256_calls":72,"coarse_plus256_calls":72,"device":str(device),"loop_seconds":time.perf_counter()-start,
        "new_network_bytes":0,"paid_gpu":False,"new_fitting":False,"reference_pixels_decoded_by_prediction":False,
        "observations_sha256":sha((args.out/"observations.json").read_bytes())})


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["information","area","rgb","model","out"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=["plan","predict","evaluate"],required=True);args=p.parse_args();args.out.mkdir(exist_ok=True)
    if (args.out/"completion_receipt.json").exists():raise ValueError("Completed output frozen; use new audit directory")
    protocol=plan(args)
    if args.stage=="predict":predict(args,protocol)
    elif args.stage=="evaluate":evaluate(args.out,args.area,protocol)
    else:print(json.dumps({"planned_calls":146,"preflight_tests":302,"new_inference":0}))


if __name__=="__main__":main()
