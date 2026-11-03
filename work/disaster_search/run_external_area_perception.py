"""Fixed offline RGB-only ChangeOS inference on18x16x2 prespecified crops."""
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS,crop_cell,public_features
from plan_search_v2 import freeze
from run_changeos_pilot import (MODEL_ID,MODEL_REVISION,WEIGHTS_SHA256,CONFIG_SHA256,CODE_COMMIT,
    prepare_tensor,predict_summary,sync)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["rgb","root","model"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execute",action="store_true");args=p.parse_args()
    rgb=json.loads((args.rgb/"rgb_manifest.json").read_text());gate=json.loads((args.root/"reference_interface_gate.json").read_text())
    if not gate["technical_interface_passed"] or sha((args.rgb/"rgb_manifest.json").read_bytes())!=gate["rgb_manifest_sha256"]:raise ValueError("Technical interface gate changed")
    selected=gate["selected_ids"];by_id={(r["sample_id"],r["mode"]):r for r in rgb}
    if len(selected)!=18 or set(by_id)!={(s,m) for s in selected for m in ["pre","post"]}:raise ValueError("Exactly18 fixed RGB pairs required")
    for r in rgb:
        if sha(Path(r["path"]).read_bytes())!=r["sha256"]:raise ValueError("RGB input changed")
    if sha((args.model/"model.safetensors").read_bytes())!=WEIGHTS_SHA256 or sha((args.model/"config.json").read_bytes())!=CONFIG_SHA256:raise ValueError("Fixed model/config changed")
    freeze(args.root/"perception_protocol.json",{"stage":"18_PAIR_RGB_ONLY_4X4_SENSOR_DIAGNOSTIC","selected_ids":selected,"cells":list(CELLS),"modes":["native","blurred"],"observations":576,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,"weights_sha256":WEIGHTS_SHA256,"config_sha256":CONFIG_SHA256,"code_commit":CODE_COMMIT,
        "runner_sha256":sha(Path(__file__).read_bytes()),"geometry_sha256":sha(Path(__file__).with_name("external_area_core.py").read_bytes()),
        "shared_rgb_helper_sha256":sha(Path(__file__).with_name("run_changeos_pilot.py").read_bytes()),
        "rgb_manifest_sha256":sha((args.rgb/"rgb_manifest.json").read_bytes()),"reference_gate_sha256":sha((args.root/"reference_interface_gate.json").read_bytes()),
        "native_crop":[128,128],"free_coarse":[64,64],"degradation":"Unchanged Gaussian1.5, BOX/2, bilinear restoration, both dates",
        "prediction_input":"RGB crops only; no reference file contents or target counts loaded by this runner",
        "fixed_observation_boolean":"At least4 predicted severe pixels; pixel-area evaluator still counts1..3 pixels without an instance threshold",
        "no_training_or_paid_gpu":True})
    if not args.execute:print(json.dumps({"frozen_observations":576,"inference_run":False}));return
    for key in ["HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"]:os.environ[key]="1"
    os.environ["WANDB_MODE"]="disabled"
    import numpy as np
    from PIL import Image
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model
    direct_url=importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if not direct_url or CODE_COMMIT not in direct_url:raise ValueError("Installed model code not pinned")
    torch.set_num_threads(4);torch.manual_seed(20261025)
    config=json.loads((args.model/"config.json").read_text())["config"];config["encoder"]["params"]["weights"]=None
    model=ChangeOS(config).eval();missing,unexpected=load_model(model,str(args.model/"model.safetensors"),strict=True)
    if missing or unexpected:raise ValueError("Checkpoint load mismatch")
    model.requires_grad_(False);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device);sync(device)
    (args.root/"prediction_cache").mkdir(exist_ok=True);(args.root/"prediction_records").mkdir(exist_ok=True)
    observations=[];features={};start=time.perf_counter();new_count=0
    for sid in selected:
        pre,post=[Image.open(by_id[sid,m]["path"]).convert("RGB") for m in ["pre","post"]]
        features[sid]=public_features(pre,post);record_file=args.root/"prediction_records"/(sid+".json")
        if record_file.exists():
            records=json.loads(record_file.read_text())
            if len(records)!=32 or {(r["cell_id"],r["mode"]) for r in records}!={(c,m) for c in CELLS for m in ["native","blurred"]}:raise ValueError("Incomplete resume scene")
            if any(sha(Path(r["cache_file"]).read_bytes())!=r["cache_sha256"] for r in records):raise ValueError("Resume mask changed")
        else:
            records=[]
            for cell in CELLS:
                pair=crop_cell(pre,post,cell)
                for mode in ["native","blurred"]:
                    t=time.perf_counter();tensor=prepare_tensor(*pair,mode).to(device)
                    with torch.inference_mode():output=model(tensor);sync(device);summary,mask=predict_summary(output)
                    if mask.dtype!=np.bool_ or mask.shape!=(128,128):raise ValueError("Unexpected canonical prediction mask")
                    path=args.root/"prediction_cache"/f"{sid}_{cell}_{mode}.npz";np.savez_compressed(path,severe_mask=mask)
                    records.append({"sample_id":sid,"event":by_id[sid,"pre"]["event"],"cell_id":cell,"mode":mode,
                        "observation_id":f"{sid}:{cell}:{mode}",**summary,"elapsed_s":time.perf_counter()-t,
                        "cache_file":str(path.resolve()),"cache_sha256":sha(path.read_bytes())})
                    new_count+=1;del output,tensor
            atomic_json(record_file,records)
        observations.extend(records);print(json.dumps({"sample_id":sid,"observations":len(observations),"device":str(device)}),flush=True)
    if len(observations)!=576:raise ValueError("Incomplete fixed inference")
    atomic_json(args.root/"observations.json",observations);atomic_json(args.root/"public_features.json",features)
    atomic_json(args.root/"inference_report.json",{"observations":576,"new_forward_calls_this_invocation":new_count,"scenes":18,"device":str(device),
        "loop_seconds_this_invocation":time.perf_counter()-start,"median_observation_seconds":statistics.median(r["elapsed_s"] for r in observations),
        "reference_contents_read_by_predict":False,"controller_training":False,"paid_gpu":False,
        "observations_sha256":sha((args.root/"observations.json").read_bytes())})


if __name__=="__main__":main()
