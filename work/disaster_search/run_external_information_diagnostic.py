"""Fixed18 native-only input-information diagnostic; no search or fitting."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from build_xbd_mini import atomic_json,sha
from external_area_core import CELLS,crop_cell
from external_context_core import stitch_native,overlap_metrics,paired_changes,summarize
from external_information_core import PATCH_MODES,COARSE_MODE,compose_inputs,coarse_canvas,canonical_mask,local_summary
from extract_external_area_masks import verify_parent,decode_reference
from plan_search_v2 import freeze
from run_changeos_pilot import crop_pair,prepare_tensor,predict_summary,sync,CODE_COMMIT
from run_external_context_diagnostic import checked_mask


def read_json(path):return json.loads(Path(path).read_text())


def plan(args):
    parents={name:verify_parent(getattr(args,name)) for name in ["context","area","rgb"]}
    prior=read_json(args.context/"protocol.json");prospect=read_json(args.context/"prospective_information_diagnostic.json")
    if prospect["fixed_ids"]!=prior["selected_ids"] or prospect["max_new_forward_calls"]!=884:raise ValueError("Fixed cohort/call limit changed")
    if sha((args.context/"protocol.json").read_bytes())!=prospect["basis_protocol_sha256"] or sha((args.context/"evidence_summary.json").read_bytes())!=prospect["basis_evidence_sha256"]:raise ValueError("Prospective basis differs")
    for name,key in [("model.safetensors","weights_sha256"),("config.json","config_sha256")]:
        if sha((args.model/name).read_bytes())!=prospect["checkpoint"][key]:raise ValueError("Checkpoint changed")
    for image in prior["rgb_inputs"]:
        if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("RGB changed")
    log=(args.out/"test_log.txt").read_text()
    if "Ran 284 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("284-test preflight required")
    names=["run_external_information_diagnostic.py","external_information_core.py","test_external_information.py",
        "external_area_core.py","external_context_core.py","run_external_context_diagnostic.py","run_changeos_pilot.py",
        "extract_external_area_masks.py","build_xbd_mini.py","plan_search_v2.py"]
    protocol={"stage":"NATIVE_ONLY_INFORMATION_BOUNDED_INPUT_DIAGNOSTIC","client_report_date":"2026-10-26",
        "parent_receipt_sha256":parents,"prospective_protocol_sha256":sha((args.context/"prospective_information_diagnostic.json").read_bytes()),
        "prospective_protocol":prospect,"selected_ids":prior["selected_ids"],"positive_controls":prior["positive_controls"],
        "rgb_inputs":prior["rgb_inputs"],"previous_native_observations":prior["previous_native_observations"],
        "offline_whole_observations":read_json(args.context/"whole_observations.json"),
        "reference_manifest_sha256":sha((args.area/"reference_manifest.json").read_bytes()),
        "checkpoint":prospect["checkpoint"],"model_directory":str(args.model.resolve()),
        "call_inventory":{"positive_controls":2,"patch128_to512":288,"blank512_plus_native_cell":288,"free_coarse512_plus_native_cell":288,"free_coarse512_only":18},
        "max_new_forward_calls":884,"forward_order":"Two locked controls, then sorted scene IDs; within scene coarse-only first, then3 declared patch conditions in order, each16 row-major cells.",
        "raw_output_scope":"512 output retained privately for cropping audit; only canonical128 ROI and its >=4px declaration are a paid observation. Coarse-only baseline remains free information.",
        "coarse_recipe":"Each native512 RGB date resized64 BOX once, then512 BILINEAR. No high-resolution pixels outside selected patch enter compose_inputs.",
        "canonical_scale_recipe":"512 bool severe output reduced128 NEAREST; canonical >=4px declaration recomputed after reduction.",
        "coarse_increment_operationalization":"Adds reference coverage beyond coarse-only means strictly positive NET event-summed covered-reference difference in EACH reference-positive event. Also report uniquely gained, lost and preserved pixels.",
        "primary_domain":"all_published_pixels","secondary_domain":"both_dates_nonzero","search_policy_runs":0,
        "reference_pixels_decoded_by_prediction":False,"fit_enabled":False,"paid_gpu":False,"new_downloads":False,
        "source_code_sha256":{name:sha(Path(__file__).with_name(name).read_bytes()) for name in names},
        "preflight_test_log_sha256":sha((args.out/"test_log.txt").read_bytes()),"tests_passed_before_inference":284}
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
    config=read_json(args.model/"config.json")["config"];config["encoder"]["params"]["weights"]=None
    model=ChangeOS(config).eval();missing,unexpected=load_model(model,str(args.model/"model.safetensors"),strict=True)
    if missing or unexpected:raise ValueError("Strict checkpoint load failed")
    model.requires_grad_(False);device=torch.device("mps" if torch.backends.mps.is_available() else "cpu");model.to(device);sync(device)
    for folder in ["cache","calls","scene_records"]:(args.out/folder).mkdir(exist_ok=True)
    index=0;new_calls=0;start=time.perf_counter()

    def forward(job,input_pair,mode,cell=None):
        nonlocal index,new_calls
        index+=1
        if index>protocol["max_new_forward_calls"]:raise ValueError("Forward limit exceeded")
        tensor=prepare_tensor(*input_pair,"native");tensor_hash=sha(tensor.numpy().tobytes())
        path=args.out/"cache"/(job+".npz");record_path=args.out/"calls"/f"call_{index:04d}.json"
        identity={"index":index,"job":job,"input_mode":mode,"cell_id":cell,"input_tensor_shape":list(tensor.shape),"input_tensor_sha256":tensor_hash}
        if record_path.exists():
            saved=read_json(record_path)
            if saved["status"]!="COMPLETE" or any(saved.get(k)!=v for k,v in identity.items()):raise ValueError("Cannot replay unfinished or different call reservation")
            shape=(128,128) if mode in PATCH_MODES else (512,512) if mode==COARSE_MODE else (128,128)
            mask=checked_mask(Path(saved["cache_file"]),saved["cache_sha256"],shape)
            return saved,mask
        freeze(record_path,{**identity,"status":"RESERVED"})
        t=time.perf_counter()
        with torch.inference_mode():output=model(tensor.to(device));sync(device);full_summary,raw=predict_summary(output)
        new_calls+=1
        if mode in PATCH_MODES:
            mask=canonical_mask(raw,mode,cell);summary=local_summary(mask)
            np.savez_compressed(path,severe_mask=mask,raw_severe_mask=raw)
        else:
            mask=raw;summary={"predicted_severe_pixels":int(mask.sum()),"declared_target":int(mask.sum())>=4}
            np.savez_compressed(path,severe_mask=mask)
        record={**identity,"status":"COMPLETE",**summary,"full_output_summary":full_summary,
            "cache_file":str(path.resolve()),"cache_sha256":sha(path.read_bytes()),"elapsed_seconds":time.perf_counter()-t}
        atomic_json(record_path,record);del output,tensor
        return record,mask

    controls=[]
    for control in protocol["positive_controls"]:
        pair=[]
        for image in control["images"]:
            if sha(Path(image["path"]).read_bytes())!=image["sha256"]:raise ValueError("Control RGB changed")
            with Image.open(image["path"]) as im:pair.append(im.convert("RGB"))
        patches=crop_pair(*pair,control["cell_id"])
        if sha(prepare_tensor(*patches,"native").numpy().tobytes())!=control["prepared_tensor_sha256"]:raise ValueError("Control input changed")
        record,mask=forward(control["sample_id"]+"_positive_control",patches,"positive_control")
        old=checked_mask(Path(control["legacy_cache_file"]),control["legacy_cache_sha256"],(128,128));different=int((old!=mask).sum())
        controls.append({"observation_id":control["observation_id"],"exact_mask_equal":different==0,"different_pixels_from_legacy":different,**record})
        atomic_json(args.out/"positive_control_results.json",controls)
        if different:raise ValueError("Positive control mismatch; diagnostic stopped")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};observations=[]
    for sid in protocol["selected_ids"]:
        pair=[]
        for mode in ["pre","post"]:
            row=images[sid,mode]
            if sha(Path(row["path"]).read_bytes())!=row["sha256"]:raise ValueError("Scene RGB changed")
            with Image.open(row["path"]) as im:pair.append(im.convert("RGB"))
        coarse=tuple(im.resize((64,64),Image.Resampling.BOX) for im in pair)
        coarse_hashes=[sha(im.tobytes()) for im in coarse];rows=[]
        record,_=forward(sid+"_"+COARSE_MODE,coarse_canvas(coarse),COARSE_MODE)
        rows.append({"sample_id":sid,"event":images[sid,"pre"]["event"],"coarse_rgb_sha256":coarse_hashes,**record})
        for mode in PATCH_MODES:
            for cell in CELLS:
                patches=crop_cell(*pair,cell)
                inputs=compose_inputs(mode,patches,cell,coarse if mode==PATCH_MODES[2] else None)
                record,_=forward(sid+"_"+cell+"_"+mode,inputs,mode,cell)
                rows.append({"sample_id":sid,"event":images[sid,"pre"]["event"],"patch_rgb_sha256":[sha(im.tobytes()) for im in patches],
                    "coarse_rgb_sha256":coarse_hashes if mode==PATCH_MODES[2] else None,**record})
        atomic_json(args.out/"scene_records"/(sid+".json"),rows);observations.extend(rows)
        print(json.dumps({"scene":sid,"scenes_complete":len(observations)//49,"completed_or_reused_calls":index,"new_calls_this_invocation":new_calls,"device":str(device)}),flush=True)
    if index!=884 or len(observations)!=882:raise ValueError("Inventory incomplete")
    atomic_json(args.out/"observations.json",observations)
    atomic_json(args.out/"inference_report.json",{"status":"COMPLETE","total_forward_calls":884,"new_calls_this_invocation":new_calls,
        "positive_control_calls":2,"native_patch_calls":864,"coarse_only_calls":18,"observations":882,"device":str(device),
        "loop_seconds_this_invocation":time.perf_counter()-start,"reference_pixels_decoded_by_prediction":False,"new_fitting":False,
        "paid_gpu":False,"new_network_bytes":0,"new_search_trajectories":0,"observations_sha256":sha((args.out/"observations.json").read_bytes())})


def evaluate(args,protocol):
    if not all(r["exact_mask_equal"] for r in read_json(args.out/"positive_control_results.json")):raise ValueError("Controls failed")
    if sha((args.area/"reference_manifest.json").read_bytes())!=protocol["reference_manifest_sha256"]:raise ValueError("Reference manifest changed")
    refs={r["sample_id"]:r for r in read_json(args.area/"reference_manifest.json")}
    observations=read_json(args.out/"observations.json")
    if len(observations)!=882 or len({r["job"] for r in observations})!=882:raise ValueError("Wrong observation inventory")
    images={(r["sample_id"],r["mode"]):r for r in protocol["rgb_inputs"]};whole={r["sample_id"]:r for r in protocol["offline_whole_observations"]}
    scores=[];changes=[];comparisons=[("crop128_mosaic_native",m) for m in PATCH_MODES]+[(PATCH_MODES[1],PATCH_MODES[2]),(COARSE_MODE,PATCH_MODES[2])]
    for sid in protocol["selected_ids"]:
        ref=refs[sid]
        if sha(Path(ref["path"]).read_bytes())!=ref["sha256"]:raise ValueError("Reference PNG changed")
        label,_=decode_reference(Path(ref["path"]).read_bytes());target=label>=3
        masks={"crop128_mosaic_native":stitch_native({r["cell_id"]:checked_mask(Path(r["cache_file"]),r["cache_sha256"],(128,128)) for r in protocol["previous_native_observations"] if r["sample_id"]==sid}),
            "whole512_native":checked_mask(Path(whole[sid]["cache_file"]),whole[sid]["cache_sha256"],(512,512))}
        for mode in PATCH_MODES:
            masks[mode]=stitch_native({r["cell_id"]:checked_mask(Path(r["cache_file"]),r["cache_sha256"],(128,128)) for r in observations if r["sample_id"]==sid and r["input_mode"]==mode})
        coarse=next(r for r in observations if r["sample_id"]==sid and r["input_mode"]==COARSE_MODE)
        masks[COARSE_MODE]=checked_mask(Path(coarse["cache_file"]),coarse["cache_sha256"],(512,512))
        pair=[]
        for mode in ["pre","post"]:
            with Image.open(images[sid,mode]["path"]) as im:pair.append(np.asarray(im.convert("RGB")))
        valid=np.any(pair[0]>0,axis=2)&np.any(pair[1]>0,axis=2)
        for domain_name,domain in [("all_published_pixels",np.ones_like(target)),("both_dates_nonzero",valid)]:
            for mode,mask in masks.items():scores.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"input_mode":mode,**overlap_metrics(mask,target,domain)})
            for before,after in comparisons:changes.append({"sample_id":sid,"event":ref["event"],"domain":domain_name,"before":before,"after":after,**paired_changes(masks[before],masks[after],target,domain)})
    atomic_json(args.out/"scene_scores.json",scores);atomic_json(args.out/"paired_changes.json",changes);summary=summarize(scores)
    atomic_json(args.out/"summary.json",summary)
    gates=[]
    for domain in ["all_published_pixels","both_dates_nonzero"]:
        totals={r["input_mode"]:r for r in summary["totals"] if r["domain"]==domain}
        event={(r["event"],r["input_mode"]):r for r in summary["event_rows"] if r["domain"]==domain}
        positive=sorted({r["event"] for r in summary["event_rows"] if r["domain"]==domain and r["reference_pixels"]>0})
        main=totals[PATCH_MODES[2]];old=totals["crop128_mosaic_native"]
        checks={"both_reference_positive_events_have_overlap":len(positive)==2 and all(event[e,PATCH_MODES[2]]["covered_reference_pixels"]>0 for e in positive),
            "pooled_coverage_exceeds_cached_native128":main["covered_reference_pixels"]>old["covered_reference_pixels"],
            "pooled_reference_precision_not_lower":main["pooled_reference_precision"] is not None and main["pooled_reference_precision"]>=old["pooled_reference_precision"],
            "pooled_coverage_exceeds_blank_frame":main["covered_reference_pixels"]>totals[PATCH_MODES[1]]["covered_reference_pixels"],
            "positive_net_coverage_beyond_free_coarse_in_each_positive_event":len(positive)==2 and all(event[e,PATCH_MODES[2]]["covered_reference_pixels"]>event[e,COARSE_MODE]["covered_reference_pixels"] for e in positive)}
        gates.append({"domain":domain,"checks":checks,"passed":all(checks.values()),"reference_positive_events":positive})
    decision={"primary_domain":"all_published_pixels","domain_screens":gates,"primary_screen_passed":gates[0]["passed"],
        "decision":"REGISTER_MIXED_QUALITY_CACHE_EXTENSION" if gates[0]["passed"] else "STOP_CURRENT_FREE_COARSE_CONTEXT_CANDIDATE_BEFORE_SEARCH",
        "new_search_policy_selected":False,"new_fitting_enabled":False,"budget8_score_created":False,"independent_efficacy_claim":False}
    atomic_json(args.out/"next_step_decision.json",decision)
    print(json.dumps({"scene_domain_scores":len(scores),"paired_changes":len(changes),"decision":decision["decision"]}))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["context","area","rgb","model","out"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--stage",choices=["plan","predict","evaluate"],required=True);args=p.parse_args();args.out.mkdir(exist_ok=True)
    if (args.out/"completion_receipt.json").exists():raise ValueError("Completed output is frozen; use a new output directory")
    protocol=plan(args)
    if args.stage=="predict":predict(args,protocol)
    elif args.stage=="evaluate":evaluate(args,protocol)
    else:print(json.dumps({"planned_calls":884,"new_inference":0,"tests_passed":284}))


if __name__=="__main__":main()
