"""Offline replay of selected reference ranges, paid decisions and area scores."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from build_xbd_mini import atomic_json,sha
from external_area_core import (CELLS,POLICIES,BUDGETS,QUALITY_SEEDS,ORDER_SEEDS,
    BinaryEnvironment,run_policy,evaluate_trace,exact_random_survey)
from extract_external_area_masks import selected_members,verify_parent,decode_reference
from extract_external_rgb18 import parse_local_header,decompress_rgb as decompress_png
from run_external_area_controls import (DOMAINS,load_scene,average,aggregate,opportunity)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["rgb","metadata","root"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();root=args.root;get=lambda n:json.loads((root/n).read_text())
    verify_parent(args.rgb);verify_parent(args.metadata)
    old=json.loads(Path("output/disaster_search_serial_multifidelity_v1/completion_receipt.json").read_text())
    hashes={Path(n).name:d for n,d in old["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    for parent in [args.metadata,args.rgb]:hashes.update(json.loads((parent/"completion_receipt.json").read_text())["source_code_sha256"])
    for name,digest in hashes.items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Old frozen code changed: "+name)
    mask_protocol=get("mask_extraction_protocol.json");members=selected_members(args.rgb,args.metadata)
    if members!=mask_protocol["members"] or sha(Path(__file__).with_name("extract_external_area_masks.py").read_bytes())!=mask_protocol["extractor_sha256"]:raise ValueError("Mask plan/source changed")
    data_root=Path(mask_protocol["data_root"]);refs={r["sample_id"]:r for r in get("reference_manifest.json")}
    total=0;intervals=set()
    for member in members:
        row=refs[member["sample_id"]];parts=[]
        for receipt in row["ranges"]:
            identity={k:receipt[k] for k in ["url","member","offset","bytes","role"]}
            if identity["url"]!=member["archive_url"] or identity["member"]!=member["path"]:raise ValueError("Reference source differs")
            key=sha(json.dumps(identity,sort_keys=True).encode());raw=(data_root/"ranges"/(key+".bin")).read_bytes()
            saved=json.loads((data_root/"ranges"/(key+".json")).read_text())
            if any(saved[k]!=v for k,v in identity.items()) or saved["sha256"]!=sha(raw) or receipt["sha256"]!=sha(raw) or len(raw)!=receipt["bytes"]:raise ValueError("Reference range cache changed")
            if saved["status"]!=206 or saved["content_range"]!=f"bytes {identity['offset']}-{identity['offset']+identity['bytes']-1}/{member['archive_size']}":raise ValueError("Nonexact reference response")
            parts.append(raw);total+=len(raw);intervals.add((identity["member"],identity["offset"],identity["bytes"],identity["role"]))
        header,variable,packed=parts;n,extra,start=parse_local_header(header,member)
        expected=[(member["local_header_offset"],30,"reference_header30"),(member["local_header_offset"]+30,n+extra,"reference_name_extra"),(start,member["compressed_bytes"],"published_damage_mask_compressed")]
        if [(r["offset"],r["bytes"],r["role"]) for r in row["ranges"]]!=expected:raise ValueError("Reference intervals differ")
        if variable[:n].decode("utf-8" if member["flag_bits"]&2048 else "cp437")!=member["path"]:raise ValueError("Reference local name differs")
        expanded=decompress_png(packed,member)
        if expanded!=Path(row["path"]).read_bytes() or sha(expanded)!=row["sha256"]:raise ValueError("Reference PNG replay differs")
        label,encoding=decode_reference(expanded)
        if encoding!=row["encoding"] or {str(k):int((label==k).sum()) for k in range(5)}!=row["class_pixels"]:raise ValueError("Reference decode/counts differ")
    ledger=json.loads((data_root/"ranges/transfer_ledger.json").read_text())
    if len(ledger)!=54 or total!=30673 or sum(r["received"] for r in ledger)!=total or any(r["status"]!="EXACT_RANGE_RECEIVED" for r in ledger):raise ValueError("Reference network ledger differs")
    if {(r["member"],r["offset"],r["bytes"],r["role"]) for r in ledger}!=intervals:raise ValueError("Unexpected reference access")
    for protocol_name,key,file in [("perception_protocol.json","runner_sha256","run_external_area_perception.py"),
        ("perception_protocol.json","geometry_sha256","external_area_core.py"),("perception_protocol.json","shared_rgb_helper_sha256","run_changeos_pilot.py"),
        ("controls_execution_protocol.json","runner_sha256","run_external_area_controls.py"),("reference_inspection_protocol.json","inspector_sha256","inspect_external_area_reference.py")]:
        if sha(Path(__file__).with_name(file).read_bytes())!=get(protocol_name)[key]:raise ValueError("Frozen new code changed: "+file)
    controls=get("controls_execution_protocol.json")
    for name,path in {"rgb_manifest":args.rgb/"rgb_manifest.json","references":root/"reference_manifest.json","observations":root/"observations.json","features":root/"public_features.json"}.items():
        if sha(path.read_bytes())!=controls["source_sha256"][name]:raise ValueError("Control input changed")
    if sha((root/"area_trajectories.jsonl.gz").read_bytes())!=get("controls_report.json")["trajectory_sha256"]:raise ValueError("Trajectory bytes changed")
    observations=get("observations.json");features=get("public_features.json");rgb={(r["sample_id"],r["mode"]):r for r in json.loads((args.rgb/"rgb_manifest.json").read_text())}
    if len(observations)!=576 or len({r["observation_id"] for r in observations})!=576:raise ValueError("Unexpected sensor inventory")
    grouped=defaultdict(list);opportunities=[];seen=set();count=0;action_count=0;last_sid=None
    action_pairs=defaultdict(dict)
    with gzip.open(root/"area_trajectories.jsonl.gz","rt") as stream:
        for line in stream:
            row=json.loads(line);sid=row["sample_id"]
            if sid!=last_sid:
                target,domains,masks,booleans=load_scene(sid,rgb,refs,observations,features);last_sid=sid
                opportunities.extend({"sample_id":sid,"event":refs[sid]["event"],"domain":name,**opportunity(target,domain,masks)} for name,domain in domains.items())
            key=(sid,row["budget"],row["quality_seed"],row["order_seed"],row["policy"])
            if key in seen or row["policy"] not in POLICIES or row["budget"] not in BUDGETS or row["quality_seed"] not in QUALITY_SEEDS or row["order_seed"] not in ORDER_SEEDS:raise ValueError("Unplanned/duplicate trajectory")
            seen.add(key)
            trace=run_policy(features[sid],BinaryEnvironment(booleans,sid,row["quality_seed"],2),row["policy"],row["budget"],row["order_seed"],2)
            if trace!=row["trace"]:raise ValueError("Decision replay differs")
            spent=0;states={}
            for a in trace:
                if a["kind"]=="survey":
                    if a["cell_id"] in states or a["cost"]!=1:raise ValueError("Repeated/incorrect survey")
                else:
                    if states.get(a["cell_id"])!=("blurred",False) or row["policy"]!="prior_gate" or a["cost"]!=2:raise ValueError("Invalid negative upgrade")
                spent+=a["cost"];states[a["cell_id"]]=(a["mode"],a["declared_target"])
                if spent!=a["cumulative_cost"] or spent>row["budget"]:raise ValueError("Overspent budget")
            action_count+=len(trace);count+=1
            action_pairs[sid,row["budget"],row["quality_seed"],row["order_seed"]][row["policy"]]=trace
            for name,domain in domains.items():
                result=evaluate_trace(trace,target,masks,domain)
                if result!=row["scores"][name]:raise ValueError("Area union/curve replay differs")
                grouped[sid,row["event"],name,row["budget"],row["policy"]].append(result)
    if count!=8100:raise ValueError("Incomplete trajectory set")
    for sid in sorted(refs):
        target,domains,masks,_=load_scene(sid,rgb,refs,observations,features)
        for name,domain in domains.items():
            for budget in BUDGETS:
                for seed in QUALITY_SEEDS:
                    exact=exact_random_survey(target,masks,sid,seed,budget,domain)
                    grouped[sid,refs[sid]["event"],name,budget,"random_exact_expectation"].append({"reference_pixels":int((target&domain).sum()),"covered_reference_pixels":exact["expected_covered_reference_pixels"],"reference_false_positive_pixels":exact["expected_reference_false_positive_pixels"],"surveys":budget,"upgrades":0,"spent":budget})
    scenes=[{"sample_id":sid,"event":event,"domain":domain,"budget":budget,"policy":policy,**average(rows)} for (sid,event,domain,budget,policy),rows in grouped.items()]
    key=lambda r:(r["sample_id"],r["event"],r["domain"],r["budget"],r["policy"])
    if sorted(scenes,key=key)!=sorted(get("area_scene_means.json"),key=key) or aggregate(scenes)!=get("area_summary.json"):raise ValueError("Aggregation replay differs")
    if opportunities!=get("sensor_area_opportunity.json"):raise ValueError("Full-coverage opportunity replay differs")
    changed=Counter()
    for (_,budget,_,_),policies in action_pairs.items():
        if policies["prior_gate"]!=policies["prior_survey"]:changed[budget]+=1
    log=(root/"test_log.txt").read_text()
    if "Ran 262 tests" not in log or not log.rstrip().endswith("OK"):raise ValueError("Current262-test evidence missing")
    atomic_json(root/"audit.json",{"status":"REFERENCE_RANGES_DECISIONS_AND_PIXEL_UNION_SCORES_REPLAYED", "reference_masks":18,"reference_exact_ranges":54,"new_network_bytes":total,
        "sensor_cache_masks_verified":576,"trajectories_replayed":count,"domain_scores_replayed":count*2,"paid_actions_verified":action_count,
        "gate_vs_survey_changed_action_traces_by_budget":dict(changed),"old_frozen_sources_unchanged":len(hashes),
        "tests_passed":262,"new_tests":18,"new_inference":576,"new_controller_fit":False,"paid_gpu":False,
        "human_annotations_created":0,"instances_created":0,"independent_efficacy_claim":False,"old_reserved_sets_scored":False})
    print(json.dumps(get("audit.json"),ensure_ascii=False))


if __name__=="__main__":main()
