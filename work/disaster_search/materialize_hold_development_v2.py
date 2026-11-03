"""Materialize exactly the frozen 72 development rows under amendment 02.

Shared damage data remain transient. No validation/confirmation outcomes leave
the trusted extractor. Original six-scene code and artifacts are unchanged.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from audit_hold_taco import CachedRanges, MetadataView, REVISION, column_ranges, read_index
from build_xbd_mini import atomic_json, sha
from materialize_hold_probes import (CLASSES, COLUMNS, CHILD_COLUMNS, TransientDamageRanges,
    extract_allowed_rows, intersects)
from hold_development_geometry import build_features, TOLERANCE_PIXELS
from plan_search_v2 import freeze


def validate_development(selected, cohort, amendment):
    expected = [r for r in cohort if r["split"] == "development"]
    allowed = amendment["allowed_returned_damage_rows"]["development"]
    if selected != expected or len(selected) != 72 or len(set(allowed)) != 72:
        raise ValueError("Only the exact original 72 development rows are allowed")
    if [r["sample_id"] for r in selected] != allowed or len({r["sample_id"] for r in selected}) != 72:
        raise ValueError("Changed, repeated or reordered development identities")
    if Counter(r["event"] for r in selected) != {"hurricane-harvey":24,"santa-rosa-wildfire":24,"mexico-earthquake":24}:
        raise ValueError("Development event composition changed")


def extract_development(parquet, identities, selected, cohort, amendment):
    validate_development(selected, cohort, amendment)
    return extract_allowed_rows(parquet, identities, [r["sample_id"] for r in selected])


def read_image(item, path, root, budget, protected, dimensions):
    if not 0 < item["internal:size"] <= 8*1024*1024 or intersects(item["internal:offset"],item["internal:size"],protected):
        raise ValueError("Invalid image interval")
    expected_receipt = {"offset":item["internal:offset"],"bytes":item["internal:size"],"revision":REVISION}
    accesses=[]
    if path.exists():
        raw=path.read_bytes()
        if json.loads(path.with_suffix(".receipt.json").read_text()) != {**expected_receipt,"sha256":sha(raw)}:
            raise ValueError("Changed resumed image")
    else:
        reader=CachedRanges(root)
        reader.remote.budget=budget  # Shared thread-safe cap; independent seek state.
        raw=reader.read(item["internal:offset"],item["internal:size"])
        accesses=reader.accesses
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        atomic_json(path.with_suffix(".receipt.json"),{**expected_receipt,"sha256":sha(raw)})
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB" or image.size != dimensions:
            raise ValueError("RGB mode/dimensions differ from source metadata")
        pixels=np.asarray(image)
        zero=np.all(pixels==0,axis=2)
        summary={"pixel_sha256_rgb":sha(pixels.tobytes()),"zero_rgb_fraction":float(zero.mean())}
    return {"path":str(path),"sha256":sha(raw),**summary},accesses


def materialize(args, selected, cohort, amendment, budget_limit, network_path):
    import pyarrow.parquet as pq
    cached=CachedRanges(args.metadata_audit/"ranges",budget_limit)
    try:
        index=read_index(cached)
        schema=pq.ParquetFile(MetadataView(cached,index["METADATA/sample.parquet"]))
        protected=column_ranges(schema,index["METADATA/sample.parquet"],{"ml:damage"})
        prefix=list(cached.accesses)
        reader=TransientDamageRanges(cached,protected)
        sample=pq.ParquetFile(MetadataView(reader,index["METADATA/sample.parquet"]))
        identities=json.loads((args.metadata_audit/"identity_rows.json").read_text())
        source={r["ml:source_file"]:r for r in identities}
        if any(source[r["sample_id"]]["ml:split_original"] != "hold" for r in selected):
            raise ValueError("Selected row is not original hold")
        rows=extract_development(sample,identities,selected,cohort,amendment)
        features={sid:build_features(sid,row) for sid,row in rows.items()}
        if any((r["ml:width"],r["ml:height"]) != (1024,1024) for r in rows.values()):
            raise ValueError("The fixed grid protocol requires 1024x1024 RGB")
        children=pq.ParquetFile(MetadataView(reader,index["METADATA/children.parquet"])).read(columns=CHILD_COLUMNS,use_threads=False).to_pylist()
        child_lookup=defaultdict(list)
        for item in children: child_lookup[item["internal:parent_id"]].append(item)
        images, image_accesses = {},[]
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks={}
            for r in selected:
                sid=r["sample_id"]
                for mode in ["pre","post"]:
                    found=[v for v in child_lookup[source[sid]["internal:current_id"]] if v["internal:relative_path"].endswith(f"/{mode}.png")]
                    if len(found) != 1: raise ValueError("Ambiguous source image mapping")
                    path=args.out/"samples"/sid/f"{sid}_{mode}_disaster.png"
                    future=executor.submit(read_image,found[0],path,args.metadata_audit/"ranges",cached.remote.budget,protected,(1024,1024))
                    tasks[future]=(sid,mode)
            for future in as_completed(tasks):
                key=tasks[future]
                image,accesses=future.result()
                images[key]=image
                image_accesses.extend(accesses)
                if len(images)%12 == 0:
                    print(json.dumps({"stage":"DEVELOPMENT_RGB_DOWNLOAD","images":len(images),"total":144,"network_bytes":cached.remote.budget.received}),flush=True)
        manifest_samples, file_records, records, signatures=[],[],[],defaultdict(list)
        for r in selected:
            sid=r["sample_id"]
            files={}
            zero={}
            for mode in ["pre","post"]:
                image=images[sid,mode]
                relative=str(Path(image["path"]).relative_to(args.out))
                files[mode+"_image"]=relative
                file_records.append({"path":relative,"sha256":image["sha256"],"pixel_sha256_rgb":image["pixel_sha256_rgb"]})
                signatures[image["pixel_sha256_rgb"]].append(f"{sid}:{mode}")
                zero[mode]=image["zero_rgb_fraction"]
            label={"metadata":{"width":1024,"height":1024,"img_name":f"{sid}_post_disaster.png","disaster":r["event"]},
                "features":{"xy":features[sid]},"provenance":{"revision":REVISION,"identity":"DERIVED_SOURCE_OBJECT_INDEX","original_uid_retained":False}}
            path=args.out/"samples"/sid/f"{sid}_post_disaster.json"
            freeze(path,label)
            files["post_label"]=str(path.relative_to(args.out))
            file_records.append({"path":files["post_label"],"sha256":sha(path.read_bytes())})
            manifest_samples.append({**r,"files":files})
            counts=Counter(f["properties"]["subtype"] for f in features[sid])
            records.append({"sample_id":sid,"event":r["event"],"buildings":len(features[sid]),"class_counts":dict(counts),
                "targets":counts["major-damage"]+counts["destroyed"],"unknown":counts["un-classified"],"zero_rgb_fraction":zero})
        duplicates=[{"pixel_sha256":h,"observations":v} for h,v in signatures.items() if len(v)>1]
        old_overlap=[]
        old_manifest=json.loads((args.mini_data/"manifest.json").read_text())
        for item in old_manifest["files"]:
            if item["path"].endswith(".png"):
                with Image.open(args.mini_data/item["path"]) as im: pixel_hash=sha(np.asarray(im.convert("RGB")).tobytes())
                if pixel_hash in signatures: old_overlap.append({"mini_file":item["path"],"development":signatures[pixel_hash]})
        if duplicates or old_overlap:
            atomic_json(args.out/"duplicate_failure.json",{"within_development":duplicates,"mini72_overlap":old_overlap})
            raise ValueError("Pixel duplicate: retain fixed cohort and investigate; never replace automatically")
        probe_manifest=json.loads((args.probes/"manifest.json").read_text())
        old_files={f["path"]:f["sha256"] for f in probe_manifest["files"]}
        new_files={f["path"]:f["sha256"] for f in file_records}
        if any(new_files[path] != digest for path,digest in old_files.items()):
            raise ValueError("Existing six probes changed under 72-row adapter")
        manifest={"dataset_id":"hold_fixed_development72_v1","source_revision":REVISION,
            "samples":manifest_samples,"files":file_records,"protocol_sha256":sha((args.out/"materialization_protocol.json").read_bytes())}
        freeze(args.out/"manifest.json",manifest)
        spatial=json.loads(args.spatial.read_text())
        by_spatial={r["sample_id"]:r for r in spatial}
        event_summary={event:{"scenes":sum(r["event"]==event for r in records),
            **{k:sum(r[k] for r in records if r["event"]==event) for k in ["buildings","targets","unknown"]},
            "target_bearing_scenes":sum(r["event"]==event and r["targets"]>0 for r in records),
            "training_envelope_overlap_scenes":sum(by_spatial[r["sample_id"]]["training_envelope_overlap"] for r in records if r["event"]==event)}
            for event in sorted({r["event"] for r in selected})}
        accesses=prefix+reader.accesses+image_accesses
        # A persistent range overlapping the shared damage column is forbidden.
        for receipt in (args.metadata_audit/"ranges").glob("*.json"):
            saved=json.loads(receipt.read_text())
            if intersects(saved["offset"],saved["bytes"],protected) and receipt.with_suffix(".bin").exists():
                raise ValueError("Damage payload persisted in range cache")
        report={"stage":"DEVELOPMENT72_DATA_INTERFACE_PASSED","pairs":72,"samples":records,"by_event":event_summary,
            "buildings":sum(r["buildings"] for r in records),"targets":sum(r["targets"] for r in records),"unknown":sum(r["unknown"] for r in records),
            "existing_six_probe_files_identical":True,"within_development_pixel_duplicates":duplicates,"mini72_pixel_duplicates":old_overlap,
            "shared_reserved_damage_payload_physically_read":True,"only_allowed_development_damage_retained":True,
            "unselected_damage_values_or_summaries_exposed":False,"shared_damage_payload_cached":False,
            "validation_or_confirmation_predictions_or_policy_scores_run":False,"data_interface_passed":True,
            "full_image_spatial_independence_verified":False,"whole_216_cohort_admitted":False,
            "membership_changed":False,"network_bytes_this_invocation":cached.remote.budget.received,
            "read_receipts":accesses,"protected_ranges":protected,
            "runner_sha256":sha(Path(__file__).read_bytes()),"manifest_sha256":sha((args.out/"manifest.json").read_bytes())}
        atomic_json(args.out/"data_report.json",report)
        atomic_json(args.out/"interface_gate.json",{"version":"DEVELOPMENT72_RGB_ACCESS_GATE","data_interface_passed":True,
            "manifest_sha256":report["manifest_sha256"],"sample_ids":[r["sample_id"] for r in selected],"scope":"No class counts, labels or evaluator results."})
        print(json.dumps({k:report[k] for k in ["stage","pairs","buildings","targets","unknown","by_event","network_bytes_this_invocation"]}),flush=True)
    finally:
        atomic_json(network_path,{"reserved_bytes":cached.remote.budget.reserved,"received_bytes":cached.remote.budget.received})


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["design","metadata-audit","mini-data","probes","out","spatial","previous-attempt"]:
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((args.design/"next_development_batch.json").read_text())
    cohort=json.loads((args.design/"cohort_selection.json").read_text())
    amendment=json.loads((args.design/"contract_amendment_02.json").read_text())
    contract=json.loads((args.design/"experiment_contract_v2_2.json").read_text())
    validate_development(selected,cohort,amendment)
    for path,key in [(args.design/"cohort_selection.json","cohort_selection_sha256"),(args.design/"next_development_batch.json","development_rows_sha256")]:
        if sha(path.read_bytes()) != amendment[key]: raise ValueError("Frozen access allowlist changed")
    if contract["version"] != "search_v2_2_design_20261025" or contract["amendment_sha256"] != sha((args.design/"contract_amendment_02.json").read_bytes()):
        raise ValueError("Access amendment does not match effective contract")
    legend=json.loads((args.metadata_audit/"collection.json").read_text())
    target=next(v for v in legend["ml:contract"]["targets"] if v["name"]=="damage")
    if target["classes"] != CLASSES or target["ignore_index"] != 4: raise ValueError("Source legend changed")
    probe_report=json.loads((args.probes/"data_report.json").read_text())
    if not probe_report["data_interface_passed"] or probe_report["class_mapping_verified_on_old_objects"] != 64:
        raise ValueError("Previous six-scene class bridge missing")
    helpers=["materialize_hold_probes.py","audit_hold_taco.py","audit_xbd_membership.py","verify_taco_polygon_bridge.py","hold_development_geometry.py"]
    freeze(args.out/"materialization_protocol.json",{"version":"fixed_hold_development72_v2","selected":selected,"source_revision":REVISION,
        "contract_sha256":sha((args.design/"experiment_contract_v2_2.json").read_bytes()),"amendment_sha256":contract["amendment_sha256"],
        "runner_sha256":sha(Path(__file__).read_bytes()),"helper_sha256":{n:sha(Path(__file__).with_name(n).read_bytes()) for n in helpers},
        "classes":CLASSES,"target_codes":[2,3],"unknown_code":4,"max_new_network_bytes":amendment["max_new_network_bytes"],
        "geometry_exception_audit_sha256":sha((args.previous_attempt/"geometry_exception_audit.json").read_bytes()),
        "geometry_tolerance_pixels":TOLERANCE_PIXELS,"geometry_storage":"Retain exact original coordinates. No clipping or matching rule change.",
        "projected_columns":COLUMNS,"pre_labels_available":False,"original_uids_available":False,
        "existing_class_bridge_report_sha256":sha((args.probes/"data_report.json").read_bytes()),
        "spatial_groups_sha256":sha(args.spatial.read_bytes()),"zero_rgb_rule":"QA only; no input masking, filtering or new policy feature.",
        "scope":"Development data/QA only. No validation/confirmation outcome summaries, inference or policy scoring."})
    attempts=args.out/"network_attempts"; attempts.mkdir(exist_ok=True)
    used=sum(json.loads(p.read_text())["reserved_bytes"] for root in [attempts,args.previous_attempt/"network_attempts"] for p in root.glob("*.json"))
    remaining=amendment["max_new_network_bytes"]-used
    if remaining <= 0: raise ValueError("Cumulative development download allowance exhausted")
    materialize(args,selected,cohort,amendment,remaining,attempts/(str(uuid.uuid4())+".json"))


if __name__ == "__main__": main()
