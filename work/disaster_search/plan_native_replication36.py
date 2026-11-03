"""Lock the next36 paired scenes from already cached ZIP metadata only."""
import json
from pathlib import Path

from build_xbd_mini import sha
from extract_external_area_masks import verify_parent
from plan_independent_feasibility import EVENTS,SOURCE_VERSION,pair_inventory,rank_hash
from plan_search_v2 import freeze


def read(path):return json.loads(Path(path).read_text())


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--metadata",type=Path,required=True);p.add_argument("--root",type=Path,required=True);args=p.parse_args();root=args.root
    parent=verify_parent(args.metadata);current=read(root/"protocol.json");decision=read(root/"next_step_decision.json")
    if decision["primary_screen_passed"]:raise ValueError("This replication plan follows the stopped blur candidate")
    old=read(args.metadata/"tentative_cohort.json");old_ids={r["sample_id"] for r in old}
    if old_ids!=set(current["selected_ids"]):raise ValueError("Original18 cohort identity differs")
    source=read(args.metadata/"sources/ebd_figshare_v2.json");archives={r["name"]:r for r in source["files"]}
    selected=[];queue=[];inventory_hashes={};counts={}
    for event in EVENTS:
        path=args.metadata/"sources"/(event+"_index.json");inventory_hashes[event]=sha(path.read_bytes());inventory=read(path)
        pairs=pair_inventory(event,inventory);ordered=sorted(pairs,key=lambda sid:(rank_hash(event,sid),sid));counts[event]=len(pairs)
        original=[r["sample_id"] for r in sorted([r for r in old if r["event"]==event],key=lambda r:r["rank_within_event"])]
        if ordered[:6]!=original:raise ValueError("Original hash ranking does not replay")
        by_path={r["path"]:r for r in inventory};archive=archives[event+".zip"]
        for rank,sid in enumerate(ordered[6:18],7):
            name=f"{event}/masks/{sid}_post_disaster.png"
            if name not in by_path:raise ValueError("Missing selected published mask identity; no scene substitution")
            row={"sample_id":sid,"event":event,"source_version":SOURCE_VERSION,"rank_within_event":rank,"rank_sha256":rank_hash(event,sid),
                "pre":pairs[sid]["pre"],"post":pairs[sid]["post"],"published_post_mask":by_path[name],"role":"PROSPECTIVE_NATIVE_MEASUREMENT_REPLICATION_DEVELOPMENT_ONLY"}
            selected.append(row)
            for mode,member in [("pre",row["pre"]),("post",row["post"]),("published_post_mask",row["published_post_mask"])]:
                queue.append({"sample_id":sid,"event":event,"mode":mode,"archive_url":archive["download_url"],"archive_size":archive["size"],"member":member,"enabled":False})
    if len(selected)!=36 or len({r["sample_id"] for r in selected})!=36 or old_ids&{r["sample_id"] for r in selected}:raise ValueError("Invalid/disjointness count")
    packed=sum(r["member"]["compressed_bytes"] for r in queue)
    if packed>48*1024*1024-1024*1024:raise ValueError("Selected payloads exceed predeclared48MiB allowance with header margin; no smaller-image replacement")
    protocol={"status":"METADATA_COHORT_FIXED_PAYLOADS_AND_INFERENCE_NOT_EXECUTED","client_report_date":"2026-10-26",
        "question":"Does the fixed native256 vs native128 equal-area advantage replicate on the next36 unexamined paired scenes?",
        "source_version":SOURCE_VERSION,"metadata_parent_receipt_sha256":parent,"inventory_sha256":inventory_hashes,
        "basis_quality_protocol_sha256":sha((root/"protocol.json").read_bytes()),"basis_quality_decision_sha256":sha((root/"next_step_decision.json").read_bytes()),
        "selector_sha256":sha(Path(__file__).read_bytes()),"existing_ranker_sha256":sha(Path(__file__).with_name("plan_independent_feasibility.py").read_bytes()),
        "selection_rule":"Same source/version/seed/hash ordering as original18; take ranks7..18 inclusive per each of3 events. Ranking never uses image bytes, label classes, sizes or detector outputs. Sizes only checked against transfer cap after selection; no substitutions.",
        "existing_complete_pair_counts":counts,"selected_pairs":selected,"old18_ids_excluded":sorted(old_ids),"new_pairs":36,"pairs_per_event":12,"total_development_pairs_if_downloaded":54,
        "payload_access_enabled":False,"download_queue_file":"native_replication36_queue_disabled.json","selected_compressed_payload_bytes":packed,"new_transfer_cap_bytes":48*1024*1024,
        "transfer_rule":"New separate extraction protocol before network: selected72 RGB plus36 post masks only, exact bounded ranges, CRC/SHA/shape/encoding validation; no full archive or other image/mask payloads. Stop on error/cap; no replacement or certificate bypass.",
        "image_correspondence":"512 RGB pairs expected; keepall selected scenes includingzero,black boundaries and failures. Checkpair geometry; no automatic label alignment, shift or connected-component instance labels.",
        "license_scope":"Carry forward source_use_record/attribution and local-research-only constraints; no new independent geography or source-chain admission, public redistribution or human annotation claim.",
        "checkpoint":current["checkpoint"],"primary_quality":"native_only","primary_condition":"native256_contiguous","control":"native128_same_new_RGB",
        "cost_contract":"128 costs1,256 costs4,512 wouldcost16. Primarybudget8 compares8cells vs2blocks. No unpurchased high-resolution context; keep64 BOX freecoarse information identical.",
        "future_forward_inventory":{"locked_old_native128_controls":2,"new_native128":576,"new_native256":144,"free64_BOX_to512_only":36},"max_future_forward_calls":758,
        "free_baseline":"Compute each new scene's fixed free-coarse-only severe output from64 BOX ->512 BILINEAR; include free baseline in bothconditions and report paid incremental pixel union without double counting.",
        "primary_comparison":"Same frozen exact uniform random expected incremental reference area at budget8,3-event macro across ALLnew36. Pooledreferenceprecision ratioexpectedcounts; same budgets4/8/16 and bothdomains. No fitting or adaptive planner.",
        "secondary_comparison":"Same old rawRGB-change ranks and10 frozen cell-order tie seeds,256blockscoremean4cellcues andminimum-member tie rank. Keep secondary separate; no additional controller variants.",
        "measurement_replication_screen":"On NEW36 only, require at least2 reference-positive events, at least6 scenes with native256 published-reference overlap (at least2 in each of2 events), higherbudget8 exact-random event-macro paid incremental coverage for256 than128, no pooledprecision drop, and nondecreasing event-level coverage in everyreference-positive event with strict gains in at least2. These are engineering sufficiency/effect screens, not significance or power calculations; report failed/inconclusive rather than replace scenes.",
        "old18_scope":"Report new36 separately first; old18 remain explorational development, no pooling to rescue a failed primarynew36 result. Anycombined54 table is descriptive andseparately labelled.",
        "screen_failure":"Stop input/selection variants on this source/checkpoint cohort; assess alternate reference/data or fixed detector as a separately scoped study. No broadercandidate hunt after unfavorable scores.",
        "forbidden_followups_in_this_plan":"No blur retuning, mixed-quality revisit, secondaryinput promotion, controller/model fit, paidGPU, oldreservedscore, fabricatedinstancegold or final-confirmationclaim.",
        "human_annotation_resources":"Unavailable","new_payload_bytes_read":0,"new_inference":0,"new_policy_scores":0}
    freeze(root/"prospective_native_replication36.json",protocol);freeze(root/"native_replication36_queue_disabled.json",queue)
    print(json.dumps({"new_pairs_fixed":36,"old18_overlap":0,"queued_RGB":72,"queued_published_post_masks":36,"selected_compressed_bytes":packed,"payloads_read":0,"future_forward_cap":758}))


if __name__=="__main__":main()
