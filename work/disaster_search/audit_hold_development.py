"""Verify development artifacts and summarize target/sensor bottlenecks."""
from collections import Counter
import json
from pathlib import Path

from associate_buildings import discovery_summary
from build_xbd_mini import atomic_json,sha
from materialize_hold_probes import intersects


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["data","perception","fit","design","metadata-audit","previous-attempt","out"]:
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/"manifest.json").read_text())
    selected=json.loads((args.design/"next_development_batch.json").read_text())
    expected={r["sample_id"] for r in selected}
    if len(expected)!=72 or {r["sample_id"] for r in manifest["samples"]}!=expected:raise ValueError("Wrong cohort")
    for item in manifest["files"]:
        if sha((args.data/item["path"]).read_bytes())!=item["sha256"]:raise ValueError("Dataset file changed")
    data_protocol=json.loads((args.data/"materialization_protocol.json").read_text())
    if data_protocol["runner_sha256"]!=sha(Path(__file__).with_name("materialize_hold_development_v2.py").read_bytes()):
        raise ValueError("Materializer changed")
    for name,digest in data_protocol["helper_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes())!=digest:raise ValueError("Materializer helper changed")
    observations=json.loads((args.perception/"observations.json").read_text())
    identities={f"{sid}:r{r:02d}c{c:02d}:{mode}" for sid in expected for r in range(8) for c in range(8) for mode in ["native","blurred"]}
    if len(observations)!=9216 or {r["observation_id"] for r in observations}!=identities:raise ValueError("Incomplete observations")
    for row in observations:
        if row["split"]!="development" or sha(Path(row["cache_file"]).read_bytes())!=row["cache_sha256"]:
            raise ValueError("Wrong split or mask cache")
    protocol=json.loads((args.perception/"perception_protocol.json").read_text())
    if protocol["runner_sha256"]!=sha(Path(__file__).with_name("run_hold_development_perception.py").read_bytes()):raise ValueError("Perception runner changed")
    fit_protocol=json.loads((args.fit/"fit_protocol.json").read_text())
    if fit_protocol["runner_sha256"]!=sha(Path(__file__).with_name("fit_hold_development.py").read_bytes()):raise ValueError("Fit runner changed")
    data=json.loads((args.data/"data_report.json").read_text())
    for p in (args.metadata_audit/"ranges").glob("*.json"):
        record=json.loads(p.read_text())
        if intersects(record["offset"],record["bytes"],data["protected_ranges"]) and p.with_suffix(".bin").exists():
            raise ValueError("Shared damage payload persists")
    legacy=Path("output/disaster_search_cross_cell_smoke_v1")
    receipt=json.loads((legacy/"completion_receipt.json").read_text())
    if any(sha((legacy/name).read_bytes())!=digest for name,digest in receipt["files"].items()):
        raise ValueError("Previous six-scene experiment artifact changed")
    association=json.loads((args.perception/"association_report.json").read_text())
    if association["observations_sha256"]!=sha((args.perception/"observations.json").read_bytes()):
        raise ValueError("Evaluator observations changed")
    if fit_protocol["sources"]["association"]!=sha((args.perception/"association_report.json").read_bytes()):
        raise ValueError("Fitted evaluator source changed")
    by_scene={r["sample_id"]:r for r in selected}
    sensor={}
    for event,composition in data["by_event"].items():
        rows=[r for r in association["rows"] if by_scene[r["sample_id"]]["event"]==event]
        summaries={mode:discovery_summary([r for r in rows if r["mode"]==mode]) for mode in ["native","blurred"]}
        both=discovery_summary(rows)
        sensor[event]={"composition":composition,"full_grid_sensor_proxy":summaries,
            "both_modes_cache_union_ceiling":both["unique_matched_targets"],
            "scenes_with_native_matches":sum(association["per_scene"][sid]["native"]["unique_matched_targets"]>0 for sid,s in by_scene.items() if s["event"]==event)}
    attempts=[p for root in [args.data,args.previous_attempt] for p in (root/"network_attempts").glob("*.json")]
    network={key:sum(json.loads(p.read_text())[key] for p in attempts) for key in ["reserved_bytes","received_bytes"]}
    amendment=json.loads((args.design/"contract_amendment_02.json").read_text())
    if network["reserved_bytes"]>amendment["max_new_network_bytes"]:raise ValueError("Cumulative network cap exceeded")
    result={"stage":"DEVELOPMENT72_DATA_SENSOR_AND_AGGREGATE_FIT_VERIFIED","scenes":72,
        "source_files_verified":len(manifest["files"]),"prediction_masks_verified":len(observations),
        "by_event":sensor,"network":network,"previous_six_scene_artifacts_unchanged":True,
        "shared_damage_payload_cached":False,"validation_or_confirmation_predictions_or_policy_scores_run":False,
        "independent_efficacy_claim":False,"full_image_spatial_independence_verified":False,
        "next":"Budgeted simple controls and existence/count ablation on development; specify strong baseline before validation/new-method claim.",
        "scope":"Full-grid cache union is an engineering discoverability ceiling, never a budgeted policy score or real-world detection guarantee.",
        "runner_sha256":sha(Path(__file__).read_bytes())}
    atomic_json(args.out/"development_health.json",result)
    print(json.dumps(result))


if __name__=="__main__":main()
