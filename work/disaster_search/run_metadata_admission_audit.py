"""Offline geometry/identity admission audit, without reserved outcomes or scores."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from audit_hold_taco import (MetadataView, REVISION, read_index, column_ranges, assert_no_protected_reads)
from audit_hold_spatial import components
from build_xbd_mini import atomic_json, sha
from inferred_scene_geometry import (COLUMNS, LIMITS, OfflineGeometryRanges,
    fit_geometry, polygon_overlap, candidate_intersections)
from plan_search_v2 import freeze
from run_utility_tie_controls import verify_receipt

SOURCE_NAMES = ["inferred_scene_geometry.py", "run_metadata_admission_audit.py",
    "audit_metadata_admission.py", "test_inferred_scene_geometry.py", "audit_hold_taco.py",
    "audit_hold_spatial.py", "audit_xbd_membership.py", "build_xbd_mini.py", "plan_search_v2.py"]


def overlap_rows(candidate, pool_ids, geometries, padded=False):
    prefix = "padded_" if padded else ""
    if not candidate["usable_inferred_range"]:
        return None
    valid = [sid for sid in pool_ids if geometries[sid]["usable_inferred_range"]]
    if not valid:
        return []
    boxes = [geometries[s][prefix + "inferred_bounds_wgs84"] for s in valid]
    result = []
    for j in candidate_intersections(candidate[prefix + "inferred_bounds_wgs84"], boxes):
        sid = valid[j]
        fraction = polygon_overlap(candidate[prefix + "inferred_corners_wgs84"],
            geometries[sid][prefix + "inferred_corners_wgs84"])
        if fraction > LIMITS["min_relative_overlap_area"]:
            result.append({"sample_id": sid, "fraction_of_smaller_inferred_post_range": fraction})
    return result


def summarize_candidates(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"], r["event"]].append(r)
    result = []
    for (split, event), part in sorted(groups.items()):
        result.append({"split": split, "event": event, "candidates": len(part),
            "usable_inferred_ranges": sum(r["usable_inferred_range"] for r in part),
            "unknown_inferred_ranges": sum(not r["usable_inferred_range"] for r in part),
            "original_training_building_envelope_flags": sum(bool(r["old_training_annotation_envelope_overlap_ids"]) for r in part),
            "inferred_training_overlap_flags": sum(bool(r["training_inferred_overlaps"]) for r in part),
            "padded_inferred_training_overlap_flags": sum(bool(r["training_padded_inferred_overlaps"]) for r in part),
            "inferred_selection_test_overlap_flags": sum(bool(r["selection_test_inferred_overlaps"]) for r in part),
            "new_inferred_training_flags_beyond_old_envelopes": sum(bool(r["training_inferred_overlaps"]) and not r["old_training_annotation_envelope_overlap_ids"] for r in part),
            "independently_admitted_scenes": 0})
    return result


def main():
    import argparse
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Fresh output directory required")
    base = Path.cwd()
    paths = {"cohort": base / "output/disaster_search_v2_design/cohort_selection.json",
        "selection_receipt": base / "output/disaster_search_v2_design/selection_receipt.json",
        "contract": base / "output/disaster_search_v2_design/experiment_contract_v2_2.json",
        "identities": base / "output/disaster_search_hold_taco_v1/identity_rows.json",
        "author_ids": base / "output/disaster_search_data_admission_v1/author_split_ids.json",
        "membership": base / "output/disaster_search_data_admission_v1/membership_report.json",
        "source_identity_audit": base / "output/disaster_search_hold_taco_v1/report.json",
        "old_spatial": base / "output/disaster_search_hold_spatial_v1/report.json",
        "source_card": base / "output/disaster_search_hold_taco_v1/dataset_card.md",
        "collection": base / "output/disaster_search_hold_taco_v1/collection.json",
        "model_card": base / "work/disaster_search/attachments/external/huggingface/changeos_swint/README.md",
        "model_manifest": base / "work/disaster_search/attachments/external/huggingface/changeos_swint/model_manifest.json",
        "development_health": base / "output/disaster_search_hold_development72_fit_v1/development_health.json",
        "prior_receipt": base / "output/disaster_search_rank_calibration_v1/completion_receipt.json"}
    sources = {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k, p in paths.items()}
    cache_root = base / "output/disaster_search_hold_taco_v1/ranges"
    freeze(args.out / "protocol.json", {"stage": "OFFLINE_METADATA_ADMISSION_AUDIT",
        "source_revision": REVISION, "sources": sources,
        "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in SOURCE_NAMES},
        "cache_root": str(cache_root.resolve()), "network_byte_cap": 0,
        "projected_columns": COLUMNS, "protected_columns": ["ml:damage", "ml:n_buildings", "ml:n_buildings_pre"],
        "geometry_criteria": LIMITS,
        "source_scope": "Project cached paired polygon and dimension columns only. Fit geometric transforms for the exact216 candidate scenes plus author's train/tier3 and checkpoint-selection test identities. No new images, damage, counts, detector inference, controller fitting or policy scores.",
        "transform": "One affine normalized pixel-to-centered lon/lat least-squares fit using original vertex/ring order; no matching/rotation search. Object even/odd cross-prediction in both directions; inverse pixel residual and spread criteria. Corners [0,width]x[0,height], plus a separately reported2px padded screen. Thresholds fixed before fitting.",
        "interpretation": "INFERRED post-image ranges ONLY. Residuals and cross-prediction check internal geometric consistency, not official geolocation accuracy or full pre-image alignment. Overlap flags are screening evidence; no overlap never certifies pixel independence, especially with unavailable training ranges. No candidate replacement or score-based exclusion.",
        "identity_roles": "Use author's exact6177 train/tier3 IDs, not the larger TACO tier3 pool; selection-test933 separate. Original hold933 remains identity-disjoint from both. Published recipe is evidence, not a per-SGD training log.",
        "license_evidence": "Inventory pinned source/model card declarations only. No legal permission or redistribution clearance inferred from a repackager's card.",
        "event_diversity": "Reuse already published development-only sensor health; do not inspect reserved damage or targets to enlarge event diversity.",
        "outputs": ["geometry_models.json", "candidate_ledger.json", "report.json", "admission_gates.json"],
        "whole_cohort_automatically_admitted": False, "validation_or_confirmation_scoring": False,
        "new_inference_calls": 0, "paid_gpu": False})
    verify_receipt(paths["prior_receipt"].parent)
    cohort = json.loads(paths["cohort"].read_text())
    selection_receipt = json.loads(paths["selection_receipt"].read_text())
    if sha(paths["cohort"].read_bytes()) != selection_receipt["selection_sha256"] or len(cohort) != 216:
        raise ValueError("Fixed cohort changed")
    source_ids = json.loads(paths["identities"].read_text())
    lookup = {r["ml:source_file"]: i for i, r in enumerate(source_ids)}
    author = json.loads(paths["author_ids"].read_text())
    training = sorted(set(author["train"]) | set(author["tier3"]))
    selection = sorted(set(author["test"]))
    candidates = [r["sample_id"] for r in cohort]
    if len(training) != 6177 or len(selection) != 933 or not set(candidates).issubset(author["hold"]):
        raise ValueError("Published source roles differ")
    if set(candidates) & (set(training) | set(selection)):
        raise ValueError("Candidate canonical-ID exposure")
    wanted = sorted(set(training) | set(selection) | set(candidates))
    if len(wanted) != 7326 or not set(wanted).issubset(lookup):
        raise ValueError("Expected exact metadata cohort missing")
    ranges = OfflineGeometryRanges(cache_root)
    index = read_index(ranges)
    parquet = pq.ParquetFile(MetadataView(ranges, index["METADATA/sample.parquet"]))
    protected = column_ranges(parquet, index["METADATA/sample.parquet"], {"ml:damage", "ml:n_buildings", "ml:n_buildings_pre"})
    assert_no_protected_reads(ranges.accesses, protected)
    ranges.protected = protected
    ranges.allowed = column_ranges(parquet, index["METADATA/sample.parquet"], set(COLUMNS)) + list(ranges.accesses)
    print(json.dumps({"stage": "REPLAY_CACHED_GEOMETRY_ONLY", "geometric_models_planned": len(wanted)}), flush=True)
    table = parquet.read(columns=COLUMNS, use_threads=False)
    if table.num_rows != len(source_ids):
        raise ValueError("Geometry/identity alignment mismatch")
    lists = {}
    for key in COLUMNS[:4]:
        column = table[key].combine_chunks()
        lists[key] = column.values.to_numpy(zero_copy_only=False), column.offsets.to_numpy(zero_copy_only=False)
    dimensions = {k: table[k].to_pylist() for k in COLUMNS[4:]}
    def flat(key, row):
        values, offsets = lists[key]
        return values[offsets[row]:offsets[row + 1]]
    geometries = {}
    started = time.perf_counter()
    for number, sid in enumerate(wanted, 1):
        i = lookup[sid]
        geometries[sid] = fit_geometry(*(flat(k, i) for k in COLUMNS[:4]),
            dimensions["ml:width"][i], dimensions["ml:height"][i])
        if number % 1000 == 0:
            print(json.dumps({"stage": "GEOMETRY_FIT_PROGRESS", "processed": number}), flush=True)
    atomic_json(args.out / "geometry_models.json", geometries)
    old = {r["sample_id"]: r for r in json.loads(paths["old_spatial"].read_text())["per_sample"]}
    ledger = []
    for sample in cohort:
        sid = sample["sample_id"]
        g = geometries[sid]
        ledger.append({**sample, "geometry_status": g["status"], "usable_inferred_range": g["usable_inferred_range"],
            "old_training_annotation_envelope_overlap_ids": old[sid]["training_annotation_envelope_overlap_ids"],
            "training_inferred_overlaps": overlap_rows(g, training, geometries),
            "training_padded_inferred_overlaps": overlap_rows(g, training, geometries, padded=True),
            "selection_test_inferred_overlaps": overlap_rows(g, selection, geometries),
            "official_footprint_verified": False, "pixel_independence_verified": False,
            "candidate_preserved": True, "independently_admitted": False})
    atomic_json(args.out / "candidate_ledger.json", ledger)
    pairs, cross_split = [], []
    by_id = {r["sample_id"]: r for r in cohort}
    for i, sid in enumerate(candidates):
        for overlap in overlap_rows(geometries[sid], candidates[i + 1:], geometries) or []:
            other = overlap["sample_id"]
            pair = {"sample_a": sid, "split_a": by_id[sid]["split"], "sample_b": other,
                "split_b": by_id[other]["split"], "fraction_of_smaller_inferred_post_range": overlap["fraction_of_smaller_inferred_post_range"]}
            pairs.append(pair)
            if pair["split_a"] != pair["split_b"]:
                cross_split.append(pair)
    health = json.loads(paths["development_health"].read_text())
    known_development = [{"event": e, "scenes": r["composition"]["scenes"],
        "target_bearing_scenes": r["composition"]["target_bearing_scenes"],
        "known_native_cache_detectable": r["full_grid_sensor_proxy"]["native"]["unique_matched_targets"] > 0}
        for e, r in sorted(health["by_event"].items())]
    model_manifest = json.loads(paths["model_manifest"].read_text())
    for entry in model_manifest["files"]:
        path = paths["model_manifest"].parent / entry["file"]
        if sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError("Pinned model artifact changed")
    source_identity = json.loads(paths["source_identity_audit"].read_text())
    identity_evidence = {"published_author_train": len(author["train"]), "published_author_tier3": len(author["tier3"]),
        "published_recipe_training_total": len(training), "checkpoint_selection_test": len(selection),
        "taco_tier3_total": source_identity["author_pool_comparisons"]["tier3"]["source_count"],
        "extra_taco_tier3_not_in_author_training_inventory": source_identity["author_pool_comparisons"]["tier3"]["source_count"] - len(author["tier3"]),
        "candidate_training_or_selection_ID_intersections": 0,
        "candidate_events_seen_in_published_training_pool": sorted({s.rsplit("_", 1)[0] for s in candidates} & {s.rsplit("_", 1)[0] for s in training}),
        "actual_per_optimizer_step_training_manifest_available": False,
        "model_card_recipe_verified_against_author_inventory": True,
        "model_files_sha256_verified": True}
    collection = json.loads(paths["collection"].read_text())
    assert_no_protected_reads(ranges.accesses, protected)
    if ranges.remote.budget.received or ranges.remote.budget.reserved or not all(r["cache_hit"] for r in ranges.accesses):
        raise ValueError("Offline cache boundary violated")
    report = {"stage": "OFFLINE_METADATA_ADMISSION_SCREEN_COMPLETE", "candidate_scenes": len(cohort),
        "geometric_model_attempts": len(wanted), "geometry_status_counts": dict(Counter(g["status"] for g in geometries.values())),
        "training_usable_inferred_ranges": sum(geometries[s]["usable_inferred_range"] for s in training),
        "training_missing_or_rejected_inferred_ranges": sum(not geometries[s]["usable_inferred_range"] for s in training),
        "selection_test_usable_inferred_ranges": sum(geometries[s]["usable_inferred_range"] for s in selection),
        "candidate_usable_inferred_ranges": sum(r["usable_inferred_range"] for r in ledger),
        "candidate_unknown_inferred_ranges": sum(not r["usable_inferred_range"] for r in ledger),
        "by_split_and_event": summarize_candidates(ledger), "identity_exposure": identity_evidence,
        "inferred_candidate_overlap_pairs": pairs, "inferred_cross_policy_split_overlaps": cross_split,
        "inferred_candidate_components": components(candidates, [(r["sample_a"], r["sample_b"]) for r in pairs]),
        "license_declarations": {"pinned_taco_collection_licenses": collection["licenses"],
            "pinned_model_manifest_license_declaration": model_manifest["license_from_model_card"],
            "original_publisher_terms_reverified_this_run": False, "redistribution_clearance_established": False},
        "existing_development_event_diversity": known_development,
        "reserved_target_composition_inspected": False, "source_geometry_columns_cached_only": COLUMNS,
        "protected_ranges": protected, "allowed_ranges": ranges.allowed, "range_receipts": ranges.accesses,
        "no_protected_payload_intersections": True, "network_bytes_received": 0, "image_bytes_read": 0,
        "new_damage_column_reads": 0, "new_detector_or_policy_fit": False,
        "geometric_transform_fits_only": True, "new_inference_calls": 0, "gpu_rented": False,
        "validation_or_confirmation_scores_run": False, "candidate_membership_changed": False,
        "official_full_image_footprints_verified": False, "full_training_pixel_hashes_verified": False,
        "whole_216_cohort_independently_admitted": False, "metadata_compute_s": time.perf_counter() - started,
        "source_hashes_preserved": {k: sha(p.read_bytes()) == sources[k]["sha256"] for k, p in paths.items()}}
    if not all(report["source_hashes_preserved"].values()):
        raise ValueError("Prior source changed")
    atomic_json(args.out / "report.json", report)
    gates = {"stage": "METADATA_ADMISSION_REMAINS_CONDITIONAL", "candidate_identity_gate": "PASS_AUTHOR_HOLD_ID_DISJOINT_FROM_TRAIN_AND_SELECTION_TEST",
        "geometry_gate": "INFERRED_SCREEN_ONLY_OFFICIAL_FOOTPRINT_AND_MISSING_TRAINING_RANGES_UNRESOLVED",
        "pixel_duplicate_gate": "FULL_TRAINING_POOL_PIXEL_HASHES_UNVERIFIED",
        "checkpoint_exposure_gate": "PUBLISHED_RECIPE_IDENTITIES_CHECKED_ALL_NINE_CANDIDATE_EVENTS_SEEN_NO_UNSEEN_EVENT_PERCEPTION_CLAIM",
        "license_gate": "PINNED_DECLARATIONS_INVENTORIED_ORIGINAL_PERMISSION_CHAIN_NOT_REVERIFIED",
        "development_diversity_gate": "TWO_EVENTS_WITH_CACHE_DETECTIONS_MEXICO_RETAINED_NO_REPLACEMENT",
        "measurement_gate": "DEVELOPMENT_ENGINEERING_PROXY_CHECKED_REAL_WORLD_IDENTIFICATION_UNVERIFIED",
        "strong_baseline_gate": "BUDGET_COMPATIBLE_STRONG_BASELINE_STILL_REQUIRED_NOT_FULL_MF_ENS",
        "new_method_selected": False, "independent_validation_admitted": False,
        "validation_or_confirmation_scoring_enabled": False, "automatic_scene_exclusion_or_replacement": False,
        "next": "Use the inferred overlap ledger to define the limits of fixed-detector policy evaluation. Do not rerun heuristics or call this cohort geographically independent. Next specify a budget-compatible strong baseline and a feasible independent-data claim before new controller training or reserved scoring."}
    atomic_json(args.out / "admission_gates.json", gates)
    print(json.dumps({k: report[k] for k in ["stage", "candidate_usable_inferred_ranges", "candidate_unknown_inferred_ranges", "training_usable_inferred_ranges", "network_bytes_received", "metadata_compute_s"]}), flush=True)


if __name__ == "__main__":
    main()
