"""Follow-up to near-complete inferred overlap; geometry and payload metadata only."""
import json
from pathlib import Path

import numpy as np

from audit_hold_taco import MetadataView, read_index, column_ranges, assert_no_protected_reads
from build_xbd_mini import atomic_json, sha
from inferred_scene_geometry import COLUMNS, OfflineGeometryRanges
from plan_search_v2 import freeze
from verify_taco_polygon_bridge import CHILD_COLUMNS, split_rings, ring_key


def main():
    import argparse
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main_protocol = json.loads((args.run / "protocol.json").read_text())
    ledger = json.loads((args.run / "candidate_ledger.json").read_text())
    pairs = [{"candidate_id": r["sample_id"], "training_id": x["sample_id"],
        "fraction_of_smaller_inferred_post_range": x["fraction_of_smaller_inferred_post_range"]}
        for r in ledger for x in (r["training_inferred_overlaps"] or [])
        if x["fraction_of_smaller_inferred_post_range"] >= .99]
    # This supplementary probe is explicitly frozen AFTER the spatial screen.
    # It is not a prospective efficacy test or a changed admission cutoff.
    freeze(args.run / "geometry_duplicate_probe_protocol.json", {
        "stage": "POST_SCREEN_GEOMETRY_DUPLICATE_PROBE", "selection": "ALL candidate/training pairs with inferred overlap >=.99; metadata only, no policy/target-outcome selection.",
        "basis_candidate_ledger_sha256": sha((args.run / "candidate_ledger.json").read_bytes()),
        "pairs": pairs, "geometry_columns": COLUMNS[:4], "child_metadata_columns": CHILD_COLUMNS,
        "network_byte_cap": 0, "no_image_payload_reads": True, "no_damage_or_count_columns": True,
        "comparison": "Exact array equality and full per-ring XY fingerprint multisets at round6, ignoring ring start/winding via the established bridge helper. Source image offset/size equality is reported separately from size alone.",
        "limits": "Identical geometry or equal PNG lengths does NOT establish pixel equality. No geographic independence or actual optimizer exposure claim.",
        "source_code_sha256": {n: sha(Path(__file__).with_name(n).read_bytes()) for n in
            ["probe_metadata_geometry_duplicates.py", "inferred_scene_geometry.py", "verify_taco_polygon_bridge.py", "audit_hold_taco.py"]}})
    identities_path = Path(main_protocol["sources"]["identities"]["path"])
    identities = json.loads(identities_path.read_text())
    lookup = {r["ml:source_file"]: (i, r["internal:current_id"]) for i, r in enumerate(identities)}
    ranges = OfflineGeometryRanges(Path(main_protocol["cache_root"]))
    index = read_index(ranges)
    sample = pq.ParquetFile(MetadataView(ranges, index["METADATA/sample.parquet"]))
    protected = column_ranges(sample, index["METADATA/sample.parquet"], {"ml:damage", "ml:n_buildings", "ml:n_buildings_pre"})
    assert_no_protected_reads(ranges.accesses, protected)
    ranges.protected = protected
    child_entry = index["METADATA/children.parquet"]
    footer_bytes = min(65536, child_entry["size"])
    ranges.allowed = column_ranges(sample, index["METADATA/sample.parquet"], set(COLUMNS[:4])) + list(ranges.accesses) + [
        {"offset": child_entry["offset"] + child_entry["size"] - footer_bytes, "bytes": footer_bytes}]
    child = pq.ParquetFile(MetadataView(ranges, child_entry))
    ranges.allowed += column_ranges(child, child_entry, set(CHILD_COLUMNS))
    children = child.read(columns=CHILD_COLUMNS, use_threads=False).to_pylist()
    child_lookup = {}
    for r in children:
        mode = Path(r["internal:relative_path"]).name.removesuffix(".png")
        key = r["internal:parent_id"], mode
        if key in child_lookup:
            raise ValueError("Duplicate child image metadata")
        child_lookup[key] = {"offset": r["internal:offset"], "bytes": r["internal:size"]}
    table = sample.read(columns=COLUMNS[:4], use_threads=False)
    projected = {}
    for sid in {p[k] for p in pairs for k in ["candidate_id", "training_id"]}:
        i, _ = lookup[sid]
        projected[sid] = table.slice(i, 1).to_pylist()[0]
    results = []
    for pair in pairs:
        a, b = projected[pair["candidate_id"]], projected[pair["training_id"]]
        keys_a = sorted(ring_key(r) for r in split_rings(a["ml:polygon_xy"], a["ml:polygon_len"]))
        keys_b = sorted(ring_key(r) for r in split_rings(b["ml:polygon_xy"], b["ml:polygon_len"]))
        # Geometry counts are for only the six already-selected spatial probe
        # rows. They contain no damage grades or target composition.
        images = {}
        for mode in ["pre", "post"]:
            ca = child_lookup[lookup[pair["candidate_id"]][1], mode]
            cb = child_lookup[lookup[pair["training_id"]][1], mode]
            images[mode] = {"candidate_archive_reference": ca, "training_archive_reference": cb,
                "encoded_size_equal": ca["bytes"] == cb["bytes"], "same_archive_payload_reference": ca == cb,
                "image_bytes_read": 0, "pixel_hash_equality_verified": False}
        results.append({**pair, "xy_polygon_fingerprint_multisets_equal_round6": keys_a == keys_b,
            "candidate_geometry_rings": len(keys_a), "training_geometry_rings": len(keys_b),
            "raw_xy_array_equal": np.array_equal(a["ml:polygon_xy"], b["ml:polygon_xy"]),
            "raw_wgs84_array_equal": np.array_equal(a["ml:polygon_lnglat"], b["ml:polygon_lnglat"]),
            "images": images, "pixel_hash_equality_verified": False})
    assert_no_protected_reads(ranges.accesses, protected)
    if ranges.remote.budget.received or ranges.remote.budget.reserved or not all(x["cache_hit"] for x in ranges.accesses):
        raise ValueError("Offline probe boundary violated")
    report = {"stage": "POST_SCREEN_GEOMETRY_DUPLICATE_PROBE_COMPLETE", "pairs": results,
        "range_receipts": ranges.accesses, "protected_ranges": protected,
        "no_protected_payload_intersections": True, "network_bytes_received": 0,
        "image_payload_bytes_read": 0, "new_damage_reads": 0,
        "validation_or_confirmation_scores_run": False, "pixel_hash_equality_verified": False,
        "all_original_candidates_retained": True}
    atomic_json(args.run / "geometry_duplicate_probe.json", report)
    print(json.dumps({"stage": report["stage"], "pairs": [{k: v for k, v in r.items() if k != "images"} for r in results]}), flush=True)


if __name__ == "__main__":
    main()
