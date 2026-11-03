"""Validate polygon identity and PNG correspondence on OLD calibration data only.

The source has a single Parquet row group; projecting XY needs ~57 MB. Damage
is never projected. The output contains mappings for the two selected old
calibration scenes only, and is not evidence that every hold label is correct.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from audit_hold_taco import CachedRanges, MetadataView, REVISION, read_index, column_ranges, assert_no_protected_reads
from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

MAX_NETWORK = 80 * 1024 * 1024
GEOMETRY_COLUMNS = ["ml:polygon_xy", "ml:polygon_len"]
CHILD_COLUMNS = ["internal:parent_id", "internal:relative_path", "internal:offset", "internal:size"]


def ring_key(points):
    """Stable pixel-ring fingerprint, invariant to first vertex and winding."""
    points = [tuple(map(float, point)) for point in points]
    if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError("Expected finite XY coordinate pairs")
    points = [tuple(round(x, 6) for x in point) for point in points]
    if points and points[-1] == points[0]:
        points.pop()
    if len(points) < 3:
        raise ValueError("Degenerate ring")
    variants = [tuple(p[i:] + p[:i]) for p in (points, list(reversed(points))) for i in range(len(p))]
    return sha(json.dumps(min(variants), separators=(",", ":")).encode())


def wkt_points(wkt):
    if not wkt.startswith("POLYGON ((") or not wkt.endswith("))"):
        raise ValueError("Expected original simple polygon WKT")
    return [tuple(map(float, p.strip().split())) for p in wkt.split("((", 1)[1][:-2].split(",")]


def split_rings(flat, lengths):
    if len(flat) != 2 * sum(lengths) or any(n < 3 for n in lengths):
        raise ValueError("Malformed per-object polygon array")
    rings, offset = [], 0
    for count in lengths:
        ring = list(zip(flat[offset:offset + 2 * count:2], flat[offset + 1:offset + 2 * count:2]))
        rings.append(ring)
        offset += 2 * count
    return rings


def pair_geometry(features, flat, lengths):
    original = {ring_key(wkt_points(f["wkt"])): f["properties"]["uid"] for f in features}
    if len(original) != len(features):
        raise ValueError("Ambiguous duplicate original polygon; do not invent UID correspondence")
    rings = split_rings(flat, lengths)
    hashes = [ring_key(ring) for ring in rings]
    if len(set(hashes)) != len(hashes) or Counter(hashes) != Counter(original.keys()):
        raise ValueError("Repackaged polygons are not one-to-one with the original")
    return [{"source_object_index": i, "polygon_sha256_round6": key,
             "original_uid_on_this_known_scene": original[key]} for i, key in enumerate(hashes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-data", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.mini_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    chosen = [min((s for s in manifest["samples"] if s["split"] == "calibration" and s["event"] == event),
                  key=lambda s: s["sample_id"]) for event in ["hurricane-harvey", "santa-rosa-wildfire"]]
    plan = {"version": "taco_old_calibration_polygon_bridge_v1", "revision": REVISION,
        "mini_manifest_sha256": sha(manifest_path.read_bytes()),
        "selected_existing_calibration_ids": [s["sample_id"] for s in chosen],
        "geometry_columns": GEOMETRY_COLUMNS, "child_columns": CHILD_COLUMNS,
        "max_network_bytes": MAX_NETWORK,
        "damage_column_access": False,
        "scope": "Full XY columns must be projected because source has one row group. Only two old calibration scenes are inspected/reported. No damage column or new hold image payload is read.",
        "identity_interpretation": "Derived per-source-polygon identity is possible only if original object correspondence is established; not an original UID retained in the repackaging.",
    }
    freeze(args.out / "bridge_protocol.json", plan)
    audit = json.loads((args.metadata_audit / "report.json").read_text())
    if audit["revision"] != REVISION or not audit["hold_exact_name_match"]:
        raise ValueError("Run the pinned identity audit first")
    identity_rows = json.loads((args.metadata_audit / "identity_rows.json").read_text())
    identity_lookup = {r["ml:source_file"]: (i, r["internal:current_id"]) for i, r in enumerate(identity_rows)}
    ranges = CachedRanges(args.metadata_audit / "ranges", max_network_bytes=MAX_NETWORK)
    index = read_index(ranges)
    sample = pq.ParquetFile(MetadataView(ranges, index["METADATA/sample.parquet"]))
    expected_bytes = sum(sample.metadata.row_group(0).column(c).total_compressed_size
        for c in range(sample.metadata.num_columns)
        if sample.metadata.row_group(0).column(c).path_in_schema.split(".list.", 1)[0] in GEOMETRY_COLUMNS)
    if sample.metadata.num_row_groups != 1 or expected_bytes > 64 * 1024 * 1024:
        raise ValueError("Unexpected geometry layout or size")
    print(json.dumps({"stage": "PROJECT_GEOMETRY_ONLY", "compressed_bytes": expected_bytes}), flush=True)
    table = sample.read(columns=GEOMETRY_COLUMNS, use_threads=False)
    protected = column_ranges(sample, index["METADATA/sample.parquet"], {"ml:damage"})
    child = pq.ParquetFile(MetadataView(ranges, index["METADATA/children.parquet"]))
    children = child.read(columns=CHILD_COLUMNS, use_threads=False).to_pylist()
    expected_hashes = {r["path"]: r["sha256"] for r in manifest["files"]}
    results = []
    for selected in chosen:
        sid = selected["sample_id"]
        row_index, internal_id = identity_lookup[sid]
        row = table.slice(row_index, 1).to_pylist()[0]
        label_rel = selected["files"]["post_label"]
        label_bytes = (args.mini_data / label_rel).read_bytes()
        if sha(label_bytes) != expected_hashes[label_rel]:
            raise ValueError("Original label hash differs")
        labels = json.loads(label_bytes)
        mapping = pair_geometry(labels["features"]["xy"], row["ml:polygon_xy"], row["ml:polygon_len"])
        images = []
        for mode in ["pre", "post"]:
            candidates = [r for r in children if r["internal:parent_id"] == internal_id and r["internal:relative_path"].endswith(f"/{mode}.png")]
            if len(candidates) != 1:
                raise ValueError("Ambiguous image payload mapping")
            entry = candidates[0]
            if not 0 < entry["internal:size"] <= 8 * 1024 * 1024:
                raise ValueError("Unexpected PNG payload size")
            raw = ranges.read(entry["internal:offset"], entry["internal:size"])
            original_rel = selected["files"][f"{mode}_image"]
            original = (args.mini_data / original_rel).read_bytes()
            if sha(original) != expected_hashes[original_rel] or sha(raw) != sha(original):
                raise ValueError("Repackaged PNG is not byte-identical to original mirror")
            images.append({"mode": mode, "bytes": len(raw), "sha256": sha(raw), "byte_identical_to_known_original": True})
        results.append({"sample_id": sid, "original_polygon_count": len(mapping), "all_polygons_one_to_one_round6": True,
                        "original_label_sha256": sha(label_bytes), "mapping": mapping, "images": images})
        print(json.dumps({"sample_id": sid, "polygons_matched": len(mapping), "images_byte_identical": 2}), flush=True)
    assert_no_protected_reads(ranges.accesses, protected)
    report = {"stage": "OLD_CALIBRATION_GEOMETRY_AND_IMAGE_BRIDGE", "protocol_sha256": sha((args.out / "bridge_protocol.json").read_bytes()),
        "revision": REVISION, "results": results, "new_hold_damage_read": False,
        "new_hold_images_read": False, "damage_column_read": False, "has_original_uid_column": False,
        "network_bytes_this_runner": ranges.remote.budget.received, "read_receipts": ranges.accesses,
        "protected_column_ranges": protected, "no_damage_payload_intersections": True,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "whole_hold_annotation_validated": False,
        "interpretation": "Object polygons and PNGs match two known original calibration examples. Derived IDs for preserved source polygons are defensible as a separately documented adapter; original UIDs have not been recovered for hold. New development geometry/mask/unknown handling and spatial checks remain required.",
    }
    atomic_json(args.out / "report.json", report)
    print(json.dumps({"stage": report["stage"], "scenes": len(results),
        "network_bytes_this_runner": report["network_bytes_this_runner"], "new_hold_damage_read": False}), flush=True)


if __name__ == "__main__":
    main()
