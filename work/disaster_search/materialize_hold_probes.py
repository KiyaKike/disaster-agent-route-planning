"""Materialize ONLY six frozen development probes from the pinned TACO source.

The shared damage column is decoded transiently. No unselected damage values
are returned, printed, summarized or cached. Two OLD calibration scenes may
be inspected inside this materializer solely to verify the class mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_hold_taco import CachedRanges, MetadataView, REVISION, column_ranges, read_index
from audit_xbd_membership import MAX_READ
from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from prepare_xbd_sample import polygon_centroid
from verify_taco_polygon_bridge import ring_key, split_rings

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
MAX_NETWORK = 96 * 1024 * 1024
COLUMNS = ["ml:polygon_xy", "ml:polygon_len", "ml:damage", "ml:height", "ml:width"]
CHILD_COLUMNS = ["internal:parent_id", "internal:relative_path", "internal:offset", "internal:size"]


def intersects(offset, size, protected):
    return any(max(offset, p["offset"]) < min(offset + size, p["offset"] + p["bytes"]) for p in protected)


class TransientDamageRanges:
    """Use persistent caches for images/geometry, never for damage payloads."""
    def __init__(self, cached, protected):
        self.cached, self.protected, self.accesses = cached, protected, []
        for receipt in cached.root.glob("*.json"):
            r = json.loads(receipt.read_text())
            if intersects(r["offset"], r["bytes"], protected) and receipt.with_suffix(".bin").exists():
                raise ValueError("Shared damage bytes already persist in the range cache")

    def read(self, offset, size):
        if not intersects(offset, size, self.protected):
            before = len(self.cached.accesses)
            result = self.cached.read(offset, size)
            self.accesses.extend(self.cached.accesses[before:])
            return result
        parts = []
        while size:
            count = min(size, MAX_READ)
            self.cached.remote.seek(offset)
            data = self.cached.remote.read(count)
            self.accesses.append({"offset": offset, "bytes": count, "sha256": sha(data),
                                  "transient_damage_payload": True, "persisted": False})
            parts.append(data)
            offset += count
            size -= count
        return b"".join(parts)


def validate_probes(selected, cohort):
    expected = [r for r in cohort if r["interface_probe"]]
    if selected != expected or len(selected) != 6 or any(r["split"] != "development" for r in selected):
        raise ValueError("Only the six frozen development probes may be materialized")
    if len({r["sample_id"] for r in selected}) != 6:
        raise ValueError("Repeated probe identity")


def grade_name(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < len(CLASSES):
        raise ValueError("Unknown source damage code; do not reinterpret as negative")
    return CLASSES[int(value)]


def build_features(sid, row):
    rings = split_rings(row["ml:polygon_xy"], row["ml:polygon_len"])
    if len(rings) != len(row["ml:damage"]):
        raise ValueError("Object/grade arrays differ in length")
    width, height = row["ml:width"], row["ml:height"]
    seen, features = set(), []
    for index, (ring, code) in enumerate(zip(rings, row["ml:damage"])):
        fingerprint = ring_key(ring)
        if fingerprint in seen:
            raise ValueError("Duplicate per-object polygon; do not double-count")
        seen.add(fingerprint)
        if ring[-1] != ring[0]:
            raise ValueError("Unclosed source polygon")
        wkt = "POLYGON ((" + ", ".join(f"{x:.17g} {y:.17g}" for x, y in ring) + "))"
        cx, cy = polygon_centroid(wkt)
        if not 0 <= cx <= width or not 0 <= cy <= height:
            raise ValueError("Polygon centroid outside image")
        if any(not (-1e-6 <= x <= width + 1e-6 and -1e-6 <= y <= height + 1e-6) for x, y in ring):
            raise ValueError("Source polygon vertex outside image; inspect rather than clip silently")
        uid = "derived-taco-" + sha(f"{REVISION}|{sid}|{index}".encode())
        features.append({"wkt": wkt, "properties": {"uid": uid, "uid_origin": "DERIVED_SOURCE_OBJECT_INDEX",
            "source_object_index": index, "polygon_sha256_round6": fingerprint, "subtype": grade_name(code)}})
    return features


def extract_allowed_rows(parquet, identities, wanted):
    """Trusted boundary: the full Arrow column never escapes this function."""
    lookup = {r["ml:source_file"]: i for i, r in enumerate(identities)}
    if len(lookup) != len(identities) or len(set(wanted)) != len(wanted):
        raise ValueError("Ambiguous source or requested row identities")
    table = parquet.read(columns=COLUMNS, use_threads=False)
    if table.num_rows != len(identities):
        raise ValueError("Projected rows/identity inventory mismatch")
    result = {sid: table.slice(lookup[sid], 1).to_pylist()[0] for sid in wanted}
    del table
    return result


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--mini-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.design / "next_development_probes.json").read_text())
    cohort = json.loads((args.design / "cohort_selection.json").read_text())
    validate_probes(selected, cohort)
    selection_receipt = json.loads((args.design / "selection_receipt.json").read_text())
    if selection_receipt["selection_sha256"] != sha((args.design / "cohort_selection.json").read_bytes()):
        raise ValueError("Frozen cohort checksum differs")
    contract = json.loads((args.design / "experiment_contract_v2_1.json").read_text())
    if contract["version"] != "search_v2_1_design_20261025" or "annotation_access_after_this_design_freeze" not in contract["sampling"]:
        raise ValueError("Explicit shared-column access amendment required")
    bridge = json.loads((args.bridge / "report.json").read_text())
    amendment = json.loads((args.design / "contract_amendment_01.json").read_text())
    if amendment["bridge_report_sha256"] != sha((args.bridge / "report.json").read_bytes()):
        raise ValueError("Previously verified bridge differs from the design amendment")
    old_ids = [r["sample_id"] for r in bridge["results"]]
    collection = json.loads((args.metadata_audit / "collection.json").read_text())
    damage = next(t for t in collection["ml:contract"]["targets"] if t["name"] == "damage")
    if damage["classes"] != CLASSES or damage["ignore_index"] != 4:
        raise ValueError("Frozen source legend differs")
    protocol = {"version": "six_development_hold_probes_v1", "source_revision": REVISION,
        "design_sha256": sha((args.design / "experiment_contract_v2_1.json").read_bytes()),
        "selected": selected, "old_calibration_class_bridge": old_ids,
        "projected_columns": COLUMNS, "classes": CLASSES, "unknown_code": 4, "target_codes": [2, 3],
        "max_network_bytes": MAX_NETWORK,
        "shared_damage_access": "Shared full-column payload decoded transiently by trusted extractor. Return only six frozen development rows plus two previously used calibration rows. Never persist shared damage column or expose other outcomes.",
        "identity": "Derived per-source-object index plus polygon fingerprint; not recovered original UID.",
        "duplicates": "Reject duplicate object polygons in a scene; report exact image/pixel duplicate checks within probes and versus downloaded mini72. Whole training pool pixel/geographic duplicate checks remain pending.",
        "pre_labels": "Not available in this repackaging; do not synthesize a pre-disaster annotation file.",
        "scope": "Data interface/annotation QA only, not a policy performance experiment."}
    freeze(args.out / "materialization_protocol.json", protocol)
    identity_rows = json.loads((args.metadata_audit / "identity_rows.json").read_text())
    source_lookup = {r["ml:source_file"]: r for r in identity_rows}
    for r in selected:
        if source_lookup[r["sample_id"]]["ml:split_original"] != "hold":
            raise ValueError("Probe is not in original hold")
    cached = CachedRanges(args.metadata_audit / "ranges", MAX_NETWORK)
    index = read_index(cached)
    schema_reader = pq.ParquetFile(MetadataView(cached, index["METADATA/sample.parquet"]))
    protected = column_ranges(schema_reader, index["METADATA/sample.parquet"], {"ml:damage"})
    prefix_accesses = list(cached.accesses)
    reader = TransientDamageRanges(cached, protected)
    sample = pq.ParquetFile(MetadataView(reader, index["METADATA/sample.parquet"]))
    wanted = [r["sample_id"] for r in selected] + old_ids
    rows = extract_allowed_rows(sample, identity_rows, wanted)
    mini_manifest = json.loads((args.mini_data / "manifest.json").read_text())
    known = {s["sample_id"]: s for s in mini_manifest["samples"]}
    expected_hashes = {f["path"]: f["sha256"] for f in mini_manifest["files"]}
    class_bridge = []
    for result in bridge["results"]:
        sid = result["sample_id"]
        path = known[sid]["files"]["post_label"]
        raw = (args.mini_data / path).read_bytes()
        if sha(raw) != expected_hashes[path]:
            raise ValueError("Old reference label changed")
        original = {f["properties"]["uid"]: f["properties"].get("subtype", "un-classified") for f in json.loads(raw)["features"]["xy"]}
        if len(rows[sid]["ml:damage"]) != len(result["mapping"]):
            raise ValueError("Old reference grade length differs")
        for mapped in result["mapping"]:
            index_number = mapped["source_object_index"]
            if grade_name(rows[sid]["ml:damage"][index_number]) != original[mapped["original_uid_on_this_known_scene"]]:
                raise ValueError("Published class ordering does not match known original labels")
        class_bridge.append({"sample_id": sid, "matched_grades": len(result["mapping"]), "all_equal": True})
    print(json.dumps({"stage": "OLD_CLASS_MAPPING_VERIFIED", "objects": sum(r["matched_grades"] for r in class_bridge)}), flush=True)
    children = pq.ParquetFile(MetadataView(reader, index["METADATA/children.parquet"])).read(columns=CHILD_COLUMNS, use_threads=False).to_pylist()
    records, file_records, samples, image_signatures = [], [], [], {}
    for selected_row in selected:
        sid = selected_row["sample_id"]
        source_row = rows[sid]
        features = build_features(sid, source_row)
        folder = args.out / "samples" / sid
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for mode in ["pre", "post"]:
            items = [r for r in children if r["internal:parent_id"] == source_lookup[sid]["internal:current_id"] and r["internal:relative_path"].endswith(f"/{mode}.png")]
            if len(items) != 1 or not 0 < items[0]["internal:size"] <= 8 * 1024 * 1024:
                raise ValueError("Invalid image payload mapping or size")
            item = items[0]
            path = folder / f"{sid}_{mode}_disaster.png"
            if path.exists():
                raw = path.read_bytes()
                receipt_path = path.with_suffix(".receipt.json")
                receipt = json.loads(receipt_path.read_text())
                if receipt != {"offset": item["internal:offset"], "bytes": item["internal:size"], "sha256": sha(raw), "revision": REVISION}:
                    raise ValueError("Existing image differs from source receipt")
            else:
                raw = reader.read(item["internal:offset"], item["internal:size"])
                path.write_bytes(raw)
                atomic_json(path.with_suffix(".receipt.json"), {"offset": item["internal:offset"], "bytes": item["internal:size"], "sha256": sha(raw), "revision": REVISION})
            with Image.open(io.BytesIO(raw)) as im:
                if im.size != (source_row["ml:width"], source_row["ml:height"]) or im.mode != "RGB":
                    raise ValueError("PNG shape/mode differs from metadata")
                pixels = np.asarray(im)
                pixel_hash = sha(pixels.tobytes())
            relative = str(path.relative_to(args.out))
            files[f"{mode}_image"] = relative
            file_records.append({"path": relative, "sha256": sha(raw), "pixel_sha256_rgb": pixel_hash})
            image_signatures.setdefault(pixel_hash, []).append(f"{sid}:{mode}")
        label_path = folder / f"{sid}_post_disaster.json"
        label = {"metadata": {"width": source_row["ml:width"], "height": source_row["ml:height"],
                  "img_name": f"{sid}_post_disaster.png", "disaster": selected_row["event"]},
                 "features": {"xy": features},
                 "provenance": {"revision": REVISION, "identity": "DERIVED_SOURCE_OBJECT_INDEX", "original_uid_retained": False}}
        freeze(label_path, label)
        files["post_label"] = str(label_path.relative_to(args.out))
        file_records.append({"path": files["post_label"], "sha256": sha(label_path.read_bytes())})
        samples.append({**selected_row, "files": files})
        counts = Counter(f["properties"]["subtype"] for f in features)
        records.append({"sample_id": sid, "event": selected_row["event"], "buildings": len(features),
                        "class_counts": dict(counts), "targets": counts["major-damage"] + counts["destroyed"],
                        "unknown": counts["un-classified"], "dimensions": [source_row["ml:width"], source_row["ml:height"]]})
        print(json.dumps({"stage": "DEVELOPMENT_PROBE_MATERIALIZED", "sample_id": sid}), flush=True)
    duplicate_pairs = [{"pixel_sha256": h, "observations": values} for h, values in image_signatures.items() if len(values) > 1]
    old_overlap = []
    for item in mini_manifest["files"]:
        if item["path"].endswith(".png"):
            with Image.open(args.mini_data / item["path"]) as im:
                pixel_hash = sha(np.asarray(im.convert("RGB")).tobytes())
            if pixel_hash in image_signatures:
                old_overlap.append({"mini_file": item["path"], "probes": image_signatures[pixel_hash], "pixel_sha256": pixel_hash})
    if duplicate_pairs or old_overlap:
        atomic_json(args.out / "duplicate_failure.json", {"within_probes": duplicate_pairs, "mini_overlap": old_overlap})
        raise ValueError("Exact pixel duplicates found; keep cohort and investigate, never swap automatically")
    manifest = {"dataset_id": "hold_six_frozen_development_probes_v1", "samples": samples,
                "files": file_records, "source_revision": REVISION,
                "protocol_sha256": sha((args.out / "materialization_protocol.json").read_bytes())}
    freeze(args.out / "manifest.json", manifest)
    report = {"stage": "SIX_DEVELOPMENT_PROBE_DATA_INTERFACE_CHECK", "samples": records,
        "pairs": len(records), "buildings": sum(r["buildings"] for r in records),
        "targets": sum(r["targets"] for r in records), "unknown": sum(r["unknown"] for r in records),
        "old_class_bridge": class_bridge, "class_mapping_verified_on_old_objects": sum(r["matched_grades"] for r in class_bridge),
        "within_probe_pixel_duplicates": duplicate_pairs, "mini72_pixel_duplicates": old_overlap,
        "shared_confirmation_damage_payload_physically_read": True,
        "only_selected_development_and_old_calibration_damage_retained": True,
        "unselected_damage_values_or_summaries_exposed": False, "shared_damage_payload_cached": False,
        "confirmation_model_or_policy_scores_run": False, "data_interface_passed": True,
        "geographic_overlap_audited": False, "full_training_pool_pixel_duplicates_audited": False,
        "whole_216_cohort_admitted": False, "sensor_or_policy_efficacy_validated": False,
        "network_bytes_this_run": cached.remote.budget.received,
        "read_receipts": prefix_accesses + reader.accesses,
        "runner_sha256": sha(Path(__file__).read_bytes()), "manifest_sha256": sha((args.out / "manifest.json").read_bytes())}
    atomic_json(args.out / "data_report.json", report)
    atomic_json(args.out / "interface_gate.json", {
        "version": "SIX_DEVELOPMENT_RGB_ACCESS_GATE", "data_interface_passed": True,
        "sample_ids": [s["sample_id"] for s in selected],
        "manifest_sha256": report["manifest_sha256"], "scope": "Prediction gate contains no class counts or label outcomes."})
    print(json.dumps({k: report[k] for k in ["pairs", "buildings", "targets", "unknown", "data_interface_passed", "network_bytes_this_run"]}), flush=True)


if __name__ == "__main__":
    main()
