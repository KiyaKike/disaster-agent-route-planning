"""Geometry-only overlap screening; annotation envelopes are NOT image footprints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_hold_taco import CachedRanges, MetadataView, REVISION, read_index, column_ranges, assert_no_protected_reads
from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

COLUMNS = ["ml:polygon_lnglat", "ml:polygon_lnglat_len"]


def ring_envelope(flat):
    flat = np.asarray(flat, dtype=float)
    if not flat.size:
        return None
    if flat.size % 2 or not np.isfinite(flat).all():
        raise ValueError("Invalid geographic coordinates")
    points = flat.reshape(-1, 2)
    if (np.abs(points[:, 0]) > 180).any() or (np.abs(points[:, 1]) > 90).any():
        raise ValueError("Geographic coordinate range differs")
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


def overlap_indices(box, all_boxes):
    if box is None:
        return np.array([], dtype=int)
    b = np.asarray(box)
    return np.flatnonzero((all_boxes[:, 0] < b[2]) & (all_boxes[:, 2] > b[0]) &
                          (all_boxes[:, 1] < b[3]) & (all_boxes[:, 3] > b[1]))


def components(ids, pairs):
    parent = {sid: sid for sid in ids}
    def root(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    for a, b in pairs:
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for sid in ids:
        groups.setdefault(root(sid), []).append(sid)
    return [sorted(values) for _, values in sorted(groups.items())]


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["design", "metadata-audit", "identities", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((args.design / "cohort_selection.json").read_text())
    receipt = json.loads((args.design / "selection_receipt.json").read_text())
    if sha((args.design / "cohort_selection.json").read_bytes()) != receipt["selection_sha256"]:
        raise ValueError("Frozen candidate cohort differs")
    freeze(args.out / "spatial_protocol.json", {
        "version": "hold_annotation_envelope_screen_v1", "revision": REVISION,
        "selection_sha256": receipt["selection_sha256"], "projected_columns": COLUMNS,
        "max_network_bytes": 64 * 1024 * 1024,
        "overlap": "Strict positive-area rectangle intersection of WGS84 building-annotation envelopes, not full-scene footprints.",
        "missing_geometry": "Record missing envelope; do not treat it as geographic independence.",
        "training": "Report envelope intersections against the audited published detector train+tier3 identity pool; no damage outcomes or perception scores.",
        "policy_split": "Retain every prespecified candidate; flag cross-split intersections and connected candidate groups. No automatic replacement.",
        "conclusion_limit": "No intersection does not certify absence of full-image overlap or nearby pixels."})
    identities = json.loads((args.metadata_audit / "identity_rows.json").read_text())
    lookup = {r["ml:source_file"]: i for i, r in enumerate(identities)}
    author = json.loads(args.identities.read_text())
    train_ids = set(author["train"]) | set(author["tier3"])
    if not train_ids.issubset(lookup):
        raise ValueError("Training identities missing from source")
    cached = CachedRanges(args.metadata_audit / "ranges", 64 * 1024 * 1024)
    index = read_index(cached)
    parquet = pq.ParquetFile(MetadataView(cached, index["METADATA/sample.parquet"]))
    protected = column_ranges(parquet, index["METADATA/sample.parquet"], {"ml:damage"})
    print(json.dumps({"stage": "PROJECT_GEOGRAPHIC_METADATA_ONLY"}), flush=True)
    table = parquet.read(columns=COLUMNS, use_threads=False)
    geo = table["ml:polygon_lnglat"].combine_chunks()
    values, offsets = geo.values.to_numpy(zero_copy_only=False), geo.offsets.to_numpy(zero_copy_only=False)
    boxes = []
    for i in range(len(identities)):
        envelope = ring_envelope(values[offsets[i]:offsets[i+1]])
        boxes.append(envelope if envelope is not None else [float("nan")] * 4)
    all_boxes = np.asarray(boxes)
    training_order = sorted(train_ids)
    train_boxes = all_boxes[[lookup[sid] for sid in training_order]]
    selected_ids = [r["sample_id"] for r in cohort]
    selected_boxes = all_boxes[[lookup[sid] for sid in selected_ids]]
    by_id = {r["sample_id"]: r for r in cohort}
    selected_pairs, cross_split, per_sample = [], [], []
    missing = []
    for i, row in enumerate(cohort):
        sid = row["sample_id"]
        box = selected_boxes[i].tolist() if np.isfinite(selected_boxes[i]).all() else None
        if box is None:
            missing.append(sid)
        for j in overlap_indices(box, selected_boxes):
            if j <= i:
                continue
            other = selected_ids[j]
            selected_pairs.append((sid, other))
            if row["split"] != by_id[other]["split"]:
                cross_split.append({"sample_a": sid, "split_a": row["split"], "sample_b": other, "split_b": by_id[other]["split"]})
        overlaps = [training_order[j] for j in overlap_indices(box, train_boxes)]
        per_sample.append({"sample_id": sid, "split": row["split"], "annotation_envelope_wgs84": box,
                           "training_annotation_envelope_overlap_ids": overlaps})
    assert_no_protected_reads(cached.accesses, protected)
    report = {"stage": "ANNOTATION_ENVELOPE_SCREEN_ONLY", "candidate_scenes": len(cohort),
        "source_training_id_count": len(training_order), "candidates_without_geometry": missing,
        "selected_annotation_envelope_overlap_pairs": selected_pairs,
        "cross_policy_split_annotation_envelope_overlaps": cross_split,
        "candidate_envelope_components": components(selected_ids, selected_pairs), "per_sample": per_sample,
        "candidates_with_training_annotation_envelope_overlap": sum(bool(r["training_annotation_envelope_overlap_ids"]) for r in per_sample),
        "damage_payload_read": False, "protected_ranges": protected, "no_damage_payload_intersections": True,
        "full_image_footprints_available": False, "geographic_image_overlap_excluded": False,
        "whole_216_cohort_admitted": False, "network_bytes_this_run": cached.remote.budget.received,
        "read_receipts": cached.accesses, "runner_sha256": sha(Path(__file__).read_bytes())}
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ["candidate_scenes", "candidates_without_geometry", "candidates_with_training_annotation_envelope_overlap", "cross_policy_split_annotation_envelope_overlaps", "network_bytes_this_run"]}), flush=True)


if __name__ == "__main__":
    main()
