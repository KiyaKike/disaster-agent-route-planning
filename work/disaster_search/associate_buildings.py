"""Evaluator-only raster-visible building association for the frozen RGB pilot.

This is a posthoc protocol correction, not a replacement of the original scores.
No object IDs or label-derived fields are written into planner observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_changeos_pilot import counts
from prepare_xbd_sample import polygon_centroid, TARGET_SUBTYPES, NEGATIVE_SUBTYPES
from run_changeos_pilot import freeze_plan


PROTOCOL = {
    "version": "raster_visibility_v2",
    "use": "EVALUATOR_ONLY_NEVER_PLANNER_INPUT",
    "timing": "Posthoc correction motivated by the inspected calibration pilot boundary errors; not preregistered pilot evaluation.",
    "rasterization": "Pillow ImageDraw full-scene pixel polygon, then half-open crop slices; shared pixels have one cell owner.",
    "cell_truth": "True if any severe building has >=4 visible raster pixels. Otherwise null if any severe fragment or unknown building is visible, else false.",
    "discovery_rule": {"min_overlap_pixels": 4, "min_visible_footprint_coverage": 0.1},
    "sensitivity_coverage": [0.05, 0.1, 0.25],
    "matching": "For each severe building, intersect its visible footprint with the unchanged canonical severe_mask. This is a coverage proxy, not standard instance AP or IoU matching.",
    "deduplication": "Union (sample_id, building_uid) across visited cells; repeated quality modes cannot create extra discoveries.",
    "denominator": "Unique severe buildings with >=4 visible pixels in at least one selected cell, not all targets in the scene.",
    "unknowns": "Unknown subtype footprints remain ambiguous, even if their centroid is outside this cell. Positives take precedence for cell presence.",
    "pixel_diagnostic": "Severe-pixel precision against union of severe footprints; pixels overlapping unknown-only footprints excluded. Broad masks can match buildings, so pixel precision must also be reported.",
    "scope": "Engineering diagnostics on calibration pilot only; checkpoint training exposure expected. No thresholds selected by best score. No new inference or likelihood calibration.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rasterize(wkt, size):
    polygon_centroid(wkt)  # Reuse strict validation of closed, nondegenerate simple XY polygons.
    coordinates = wkt.strip().split("((", 1)[1].split("))", 1)[0]
    points = [tuple(map(float, pair.split())) for pair in coordinates.split(",")]
    if not all(np.isfinite(p).all() for p in points):
        raise ValueError("Non-finite polygon coordinates")
    image = Image.new("1", size, 0)
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


def visible_objects(features, size, cells):
    """Build evaluator-only crops; rasterize once globally to avoid edge artifacts."""
    result = {cell: [] for cell in cells}
    seen = set()
    for feature in features:
        uid = feature["properties"]["uid"]
        if uid in seen:
            raise ValueError("Duplicate building UID")
        seen.add(uid)
        subtype = feature["properties"].get("subtype", "un-classified")
        if subtype in NEGATIVE_SUBTYPES:
            continue
        mask = rasterize(feature["wkt"], size)
        for cell, (x0, y0, x1, y1) in cells.items():
            part = mask[y0:y1, x0:x1]
            if part.any():
                result[cell].append({"uid": uid, "target": subtype in TARGET_SUBTYPES,
                                     "mask": part.copy()})
    return result


def associate(severe_mask, objects, coverage=0.1):
    severe_mask = np.asarray(severe_mask, dtype=bool)
    target_union = np.zeros_like(severe_mask)
    unknown_union = np.zeros_like(severe_mask)
    visible, matches, overlap_records = [], [], []
    has_fragment = False
    for obj in objects:
        mask = obj["mask"]
        if mask.shape != severe_mask.shape:
            raise ValueError("Mask shape mismatch")
        area = int(mask.sum())
        if obj["target"]:
            target_union |= mask
            has_fragment |= 0 < area < 4
            if area >= 4:
                visible.append(obj["uid"])
            overlap = int((mask & severe_mask).sum())
            overlap_records.append({"uid": obj["uid"], "visible_pixels": area,
                                    "severe_overlap_pixels": overlap,
                                    "visible_footprint_coverage": overlap / area if area else 0})
            if area >= 4 and overlap >= 4 and overlap / area >= coverage:
                matches.append(obj["uid"])
        else:
            unknown_union |= mask
    target = True if visible else (None if has_fragment or unknown_union.any() else False)
    ignored = severe_mask & unknown_union & ~target_union
    return {"target": target, "visible_target_uids": sorted(visible), "matched_target_uids": sorted(matches),
            "overlap_records": overlap_records,
            "severe_pixels_on_targets": int((severe_mask & target_union).sum()),
            "severe_pixels_on_unknown_only": int(ignored.sum()),
            "severe_pixels_on_known_negative_or_background": int((severe_mask & ~target_union & ~unknown_union).sum())}


def discovery_summary(rows):
    visible, found = set(), set()
    match_events = 0
    for row in rows:
        visible.update((row["sample_id"], uid) for uid in row["visible_target_uids"])
        found.update((row["sample_id"], uid) for uid in row["matched_target_uids"])
        match_events += len(row["matched_target_uids"])
    on_target = sum(r["severe_pixels_on_targets"] for r in rows)
    on_negative = sum(r["severe_pixels_on_known_negative_or_background"] for r in rows)
    return {"unique_visible_targets": len(visible), "unique_matched_targets": len(found),
            "matched_observation_building_pairs": match_events,
            "repeated_matches_not_counted_again": match_events - len(found),
            "visible_building_coverage_proxy": len(found) / len(visible) if visible else None,
            "severe_pixel_precision_known_only": on_target / (on_target + on_negative) if on_target + on_negative else None,
            "severe_pixels_on_targets": on_target, "severe_pixels_on_known_negative_or_background": on_negative,
            "severe_pixels_on_unknown_only": sum(r["severe_pixels_on_unknown_only"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--full-calibration", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    if args.validation and args.full_calibration:
        raise ValueError("Choose one evaluation split")
    selected_split = "validation" if args.validation else "calibration"
    if args.full_calibration:
        protocol["scope"] = "All 24 frozen calibration scenes and 64 cells, using the pilot-motivated v2 rules; no validation/test evaluation. Model training exposure expected."
    if args.validation:
        protocol["scope"] = "All 24 frozen validation scenes, unchanged raster_visibility_v2 rules. No threshold tuning or test evaluation. Model training exposure expected."
    freeze_plan(args.out / "association_protocol.json", protocol)
    if args.plan_only:
        print("Frozen raster_visibility_v2 before applying the new association rules")
        return
    manifest = json.loads((args.data / "manifest.json").read_text())
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    source = args.pilot / "observations.json"
    observations = json.loads(source.read_text())
    if args.full_calibration or args.validation:
        expected_ids = {f"{s['sample_id']}:r{r:02d}c{c:02d}:{m}" for s in samples.values()
                        if s["split"] == selected_split for r in range(8) for c in range(8) for m in ("native", "blurred")}
        if len(observations) != 3072 or {o["observation_id"] for o in observations} != expected_ids:
            raise ValueError("Full calibration requires the complete frozen observation grid")
    old_report = args.pilot / "evaluation_report.json"
    old_hash = sha(old_report)
    objects_by_scene, centroid_by_scene, label_hashes = {}, {}, {}
    rows = []
    seen = set()
    for observation in observations:
        sid, cid = observation["sample_id"], observation["cell_id"]
        if observation["split"] != selected_split or samples[sid]["split"] != selected_split:
            raise ValueError("Observation/evaluator split mismatch")
        if observation["observation_id"] in seen:
            raise ValueError("Duplicate observation ID")
        seen.add(observation["observation_id"])
        if sid not in objects_by_scene:
            label_path = samples[sid]["files"]["post_label"]
            if sha(args.data / label_path) != expected[label_path]:
                raise ValueError("Raw label hash mismatch")
            label_hashes[label_path] = expected[label_path]
            labels = json.loads((args.data / label_path).read_text())
            index = json.loads((args.prepared_data / sid / "local_observation_index.json").read_text())
            cells = {c["cell_id"]: c["pixel_bbox"] for c in index["cells"]
                     if any(o["sample_id"] == sid and o["cell_id"] == c["cell_id"] for o in observations)}
            size = labels["metadata"]["width"], labels["metadata"]["height"]
            for x0, y0, x1, y1 in cells.values():
                if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
                    raise ValueError("Invalid crop")
            objects_by_scene[sid] = visible_objects(labels["features"]["xy"], size, cells)
            centroid_by_scene[sid] = json.loads((args.prepared_data / sid / "evaluation_truth.json").read_text())["cells"]
        cache_path = Path(observation["cache_file"])
        if sha(cache_path) != observation["cache_sha256"]:
            raise ValueError("Inference cache hash mismatch")
        with np.load(cache_path, allow_pickle=False) as cache:
            mask = cache["severe_mask"]
        if mask.dtype != np.bool_ or int(mask.sum()) != observation["predicted_severe_pixels"]:
            raise ValueError("Canonical severe mask does not match frozen observation")
        if bool(mask.sum() >= 4) != observation["declared_target"]:
            raise ValueError("Frozen declaration rule mismatch")
        row = {key: observation[key] for key in ("sample_id", "cell_id", "mode", "observation_id", "evidence_group_id")}
        row.update(associate(mask, objects_by_scene[sid][cid]))
        row["predicted"] = observation["declared_target"]
        row["centroid_target"] = centroid_by_scene[sid][cid]["target_present"]
        rows.append(row)
    modes = sorted({r["mode"] for r in rows})
    summaries = {}
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode]
        known = [r for r in mode_rows if r["target"] is not None]
        summaries[mode] = {"visible_cell_confusion": {**counts(known), "unknown_excluded": len(mode_rows) - len(known)},
                           "building_proxy": discovery_summary(mode_rows)}
    sensitivity = {}
    for threshold in PROTOCOL["sensitivity_coverage"]:
        changed_rows = []
        for row in rows:
            changed = dict(row)
            changed["matched_target_uids"] = [r["uid"] for r in row["overlap_records"]
                if r["visible_pixels"] >= 4 and r["severe_overlap_pixels"] >= 4 and r["visible_footprint_coverage"] >= threshold]
            changed_rows.append(changed)
        sensitivity[str(threshold)] = {m: discovery_summary([r for r in changed_rows if r["mode"] == m]) for m in modes}
    stage = "FULL_VALIDATION_BOUNDARY_DIAGNOSTICS" if args.validation else ("FULL_CALIBRATION_BOUNDARY_DIAGNOSTICS" if args.full_calibration else "POSTHOC_CALIBRATION_BOUNDARY_DIAGNOSTICS")
    report = {"stage": stage, "test_evaluation_run": False,
              "protocol_sha256": sha(args.out / "association_protocol.json"),
              "observations_sha256": sha(source), "original_centroid_report_sha256": old_hash,
              "label_hashes": label_hashes, "results_by_mode": summaries,
              "combined_modes_deduplication_check_only": discovery_summary(rows),
              "sensitivity_not_for_threshold_selection": sensitivity,
              "cell_truth_changes": [{k: r[k] for k in ("sample_id", "cell_id", "mode", "centroid_target", "target")}
                                     for r in rows if r["centroid_target"] != r["target"]],
              "rows": rows, "limitations": [protocol["timing"], protocol["scope"],
                  "The 4-pixel and 10% coverage rules are transparent engineering choices, not validated discovery thresholds.",
                  "A broad severe mask may cover multiple buildings. Coverage proxy needs the accompanying pixel-precision diagnostic; it is not instance-detection AP.",
                  "Repeated quality modes share a base image. Combined union is a deduplication check, not an independent trial.",
                  "Only selected cells are covered; not a search-policy experiment, full-scene recall, or calibrated likelihood."]}
    if sha(old_report) != old_hash:
        raise RuntimeError("Original centroid report unexpectedly changed")
    (args.out / "association_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"results_by_mode": summaries, "combined_dedup_check": report["combined_modes_deduplication_check_only"],
                      "truth_changes": len(report["cell_truth_changes"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
