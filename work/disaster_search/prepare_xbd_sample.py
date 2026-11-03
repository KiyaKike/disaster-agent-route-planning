"""Build separate raw-cue and evaluation-label files for a pinned xBD manifest.

Requires Pillow and numpy. The bundled desktop Python supplies both.
No label is used to construct the coarse image change cue.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


TARGET_SUBTYPES = {"major-damage", "destroyed"}
NEGATIVE_SUBTYPES = {"no-damage", "minor-damage"}


def polygon_centroid(wkt):
    match = re.fullmatch(r"POLYGON\s*\(\(([^()]+)\)\)", wkt.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Only simple original pixel polygons are supported; do not silently guess")
    points = [tuple(float(v) for v in pair.split()) for pair in match[1].split(",")]
    if len(points) < 4 or points[0] != points[-1] or any(len(p) != 2 for p in points):
        raise ValueError("Expected a closed XY polygon")
    cross = [a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:])]
    twice_area = sum(cross)
    if abs(twice_area) < 1e-9:
        raise ValueError("Degenerate building polygon")
    return tuple(sum((a[k] + b[k]) * c for a, b, c in zip(points, points[1:], cross)) / (3 * twice_area)
                 for k in (0, 1))


def grid_truth(features, width, height, side):
    truth = {f"r{y:02d}c{x:02d}": {"target_building_ids": [], "unknown_building_ids": [], "building_ids": []}
             for y in range(side) for x in range(side)}
    seen = set()
    for feature in features:
        properties = feature["properties"]
        uid = properties["uid"]
        if uid in seen:
            raise ValueError("Duplicate building UID")
        seen.add(uid)
        x, y = polygon_centroid(feature["wkt"])
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError("Pixel centroid outside image")
        col = min(side - 1, int(x / width * side))
        row = min(side - 1, int(y / height * side))
        item = truth[f"r{row:02d}c{col:02d}"]
        item["building_ids"].append(uid)
        subtype = properties.get("subtype", "un-classified")
        if subtype in TARGET_SUBTYPES:
            item["target_building_ids"].append(uid)
        elif subtype not in NEGATIVE_SUBTYPES:
            item["unknown_building_ids"].append(uid)
    for item in truth.values():
        item["target_present"] = True if item["target_building_ids"] else (None if item["unknown_building_ids"] else False)
    return truth


def main():
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    # Refuse changed downloaded inputs, including labels.
    for item in manifest["files"]:
        actual = hashlib.sha256((args.data / item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError("Input hash mismatch: " + item["path"])
    reports = []
    png_by_hash = {}
    cross_split_duplicates = []
    for item in manifest["files"]:
        if item["path"].endswith(".png"):
            png_by_hash.setdefault(item["sha256"], []).append(item["path"])
    path_split = {path: s.get("split", "development") for s in manifest["samples"] for path in s["files"].values()}
    for digest, paths in png_by_hash.items():
        if len({path_split[p] for p in paths}) > 1:
            cross_split_duplicates.append({"sha256": digest, "paths": paths})
    uid_splits = {}
    for sample in manifest["samples"]:
        target = args.out / sample["sample_id"]
        target.mkdir(exist_ok=True)
        files = sample["files"]
        with Image.open(args.data / files["pre_image"]) as image:
            pre = image.convert("RGB")
        with Image.open(args.data / files["post_image"]) as image:
            post = image.convert("RGB")
        if pre.size != post.size:
            raise ValueError("Pre/post dimensions differ")
        width, height = pre.size
        side, coarse_side = 8, 128
        a = pre.resize((coarse_side, coarse_side), Image.Resampling.BOX)
        b = post.resize((coarse_side, coarse_side), Image.Resampling.BOX)
        a.save(target / "coarse_pre.png")
        b.save(target / "coarse_post.png")
        delta = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean(axis=2) / 255
        cells = []
        for row in range(side):
            for col in range(side):
                tile = delta[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
                cells.append({"cell_id": f"r{row:02d}c{col:02d}", "row": row, "col": col,
                              "raw_change_score": float(np.percentile(tile, 75)),
                              "prior_probability": None, "prior_quality": None})
        # Write image-only public data before reading labels.
        public = {"sample_id": sample["sample_id"], "event": sample["event"], "grid_side": side,
                  "split": sample.get("split", "development"),
                  "coarse_shape": [128, 128], "coarse_resolution_m": None,
                  "status": "UNCALIBRATED_IMAGE_CUES_ONLY", "cells": cells,
                  "limitations": ["Downsampling is a proxy, not an actual PlanetScope observation.",
                                  "Raw RGB differences are not target probabilities and lack registration/illumination correction.",
                                  "Image-quality likelihoods and the local detector are not implemented."]}
        (target / "public_cues.json").write_text(json.dumps(public, indent=2))
        labels = json.loads((args.data / files["post_label"]).read_text())
        features = labels["features"]["xy"]
        pre_label = json.loads((args.data / files["pre_label"]).read_text())
        for label, image_name in ((pre_label, files["pre_image"]), (labels, files["post_image"])):
            metadata = label.get("metadata", {})
            if metadata.get("width") != width or metadata.get("height") != height:
                raise ValueError("Label metadata/image dimensions disagree: " + image_name)
            if metadata.get("img_name") != Path(image_name).name:
                raise ValueError("Label metadata/image filename mismatch: " + image_name)
            if metadata.get("disaster") != sample["event"]:
                raise ValueError("Label event does not match frozen sample entry: " + image_name)
        for feature in features:
            uid_splits.setdefault(feature["properties"]["uid"], set()).add(sample.get("split", "development"))
        truth = grid_truth(features, width, height, side)
        evaluation = {"sample_id": sample["sample_id"], "use": "EVALUATOR_ONLY_NEVER_PLANNER_INPUT",
                      "target_subtypes": sorted(TARGET_SUBTYPES),
                      "assignment": "Each building counted once, assigned by pixel-polygon area centroid.",
                      "metric_unit": "target-positive cells; not a building-instance detection score", "cells": truth}
        (target / "evaluation_truth.json").write_text(json.dumps(evaluation, indent=2))
        # The observation provider may crop only selected cells from these paths.
        # This index stays out of public_cues.json and contains no damage labels.
        observation_index = {"sample_id": sample["sample_id"], "use": "ENVIRONMENT_ONLY_SELECTED_REGION_ACCESS",
                             "pre_image": str((args.data / files["pre_image"]).resolve()),
                             "post_image": str((args.data / files["post_image"]).resolve()),
                             "independence_note": "Repeated reads of one base image are correlated evidence, not independent revisits.",
                             "cells": [{"cell_id": c["cell_id"],
                                        "pixel_bbox": [c["col"] * width // side, c["row"] * height // side,
                                                       (c["col"] + 1) * width // side, (c["row"] + 1) * height // side]}
                                       for c in cells]}
        (target / "local_observation_index.json").write_text(json.dumps(observation_index, indent=2))
        reports.append({"sample_id": sample["sample_id"], "event": sample["event"],
                        "split": sample.get("split", "development"),
                        "label_selected_diagnostic": sample.get("label_selected_diagnostic", False),
                        "dimensions": [width, height], "building_count": len(features),
                        "subtypes": dict(Counter(f["properties"].get("subtype", "un-classified") for f in features)),
                        "target_buildings": sum(len(c["target_building_ids"]) for c in truth.values()),
                        "positive_cells": sum(c["target_present"] is True for c in truth.values()),
                        "unknown_cells": sum(c["target_present"] is None for c in truth.values())})
    splits = {}
    for sample in reports:
        aggregate = splits.setdefault(sample["split"], {"pairs": 0, "events": [], "buildings": 0,
                                                        "target_buildings": 0, "positive_scenes": 0,
                                                        "negative_scenes": 0, "indeterminate_scenes": 0,
                                                        "positive_cells": 0, "unknown_cells": 0})
        aggregate["pairs"] += 1
        if sample["event"] not in aggregate["events"]:
            aggregate["events"].append(sample["event"])
        aggregate["buildings"] += sample["building_count"]
        aggregate["target_buildings"] += sample["target_buildings"]
        aggregate["positive_cells"] += sample["positive_cells"]
        aggregate["unknown_cells"] += sample["unknown_cells"]
        if sample["positive_cells"]:
            aggregate["positive_scenes"] += 1
        elif sample["unknown_cells"]:
            aggregate["indeterminate_scenes"] += 1
        else:
            aggregate["negative_scenes"] += 1
    duplicate_uids = [uid for uid, groups in uid_splits.items() if len(groups) > 1]
    report = {"stage": "REAL_DATA_ADAPTER_CHECK_ONLY", "real_efficacy_evaluation_run": False,
              "dataset_id": manifest.get("dataset_id", "xbd_development_samples"),
              "grid_cells_per_scene": 64, "total_pairs": len(reports), "split_summary": splits,
              "protocol_sha256": manifest.get("protocol_sha256"),
              "cross_split_duplicate_images": cross_split_duplicates,
              "cross_split_duplicate_building_uids": duplicate_uids,
              "integrity_checks_passed": not cross_split_duplicates and not duplicate_uids,
              "geographic_overlap_audited": False,
              "preparation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "download_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
              "samples": reports,
              "pending": ["Compare raw mirror files with original distribution if official checksums become available.",
                          "Audit frozen perception checkpoint training-event overlap before unseen-event claims.",
                          "Choose frozen detector and calibrate its quality-conditional error rates.",
                          "Calibrate raw cue into an initial probability using calibration events only.",
                          "Define actual additional observations; repeated reads of one image are not independent evidence.",
                          "Add travel/return costs after the observation-budget experiment."]}
    (args.out / "adapter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (args.out / "sample_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "event", "split", "building_count", "target_buildings",
                                              "positive_cells", "unknown_cells", "label_selected_diagnostic"])
        writer.writeheader()
        for sample in reports:
            writer.writerow({k: sample[k] for k in writer.fieldnames})
    print(json.dumps({k: report[k] for k in ("stage", "total_pairs", "split_summary", "integrity_checks_passed")}, ensure_ascii=False))
    if not report["integrity_checks_passed"]:
        raise ValueError("Cross-split duplicate inputs found; preserve the protocol and inspect audit report")


if __name__ == "__main__":
    main()
