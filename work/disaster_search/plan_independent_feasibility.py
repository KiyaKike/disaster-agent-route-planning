"""Fix an external-event feasibility cohort from ZIP directory metadata only.

This program has no network, image, annotation or detector imports. The result is
a prospective queue, not admission of independent evaluation data.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

EVENTS = ("EARTHQUAKE-TURKEY", "PAKISTAN-FLOODING", "TEXAS-TORNADOES")
SEED = 20261025
PER_EVENT = 6
SOURCE_VERSION = "10.6084/m9.figshare.25285009.v2"
INDEX_FIELDS = {"path", "uncompressed_bytes", "compressed_bytes",
                "crc32_unverified", "compression_method", "directory"}


def pair_inventory(event, rows):
    """Require unambiguous complete image pairs; do not use masks or sizes to rank."""
    if event not in EVENTS:
        raise ValueError("Unplanned event")
    seen, pairs = set(), defaultdict(dict)
    pattern = re.compile(re.escape(event) + r"/images/(" + re.escape(event)
                         + r"_\d+)_(pre|post)_disaster\.png")
    for row in rows:
        if set(row) != INDEX_FIELDS:
            raise ValueError("Only the frozen ZIP metadata schema is accepted")
        name = row["path"]
        path = PurePosixPath(name)
        if name in seen or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe or duplicate member identity")
        seen.add(name)
        if row["directory"]:
            continue
        if not name.startswith(event + "/images/"):
            continue
        match = pattern.fullmatch(name)
        if not match:
            raise ValueError("Unexpected image identity; do not silently drop it")
        sample_id, mode = match.groups()
        if mode in pairs[sample_id]:
            raise ValueError("Duplicate pair member")
        if any(type(row[k]) is not int or row[k] <= 0
               for k in ["compressed_bytes", "uncompressed_bytes"]):
            raise ValueError("Invalid image size metadata")
        pairs[sample_id][mode] = row
    if not pairs or any(set(pair) != {"pre", "post"} for pair in pairs.values()):
        raise ValueError("Incomplete image inventory; no scene substitution")
    return dict(pairs)


def rank_hash(event, sample_id):
    text = f"disaster-search-v3-feasibility|{SOURCE_VERSION}|{SEED}|{event}|{sample_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_pairs(event, rows):
    pairs = pair_inventory(event, rows)
    if len(pairs) < PER_EVENT:
        raise ValueError("Insufficient metadata candidates")
    ordered = sorted(pairs, key=lambda sid: (rank_hash(event, sid), sid))[:PER_EVENT]
    result = []
    for rank, sid in enumerate(ordered, 1):
        result.append({"sample_id": sid, "event": event,
            "source_version": SOURCE_VERSION, "archive": event + ".zip",
            "rank_within_event": rank, "rank_sha256": rank_hash(event, sid),
            "role": "FEASIBILITY_ONLY_NEVER_FORMAL_CONFIRMATION",
            "pre": dict(pairs[sid]["pre"]), "post": dict(pairs[sid]["post"]),
            "source_dimensions_expected_not_decoded": [512, 512, 3],
            "geographic_independence": "UNVERIFIED",
            "original_imagery_terms": "PENDING_SOURCE_CHAIN_CHECK",
            "instance_gold": "PENDING_HUMAN_REVIEW",
            "payload_download_enabled": False,
            "inference_and_policy_scoring_enabled": False})
    return len(pairs), result


def annotation_template(row):
    return {"sample_id": row["sample_id"], "event": row["event"],
        "status": "NOT_STARTED", "image_sha256": {"pre": None, "post": None},
        "reviewer": None, "reviewed_at_utc": None, "review_seconds": None,
        "blind_to_detector_predictions": True,
        "whole_scene_reviewed": False, "empty_scene_confirmed": False,
        "buildings": [], "second_review": {"reviewer": None, "status": "PENDING"},
        "notes": "Empty list is an unfilled form, never a negative scene label."}


def validate_gold_record(record):
    """Structural check only; never authenticates a reviewer or label accuracy."""
    if record.get("status") != "REVIEWED" or record.get("whole_scene_reviewed") is not True:
        raise ValueError("An unfilled record cannot become negative gold")
    if not record.get("reviewer") or not record.get("reviewed_at_utc"):
        raise ValueError("Review provenance missing")
    if record.get("blind_to_detector_predictions") is not True:
        raise ValueError("Prediction-blind review required")
    digests = record.get("image_sha256", {})
    if set(digests) != {"pre", "post"} or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for v in digests.values()):
        raise ValueError("Both image hashes required")
    if record.get("second_review", {}).get("status") != "ADJUDICATED" or not record["second_review"].get("reviewer"):
        raise ValueError("Second review and disagreement resolution missing")
    if record["second_review"]["reviewer"] == record["reviewer"]:
        raise ValueError("Two distinct reviewers required")
    buildings = record.get("buildings")
    if not isinstance(buildings, list) or (not buildings and record.get("empty_scene_confirmed") is not True):
        raise ValueError("Explicit complete-scene emptiness review required")
    seen = set()
    for building in buildings:
        uid = building.get("instance_id")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("Unique human-reviewed instance IDs required")
        seen.add(uid)
        if building.get("damage") not in ["no-damage", "minor-damage", "major-damage", "destroyed", "unknown"]:
            raise ValueError("Unknown damage must stay explicit")
        if building.get("identity_source") not in ["human_traced", "original_polygon_human_verified"]:
            raise ValueError("Connected components are not instance gold")
        polygon = building.get("post_polygon_xy")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("Post-image instance polygon required")
        import math
        if any(not isinstance(point, list) or len(point) != 2 or any(
                type(v) not in [int, float] or not math.isfinite(v) or not 0 <= v <= 512
                for v in point) for point in polygon):
            raise ValueError("Invalid polygon coordinates")
        area = abs(sum(polygon[i][0] * polygon[(i+1) % len(polygon)][1]
                       - polygon[(i+1) % len(polygon)][0] * polygon[i][1]
                       for i in range(len(polygon)))) / 2
        if area <= 0:
            raise ValueError("Degenerate polygon")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source_paths = [root / "discovery_protocol.json", root / "zip_index_protocol.json",
        root / "archive_index_report.json", root / "sources/ebd_figshare_v2.json"]
    source_paths.extend(root / "sources" / (e + "_index.json") for e in EVENTS)
    protocol = {"stage": "METADATA_ONLY_TENTATIVE_COHORT_SELECTION",
        "source_version": SOURCE_VERSION, "seed": SEED, "events": list(EVENTS),
        "pairs_per_event": PER_EVENT,
        "ranking": "SHA256(disaster-search-v3-feasibility|source_version|seed|event|sample_id), ascending; sizes, masks, outcomes never rank candidates",
        "source_sha256": {str(p.relative_to(root)): sha(p.read_bytes()) for p in source_paths},
        "selector_sha256": sha(Path(__file__).read_bytes()),
        "no_outcome_reads_or_replacement": True, "image_payload_access_enabled": False}
    freeze(root / "cohort_selection_protocol.json", protocol)
    reports = json.loads((root / "archive_index_report.json").read_text())
    zip_protocol = json.loads((root / "zip_index_protocol.json").read_text())
    metadata = json.loads((root / "sources/ebd_figshare_v2.json").read_text())
    if metadata["version"] != 2 or reports["image_mask_payload_bytes_read"] != 0:
        raise ValueError("Wrong source version or outcome access")
    for report in reports["archives"]:
        if report["status"] != "INDEX_PARSED_NO_PAYLOAD":
            raise ValueError("All three metadata inventories must complete")
    if Counter(r["archive"] for r in reports["archives"]) != Counter(e + ".zip" for e in EVENTS):
        raise ValueError("Archive set differs")
    selected, counts = [], {}
    for event in EVENTS:
        report = next(r for r in reports["archives"] if r["archive"] == event + ".zip")
        files = {f["name"]: f for f in zip_protocol["archives"]}
        f = files[event + ".zip"]
        raw_root = root / "archive_metadata" / event
        end = (raw_root / f"{f['size']-22}_22.bin").read_bytes()
        from index_external_rgb_metadata import central_directory_location, parse_central_directory
        offset, size, entries = central_directory_location(end, f["size"])
        raw = (raw_root / f"{offset}_{size}.bin").read_bytes()
        if sha(raw) != report["index_sha256"]:
            raise ValueError("ZIP directory metadata changed")
        inventory = json.loads((root / "sources" / (event + "_index.json")).read_text())
        if parse_central_directory(raw, entries) != inventory:
            raise ValueError("Stored inventory differs from exact ZIP directory")
        counts[event], rows = select_pairs(event, inventory)
        selected.extend(rows)
    freeze(root / "tentative_cohort.json", selected)
    queue = []
    for row in selected:
        file = files[row["archive"]]
        for mode in ["pre", "post"]:
            queue.append({"sample_id": row["sample_id"], "event": row["event"], "mode": mode,
                "archive_url": file["download_url"], "source_version": SOURCE_VERSION,
                "member": row[mode], "enabled": False,
                "reason": "Metadata-only stage; original terms, RGB mapping and extraction protocol pending"})
    freeze(root / "rgb_download_queue_disabled.json", queue)
    freeze(root / "annotation_queue_unfilled.json", [annotation_template(r) for r in selected])
    atomic_json(root / "cohort_report.json", {"stage": "18_PAIRS_TENTATIVELY_FIXED_NOT_DATA_ADMITTED",
        "available_complete_image_pairs": counts, "selected_pairs": len(selected),
        "events": len(EVENTS), "image_members_queued_disabled": len(queue),
        "selected_compressed_image_bytes_from_index": sum(r["member"]["compressed_bytes"] for r in queue),
        "selected_uncompressed_png_file_bytes_from_index": sum(r["member"]["uncompressed_bytes"] for r in queue),
        "primary_expected_grid": [4, 4], "primary_expected_budget": 8,
        "future_all_cells_all_qualities_observations_if_admitted": len(selected) * 16 * 2,
        "independent_validation_admitted": False, "human_gold_complete": False,
        "new_image_or_label_payload_reads": 0, "new_inference": 0,
        "new_detector_or_policy_fit": False, "paid_gpu": False})
    print(json.dumps({"complete_pairs": counts, "selected": len(selected),
                      "all_downloads_disabled": True}))


if __name__ == "__main__":
    main()
