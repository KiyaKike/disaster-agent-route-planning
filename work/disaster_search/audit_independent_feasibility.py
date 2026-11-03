"""Audit metadata-only external cohort, disabled queues and unchanged lineage."""
from collections import Counter
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from index_external_rgb_metadata import central_directory_location, parse_central_directory
from plan_independent_feasibility import (EVENTS, select_pairs, annotation_template,
    validate_gold_record)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    get = lambda name: json.loads((root / name).read_text())
    basis = get("discovery_protocol.json")["basis"]
    for name, expected in basis.items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("Prior research basis changed: " + name)
    # Old immutable source snapshots must continue to match current helpers.
    previous = json.loads(Path("output/disaster_search_serial_multifidelity_v1/completion_receipt.json").read_text())
    old_code = {Path(name).name: digest for name, digest in previous["files"].items()
                if name.startswith("source_snapshots/") and name.endswith(".py")}
    for name, expected in old_code.items():
        if sha((Path(__file__).parent/name).read_bytes()) != expected:
            raise ValueError("Previously frozen helper changed: " + name)
    selection = get("cohort_selection_protocol.json")
    for name, expected in selection["source_sha256"].items():
        if sha((root/name).read_bytes()) != expected:
            raise ValueError("Frozen metadata source changed: " + name)
    if sha(Path(__file__).with_name("plan_independent_feasibility.py").read_bytes()) != selection["selector_sha256"]:
        raise ValueError("Selector changed after selection")
    zip_protocol = get("zip_index_protocol.json")
    if sha(Path(__file__).with_name("index_external_rgb_metadata.py").read_bytes()) != zip_protocol["reader_sha256"]:
        raise ValueError("Metadata reader changed after range acquisition")
    report = get("archive_index_report.json")
    files = {f["name"]: f for f in zip_protocol["archives"]}
    rebuilt, counts, total = [], {}, 0
    for event in EVENTS:
        file = files[event+".zip"]
        row = next(r for r in report["archives"] if r["archive"] == event+".zip")
        folder = root/"archive_metadata"/event
        end = (folder/f"{file['size']-22}_22.bin").read_bytes()
        offset, size, entries = central_directory_location(end, file["size"])
        directory = (folder/f"{offset}_{size}.bin").read_bytes()
        expected_ranges = [("eocd22", file["size"]-22, 22, sha(end)),
                           ("central_directory", offset, size, sha(directory))]
        actual_ranges = [(r["role"], r["offset"], r["bytes"], r["sha256"])
                         for r in row["read_receipts"]]
        if actual_ranges != expected_ranges:
            raise ValueError("Read ranges differ from exact metadata allowlist")
        for receipt in row["read_receipts"]:
            if receipt["url"] != file["download_url"]:
                raise ValueError("Archive source changed")
        if sha(directory) != row["index_sha256"]:
            raise ValueError("Index hash changed")
        inventory = parse_central_directory(directory, entries)
        if inventory != get("sources/"+event+"_index.json"):
            raise ValueError("Inventory replay differs")
        counts[event], selected = select_pairs(event, inventory)
        rebuilt.extend(selected)
        total += len(end)+len(directory)
    if rebuilt != get("tentative_cohort.json"):
        raise ValueError("Cohort selection replay differs")
    if Counter(r["event"] for r in rebuilt) != Counter({e: 6 for e in EVENTS}):
        raise ValueError("Event allocation differs")
    queue = get("rgb_download_queue_disabled.json")
    expected_queue = []
    for row in rebuilt:
        for mode in ["pre", "post"]:
            expected_queue.append((row["sample_id"], row["event"], mode, row[mode]))
    actual_queue = [(r["sample_id"], r["event"], r["mode"], r["member"]) for r in queue]
    if actual_queue != expected_queue or any(r["enabled"] is not False for r in queue):
        raise ValueError("Download queue differs or has been enabled")
    forms = get("annotation_queue_unfilled.json")
    if forms != [annotation_template(r) for r in rebuilt]:
        raise ValueError("Blank annotation queue changed")
    rejected = 0
    for row in forms:
        try:
            validate_gold_record(row)
        except ValueError:
            rejected += 1
    if rejected != 18:
        raise ValueError("Unfilled annotation forms accepted as gold")
    protocol = get("prospective_feasibility_v3.json")
    for key, name in [("selection_protocol_sha256", "cohort_selection_protocol.json"),
                      ("tentative_cohort_sha256", "tentative_cohort.json"),
                      ("source_decision_sha256", "source_decision.json")]:
        if sha((root/name).read_bytes()) != protocol[key]:
            raise ValueError("Prospective protocol source changed")
    for stage in ["stage_b_next_rgb_and_gold", "stage_c_future_measurement", "stage_d_future_descriptive_policy_replay"]:
        if protocol[stage]["enabled_now"] is not False:
            raise ValueError("Future stage enabled by metadata audit")
    if report["image_mask_payload_bytes_read"] != 0 or report["image_or_label_decode"] is not False:
        raise ValueError("This stage contains payload access")
    if total != 2917031 or total > zip_protocol["max_metadata_bytes"]:
        raise ValueError("Metadata bytes differ or exceed cap")
    # Public structured metadata API receipts are separate from archive ranges.
    api_bytes = 0
    for name in ["ebd_figshare_v2", "bright_zenodo_20072020"]:
        receipt = get("sources/"+name+".receipt.json")
        data = (root/"sources"/(name+".json")).read_bytes()
        if sha(data) != receipt["sha256"]:
            raise ValueError("Structured source metadata changed")
        api_bytes += len(data)
    if total+api_bytes > 8*1024*1024:
        raise ValueError("Discovery structured metadata cap exceeded")
    log = (root/"test_log.txt").read_text()
    if "Ran 229 tests" not in log or not log.rstrip().endswith("OK"):
        raise ValueError("Current full test evidence missing")
    if any(p.suffix.lower() in [".png", ".jpg", ".tif", ".pt", ".pth", ".zip"]
           for p in root.rglob("*") if p.is_file()):
        raise ValueError("Unexpected image/model/archive payload file")
    atomic_json(root/"audit.json", {"status": "METADATA_AND_PROSPECTIVE_PROTOCOL_VERIFIED",
        "complete_image_pairs": counts, "selected_pairs": 18, "events": 3,
        "disabled_rgb_queue": 36, "blank_gold_records_correctly_rejected": rejected,
        "exact_archive_metadata_bytes": total, "structured_metadata_api_bytes": api_bytes,
        "browser_articles_read_separately": True, "all_inventory_and_selection_replayed": True,
        "previous_frozen_helpers_unchanged": len(old_code), "basis_hashes_verified": len(basis),
        "tests_passed": 229, "new_tests": 17,
        "image_or_label_payload_bytes_this_stage": 0, "new_inference": 0,
        "new_detector_or_policy_fit": False, "paid_gpu": False,
        "human_gold_complete": False, "independent_data_admitted": False,
        "new_method_selected": False, "reserved_scores_opened": False})
    print(json.dumps(get("audit.json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
