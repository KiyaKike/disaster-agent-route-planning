"""Replay bounded extraction and record limits of the external RGB feasibility set."""
from collections import Counter
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha
from extract_external_rgb18 import decompress_rgb, make_members, parse_local_header


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    parent, root = args.parent, args.root
    get = lambda name: json.loads((root/name).read_text())
    previous = json.loads((parent/"completion_receipt.json").read_text())
    for name, digest in previous["files"].items():
        if sha((parent/name).read_bytes()) != digest:
            raise ValueError("Frozen parent changed: "+name)
    # Preserve all earlier frozen code, including the102 prior planner helpers.
    old = json.loads(Path("output/disaster_search_serial_multifidelity_v1/completion_receipt.json").read_text())
    source_hashes = {Path(n).name: d for n, d in old["files"].items() if n.startswith("source_snapshots/") and n.endswith(".py")}
    source_hashes.update(previous["source_code_sha256"])
    for name, digest in source_hashes.items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Previously frozen source changed: "+name)
    protocol = get("extraction_protocol.json")
    if sha((parent/"completion_receipt.json").read_bytes()) != protocol["parent_completion_sha256"]:
        raise ValueError("Parent receipt differs")
    if sha((root/"source_use_record.json").read_bytes()) != protocol["source_use_record_sha256"]:
        raise ValueError("Source-use record changed after extraction")
    if sha(Path(__file__).with_name("extract_external_rgb18.py").read_bytes()) != protocol["extractor_sha256"]:
        raise ValueError("Extractor changed after protocol freeze")
    members = make_members(parent)
    if members != protocol["members"]:
        raise ValueError("Selected member metadata replay differs")
    data_root = Path(protocol["data_root"])
    manifests = get("rgb_manifest.json")
    if len(manifests) != 36:
        raise ValueError("All36 fixed images required")
    by_id = {(r["sample_id"], r["mode"]): r for r in manifests}
    if len(by_id) != 36:
        raise ValueError("Repeated image identity")
    total = 0
    expected_requests = set()
    for member in members:
        row = by_id[(member["sample_id"], member["mode"])]
        if row["source_member"] != member["path"] or "/images/" not in row["source_member"]:
            raise ValueError("Not a selected RGB image")
        payloads = []
        for receipt in row["ranges"]:
            identity = {k: receipt[k] for k in ["url", "member", "offset", "bytes", "role"]}
            if identity["member"] != member["path"] or identity["url"] != member["archive_url"]:
                raise ValueError("Range outside fixed RGB whitelist")
            key = sha(json.dumps(identity, sort_keys=True).encode())
            raw = (data_root/"ranges"/(key+".bin")).read_bytes()
            saved = json.loads((data_root/"ranges"/(key+".json")).read_text())
            if len(raw) != identity["bytes"] or sha(raw) != receipt["sha256"] or any(saved[k] != v for k,v in identity.items()):
                raise ValueError("Received range cache changed")
            if saved["sha256"] != sha(raw) or saved["http_status"] != 206:
                raise ValueError("Exact-range receipt differs")
            expected_range = f"bytes {identity['offset']}-{identity['offset']+identity['bytes']-1}/{member['archive_size']}"
            if saved["content_range"] != expected_range or saved["content_encoding"] != "identity":
                raise ValueError("Unbounded or encoded response")
            payloads.append(raw)
            expected_requests.add((identity["member"],identity["offset"],identity["bytes"],identity["role"]))
            total += len(raw)
        header, variable, packed = payloads
        n, extra, start = parse_local_header(header, member)
        expected = [(member["local_header_offset"],30,"local_header30"),
            (member["local_header_offset"]+30,n+extra,"local_name_extra"),
            (start,member["compressed_bytes"],"selected_rgb_compressed")]
        if [(r["offset"],r["bytes"],r["role"]) for r in row["ranges"]] != expected:
            raise ValueError("Read intervals differ from parsed selected member")
        encoding = "utf-8" if member["flag_bits"] & 2048 else "cp437"
        if variable[:n].decode(encoding) != member["path"]:
            raise ValueError("Local member name differs")
        expanded = decompress_rgb(packed, member)
        target = Path(row["path"])
        if expanded != target.read_bytes() or sha(expanded) != row["sha256"]:
            raise ValueError("RGB file differs from source member replay")
    ledger = json.loads((data_root/"ranges/transfer_ledger.json").read_text())
    if len(ledger) != 108 or any(r["status"] != "EXACT_RANGE_RECEIVED" for r in ledger):
        raise ValueError("Unexpected incomplete/additional network requests")
    observed_requests = {(r["member"],r["offset"],r["bytes"],r["role"]) for r in ledger}
    if observed_requests != expected_requests or sum(r["received_bytes"] for r in ledger) != total:
        raise ValueError("Network ledger differs from exact RGB ranges")
    if total != 13694435 or total > protocol["max_transfer_bytes"]:
        raise ValueError("Body byte total differs or exceeds budget")
    extraction = get("extraction_report.json")
    if extraction["failures"] or extraction["mask_payload_bytes_read"] or extraction["new_inference"]:
        raise ValueError("Unexpected failures, masks or inference")
    inspection = get("rgb_inspection_protocol.json")
    if sha((root/"rgb_manifest.json").read_bytes()) != inspection["manifest_sha256"] or sha(Path(__file__).with_name("inspect_external_rgb18.py").read_bytes()) != inspection["inspector_sha256"]:
        raise ValueError("RGB inspection lineage changed")
    quality = get("rgb_quality_report.json")
    from PIL import Image
    for row in manifests:
        with Image.open(row["path"]) as image:
            image.load()
            if image.mode != "RGB" or image.size != (512,512):
                raise ValueError("RGB shape check differs")
    review = get("review_tool_protocol.json")
    for key, path in [("index_sha256",root/"review_packet_index.json"),
        ("template_sha256",Path(__file__).with_name("rgb18_review_template.html")),
        ("builder_sha256",Path(__file__).with_name("build_rgb18_review.py"))]:
        if sha(path.read_bytes()) != review[key]:
            raise ValueError("Review tool lineage changed")
    if sha((root/"review.html").read_bytes()) != get("review_tool_build.json")["review_html_sha256"]:
        raise ValueError("Rendered review tool changed")
    if get("annotation_resource_decision.json")["human_answer"] != "暂时没有人工标注资源":
        raise ValueError("User resource constraint not preserved")
    area = get("prospective_area_reference_v3_1.json")
    masks = get("published_mask_queue_metadata_only.json")
    if len(masks) != 18 or any(r["payload_access_enabled"] is not False for r in masks) or area["executed_now"] is not False:
        raise ValueError("Prospective area reference stage has been opened")
    if sha((root/"published_mask_queue_metadata_only.json").read_bytes()) != area["mask_queue_sha256"]:
        raise ValueError("Prospective mask identities changed")
    ui = get("ui_qa/browser_interaction.json")
    if ui["fixture"] != "SYNTHETIC_ONLY_NOT_HUMAN_LABELS" or ui["final_gold_marked_complete"] is not False:
        raise ValueError("UI test data mistaken for human gold")
    loading = get("ui_qa/real_image_loading.json")
    if len(loading["checks"]) != 36 or any(r.get("error") or (r["width"],r["height"]) != (512,512) for r in loading["checks"]):
        raise ValueError("Real RGB review packet did not load all36 images")
    if loading["initialProgress"] != "首轮标记完成 0 / 18；最终真值 0 / 18":
        raise ValueError("Real review packet contains unexpected completed labels")
    log = (root/"test_log.txt").read_text()
    if "Ran 244 tests" not in log or not log.rstrip().endswith("OK"):
        raise ValueError("Current full test evidence missing")
    atomic_json(root/"audit.json", {"status":"BOUNDED_RGB_EXTRACTION_AND_PREPARATION_VERIFIED",
        "pairs":18,"events":3,"rgb_files":36,"network_exact_ranges_replayed":108,
        "received_rgb_archive_body_bytes":total,
        "compressed_png_body_bytes":sum(m["compressed_bytes"] for m in members),
        "native_png_file_bytes":sum(r["bytes"] for r in manifests),
        "files_by_event":dict(Counter(r["event"] for r in manifests)),
        "all_rgb_crc_sha_shape_checks":True,"old_frozen_source_files_unchanged":len(source_hashes),
        "tests_passed":244,"new_tests":15,"browser_test":"Synthetic interaction plus real-image loading; no human labels created",
        "black_border_and_translation_hints_not_admission":True,
        "within36_pixel_duplicate_groups":len(quality["exact_pixel_duplicate_groups_within36"]),
        "human_gold_complete":False,"instance_count_route":"DEFERRED_NO_ANNOTATION_RESOURCES",
        "area_reference_stage":"PROSPECTIVE_ONLY",
        "mask_payload_bytes_read":0,"new_inference":0,"new_fit":False,"paid_gpu":False,
        "independent_data_admitted":False,"validation_or_confirmation_scored":False})
    print(json.dumps(get("audit.json"),ensure_ascii=False))


if __name__ == "__main__":
    main()
