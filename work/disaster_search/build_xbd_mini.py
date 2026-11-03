"""Freeze and download a label-independent, event-disjoint 72-pair pilot subset.

plan reads directory metadata only, before any selected damage labels are opened.
download follows the frozen plan, supports checked resume, and never substitutes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request


REPO = "aryananand/xBD"
REVISION = "b98724026e844e1fccd879c95e9f09f03c56e6f8"
SEED = 20261024
EVENTS = {
    "calibration": ["hurricane-harvey", "santa-rosa-wildfire"],
    "validation": ["hurricane-michael", "mexico-earthquake"],
    "test": ["midwest-flooding", "socal-fire"],
}
DIAGNOSTIC_IDS = {"guatemala-volcano_00000000", "hurricane-florence_00000000", "santa-rosa-wildfire_00000004"}
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_DATA_BYTES = 600 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def paths_for(sample_id):
    return {period + "_" + kind: f"train/{directory}/{sample_id}_{period}_disaster.{extension}"
            for period in ("pre", "post")
            for kind, directory, extension in (("image", "images", "png"), ("label", "labels", "json"))}


def complete_inventory(source_index):
    files = {f["rfilename"] for f in source_index["siblings"]}
    groups = defaultdict(list)
    for path in sorted(files):
        if path.startswith("train/labels/") and path.endswith("_post_disaster.json"):
            sample_id = Path(path).name.removesuffix("_post_disaster.json")
            if all(p in files for p in paths_for(sample_id).values()):
                groups[sample_id.rsplit("_", 1)[0]].append(sample_id)
    return dict(groups)


def select_samples(inventory, event_splits=EVENTS, count=12, seed=SEED, excluded=DIAGNOSTIC_IDS):
    flat_events = [event for events in event_splits.values() for event in events]
    if len(flat_events) != len(set(flat_events)):
        raise ValueError("An event appears in more than one split")
    if count <= 0:
        raise ValueError("Per-event sample count must be positive")
    selected = []
    for split, events in event_splits.items():
        for event in events:
            eligible = [s for s in inventory.get(event, []) if s not in excluded]
            if len(eligible) < count:
                raise ValueError("Insufficient complete, unexposed pairs for event " + event)
            ranked = sorted(eligible, key=lambda s: (sha(f"{seed}|{event}|{s}".encode()), s))
            for sample_id in ranked[:count]:
                selected.append({"sample_id": sample_id, "event": event, "split": split,
                                 "upstream_split": "train", "label_selected_diagnostic": False,
                                 "selection_digest": sha(f"{seed}|{event}|{sample_id}".encode()),
                                 "files": paths_for(sample_id)})
    if len({s["sample_id"] for s in selected}) != len(selected):
        raise ValueError("Duplicate selected sample IDs")
    return selected


def create_plan(index_path, out):
    index_data = index_path.read_bytes()
    index = json.loads(index_data)
    if index["sha"] != REVISION:
        raise ValueError("Source index does not match pinned revision")
    inventory = complete_inventory(index)
    samples = select_samples(inventory)
    protocol = {
        "dataset_id": "xbd_mini72_v1", "stage": "FROZEN_SELECTION_BEFORE_LABEL_DOWNLOAD",
        "source_repo": REPO, "source_revision": REVISION, "source_index_sha256": sha(index_data),
        "seed": SEED, "samples_per_event": 12, "event_splits": EVENTS,
        "available_complete_pairs": {k: len(v) for k, v in sorted(inventory.items())},
        "excluded_development_samples": sorted(DIAGNOSTIC_IDS),
        "selection_rule": "Rank complete source-train pairs by SHA256(seed|event|sample_id), choose first 12 per event; no image contents or target labels used.",
        "target_subtypes": ["destroyed", "major-damage"], "unknown_subtypes": ["un-classified"],
        "grid_side": 8, "coarse_side_pixels": 128,
        "benchmark_status": "Pilot subset, not a new official benchmark or efficacy result.",
        "upstream_split_note": "All raw files come from upstream train. Mini splits separate policy calibration/tuning/evaluation events; a future frozen perception checkpoint must be audited for training-event overlap before any unseen-event claim.",
        "sampling_note": "Natural within-event label distribution is preserved; events are purposively fixed, not a representative random sample of all disasters. No relabeling, balancing or replacement after auditing.",
        "sample_count": len(samples), "expected_raw_files": len(samples) * 4, "samples": samples,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Refusing to overwrite a different frozen protocol; use a new dataset version")
    if not path.exists():
        atomic_json(path, protocol)
    receipt = out / "selection_receipt.json"
    if not receipt.exists():
        atomic_json(receipt, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                              "protocol_sha256": sha(path.read_bytes()),
                              "selected_labels_opened_by_selection_code": False})
    elif json.loads(receipt.read_text())["protocol_sha256"] != sha(path.read_bytes()):
        raise ValueError("Frozen protocol changed since receipt")
    with (out / "sample_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "event", "split", "upstream_split", "pre_image", "post_image", "pre_label", "post_label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({**{k: sample[k] for k in ("sample_id", "event", "split", "upstream_split")}, **sample["files"]})
    print(json.dumps({"stage": protocol["stage"], "pairs": len(samples), "event_splits": EVENTS,
                      "protocol_sha256": sha(path.read_bytes())}, ensure_ascii=False), flush=True)


def inspect_raw(path, data):
    record = {"path": path, "bytes": len(data), "sha256": sha(data)}
    if path.endswith(".png"):
        if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
            raise ValueError("Expected lossless PNG: " + path)
        record["width"], record["height"] = struct.unpack(">II", data[16:24])
    else:
        label = json.loads(data)
        features = label["features"]["xy"]
        if not all("wkt" in f and "properties" in f for f in features):
            raise ValueError("Unexpected xBD label schema: " + path)
        record["features_xy"] = len(features)
    return record


def fetch_one(path, out, cached):
    target = out / path
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    if target.exists() and path in cached:
        data = target.read_bytes()
        if sha(data) != cached[path]["sha256"]:
            raise ValueError("Cached raw file hash mismatch: " + path)
        return {**inspect_raw(path, data), "url": url}
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "disaster-search-pilot/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("File exceeds 12 MiB limit: " + path)
            record = {**inspect_raw(path, data), "url": url}
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Refusing to replace different raw input: " + path)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
            return record
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed without sample substitution: {path}: {error}") from error
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable")


def download(out, workers):
    if not 1 <= workers <= 4:
        raise ValueError("Use 1 to 4 network workers")
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads((out / "selection_receipt.json").read_text())
    if sha(protocol_path.read_bytes()) != receipt["protocol_sha256"]:
        raise ValueError("Frozen selection has changed")
    if protocol["source_revision"] != REVISION or protocol["source_repo"] != REPO:
        raise ValueError("Unexpected source")
    paths = [p for sample in protocol["samples"] for p in sample["files"].values()]
    cache_path = out / "download_records.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    records, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(fetch_one, p, out, cached): p for p in paths}
        for future in as_completed(pending):
            path = pending[future]
            try:
                records[path] = future.result()
            except Exception as error:
                failures.append({"path": path, "error": str(error)})
            if sum(r["bytes"] for r in records.values()) > MAX_DATA_BYTES:
                for f in pending:
                    f.cancel()
                raise ValueError("Pilot data exceeded 600 MiB size bound")
            if (len(records) + len(failures)) % 16 == 0 or len(records) + len(failures) == len(paths):
                atomic_json(cache_path, {**cached, **records})
                atomic_json(out / "download_progress.json", {"completed": len(records), "expected": len(paths),
                                                             "failures": failures,
                                                             "downloaded_bytes": sum(r["bytes"] for r in records.values())})
                print(json.dumps({"files": len(records), "expected": len(paths), "failures": len(failures),
                                  "MiB": round(sum(r["bytes"] for r in records.values()) / 2**20, 1)}), flush=True)
    if failures:
        atomic_json(out / "download_failures.json", failures)
        raise RuntimeError("Incomplete download; rerun to resume the same selection")
    manifest = {"stage": "FROZEN_MINI72_RAW_DOWNLOAD_COMPLETE", "dataset_id": protocol["dataset_id"],
                "source_repo": REPO, "source_revision": REVISION, "original_distributor": False,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": receipt["protocol_sha256"],
                "samples": protocol["samples"], "files": [records[p] for p in paths],
                "total_bytes": sum(r["bytes"] for r in records.values()),
                "license": "CC-BY-NC-SA-4.0", "license_reference": "https://docs.torchgeo.org/en/stable/api/datasets.html",
                "license_note": "TorchGeo and attributed xBD mirrors list CC BY-NC-SA 4.0. The loose-file mirror has no license card; upstream terms still apply and files have not been matched to official-distributor checksums.",
                "ready_for_efficacy_evaluation": False}
    atomic_json(out / "manifest.json", manifest)
    (out / "SOURCES.md").write_text(
        "# xBD mini72 v1 来源与抽样\n\n"
        "- 数据集：xBD / xView2，Gupta et al. 2019，https://arxiv.org/abs/1911.09296\n"
        f"- 下载镜像：https://huggingface.co/datasets/{REPO}，固定 commit `{REVISION}`\n"
        f"- UTC 下载完成时间：{manifest['downloaded_at_utc']}\n"
        "- 原发布入口：https://xview2.org/dataset\n"
        "- 许可参考：TorchGeo xBD 表项为 CC-BY-NC-SA-4.0，https://docs.torchgeo.org/en/stable/api/datasets.html\n"
        "- 本散文件镜像未声明独立许可；原始权利和条款仍适用，尚未与官方分发的校验和逐一对照。\n"
        "- 用途：非商业科研的主动搜索预实验，非新增官方基准。保留原始署名和同方式共享要求。\n"
        "- 六事件各十二对，共七十二对；protocol.json 在下载标签前固定，selection_receipt.json 记录时间与 SHA256。\n"
        "- 不按损伤比例筛选，不在审计后换图；开发诊断样例已排除。每个事件只在一个 mini 分组中。\n"
        "- 所有文件来自上游 train；后续感知模型必须审计训练事件重叠，不能直接宣称感知器从未见过测试事件。\n")
    print(json.dumps({"stage": manifest["stage"], "pairs": len(protocol["samples"]),
                      "files": len(records), "bytes": manifest["total_bytes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "download"])
    parser.add_argument("--index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "plan":
        if args.index is None:
            parser.error("plan needs --index")
        create_plan(args.index, args.out)
    else:
        download(args.out, args.workers)


if __name__ == "__main__":
    main()
