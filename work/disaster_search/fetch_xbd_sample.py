"""Download two lossless xBD community-mirror pairs for adapter checks only.

Uses pinned repository revision and records provenance, hashes and label schema.
No training, paid service, full-dataset download or label-derived prior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


REPO = "aryananand/xBD"
ATTRIBUTION_REPO = "kshitijrajsharma/xview2-xbd"
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 40 * 1024 * 1024


def get_bytes(url, maximum=MAX_FILE_BYTES):
    request = urllib.request.Request(url, headers={"User-Agent": "disaster-search-data-check/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Download exceeds explicitly bounded size")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-positive-diagnostic", action="store_true",
                        help="Add a label-selected positive integration example; never treat it as an unbiased test sample")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(get_bytes("https://huggingface.co/api/datasets/" + REPO))
    revision = metadata["sha"]
    files = {item["rfilename"] for item in metadata["siblings"]}
    selected, events = [], set()
    for name in sorted(files):
        if not name.startswith("train/labels/") or not name.endswith("_post_disaster.json"):
            continue
        stem = Path(name).name.removesuffix("_post_disaster.json")
        event = stem.rsplit("_", 1)[0]
        needed = [f"train/{kind}/{stem}_{period}_disaster.{extension}"
                  for period in ("pre", "post")
                  for kind, extension in (("images", "png"), ("labels", "json"))]
        if event not in events and all(p in files for p in needed):
            selected.append({"sample_id": stem, "event": event, "remote_paths": needed})
            events.add(event)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("Could not locate two complete pairs from different events")
    scanned_labels = []
    if args.include_positive_diagnostic:
        candidates = [p for p in sorted(files) if p.startswith("train/labels/") and
                      p.endswith("_post_disaster.json") and ("wildfire" in p or "socal-fire" in p)]
        for remote in candidates[:16]:
            label = json.loads(get_bytes(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{remote}"))
            positive_count = sum(f["properties"].get("subtype") in {"major-damage", "destroyed"}
                                 for f in label["features"]["xy"])
            scanned_labels.append({"path": remote, "positive_buildings": positive_count})
            if positive_count:
                stem = Path(remote).name.removesuffix("_post_disaster.json")
                needed = [f"train/{kind}/{stem}_{period}_disaster.{extension}"
                          for period in ("pre", "post")
                          for kind, extension in (("images", "png"), ("labels", "json"))]
                if all(p in files for p in needed):
                    selected.append({"sample_id": stem, "event": stem.rsplit("_", 1)[0],
                                     "remote_paths": needed, "label_selected_diagnostic": True})
                    break
        if len(selected) != 3:
            raise ValueError("No complete positive diagnostic found in bounded label scan")
    total, manifest_files, samples = 0, [], []
    for sample in selected:
        entry = {"sample_id": sample["sample_id"], "event": sample["event"], "files": {},
                 "label_selected_diagnostic": sample.get("label_selected_diagnostic", False)}
        for remote in sample["remote_paths"]:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{remote}"
            data = get_bytes(url)
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Sample download exceeds 40 MiB total bound")
            target = args.out / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Refusing to replace different existing sample data: " + str(target))
            target.write_bytes(data)
            item = {"path": remote, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if target.suffix == ".png":
                if data[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Expected original PNG, not a browser JPEG preview")
                item["width"], item["height"] = struct.unpack(">II", data[16:24])
            else:
                label = json.loads(data)
                features = label["features"]["xy"]
                if not all("wkt" in feature and "properties" in feature for feature in features):
                    raise ValueError("Unexpected original xBD label schema")
                item["features_xy"] = len(features)
                item["subtypes"] = dict(Counter(f["properties"].get("subtype", "not_provided") for f in features))
                item["metadata"] = label.get("metadata", {})
            manifest_files.append(item)
            period = "pre" if "_pre_disaster." in remote else "post"
            entry["files"][period + ("_image" if target.suffix == ".png" else "_label")] = remote
        samples.append(entry)
    attribution = get_bytes(f"https://huggingface.co/datasets/{ATTRIBUTION_REPO}/resolve/main/README.md")
    (args.out / "upstream_mirror_attribution.txt").write_bytes(attribution)
    manifest = {"stage": "REAL_XBD_INTEGRATION_SAMPLES_ONLY", "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_repo": REPO, "source_revision": revision, "original_distributor": False,
                "license_note": "xBD imagery is described as CC BY-NC-SA 4.0 by the attributed parquet mirror. The loose-file mirror has no license card metadata; this is not independent rights verification.",
                "original_source": "https://xview2.org/dataset", "paper": "https://arxiv.org/abs/1911.09296",
                "selection": "First complete train pair in each of first two lexicographic event groups, independent of labels; optional third positive diagnostic explicitly selected from at most 16 wildfire labels.",
                "diagnostic_label_scan": scanned_labels,
                "usage": "Schema and lossless-input checks only; two samples are not a train/test benchmark.",
                "total_bytes": total, "samples": samples, "files": manifest_files,
                "ready_for_efficacy_evaluation": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (args.out / "SOURCES.md").write_text(
        "# xBD 接入小样本来源\n\n"
        f"- 数据：xBD / xView2；社区镜像 https://huggingface.co/datasets/{REPO}\n"
        f"- 固定版本：{revision}\n"
        "- 原始来源：https://xview2.org/dataset\n"
        "- 论文：https://arxiv.org/abs/1911.09296\n"
        f"- 下载 UTC 时间：{manifest['fetched_at_utc']}\n"
        "- 授权记录：署名镜像声明 CC BY-NC-SA 4.0；本次散文件镜像没有 license 字段，尚未独立核对原发布条款。\n"
        "- 用途：train 样例的图像/JSON 接入检查，不用于性能或跨事件结论；若含阳性诊断样例，它经过标签筛选，不能作为无偏测试集。\n"
        "- 原始 PNG 与 JSON 原样保存，SHA256 见 manifest.json。未下载网页有损标签预览。\n"
        "- 尚未与官方分发文件逐一校验一致性。\n")
    print(json.dumps({"output": str(args.out), "bytes": total, "samples": [s["sample_id"] for s in samples]}))


if __name__ == "__main__":
    main()
