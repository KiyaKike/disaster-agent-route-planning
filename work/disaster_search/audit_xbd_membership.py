"""Read only pinned Parquet image_name columns to audit model/data exposure.

Range reads are capped and verified. No image bytes or mask columns are read.
ID membership is evidence about sample identities, not pixel checksum equality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen

from build_xbd_mini import atomic_json, sha


REPO = "EVER-Z/torchange_xView2"
REVISION = "cda47e44377258222166dac63a85068e8bdca446"
MAX_READ = 1024 * 1024
MAX_TOTAL = 16 * 1024 * 1024


class RangeBudget:
    def __init__(self, limit=MAX_TOTAL):
        self.limit, self.reserved, self.received = limit, 0, 0
        self.lock = threading.Lock()

    def reserve(self, count):
        with self.lock:
            if self.reserved + count > self.limit:
                raise ValueError("Metadata network budget exceeded")
            self.reserved += count

    def record(self, count):
        with self.lock:
            self.received += count


class VerifiedRangeFile(io.RawIOBase):
    def __init__(self, url, size, budget):
        self.url, self.size, self.position, self.budget = url, size, 0, budget
        self.read_receipts = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unsupported seek")
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError("Seek outside pinned file")
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if not count:
            return b""
        if count > MAX_READ:
            raise ValueError("Refusing a bulk image/mask read")
        start, end = self.position, self.position + count - 1
        self.budget.reserve(count)
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                            "User-Agent": "disaster-search-membership-audit/0.1"})
        with urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Server did not honor the exact bounded byte range")
            data = response.read(count + 1)
        if len(data) != count:
            raise ValueError("Unexpected metadata range length")
        self.budget.record(len(data))
        self.read_receipts.append({"offset": start, "bytes": count, "sha256": sha(data)})
        self.position += count
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def sample_id(image_name, expected_split):
    prefix = expected_split + "/images/"
    if not image_name.startswith(prefix):
        raise ValueError("Image name does not match author split")
    sid = image_name.removeprefix(prefix)
    if not re.fullmatch(r"[a-z0-9-]+_[0-9]{8}", sid):
        raise ValueError("Unexpected canonical sample identity")
    return sid


def inspect_shard(item, out, budget):
    import pyarrow.parquet as pq
    path = item["rfilename"]
    split = Path(path).name.split("-", 1)[0]
    cache = out / "shard_names" / (Path(path).stem + ".json")
    if cache.exists():
        result = json.loads(cache.read_text())
        if result["revision"] != REVISION or result["path"] != path or result["remote_size"] != item["size"]:
            raise ValueError("Different cached membership snapshot")
        if sha(json.dumps(result["image_names"], sort_keys=True).encode()) != result["image_names_sha256"]:
            raise ValueError("Cached membership names changed")
        return result
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    last_error = None
    for attempt in range(3):
        try:
            with VerifiedRangeFile(url, item["size"], budget) as reader:
                parquet = pq.ParquetFile(reader)
                # Project the scalar name column only; never read images/masks.
                names = parquet.read(columns=["image_name"]).column("image_name").to_pylist()
                result = {"revision": REVISION, "path": path, "author_split": split,
                    "remote_size": item["size"], "remote_lfs_sha256_not_download_verified": item["lfs"]["sha256"],
                    "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
                    "projected_columns": ["image_name"], "image_names": names,
                    "image_names_sha256": sha(json.dumps(names, sort_keys=True).encode()),
                    "range_receipts": reader.read_receipts, "read_bytes": sum(r["bytes"] for r in reader.read_receipts)}
            if result["rows"] != len(names) or len(names) != len(set(names)):
                raise ValueError("Incomplete or duplicate shard identities")
            for name in names:
                sample_id(name, split)
            atomic_json(cache, result)
            return result
        except (OSError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Metadata shard failed without substitution: {path}") from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--loose-index", type=Path, required=True)
    parser.add_argument("--mini-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.metadata_root / "repository_info.json").read_text())
    if metadata["sha"] != REVISION:
        raise ValueError("Author dataset revision changed")
    shards = [item for item in metadata["siblings"] if item["rfilename"].endswith(".parquet")]
    if len(shards) != 50:
        raise ValueError("Unexpected pinned shard inventory")
    plan = {"version": "xbd_membership_admission_v1", "source_repo": REPO, "revision": REVISION,
        "metadata_sha256": sha((args.metadata_root / "repository_info.json").read_bytes()),
        "loose_index_sha256": sha(args.loose_index.read_bytes()), "mini_protocol_sha256": sha(args.mini_protocol.read_bytes()),
        "shards": [{"path": item["rfilename"], "size": item["size"], "sha256": item["lfs"]["sha256"]} for item in shards],
        "projected_columns": ["image_name"], "max_request_bytes": MAX_READ, "max_network_bytes_per_run": MAX_TOTAL,
        "purpose": "Read scalar identities to audit reported train/tier3/test/hold roles and loose-mirror mapping; no image/mask/label outcome access or sampling on model performance."}
    path = args.out / "membership_protocol.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen membership plan differs")
    if not path.exists():
        atomic_json(path, plan)
    budget = RangeBudget()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(inspect_shard, item, args.out, budget) for item in shards]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"shard": result["path"], "rows": result["rows"], "shards_done": len(results)}), flush=True)
    identities = defaultdict(list)
    for result in sorted(results, key=lambda r: r["path"]):
        split = result["author_split"]
        identities[split].extend(sample_id(name, split) for name in result["image_names"])
    for split, names in identities.items():
        if len(names) != len(set(names)):
            raise ValueError("Repeated canonical sample ID within author split")
    id_sets = {split: set(names) for split, names in identities.items()}
    loose = json.loads(args.loose_index.read_text())
    paths = {item["rfilename"] for item in loose["siblings"]}
    loose_ids = {split: {Path(path).name.removesuffix("_post_disaster.json") for path in paths
                        if path.startswith(split + "/labels/") and path.endswith("_post_disaster.json")}
                 for split in ("train", "test")}
    mini = json.loads(args.mini_protocol.read_text())
    memberships = [{"sample_id": s["sample_id"], "mini_split": s["split"],
                    "author_splits": sorted(split for split, values in id_sets.items() if s["sample_id"] in values)} for s in mini["samples"]]
    training_ids = id_sets["train"] | id_sets["tier3"]
    report = {"stage": "AUTHOR_SPLIT_IDENTITY_ADMISSION_AUDIT", "source_repo": REPO, "revision": REVISION,
        "protocol_sha256": sha(path.read_bytes()), "shards_read": len(results), "projected_columns": ["image_name"],
        "network_bytes_this_run": budget.received, "successful_metadata_read_bytes_in_cached_shards": sum(r["read_bytes"] for r in results),
        "split_counts": {split: len(values) for split, values in sorted(id_sets.items())},
        "event_counts": {split: dict(sorted(Counter(sid.rsplit("_", 1)[0] for sid in values).items())) for split, values in sorted(id_sets.items())},
        "pairwise_id_overlaps": {f"{a}|{b}": len(id_sets[a] & id_sets[b]) for a in sorted(id_sets) for b in sorted(id_sets) if a < b},
        "loose_mirror_mapping": {loose_split: {author_split: len(values & author_values) for author_split, author_values in sorted(id_sets.items())}
                                 for loose_split, values in loose_ids.items()},
        "mini_memberships": memberships, "mini_count_in_training_pool": sum(s["sample_id"] in training_ids for s in mini["samples"]),
        "hold_count_in_training_pool": len(id_sets["hold"] & training_ids),
        "hold_count_in_checkpoint_selection_test": len(id_sets["hold"] & id_sets["test"]),
        "hold_events_also_in_training_pool": sorted({sid.rsplit("_", 1)[0] for sid in id_sets["hold"]} & {sid.rsplit("_", 1)[0] for sid in training_ids}),
        "hold_ids_present_in_loose_mirror": len(id_sets["hold"] & (loose_ids["train"] | loose_ids["test"])),
        "hold_has_original_polygon_uid_fields_in_published_schema": False,
        "image_or_mask_columns_read": False, "per_image_pixel_hashes_verified": False,
        "scope": "Exact canonical-ID membership from the pinned author file columns, combined with author's documented training/selection roles. Does not prove byte identity, geographic separation or unseen-event generalization.",
        "dataset_server_info_note": "Info endpoint marked partial; use all pinned shard row counts and IDs, not viewer subset counts."}
    atomic_json(args.out / "membership_report.json", report)
    atomic_json(args.out / "author_split_ids.json", {split: sorted(values) for split, values in sorted(id_sets.items())})
    print(json.dumps({k: report[k] for k in ("split_counts", "loose_mirror_mapping", "mini_count_in_training_pool", "hold_count_in_training_pool", "hold_count_in_checkpoint_selection_test", "hold_ids_present_in_loose_mirror", "network_bytes_this_run")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
