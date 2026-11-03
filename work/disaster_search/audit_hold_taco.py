"""Bounded, metadata-only inspection of an alternative xBD repackaging.

Never projects damage, counts or polygon columns. CoZIP locations are used to
read only collection and Parquet metadata plus explicitly whitelisted IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import struct
from urllib.request import urlopen
import zlib

from audit_xbd_membership import RangeBudget, VerifiedRangeFile, MAX_READ
from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

REPO = "isp-uv-es/xview2-taco"
REVISION = "44e219dc1d88c9d0903d6b478d7f2f3c435e574f"
SIZE = 32404817624
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/xview2.zip"
IDENTITY_COLUMNS = ["ml:source_file", "ml:split_original", "ml:disaster", "internal:current_id"]
MAX_NETWORK = 16 * 1024 * 1024


def parse_index(payload, archive_size):
    if len(payload) < 11 or payload[:4] != b"CZIP":
        raise ValueError("Not a CoZIP index")
    version, profile, count = struct.unpack_from("<HBI", payload, 4)
    if version != 1 or profile != 2 or not 1 <= count <= 32:
        raise ValueError("Unsupported CoZIP version/profile/index size")
    lengths = struct.unpack_from("<" + "H" * count, payload, 11)
    cursor = 11 + 2 * count
    if len(payload) != cursor + sum(lengths) + 16 * count:
        raise ValueError("Malformed CoZIP index length")
    names = []
    for length in lengths:
        names.append(payload[cursor:cursor + length].decode("utf-8"))
        cursor += length
    offsets = struct.unpack_from("<" + "Q" * count, payload, cursor)
    sizes = struct.unpack_from("<" + "Q" * count, payload, cursor + 8 * count)
    allowed = {"COLLECTION.json", "METADATA/children.parquet", "METADATA/sample.parquet"}
    if set(names) != allowed or len(names) != len(allowed):
        raise ValueError("Unexpected metadata inventory")
    result = {}
    for name, offset, size in zip(names, offsets, sizes):
        if size < 1 or offset < 0 or offset + size > archive_size:
            raise ValueError("Indexed metadata outside archive")
        result[name] = {"offset": offset, "size": size}
    intervals = sorted((r["offset"], r["offset"] + r["size"]) for r in result.values())
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping metadata entries")
    return result


class CachedRanges:
    def __init__(self, root, max_network_bytes=MAX_NETWORK):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.remote = VerifiedRangeFile(URL, SIZE, RangeBudget(max_network_bytes))
        self.accesses = []

    def read(self, offset, size):
        if not 0 <= offset <= offset + size <= SIZE:
            raise ValueError("Range outside pinned archive")
        parts = []
        while size:
            count = min(size, MAX_READ)
            path = self.root / f"{offset}_{count}.bin"
            receipt = path.with_suffix(".json")
            cached = path.exists() and receipt.exists()
            if cached:
                data = path.read_bytes()
                saved = json.loads(receipt.read_text())
                if saved != {"url": URL, "offset": offset, "bytes": count, "sha256": sha(data)} or len(data) != count:
                    raise ValueError("Cached range differs from its pinned receipt")
            else:
                self.remote.seek(offset)
                data = self.remote.read(count)
                path.write_bytes(data)
                atomic_json(receipt, {"url": URL, "offset": offset, "bytes": count, "sha256": sha(data)})
            self.accesses.append({"offset": offset, "bytes": count, "sha256": sha(data), "cache_hit": cached})
            parts.append(data)
            offset += count
            size -= count
        return b"".join(parts)


class MetadataView(io.RawIOBase):
    def __init__(self, ranges, entry):
        self.ranges, self.base, self.size = ranges, entry["offset"], entry["size"]
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError("Seek outside metadata member")
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        data = self.ranges.read(self.base + self.position, count)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def read_index(ranges):
    header = ranges.read(0, 30)
    signature, _, flags, method, _, _, crc, compressed, size, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or method != 0 or flags & 9 or compressed != size or size > 65536:
        raise ValueError("Unsupported first ZIP member")
    name = ranges.read(30, name_len)
    if name != b"__cozip__":
        raise ValueError("Missing first CoZIP member")
    offset = 30 + name_len + extra_len
    payload = ranges.read(offset, size)
    if zlib.crc32(payload) != crc:
        raise ValueError("CoZIP index CRC mismatch")
    return parse_index(payload, SIZE)


def project_identities(parquet):
    if not set(IDENTITY_COLUMNS).issubset(parquet.schema_arrow.names):
        raise ValueError("Required identity columns missing")
    return parquet.read(columns=IDENTITY_COLUMNS, use_threads=False).to_pylist()


def column_ranges(parquet, entry, protected):
    result = []
    for r in range(parquet.metadata.num_row_groups):
        for c in range(parquet.metadata.num_columns):
            column = parquet.metadata.row_group(r).column(c)
            field = column.path_in_schema.split(".list.", 1)[0]
            if field in protected:
                start = min(x for x in [column.dictionary_page_offset, column.data_page_offset] if x is not None and x >= 0)
                result.append({"field": field, "offset": entry["offset"] + start,
                               "bytes": column.total_compressed_size})
    return result


def assert_no_protected_reads(accesses, protected):
    for access in accesses:
        for column in protected:
            if max(access["offset"], column["offset"]) < min(access["offset"] + access["bytes"], column["offset"] + column["bytes"]):
                raise ValueError(f"Read intersects protected column {column['field']}")


def fetch_small(url, path, limit):
    if path.exists():
        return path.read_bytes()
    with urlopen(url, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Source metadata exceeds cap")
    path.write_bytes(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(args.out / "audit_protocol.json", {
        "repo": REPO, "revision": REVISION, "archive_size": SIZE,
        "projected_columns": IDENTITY_COLUMNS, "max_network_bytes": MAX_NETWORK,
        "purpose": "Inspect schema and ID mapping only; do not decode images, damage, counts, XY or WGS84 polygons.",
        "design_sha256": sha((args.design / "experiment_contract.json").read_bytes()),
        "selection_sha256": sha((args.design / "cohort_selection.json").read_bytes()),
        "exploratory_read_disclosure": "Before this runner, a 16384-byte prefix probe crossed the index into part of the first PNG's compressed payload. It was not decoded. A subsequent schema-only probe read 141049 bytes. Do not claim zero image bytes were accessed in the whole session.",
    })
    api_url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    api = json.loads(fetch_small(api_url, args.out / "repository_info.json", 1024 * 1024))
    if api["sha"] != REVISION:
        raise ValueError("Dataset revision differs")
    archive = next(f for f in api["siblings"] if f["rfilename"] == "xview2.zip")
    if archive["size"] != SIZE:
        raise ValueError("Archive size differs")
    card_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"
    card = fetch_small(card_url, args.out / "dataset_card.md", 128 * 1024)
    ranges = CachedRanges(args.out / "ranges")
    index = read_index(ranges)
    atomic_json(args.out / "cozip_index.json", index)
    entry = index["COLLECTION.json"]
    collection = json.loads(ranges.read(entry["offset"], entry["size"]))
    atomic_json(args.out / "collection.json", collection)
    schemas, footer_rows = {}, {}
    sample = None
    for name in ["METADATA/sample.parquet", "METADATA/children.parquet"]:
        parquet = pq.ParquetFile(MetadataView(ranges, index[name]))
        schemas[name] = str(parquet.schema_arrow)
        footer_rows[name] = {"rows": parquet.metadata.num_rows, "row_groups": parquet.metadata.num_row_groups,
            "columns": parquet.schema_arrow.names,
            "compressed_column_bytes": {parquet.metadata.row_group(0).column(c).path_in_schema:
                sum(parquet.metadata.row_group(r).column(c).total_compressed_size for r in range(parquet.metadata.num_row_groups))
                for c in range(parquet.metadata.num_columns)}}
        if name.endswith("sample.parquet"):
            sample = parquet
    rows = project_identities(sample)
    protected = column_ranges(sample, index["METADATA/sample.parquet"],
        {"ml:damage", "ml:polygon_xy", "ml:polygon_len", "ml:polygon_lnglat", "ml:polygon_lnglat_len"})
    assert_no_protected_reads(ranges.accesses, protected)
    ids = [r["ml:source_file"] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != 11034:
        raise ValueError("Unexpected repackaged identity inventory")
    author = json.loads(args.identities.read_text())
    hold_rows = [r for r in rows if r["ml:split_original"] == "hold"]
    hold_ids = {r["ml:source_file"] for r in hold_rows}
    selected = json.loads((args.design / "cohort_selection.json").read_text())
    lookup = {r["ml:source_file"]: r for r in rows}
    comparisons = {split: {"author_count": len(values),
        "source_count": sum(r["ml:split_original"] == split for r in rows),
        "missing_author_ids": sorted(set(values) - {r["ml:source_file"] for r in rows if r["ml:split_original"] == split})}
        for split, values in author.items()}
    report = {
        "stage": "HOLD_ALTERNATIVE_METADATA_AUDIT_NOT_DATA_ADMISSION",
        "repo": REPO, "revision": REVISION, "archive_size": SIZE,
        "archive_published_sha256": archive["lfs"]["sha256"], "whole_archive_hash_verified": False,
        "source_urls": [api_url, card_url, "https://xview2.org/dataset", "https://asterisk.coop/taco/spec/"],
        "repository_info_sha256": sha((args.out / "repository_info.json").read_bytes()),
        "dataset_card_sha256": sha(card), "collection_sha256": sha((args.out / "collection.json").read_bytes()),
        "metadata": footer_rows, "schemas": schemas,
        "split_original_counts": dict(Counter(r["ml:split_original"] for r in rows)),
        "author_pool_comparisons": comparisons,
        "hold_exact_name_match": hold_ids == set(author["hold"]), "hold_count": len(hold_ids),
        "selected_count": len(selected),
        "selected_all_in_hold": all(r["sample_id"] in hold_ids for r in selected),
        "has_original_uid_column": any("uid" in c.lower() for c in sample.schema_arrow.names),
        "has_per_object_polygons": {key: key in sample.schema_arrow.names for key in ["ml:polygon_xy", "ml:polygon_len", "ml:polygon_lnglat", "ml:polygon_lnglat_len"]},
        "damage_legend_from_collection_only": collection["ml:contract"]["targets"],
        "projected_columns": IDENTITY_COLUMNS,
        "protected_column_ranges": protected, "no_protected_payload_intersections": True,
        "damage_values_read": False, "geometry_values_read": False, "images_decoded": False,
        "network_bytes_this_runner": ranges.remote.budget.received,
        "read_receipts": ranges.accesses,
        "data_admitted": False,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "next_gate": "Polygons are preserved but original UID is absent. Verify exact one-to-one mapping against known original calibration polygons before proposing derived instance IDs in an explicit contract amendment. No connected-component substitution.",
    }
    if not report["hold_exact_name_match"] or not report["selected_all_in_hold"]:
        raise ValueError("Candidate cohort does not map to audited hold")
    atomic_json(args.out / "identity_rows.json", rows)
    atomic_json(args.out / "selected_metadata.json", [{**r, "source_internal_id": lookup[r["sample_id"]]["internal:current_id"]} for r in selected])
    atomic_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ["hold_exact_name_match", "selected_all_in_hold", "hold_count", "has_original_uid_column", "network_bytes_this_runner", "data_admitted"]}))


if __name__ == "__main__":
    main()
