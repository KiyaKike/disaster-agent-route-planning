"""Read exact ZIP index ranges only, never images or damage masks.

The source file metadata and the exact archive allowlist must be frozen first.
EOCD comments/unsupported ZIP layouts fail closed instead of tail scanning.
"""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zipfile

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

ARCHIVES = ["EARTHQUAKE-TURKEY.zip", "PAKISTAN-FLOODING.zip", "TEXAS-TORNADOES.zip"]
MAX_METADATA = 4 * 1024 * 1024


def central_directory_location(eocd, archive_size):
    if len(eocd) != 22:
        raise ValueError("Exact22byte EOCD required")
    signature, disk, directory_disk, entries_disk, entries, size, offset, comment = struct.unpack("<4s4H2IH", eocd)
    if signature != b"PK\x05\x06" or comment or disk or directory_disk or entries_disk != entries:
        raise ValueError("Unsupported commented/multidisk ZIP; no payload tail scan allowed")
    if entries == 65535 or size == 0xffffffff or offset == 0xffffffff:
        raise ValueError("ZIP64 requires a separately specified metadata reader")
    if not 0 < size <= MAX_METADATA or offset < 0 or offset + size != archive_size - 22:
        raise ValueError("Central directory bounds/layout differ")
    return offset, size, entries


def parse_central_directory(data, entries):
    # A synthetic end record makes zipfile parse ONLY this directory buffer.
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, entries, entries, len(data), 0, 0)
    with zipfile.ZipFile(io.BytesIO(data + end)) as stream:
        rows = []
        names = set()
        for info in stream.infolist():
            path = Path(info.filename)
            if info.filename in names or path.is_absolute() or ".." in path.parts:
                raise ValueError("Ambiguous or unsafe ZIP member identity")
            names.add(info.filename)
            if info.flag_bits & 1:
                raise ValueError("Encrypted archive unsupported")
            rows.append({"path": info.filename, "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size, "crc32_unverified": f"{info.CRC:08x}",
                "compression_method": info.compress_type, "directory": info.is_dir()})
    if len(rows) != entries:
        raise ValueError("Incomplete ZIP index")
    return rows


class ExactMetadataReader:
    def __init__(self, url, size, out, budget):
        self.url, self.size, self.out, self.budget = url, size, out, budget
        self.receipts = []

    def read(self, offset, count, role):
        if not 0 <= offset < offset + count <= self.size or not 0 < count <= MAX_METADATA:
            raise ValueError("Invalid bounded metadata interval")
        if role not in ["eocd22", "central_directory"]:
            raise ValueError("Image/mask/local-member reads are disabled")
        path = self.out / f"{offset}_{count}.bin"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            data = path.read_bytes()
            saved = json.loads(receipt.read_text())
            if saved["url"] != self.url or saved["sha256"] != sha(data) or len(data) != count:
                raise ValueError("Metadata cache changed")
            cache = True
        else:
            if self.budget["received"] + count > self.budget["limit"]:
                raise ValueError("Total metadata cap exceeded")
            req = Request(self.url, headers={"Range": f"bytes={offset}-{offset+count-1}",
                "Accept-Encoding": "identity", "User-Agent": "disaster-search-zip-index/0.1"})
            with urlopen(req, timeout=25) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{offset+count-1}/{self.size}":
                    raise ValueError("Exact metadata Range not honored; no body read")
                data = response.read(count + 1)
            self.budget["received"] += len(data)
            if len(data) != count:
                raise ValueError("Unexpected index range size")
            self.out.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            atomic_json(receipt, {"url": self.url, "offset": offset, "bytes": count,
                "sha256": sha(data), "role": role})
            cache = False
        self.receipts.append({"url": self.url, "offset": offset, "bytes": count,
            "sha256": sha(data), "role": role, "cache_hit": cache})
        return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.root / "sources/ebd_figshare_v2.json"
    data = json.loads(metadata.read_text())
    if data["version"] != 2:
        raise ValueError("Fixed EBDv2 required")
    files = {f["name"]: f for f in data["files"]}
    freeze(args.root / "zip_index_protocol.json", {"stage": "EBD_V2_EXACT_INDEX_ONLY",
        "source_metadata_sha256": sha(metadata.read_bytes()), "archives": [files[n] for n in ARCHIVES],
        "max_metadata_bytes": MAX_METADATA, "reader_sha256": sha(Path(__file__).read_bytes()),
        "access": "Read final22byte EOCD then exactly indexed central directory, never full tail/payload/local headers. Unsupported layout or range response stops archive without broadening. No image or mask decoded.",
        "use": "Metadata-only tentative image pair ranking before any image/label/model result. No masks/class counts used for selection.",
        "license_clearance": False, "image_payload_access_enabled": False,
        "inference_or_scores_enabled": False})
    budget = {"limit": MAX_METADATA, "received": 0}
    result = []
    for name in ARCHIVES:
        f = files[name]
        reader = ExactMetadataReader(f["download_url"], f["size"], args.root / "archive_metadata" / name.removesuffix(".zip"), budget)
        row = {"archive": name, "published_size": f["size"], "published_md5_not_download_verified": f["computed_md5"]}
        try:
            end = reader.read(f["size"] - 22, 22, "eocd22")
            offset, size, entries = central_directory_location(end, f["size"])
            directory = reader.read(offset, size, "central_directory")
            members = parse_central_directory(directory, entries)
            atomic_json(args.root / "sources" / (name.removesuffix(".zip") + "_index.json"), members)
            row.update(status="INDEX_PARSED_NO_PAYLOAD", member_entries=len(members),
                central_directory_offset=offset, central_directory_bytes=size,
                index_sha256=sha(directory))
        except (OSError, ValueError, TimeoutError) as error:
            row.update(status="INDEX_UNAVAILABLE_NO_FALLBACK_PAYLOAD", error_type=type(error).__name__, error=str(error))
        row["read_receipts"] = reader.receipts
        result.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "read_receipts"}), flush=True)
    atomic_json(args.root / "archive_index_report.json", {"stage": "METADATA_INDEX_ATTEMPT_COMPLETE",
        "archives": result, "network_bytes_received_this_run": budget["received"],
        "image_mask_payload_bytes_read": 0, "image_or_label_decode": False,
        "candidate_selection_complete": False, "new_detector_or_policy_fit": False,
        "validation_or_confirmation_scored": False,
        "at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
