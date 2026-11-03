"""Extract only the fixed36 EBD RGB members using exact bounded ZIP ranges.

No masks, replacement sampling, whole-archive fallback, prediction or scoring.
The parent metadata stage remains immutable. Each received range is replayable.
"""
import binascii
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import struct
from urllib.request import Request, urlopen
import zipfile
import zlib

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze

MAX_TRANSFER = 128 * 1024 * 1024
MAX_MEMBER = 4 * 1024 * 1024
MAX_LOCAL_EXTRA = 4096


def central_infos(directory, entries):
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, entries, entries, len(directory), 0, 0)
    with zipfile.ZipFile(io.BytesIO(directory+end)) as archive:
        infos = archive.infolist()
    if len(infos) != entries or len({i.filename for i in infos}) != entries:
        raise ValueError("Ambiguous central directory")
    return infos


def parse_local_header(header, member):
    if len(header) != 30:
        raise ValueError("Exact30byte local header required")
    sig, version, flags, method, _, _, crc, size, expanded, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    if sig != b"PK\x03\x04" or flags & 1 or method not in [0, 8] or version >= 45:
        raise ValueError("Unsupported local header")
    if flags != member["flag_bits"] or method != member["compression_method"]:
        raise ValueError("Local and central directory disagree")
    expected_name = member["path"].encode("utf-8" if flags & 2048 else "cp437")
    if name_len != len(expected_name) or extra_len > MAX_LOCAL_EXTRA:
        raise ValueError("Local name/extra length differs or exceeds cap")
    expected = [int(member["crc32_unverified"], 16), member["compressed_bytes"], member["uncompressed_bytes"]]
    actual = [crc, size, expanded]
    if any(a != b and not (flags & 8 and a == 0) for a, b in zip(actual, expected)):
        raise ValueError("Local CRC/size disagrees with directory")
    start = member["local_header_offset"] + 30 + name_len + extra_len
    if start + member["compressed_bytes"] > member["next_header_or_directory"]:
        raise ValueError("RGB range could cross another member")
    return name_len, extra_len, start


def decompress_rgb(data, member):
    if len(data) != member["compressed_bytes"]:
        raise ValueError("Compressed range length differs")
    expected = member["uncompressed_bytes"]
    if not 0 < expected <= MAX_MEMBER:
        raise ValueError("Expanded member cap exceeded")
    if member["compression_method"] == 8:
        decoder = zlib.decompressobj(-15)
        result = decoder.decompress(data, expected+1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Incomplete, trailing or oversized deflate stream")
    elif member["compression_method"] == 0:
        result = data
    else:
        raise ValueError("Unsupported compression")
    if len(result) != expected or f"{binascii.crc32(result) & 0xffffffff:08x}" != member["crc32_unverified"]:
        raise ValueError("Expanded length or CRC mismatch")
    if not result.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Selected RGB member is not PNG")
    return result


class BoundedRangeCache:
    """Conservative persistent cap counts every requested body, including failures."""
    def __init__(self, root, limit=MAX_TRANSFER):
        self.root, self.limit = root, limit
        root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = root/"transfer_ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        self.used = sum(r["reserved_bytes"] for r in self.ledger)

    def read(self, member, offset, count, role):
        if role not in ["local_header30", "local_name_extra", "selected_rgb_compressed"]:
            raise ValueError("Unplanned range role")
        lower = member["local_header_offset"]
        if not lower <= offset < offset+count <= member["next_header_or_directory"] or count > MAX_MEMBER:
            raise ValueError("Request exceeds selected member bounds")
        if role == "local_header30" and (offset != lower or count != 30):
            raise ValueError("Local fixed header range differs")
        if role == "local_name_extra" and (offset != lower+30 or count > len(member["path"].encode())+MAX_LOCAL_EXTRA):
            raise ValueError("Variable header range differs")
        if role == "selected_rgb_compressed" and count != member["compressed_bytes"]:
            raise ValueError("Compressed body range differs")
        identity = {"url": member["archive_url"], "member": member["path"],
                    "offset": offset, "bytes": count, "role": role}
        key = sha(json.dumps(identity, sort_keys=True).encode())
        payload, receipt = self.root/(key+".bin"), self.root/(key+".json")
        if payload.exists() != receipt.exists():
            raise ValueError("Incomplete range cache pair")
        if payload.exists():
            saved = json.loads(receipt.read_text())
            data = payload.read_bytes()
            if any(saved[k] != v for k, v in identity.items()) or saved["sha256"] != sha(data) or len(data) != count:
                raise ValueError("Range cache changed")
            return data, {**saved, "cache_hit": True}
        if self.used + count > self.limit:
            raise ValueError("Persistent transfer cap exceeded")
        row = {**identity, "reserved_bytes": count, "received_bytes": 0,
               "status": "REQUEST_RESERVED", "at_utc": datetime.now(timezone.utc).isoformat()}
        self.ledger.append(row)
        self.used += count
        atomic_json(self.ledger_path, self.ledger)
        req = Request(member["archive_url"], headers={"Range": f"bytes={offset}-{offset+count-1}",
            "Accept-Encoding": "identity", "User-Agent": "disaster-search-fixed-rgb-extraction/0.1"})
        try:
            with urlopen(req, timeout=25) as response:
                expected_range = f"bytes {offset}-{offset+count-1}/{member['archive_size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range or response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Exact unencoded Range not honored; response body not read")
                if response.headers.get("Content-Length") not in [None, str(count)]:
                    raise ValueError("Response size differs; body not read")
                data = bytearray()
                while len(data) < count:
                    chunk = response.read(min(65536, count-len(data)))
                    if not chunk:
                        raise ValueError("Truncated exact range")
                    data.extend(chunk)
                    row["received_bytes"] = len(data)
                data = bytes(data)
            row["status"] = "EXACT_RANGE_RECEIVED"
            payload.write_bytes(data)
            saved = {**identity, "sha256": sha(data), "http_status": 206,
                     "content_range": expected_range, "content_encoding": "identity"}
            atomic_json(receipt, saved)
            return data, {**saved, "cache_hit": False}
        except Exception as error:
            row.update(status="STOPPED_NO_BROADENING", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            atomic_json(self.ledger_path, self.ledger)


def make_members(parent):
    queue = json.loads((parent/"rgb_download_queue_disabled.json").read_text())
    reports = json.loads((parent/"archive_index_report.json").read_text())["archives"]
    members = []
    for report in reports:
        event = report["archive"].removesuffix(".zip")
        folder = parent/"archive_metadata"/event
        raw = (folder/f"{report['central_directory_offset']}_{report['central_directory_bytes']}.bin").read_bytes()
        if sha(raw) != report["index_sha256"]:
            raise ValueError("Parent central directory changed")
        infos = central_infos(raw, report["member_entries"])
        by_name = {i.filename: i for i in infos}
        offsets = sorted(set(i.header_offset for i in infos) | {report["central_directory_offset"]})
        for row in queue:
            if row["event"] != event:
                continue
            meta = row["member"]
            name = meta["path"]
            if not name.startswith(event+"/images/") or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("Only frozen image members allowed")
            info = by_name[name]
            if info.compress_size != meta["compressed_bytes"] or info.file_size != meta["uncompressed_bytes"] or f"{info.CRC:08x}" != meta["crc32_unverified"]:
                raise ValueError("Selected member metadata changed")
            if info.flag_bits & 1 or info.compress_type not in [0, 8] or not 0 < info.file_size <= MAX_MEMBER or not 0 < info.compress_size <= MAX_MEMBER:
                raise ValueError("Unsupported selected member")
            next_offset = next(o for o in offsets if o > info.header_offset)
            members.append({**meta, "sample_id": row["sample_id"], "event": event, "mode": row["mode"],
                "archive_url": row["archive_url"], "archive_size": report["published_size"],
                "local_header_offset": info.header_offset, "flag_bits": info.flag_bits,
                "next_header_or_directory": next_offset})
    return sorted(members, key=lambda r: (r["event"], r["sample_id"], r["mode"]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    receipt_path = args.parent/"completion_receipt.json"
    previous = json.loads(receipt_path.read_text())
    for name, expected in previous["files"].items():
        if sha((args.parent/name).read_bytes()) != expected:
            raise ValueError("Frozen parent artifact changed: "+name)
    use = json.loads((args.out/"source_use_record.json").read_text())
    if use["local_research_rgb_copy_enabled"] is not True or use["commercial_use_or_publication_of_raw_images_enabled"] is not False:
        raise ValueError("This protocol only admits attributed local research copies")
    members = make_members(args.parent)
    if len(members) != 36 or len({r["sample_id"] for r in members}) != 18:
        raise ValueError("Fixed18 pair cohort required")
    freeze(args.out/"extraction_protocol.json", {"stage": "FIXED36_RGB_EXTRACTION_ONLY",
        "parent_completion_sha256": sha(receipt_path.read_bytes()),
        "source_use_record_sha256": sha((args.out/"source_use_record.json").read_bytes()),
        "extractor_sha256": sha(Path(__file__).read_bytes()), "members": members,
        "data_root": str(args.data.resolve()), "max_transfer_bytes": MAX_TRANSFER,
        "roles": ["local_header30", "local_name_extra", "selected_rgb_compressed"],
        "replacement_or_full_archive_fallback": False,
        "mask_or_detector_access": False, "independent_evaluation_admitted": False})
    if not args.execute:
        print(json.dumps({"planned_rgb_members": len(members), "payload_read": False}))
        return
    from PIL import Image
    reader = BoundedRangeCache(args.data/"ranges")
    records, failures = [], []
    for member in members:
        name = member["path"]
        try:
            header, header_receipt = reader.read(member, member["local_header_offset"], 30, "local_header30")
            n, extra, start = parse_local_header(header, member)
            variable, variable_receipt = reader.read(member, member["local_header_offset"]+30, n+extra, "local_name_extra")
            encoding = "utf-8" if member["flag_bits"] & 2048 else "cp437"
            if variable[:n].decode(encoding) != name:
                raise ValueError("Local member name differs from fixed whitelist")
            packed, body_receipt = reader.read(member, start, member["compressed_bytes"], "selected_rgb_compressed")
            data = decompress_rgb(packed, member)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512) or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Unexpected image format/mode/dimensions")
                image.load()
            target = args.data/"images"/member["event"]/(member["sample_id"]+"_"+member["mode"]+".png")
            if target.exists() and sha(target.read_bytes()) != sha(data):
                raise ValueError("Existing RGB file changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes(data)
            records.append({"sample_id": member["sample_id"], "event": member["event"], "mode": member["mode"],
                "path": str(target.resolve()), "sha256": sha(data), "bytes": len(data),
                "shape": [512, 512, 3], "source_member": name, "crc32_verified": member["crc32_unverified"],
                "ranges": [header_receipt, variable_receipt, body_receipt]})
            print(json.dumps({"member": name, "status": "RGB_CRC_SHA_AND_SHAPE_VERIFIED", "complete": len(records)}), flush=True)
        except (OSError, ValueError, TimeoutError, zlib.error) as error:
            failures.append({"member": name, "sample_id": member["sample_id"], "mode": member["mode"],
                "status": "FAILED_NO_REPLACEMENT", "error": str(error), "error_type": type(error).__name__})
            print(json.dumps(failures[-1]), flush=True)
    atomic_json(args.out/"rgb_manifest.json", records)
    atomic_json(args.out/"extraction_report.json", {"stage": "RGB_EXTRACTION_COMPLETE" if not failures else "RGB_EXTRACTION_INCOMPLETE",
        "complete_rgb_files": len(records), "failures": failures,
        "received_body_bytes_cumulative": sum(r["received_bytes"] for r in reader.ledger),
        "requested_bytes_reserved_cumulative": reader.used, "transfer_cap": MAX_TRANSFER,
        "mask_payload_bytes_read": 0, "new_inference": 0, "new_fit": False,
        "human_gold_complete": False, "independent_data_admitted": False})


if __name__ == "__main__":
    main()
