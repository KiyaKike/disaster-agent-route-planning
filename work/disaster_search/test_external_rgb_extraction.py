import binascii
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from extract_external_rgb18 import (BoundedRangeCache, central_infos, decompress_rgb,
    parse_local_header)
from index_external_rgb_metadata import central_directory_location


def synthetic_member():
    image = io.BytesIO()
    Image.new("RGB", (512, 512), (11, 22, 33)).save(image, format="PNG")
    png = image.getvalue()
    name = "E/images/E_000001_pre_disaster.png"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, png)
        archive.writestr("E/masks/DO_NOT_READ.png", b"mask sentinel")
    whole = buffer.getvalue()
    cd_start, cd_bytes, entries = central_directory_location(whole[-22:], len(whole))
    infos = central_infos(whole[cd_start:cd_start+cd_bytes], entries)
    image_info, mask_info = infos
    row = {"path": name, "local_header_offset": image_info.header_offset,
        "next_header_or_directory": mask_info.header_offset,
        "flag_bits": image_info.flag_bits, "compression_method": image_info.compress_type,
        "crc32_unverified": f"{image_info.CRC:08x}", "compressed_bytes": image_info.compress_size,
        "uncompressed_bytes": image_info.file_size,
        "archive_url": "https://example.invalid/data.zip", "archive_size": len(whole)}
    header = whole[image_info.header_offset:image_info.header_offset+30]
    return whole, png, header, row


class HeaderAndBodyTests(unittest.TestCase):
    def test_offsets_are_original_archive_positions(self):
        whole, png, header, member = synthetic_member()
        n, extra, start = parse_local_header(header, member)
        self.assertEqual(whole[30:30+n].decode(), member["path"])
        self.assertEqual(decompress_rgb(whole[start:start+member["compressed_bytes"]], member), png)
        self.assertEqual(whole[member["next_header_or_directory"]:][:4], b"PK\x03\x04")

    def test_header_size_name_and_crc_disagreements_stop(self):
        _, _, header, member = synthetic_member()
        for at in [0, 14, 18, 22, 26]:
            changed = bytearray(header)
            changed[at] ^= 1
            with self.assertRaises(ValueError):
                parse_local_header(changed, member)

    def test_cross_member_and_oversized_extra_are_rejected(self):
        _, _, header, member = synthetic_member()
        member["next_header_or_directory"] = 31
        with self.assertRaises(ValueError):
            parse_local_header(header, member)
        changed = bytearray(header)
        changed[28:30] = struct.pack("<H", 4097)
        with self.assertRaises(ValueError):
            parse_local_header(changed, synthetic_member()[3])

    def test_crc_and_truncated_body_rejected(self):
        whole, _, header, member = synthetic_member()
        _, _, start = parse_local_header(header, member)
        body = whole[start:start+member["compressed_bytes"]]
        with self.assertRaises(ValueError):
            decompress_rgb(body[:-1], member)
        member["crc32_unverified"] = "00000000"
        with self.assertRaises(ValueError):
            decompress_rgb(body, member)

    def test_oversized_expansion_cannot_be_decompressed_unbounded(self):
        whole, _, header, member = synthetic_member()
        _, _, start = parse_local_header(header, member)
        member["uncompressed_bytes"] = 20
        with self.assertRaises(ValueError):
            decompress_rgb(whole[start:start+member["compressed_bytes"]], member)

    def test_valid_crc_non_png_is_not_accepted(self):
        payload = b"not an RGB PNG"
        member = {"compressed_bytes": len(payload), "uncompressed_bytes": len(payload),
            "compression_method": 0, "crc32_unverified": f"{binascii.crc32(payload)&0xffffffff:08x}"}
        with self.assertRaises(ValueError):
            decompress_rgb(payload, member)


class RangeBoundaryTests(unittest.TestCase):
    def test_full_response_is_closed_without_reading_body(self):
        member = synthetic_member()[3]
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen") as network:
            response = network.return_value.__enter__.return_value
            response.status = 200
            response.headers = {}
            reader = BoundedRangeCache(Path(tmp))
            with self.assertRaises(ValueError):
                reader.read(member, 0, 30, "local_header30")
            response.read.assert_not_called()
            self.assertEqual(reader.ledger[0]["received_bytes"], 0)

    def test_mismatched_content_range_rejected_before_read(self):
        member = synthetic_member()[3]
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen") as network:
            response = network.return_value.__enter__.return_value
            response.status = 206
            response.headers = {"Content-Range": "bytes 1-30/999"}
            with self.assertRaises(ValueError):
                BoundedRangeCache(Path(tmp)).read(member, 0, 30, "local_header30")
            response.read.assert_not_called()

    def test_roles_and_neighbor_bounds_prevent_mask_reads(self):
        member = synthetic_member()[3]
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen") as network:
            reader = BoundedRangeCache(Path(tmp))
            for offset, count, role in [(0, 30, "mask"),
                (member["next_header_or_directory"], 30, "local_header30"),
                (0, 31, "local_header30"), (0, member["compressed_bytes"]+1, "selected_rgb_compressed")]:
                with self.assertRaises(ValueError):
                    reader.read(member, offset, count, role)
            network.assert_not_called()

    def test_transfer_reservation_survives_failure_and_restart(self):
        member = synthetic_member()[3]
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen", side_effect=OSError("offline")) as network:
            with self.assertRaises(OSError):
                BoundedRangeCache(Path(tmp), limit=30).read(member, 0, 30, "local_header30")
            with self.assertRaises(ValueError):
                BoundedRangeCache(Path(tmp), limit=30).read(member, 0, 30, "local_header30")
            self.assertEqual(network.call_count, 1)

    def test_exact_range_cache_replay_needs_no_network(self):
        _, _, header, member = synthetic_member()
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen") as network:
            response = network.return_value.__enter__.return_value
            response.status = 206
            response.headers = {"Content-Range": f"bytes 0-29/{member['archive_size']}", "Content-Length": "30"}
            response.read.return_value = header
            data, receipt = BoundedRangeCache(Path(tmp)).read(member, 0, 30, "local_header30")
            self.assertEqual(data, header)
            self.assertFalse(receipt["cache_hit"])
            network.reset_mock()
            data, receipt = BoundedRangeCache(Path(tmp)).read(member, 0, 30, "local_header30")
            self.assertEqual(data, header)
            self.assertTrue(receipt["cache_hit"])
            network.assert_not_called()
            file = next(Path(tmp).glob("*.bin"))
            file.write_bytes(b"0"*30)
            with self.assertRaises(ValueError):
                BoundedRangeCache(Path(tmp)).read(member, 0, 30, "local_header30")

    def test_short_body_stops_without_creating_valid_cache(self):
        member = synthetic_member()[3]
        with tempfile.TemporaryDirectory() as tmp, patch("extract_external_rgb18.urlopen") as network:
            response = network.return_value.__enter__.return_value
            response.status = 206
            response.headers = {"Content-Range": f"bytes 0-29/{member['archive_size']}"}
            response.read.side_effect = [b"x"*10, b""]
            reader = BoundedRangeCache(Path(tmp))
            with self.assertRaises(ValueError):
                reader.read(member, 0, 30, "local_header30")
            self.assertEqual(reader.ledger[0]["received_bytes"], 10)
            self.assertFalse(list(Path(tmp).glob("*.bin")))


if __name__ == "__main__":
    unittest.main()
