import copy
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from index_external_rgb_metadata import (ExactMetadataReader, central_directory_location,
    parse_central_directory, MAX_METADATA)
from plan_independent_feasibility import (EVENTS, PER_EVENT, annotation_template,
    pair_inventory, select_pairs, validate_gold_record)


def directory_bytes(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as stream:
        for name in names:
            stream.writestr(name, b"synthetic-payload-never-read-by-parser")
    data = buffer.getvalue()
    offset, size, entries = central_directory_location(data[-22:], len(data))
    return data, data[offset:offset+size], entries


def inventory(event=EVENTS[0], n=9):
    names = [f"{event}/images/{event}_{i:06d}_{mode}_disaster.png"
             for i in range(n) for mode in ["pre", "post"]]
    _, data, count = directory_bytes(names)
    return parse_central_directory(data, count)


class IndexBoundaryTests(unittest.TestCase):
    def test_only_directory_buffer_is_needed(self):
        whole, data, count = directory_bytes(["e/images/a.png", "e/masks/a.png"])
        rows = parse_central_directory(data, count)
        self.assertEqual(len(rows), 2)
        self.assertLess(len(data), len(whole))
        self.assertNotIn("payload", rows[0])

    def test_end_record_rejects_comment_and_multidisk(self):
        for disk, comment in [(0, 1), (1, 0)]:
            end = struct.pack("<4s4H2IH", b"PK\x05\x06", disk, 0, 1, 1, 40, 100, comment)
            with self.assertRaises(ValueError):
                central_directory_location(end, 162)

    def test_end_record_rejects_zip64_and_unbounded_directory(self):
        for entries, size, offset in [(65535, 40, 100), (1, MAX_METADATA+1, 0), (1, 40, 99)]:
            end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, entries, entries, size, offset, 0)
            with self.assertRaises(ValueError):
                central_directory_location(end, 162)

    def test_unsafe_member_and_entry_count_stop(self):
        for name in ["../a.png", "/a.png"]:
            _, data, entries = directory_bytes([name])
            with self.assertRaises(ValueError):
                parse_central_directory(data, entries)
        _, data, entries = directory_bytes(["a.png"])
        with self.assertRaises(ValueError):
            parse_central_directory(data, entries+1)

    def test_disallowed_role_and_budget_never_call_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("index_external_rgb_metadata.urlopen") as network:
            reader = ExactMetadataReader("https://example.invalid/a.zip", 100, Path(tmp), {"received": 0, "limit": 1})
            for args in [(0, 2, "image"), (98, 2, "eocd22"), (99, 2, "central_directory")]:
                with self.assertRaises(ValueError):
                    reader.read(*args)
            network.assert_not_called()

    def test_nonrange_response_body_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch("index_external_rgb_metadata.urlopen") as network:
            response = network.return_value.__enter__.return_value
            response.status = 200
            response.headers = {}
            reader = ExactMetadataReader("https://example.invalid/a.zip", 100, Path(tmp), {"received": 0, "limit": 100})
            with self.assertRaises(ValueError):
                reader.read(78, 22, "eocd22")
            response.read.assert_not_called()

    def test_cache_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = ExactMetadataReader("https://example.invalid/a.zip", 100, Path(tmp), {"received": 0, "limit": 100})
            (Path(tmp)/"78_22.bin").write_bytes(b"x"*22)
            (Path(tmp)/"78_22.json").write_text('{"url":"https://example.invalid/a.zip","sha256":"bad"}')
            with self.assertRaises(ValueError):
                reader.read(78, 22, "eocd22")


class SelectionTests(unittest.TestCase):
    def test_order_invariant_and_exact_six(self):
        rows = inventory()
        count, selected = select_pairs(EVENTS[0], rows)
        self.assertEqual(count, 9)
        self.assertEqual(len(selected), PER_EVENT)
        self.assertEqual(select_pairs(EVENTS[0], list(reversed(rows)))[1], selected)

    def test_sizes_and_masks_do_not_rank(self):
        rows = inventory()
        expected = [r["sample_id"] for r in select_pairs(EVENTS[0], rows)[1]]
        modified = copy.deepcopy(rows)
        for i, row in enumerate(modified):
            row["compressed_bytes"] = 9999999-i
            row["uncompressed_bytes"] = i+1
        _, masks, n = directory_bytes([EVENTS[0]+"/masks/arbitrary_damage_name.png"])
        modified.extend(parse_central_directory(masks, n))
        self.assertEqual([r["sample_id"] for r in select_pairs(EVENTS[0], modified)[1]], expected)

    def test_incomplete_pair_stops_entire_inventory(self):
        rows = inventory()
        with self.assertRaises(ValueError):
            select_pairs(EVENTS[0], rows[:-1])

    def test_duplicate_and_unplanned_input_stop(self):
        rows = inventory()
        with self.assertRaises(ValueError):
            pair_inventory(EVENTS[0], rows+[rows[0]])
        with self.assertRaises(ValueError):
            pair_inventory("new-event", rows)
        rows[0]["damage_count"] = 1
        with self.assertRaises(ValueError):
            pair_inventory(EVENTS[0], rows)

    def test_queue_has_no_admission_or_download_authority(self):
        for event in EVENTS:
            for row in select_pairs(event, inventory(event))[1]:
                self.assertFalse(row["payload_download_enabled"])
                self.assertFalse(row["inference_and_policy_scoring_enabled"])
                self.assertEqual(row["geographic_independence"], "UNVERIFIED")


class GoldBoundaryTests(unittest.TestCase):
    def valid_record(self):
        row = select_pairs(EVENTS[0], inventory())[1][0]
        record = annotation_template(row)
        record.update(status="REVIEWED", whole_scene_reviewed=True, reviewer="reviewer-a",
            reviewed_at_utc="2026-10-25T00:00:00Z", image_sha256={"pre": "a"*64, "post": "b"*64},
            second_review={"reviewer": "reviewer-b", "status": "ADJUDICATED"})
        record["buildings"] = [{"instance_id": "b1", "identity_source": "human_traced",
            "damage": "unknown", "post_polygon_xy": [[1, 1], [3, 1], [3, 3]]}]
        return record

    def test_unfilled_template_is_not_negative_gold(self):
        row = select_pairs(EVENTS[0], inventory())[1][0]
        with self.assertRaises(ValueError):
            validate_gold_record(annotation_template(row))

    def test_unknown_damage_preserved_with_valid_geometry(self):
        self.assertTrue(validate_gold_record(self.valid_record()))

    def test_empty_scene_requires_positive_review_of_emptiness(self):
        record = self.valid_record()
        record["buildings"] = []
        with self.assertRaises(ValueError):
            validate_gold_record(record)
        record["empty_scene_confirmed"] = True
        self.assertTrue(validate_gold_record(record))

    def test_components_cannot_become_instance_gold(self):
        record = self.valid_record()
        record["buildings"][0]["identity_source"] = "connected_components"
        with self.assertRaises(ValueError):
            validate_gold_record(record)

    def test_reviewer_hash_and_geometry_fail_closed(self):
        for change in ["same_reviewer", "missing_hash", "nan_polygon", "duplicate_uid", "zero_area"]:
            record = self.valid_record()
            if change == "same_reviewer": record["second_review"]["reviewer"] = record["reviewer"]
            if change == "missing_hash": record["image_sha256"]["pre"] = None
            if change == "nan_polygon": record["buildings"][0]["post_polygon_xy"][0][0] = float("nan")
            if change == "duplicate_uid": record["buildings"].append(copy.deepcopy(record["buildings"][0]))
            if change == "zero_area": record["buildings"][0]["post_polygon_xy"] = [[1, 1], [2, 2], [3, 3]]
            with self.assertRaises(ValueError):
                validate_gold_record(record)


if __name__ == "__main__":
    unittest.main()
