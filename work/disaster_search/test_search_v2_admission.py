"""Protect split isolation and metadata-only access before building new data."""
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from audit_hold_taco import MetadataView, parse_index, project_identities, IDENTITY_COLUMNS, assert_no_protected_reads
from plan_search_v2 import EVENTS, freeze, select_cohort
from verify_taco_polygon_bridge import pair_geometry, ring_key, split_rings


def pools():
    return {"train": ["other_00000001"], "tier3": [], "test": [], "hold": [
        f"{event}_{i:08d}" for events in EVENTS.values() for event in events for i in range(30)]}


class CohortTests(unittest.TestCase):
    def test_selection_is_order_invariant_and_confirmation_events_are_separate(self):
        data = pools()
        first = select_cohort(data)
        self.assertEqual(first, select_cohort({k: list(reversed(v)) for k, v in data.items()}))
        self.assertEqual(len(first), 216)
        self.assertEqual(sum(r["interface_probe"] for r in first), 6)
        self.assertFalse(any(r["interface_probe"] for r in first if r["split"] != "development"))
        self.assertFalse(set(EVENTS["confirmation"]) & (set(EVENTS["development"]) | set(EVENTS["validation"])))

    def test_training_overlap_is_fatal(self):
        data = pools()
        data["train"].append(data["hold"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_cohort(data)

    def test_sparse_event_cannot_be_silently_filled(self):
        data = pools()
        data["hold"] = [s for s in data["hold"] if not s.startswith("palu-tsunami_")]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_cohort(data)

    def test_freeze_refuses_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"
            freeze(path, {"split": "development"})
            freeze(path, {"split": "development"})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                freeze(path, {"split": "confirmation"})
            self.assertEqual(json.loads(path.read_text()), {"split": "development"})


class MetadataAccessTests(unittest.TestCase):
    def test_range_receipts_must_not_touch_protected_damage_payload(self):
        protected = [{"field": "ml:damage", "offset": 100, "bytes": 20}]
        assert_no_protected_reads([{"offset": 50, "bytes": 50}, {"offset": 120, "bytes": 30}], protected)
        with self.assertRaisesRegex(ValueError, "protected"):
            assert_no_protected_reads([{"offset": 99, "bytes": 2}], protected)

    def test_parquet_projection_cannot_read_damage_or_geometry(self):
        class Parquet:
            schema_arrow = SimpleNamespace(names=IDENTITY_COLUMNS + ["ml:damage", "ml:polygon_xy"])
            def read(self, *, columns, use_threads):
                assert columns == IDENTITY_COLUMNS
                assert not use_threads
                return SimpleNamespace(to_pylist=lambda: [{"id": "safe"}])
        self.assertEqual(project_identities(Parquet()), [{"id": "safe"}])

    def test_subfile_reads_stay_in_declared_metadata_member(self):
        accesses = []
        class Ranges:
            def read(self, offset, size):
                accesses.append((offset, size))
                return b"x" * size
        view = MetadataView(Ranges(), {"offset": 100, "size": 10})
        view.seek(-2, 2)
        self.assertEqual(view.read(100000), b"xx")
        self.assertEqual(accesses, [(108, 2)])
        with self.assertRaises(ValueError):
            view.seek(-1)
        with self.assertRaises(ValueError):
            view.seek(11)

    def test_cozip_index_refuses_data_members_and_out_of_bounds(self):
        def payload(names, offsets=(100, 200, 300), sizes=(10, 10, 10), profile=2):
            encoded = [s.encode() for s in names]
            return (struct.pack("<4sHBI", b"CZIP", 1, profile, len(names)) +
                    struct.pack("<" + "H" * len(names), *(len(s) for s in encoded)) +
                    b"".join(encoded) + struct.pack("<3Q", *offsets) + struct.pack("<3Q", *sizes))
        names = ["COLLECTION.json", "METADATA/children.parquet", "METADATA/sample.parquet"]
        self.assertEqual(len(parse_index(payload(names), 1000)), 3)
        with self.assertRaises(ValueError):
            parse_index(payload(names, profile=1), 1000)
        with self.assertRaises(ValueError):
            parse_index(payload(names, offsets=(100, 200, 999)), 1000)
        with self.assertRaises(ValueError):
            parse_index(payload(["COLLECTION.json", "DATA/0/post.png", "METADATA/sample.parquet"]), 1000)


class PolygonBridgeTests(unittest.TestCase):
    def test_ring_identity_ignores_first_vertex_and_winding(self):
        points = [(0, 0), (2, 0), (2, 2), (0, 0)]
        self.assertEqual(ring_key(points), ring_key([(2, 0), (0, 0), (2, 2)]))

    def test_original_uid_mapping_requires_one_to_one_polygons(self):
        feature = {"wkt": "POLYGON ((0 0, 2 0, 2 2, 0 0))", "properties": {"uid": "original-u1"}}
        flat = [0, 0, 2, 0, 2, 2, 0, 0]
        mapping = pair_geometry([feature], flat, [4])
        self.assertEqual(mapping[0]["original_uid_on_this_known_scene"], "original-u1")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            pair_geometry([feature, feature], flat * 2, [4, 4])
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            pair_geometry([feature], [0, 0, 5, 0, 5, 5, 0, 0], [4])

    def test_invalid_arrays_and_nonfinite_geometry_are_rejected(self):
        with self.assertRaises(ValueError):
            split_rings([0, 1, 2], [3])
        with self.assertRaises(ValueError):
            ring_key([(0, 0), (float("nan"), 0), (2, 2)])


if __name__ == "__main__":
    unittest.main()
