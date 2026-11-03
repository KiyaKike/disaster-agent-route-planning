import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pyarrow as pa

from associate_buildings import associate, visible_objects
from materialize_hold_probes import (
    COLUMNS, TransientDamageRanges, build_features, extract_allowed_rows, grade_name, validate_probes,
)
from audit_hold_spatial import ring_envelope, overlap_indices, components


def source_row(codes=(2, 4)):
    rings = [[1, 1, 9, 1, 9, 9, 1, 1], [15, 15, 20, 15, 20, 20, 15, 15]]
    return {"ml:polygon_xy": sum(rings, []), "ml:polygon_len": [4, 4],
            "ml:damage": list(codes), "ml:width": 32, "ml:height": 32}


class AnnotationTests(unittest.TestCase):
    def test_source_code_four_is_unknown_not_destroyed(self):
        self.assertEqual(grade_name(2), "major-damage")
        self.assertEqual(grade_name(3), "destroyed")
        self.assertEqual(grade_name(4), "un-classified")
        row = source_row((4, 4))
        features = build_features("s", row)
        objects = visible_objects(features, (32, 32), {"r00c00": [0, 0, 32, 32]})["r00c00"]
        result = associate(np.zeros((32, 32), dtype=bool), objects)
        self.assertIsNone(result["target"])
        self.assertEqual(result["visible_target_uids"], [])

    def test_invalid_codes_or_object_grade_mismatch_stop_materialization(self):
        for code in [-1, 5, True, 2.5]:
            with self.assertRaises(ValueError):
                grade_name(code)
        row = source_row()
        row["ml:damage"] = [2]
        with self.assertRaisesRegex(ValueError, "differ"):
            build_features("s", row)

    def test_derived_id_is_stable_and_duplicate_polygons_are_rejected(self):
        a = build_features("s", source_row())
        b = build_features("s", source_row((3, 0)))
        self.assertEqual([f["properties"]["uid"] for f in a], [f["properties"]["uid"] for f in b])
        self.assertNotEqual(a[0]["properties"]["uid"], a[1]["properties"]["uid"])
        row = source_row()
        row["ml:polygon_xy"] = row["ml:polygon_xy"][:8] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_features("s", row)


class SplitBoundaryTests(unittest.TestCase):
    def test_confirmation_or_changed_probe_list_is_refused(self):
        selected = [{"sample_id": str(i), "split": "development", "interface_probe": True} for i in range(6)]
        validate_probes(selected, selected)
        changed = [dict(r) for r in selected]
        changed[0]["split"] = "confirmation"
        with self.assertRaises(ValueError):
            validate_probes(changed, changed)
        with self.assertRaises(ValueError):
            validate_probes(list(reversed(selected)), selected)

    def test_shared_column_extractor_exposes_only_requested_rows(self):
        table = pa.Table.from_pylist([source_row((0, 0)), source_row((2, 4)), source_row((3, 3))])
        class Parquet:
            def read(self, *, columns, use_threads):
                assert columns == COLUMNS and use_threads is False
                return table
        identities = [{"ml:source_file": s} for s in ["reserved1", "allowed", "reserved2"]]
        result = extract_allowed_rows(Parquet(), identities, ["allowed"])
        self.assertEqual(list(result), ["allowed"])
        self.assertEqual(result["allowed"]["ml:damage"], [2, 4])

    def test_damage_payload_is_transient_and_never_sent_to_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Remote:
                def seek(self, offset): self.position = offset
                def read(self, count): return b"d" * count
            cached = SimpleNamespace(root=Path(tmp), remote=Remote(), accesses=[])
            cached.read = lambda *args: (_ for _ in ()).throw(AssertionError("Damage must bypass cache"))
            ranges = TransientDamageRanges(cached, [{"offset": 100, "bytes": 20}])
            self.assertEqual(ranges.read(100, 20), b"d" * 20)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertFalse(ranges.accesses[0]["persisted"])
            (Path(tmp) / "100_20.bin").write_bytes(b"d" * 20)
            (Path(tmp) / "100_20.json").write_text(json.dumps({"offset": 100, "bytes": 20}))
            with self.assertRaisesRegex(ValueError, "persist"):
                TransientDamageRanges(cached, [{"offset": 100, "bytes": 20}])


class SpatialScreenTests(unittest.TestCase):
    def test_missing_geometry_and_touching_edges_are_not_positive_area_intersections(self):
        self.assertIsNone(ring_envelope([]))
        boxes = np.asarray([[0, 0, 1, 1], [1, 0, 2, 1], [.5, .5, 1.5, 1.5], [np.nan] * 4])
        self.assertEqual(overlap_indices([0, 0, 1, 1], boxes).tolist(), [0, 2])
        self.assertEqual(overlap_indices(None, boxes).tolist(), [])

    def test_geographic_components_preserve_singletons_and_transitive_overlap(self):
        self.assertEqual(components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")]), [["a", "b", "c"], ["d"]])
        with self.assertRaises(ValueError):
            ring_envelope([0, 95, 1, 96])


if __name__ == "__main__":
    unittest.main()
