import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_hold_taco import URL
from build_xbd_mini import sha
from inferred_scene_geometry import (COLUMNS, OfflineGeometryRanges, covered,
    fit_geometry, polygon_overlap, bounds)
from run_metadata_admission_audit import summarize_candidates


class InferredGeometryTests(unittest.TestCase):
    def fixture(self):
        xy = np.array([[0, 0], [400, 0], [400, 400], [0, 400], [0, 0],
            [600, 600], [1000, 600], [1000, 1000], [600, 1000], [600, 600]], float)
        geo = xy @ np.array([[1e-6, 1e-7], [2e-7, -1e-6]]) + [100, 30]
        return xy.reshape(-1), [5, 5], geo.reshape(-1), [5, 5], 1024, 1024

    def test_recovers_known_affine_corners_and_never_certifies(self):
        g = fit_geometry(*self.fixture())
        self.assertTrue(g["usable_inferred_range"])
        self.assertFalse(g["official_footprint_verified"])
        xy = np.array([[0, 0], [1024, 0], [1024, 1024], [0, 1024]])
        expected = xy @ np.array([[1e-6, 1e-7], [2e-7, -1e-6]]) + [100, 30]
        np.testing.assert_allclose(expected, g["inferred_corners_wgs84"], rtol=0, atol=1e-12)
        self.assertLess(g["max_inverse_residual_px"], 1e-6)

    def test_padding_expands_range_but_is_not_uncertainty_certificate(self):
        g = fit_geometry(*self.fixture())
        a, b = g["inferred_bounds_wgs84"], g["padded_inferred_bounds_wgs84"]
        self.assertTrue(b[0] < a[0] and b[1] < a[1] and b[2] > a[2] and b[3] > a[3])

    def test_unaligned_missing_and_degenerate_are_unknown(self):
        f = list(self.fixture())
        f[3] = [4, 6]
        self.assertEqual(fit_geometry(*f)["status"], "UNALIGNED_RING_LAYOUT")
        self.assertFalse(fit_geometry([], [], [], [], 1024, 1024)["usable_inferred_range"])
        f = list(self.fixture())
        f[0] = np.zeros(20)
        self.assertFalse(fit_geometry(*f)["usable_inferred_range"])

    def test_small_annotation_coverage_not_extrapolated_as_trusted_range(self):
        f = list(self.fixture())
        f[0] = f[0] / 10
        self.assertEqual(fit_geometry(*f)["status"], "INSUFFICIENT_IMAGE_COVERAGE")

    def test_bad_geographic_pairing_fails_pixel_residual(self):
        f = list(self.fixture())
        f[2] = f[2].copy()
        f[2][0] += .001
        self.assertEqual(fit_geometry(*f)["status"], "AFFINE_RESIDUAL_TOO_LARGE")

    def test_polygon_clipping_has_known_overlap_and_winding_invariance(self):
        a = [[0, 0], [2, 0], [2, 2], [0, 2]]
        b = [[1, 1], [3, 1], [3, 3], [1, 3]]
        self.assertAlmostEqual(polygon_overlap(a, b), .25)
        self.assertAlmostEqual(polygon_overlap(a[::-1], b[::-1]), .25)
        self.assertEqual(polygon_overlap(a, [[2, 0], [4, 0], [4, 2], [2, 2]]), 0.)
        self.assertAlmostEqual(polygon_overlap(a, [[.5, .5], [1, .5], [1, 1], [.5, 1]]), 1.)

    def test_clipping_is_stable_at_geographic_offset(self):
        a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]]) * 1e-4 + [179, -80]
        b = a + [1e-4, 1e-4]
        self.assertAlmostEqual(polygon_overlap(a, b), .25, places=8)

    def test_interval_union_requires_complete_coverage(self):
        self.assertTrue(covered(10, 20, [{"offset": 10, "bytes": 10}, {"offset": 20, "bytes": 10}]))
        self.assertFalse(covered(10, 20, [{"offset": 10, "bytes": 9}, {"offset": 20, "bytes": 10}]))

    def test_offline_reader_rejects_protected_and_missing_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            r = OfflineGeometryRanges(Path(folder))
            r.protected = [{"field": "ml:damage", "offset": 10, "bytes": 10}]
            with self.assertRaisesRegex(ValueError, "protected column"):
                r.read(15, 1)
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                r.read(100, 1)
            self.assertEqual(r.remote.budget.received, 0)
            self.assertEqual(r.accesses, [])

    def test_offline_reader_checks_cache_receipt_and_whitelist(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"abcd"
            (root / '100_4.bin').write_bytes(data)
            (root / '100_4.json').write_text(json.dumps({"url": URL, "offset": 100, "bytes": 4, "sha256": sha(data)}))
            r = OfflineGeometryRanges(root)
            r.allowed = [{"offset": 100, "bytes": 4}]
            self.assertEqual(r.read(100, 4), data)
            with self.assertRaisesRegex(ValueError, "outside explicitly allowed"):
                r.read(99, 4)
            (root / '100_4.bin').write_bytes(b"bad!")
            with self.assertRaisesRegex(ValueError, "differs"):
                r.read(100, 4)

    def test_projected_columns_exclude_damage_and_count_payloads(self):
        self.assertFalse(set(COLUMNS) & {"ml:damage", "ml:n_buildings", "ml:n_buildings_pre"})

    def test_missing_range_is_unknown_not_admitted(self):
        rows = [{"split": "development", "event": "e", "usable_inferred_range": False,
            "old_training_annotation_envelope_overlap_ids": [], "training_inferred_overlaps": None,
            "training_padded_inferred_overlaps": None, "selection_test_inferred_overlaps": None}]
        row = summarize_candidates(rows)[0]
        self.assertEqual(row["unknown_inferred_ranges"], 1)
        self.assertEqual(row["independently_admitted_scenes"], 0)


if __name__ == "__main__":
    unittest.main()
