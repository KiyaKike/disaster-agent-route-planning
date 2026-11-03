import unittest

from prepare_xbd_sample import grid_truth, polygon_centroid


def building(uid, subtype, wkt="POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"):
    return {"properties": {"uid": uid, "subtype": subtype}, "wkt": wkt}


class LabelChecks(unittest.TestCase):
    def test_polygon_centroid_independent_of_vertex_order(self):
        self.assertEqual(polygon_centroid("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), (5, 5))
        self.assertEqual(polygon_centroid("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"), (5, 5))

    def test_unknown_label_never_becomes_negative(self):
        truth = grid_truth([building("x", "un-classified")], 100, 100, 2)
        self.assertIsNone(truth["r00c00"]["target_present"])

    def test_multiple_target_buildings_are_counted_once_each(self):
        truth = grid_truth([building("x", "major-damage"), building("y", "destroyed")], 100, 100, 2)
        self.assertTrue(truth["r00c00"]["target_present"])
        self.assertEqual(len(truth["r00c00"]["target_building_ids"]), 2)
        self.assertEqual(sum(c["target_present"] is True for c in truth.values()), 1)

    def test_duplicates_and_unsupported_geometry_rejected(self):
        with self.assertRaises(ValueError):
            grid_truth([building("x", "destroyed"), building("x", "destroyed")], 100, 100, 2)
        with self.assertRaises(ValueError):
            polygon_centroid("MULTIPOLYGON (((0 0, 10 0, 10 10, 0 0)))")


if __name__ == "__main__":
    unittest.main()
