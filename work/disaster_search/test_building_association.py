import unittest

import numpy as np

from associate_buildings import associate, discovery_summary, visible_objects
from real_evidence import ImageEvidenceRegistry


def feature(uid="a", subtype="destroyed", wkt="POLYGON ((2 1, 5 1, 5 3, 2 3, 2 1))"):
    return {"properties": {"uid": uid, "subtype": subtype}, "wkt": wkt}


class AssociationTest(unittest.TestCase):
    def test_cross_boundary_raster_pixels_have_exactly_one_cell_owner(self):
        cells = {"left": [0, 0, 4, 4], "right": [4, 0, 8, 4]}
        objects = visible_objects([feature()], (8, 4), cells)
        # Same UID visible in both cells; global raster has 4 * 3 pixels.
        self.assertEqual(sum(o[0]["mask"].sum() for o in objects.values()), 12)
        rows = []
        for cell in cells:
            row = associate(np.ones((4, 4), dtype=bool), objects[cell])
            self.assertIs(row["target"], True)
            rows.append({"sample_id": "scene", **row})
        totals = discovery_summary(rows + rows)
        self.assertEqual(totals["unique_visible_targets"], 1)
        self.assertEqual(totals["unique_matched_targets"], 1)
        self.assertEqual(totals["repeated_matches_not_counted_again"], 3)

    def test_unknown_cross_boundary_remains_ambiguous(self):
        cells = {"left": [0, 0, 4, 4]}
        objects = visible_objects([feature(subtype="un-classified")], (8, 4), cells)
        row = associate(np.ones((4, 4), dtype=bool), objects["left"])
        self.assertIsNone(row["target"])
        self.assertEqual(row["matched_target_uids"], [])
        self.assertEqual(row["severe_pixels_on_unknown_only"], 6)

    def test_tiny_fragment_not_silently_negative_or_confirmed(self):
        obj = {"uid": "a", "target": True, "mask": np.array([[True, False]])}
        row = associate(np.ones((1, 2), dtype=bool), [obj])
        self.assertIsNone(row["target"])
        self.assertEqual(row["matched_target_uids"], [])

    def test_small_overlap_and_broad_false_positive_are_auditable(self):
        mask = np.ones((10, 10), dtype=bool)
        pred = np.zeros((10, 10), dtype=bool)
        pred[0, :4] = True  # Four pixels alone do not clear 10% of this footprint.
        row = associate(pred, [{"uid": "a", "target": True, "mask": mask}])
        self.assertEqual(row["matched_target_uids"], [])
        target = np.zeros_like(mask)
        target[:2, :2] = True
        row = {"sample_id": "s", **associate(mask, [{"uid": "a", "target": True, "mask": target}])}
        self.assertEqual(discovery_summary([row])["severe_pixel_precision_known_only"], .04)

    def test_duplicate_uid_rejected(self):
        with self.assertRaises(ValueError):
            visible_objects([feature(), feature()], (8, 4), {"x": [0, 0, 8, 4]})


class RegistryTest(unittest.TestCase):
    def observation(self, mode):
        return {"sample_id": "s", "cell_id": "r00c00", "mode": mode, "evidence_group_id": "s:r00c00",
                "observation_id": f"s:r00c00:{mode}", "declared_target": mode == "native",
                "uncalibrated_severe_score": .9 if mode == "native" else .1,
                "matched_target_uids": ["MUST_NOT_LEAK"]}

    def test_upgrade_replaces_evidence_and_order_does_not_matter(self):
        registries = []
        for modes in (("blurred", "native"), ("native", "blurred")):
            registry = ImageEvidenceRegistry()
            for mode in modes:
                registry.add(self.observation(mode))
            reps = registry.representatives()
            self.assertEqual(len(reps), 1)
            self.assertEqual(reps[0]["mode"], "native")
            self.assertNotIn("matched_target_uids", reps[0])
            registries.append(reps)
        self.assertEqual(*registries)

    def test_duplicate_observation_and_spoofed_group_rejected(self):
        registry = ImageEvidenceRegistry()
        registry.add(self.observation("native"))
        with self.assertRaises(ValueError):
            registry.add(self.observation("native"))
        obs = self.observation("blurred")
        obs["evidence_group_id"] = "fake"
        with self.assertRaises(ValueError):
            registry.add(obs)


if __name__ == "__main__":
    unittest.main()
