import unittest

from diagnose_feedback_scope import aggregate, eligible, related, scene_diagnostic


RATES = {"native": {"tpr": .85, "fpr": .01}, "blurred": {"tpr": .15, "fpr": .005}}


def cell(cid, x, y, group, target=True, uid=None):
    return {"cell_id": cid, "x": x, "y": y, "group": group, "target": target,
            "uids": [uid] if uid else [], "prior": .2,
            "observations": {"native": bool(target), "blurred": False}}


class ScopeDiagnosticsChecks(unittest.TestCase):
    def test_public_neighbors_are_independent_of_truth(self):
        a, b = cell("a", 0, 0, 0), cell("b", 1, 1, 0, False)
        self.assertTrue(related(a, b, "neighbors_same_bin"))
        b["target"] = None
        self.assertTrue(related(a, b, "neighbors_same_bin"))
        self.assertFalse(eligible(a, b, "neighbors_same_bin", False))
        self.assertFalse(related(a, a, "neighbors"))

    def test_shared_target_exclusion_is_evaluator_only(self):
        a, b = cell("a", 0, 0, 0, uid="same"), cell("b", 1, 0, 0, uid="same")
        self.assertTrue(eligible(a, b, "neighbors", False))
        self.assertFalse(eligible(a, b, "neighbors", True))
        b["uids"] = ["different"]
        self.assertTrue(eligible(a, b, "neighbors", True))

    def test_equal_scene_mean_does_not_weight_dense_scene_more(self):
        base = {"ordered_pairs": 1, "shared_uid_pairs_excluded": 0,
                "within_scene_conditional_difference": None,
                "one_step_brier_delta": {"native:positive": -.1},
                "one_step_source_counts": {"native:positive": 1}}
        dense = {**base, "ordered_pairs": 100, "one_step_brier_delta": {"native:positive": .2},
                 "one_step_source_counts": {"native:positive": 100}}
        result = aggregate([base, dense])
        self.assertAlmostEqual(result["one_step_brier_delta"]["native:positive"]["equal_scene_mean"], .05)

    def test_known_negative_scene_and_unknown_exclusion(self):
        cells = [cell("a", 0, 0, 0, False), cell("b", 1, 0, 0, False), cell("c", 2, 0, 0, None)]
        result = scene_diagnostic("s", "e", cells, "neighbors", True, RATES)
        self.assertEqual(result["ordered_pairs"], 2)
        self.assertIsNone(result["within_scene_conditional_difference"])
        self.assertLess(result["one_step_brier_delta"]["native:negative"], 0)


if __name__ == "__main__":
    unittest.main()
