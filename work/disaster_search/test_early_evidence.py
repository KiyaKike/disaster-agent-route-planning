import itertools
import math
import unittest

from diagnose_early_evidence import (counterfactual, oriented_auc, pair_alignment,
                                     prior_order, probe_features)
from run_cached_search import PublicCell
from test_cached_search import RATES, observation


class EarlyEvidenceChecks(unittest.TestCase):
    def cells(self):
        return {str(i): PublicCell(str(i), i, 0, i % 2, .1 + i / 10) for i in range(4)}

    def test_all_negative_probe_has_no_binary_alignment(self):
        obs = [observation(str(i), "native", False) for i in range(4)]
        result = probe_features(self.cells(), obs, RATES)
        self.assertIsNone(result["binary_alignment"])
        self.assertEqual(result["binary_comparable_pairs"], 0)
        self.assertLess(result["response_surplus"], 0)

    def test_alignment_does_not_compare_different_quality_modes(self):
        obs = [observation("0", "blurred", False), observation("1", "native", True)]
        self.assertEqual(pair_alignment([.1, .2], obs, "declared_target"), (None, 0))
        obs[0]["mode"] = "native"
        self.assertEqual(pair_alignment([.1, .2], obs, "declared_target"), (1., 1))

    def test_features_cannot_read_truth_or_future_fields(self):
        obs = [observation(str(i), "native", i == 3) for i in range(4)]
        result = probe_features(self.cells(), obs, RATES)
        for row in obs:
            row.update(target=False, matched_target_uids=["SECRET"], future_outcome=True)
        self.assertEqual(result, probe_features(self.cells(), obs, RATES))
        with self.assertRaises(ValueError):
            probe_features(self.cells(), [obs[0], obs[0]], RATES)

    def test_paid_prefix_and_cross_grid_discoveries_not_double_counted(self):
        matched = {"0": {"a"}, "1": {"a"}, "2": {"b"}, "3": set()}
        result = counterfactual(self.cells(), ["0"], 3, matched)
        self.assertEqual(result["prefix_discoveries"], 1)
        self.assertEqual(result["remaining_reachable_targets"], 1)
        # Draw 2 of the 3 remaining cells, including B with probability 2/3.
        self.assertAlmostEqual(result["random_expected_total"], 1 + 2 / 3)
        self.assertEqual(len(result["prior_continuation_cells"]), 2)
        self.assertEqual(result["prior_total"], 2)

    def test_random_conditional_expectation_matches_enumeration(self):
        matched = {"0": set(), "1": {"a"}, "2": {"a", "b"}, "3": {"c"}}
        result = counterfactual(self.cells(), ["0"], 3, matched)
        totals = [len(set().union(*(matched[c] for c in subset))) for subset in itertools.combinations(["1", "2", "3"], 2)]
        self.assertAlmostEqual(result["random_expected_total"], sum(totals) / len(totals))
        self.assertGreaterEqual(result["oracle_expected_total"], result["prior_total"])

    def test_no_opportunity_has_null_gain_label(self):
        result = counterfactual(self.cells(), ["0"], 3, {str(i): set() for i in range(4)})
        self.assertIsNone(result["normalized_gain"])
        self.assertEqual(result["gain"], 0)

    def test_prior_continuation_uses_only_public_prior_and_excludes_probe(self):
        self.assertEqual(prior_order(self.cells(), ["3"], 2), ["2", "1"])
        self.assertEqual(prior_order(self.cells(), [], 2), ["3", "2"])

    def test_auc_orientation_and_missing_classes(self):
        self.assertEqual(oriented_auc([0, 1], [-.2, .2], .1)["auc"], 1)
        self.assertEqual(oriented_auc([1, 0], [-.2, .2], .1)["auc"], 0)
        self.assertEqual(oriented_auc([1, 1], [-.2, .2], .1)["auc"], .5)
        self.assertIsNone(oriented_auc([1, 2], [.2, .3], .1)["auc"])


if __name__ == "__main__":
    unittest.main()
