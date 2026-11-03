"""Checks for crop isolation, conditional transfer and score evaluation."""
import unittest

import numpy as np
from PIL import Image

from run_boone_context_experiment import grouped_ap, masked_crop, metrics, score_statistics


class ExperimentTests(unittest.TestCase):
    def test_unobserved_background_cannot_change_crop_or_features(self):
        first = np.zeros((24, 24, 3), dtype=np.uint8)
        second = np.full_like(first, 255)
        first[5:16, 5:16] = 77
        second[5:16, 5:16] = 77
        polygon = [(5, 5), (15, 5), (15, 15), (5, 15)]
        crop1, values1, _, _ = masked_crop(Image.fromarray(first), polygon)
        crop2, values2, _, _ = masked_crop(Image.fromarray(second), polygon)
        self.assertTrue(np.array_equal(crop1, crop2))
        self.assertTrue(np.array_equal(values1, values2))

    def test_visual_statistics_can_distinguish_equal_distance_targets(self):
        p0 = np.array([.8, .8, .8])
        xy = np.array([[0., 0.], [1., 0.], [-1., 0.]])
        similarity = np.array([[1., .9, .1], [.9, 1., .2], [.1, .2, 1.]])
        scores = score_statistics(p0, 0, 0., xy, similarity)
        self.assertEqual(scores['S1_spatial'][1], scores['S1_spatial'][2])
        self.assertLess(scores['S2_visual_spatial'][1], scores['S2_visual_spatial'][2])

    def test_tied_ap_and_known_brier(self):
        self.assertAlmostEqual(grouped_ap([.5, .5, .2], [1, 0, 1]), 7/12)
        self.assertAlmostEqual(metrics([.2, .8], [0, 1])['brier'], .04)
        self.assertIsNone(grouped_ap([.2, .8], [0, 0]))


if __name__ == '__main__':
    unittest.main()
