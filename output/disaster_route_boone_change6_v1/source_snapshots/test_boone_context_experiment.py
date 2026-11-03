"""Checks for crop isolation, conditional transfer and score evaluation."""
import unittest

import numpy as np
from PIL import Image

from run_boone_context_experiment import feature_kernel, grouped_ap, masked_crop, metrics, score_statistics


class ExperimentTests(unittest.TestCase):
    def test_change_only_kernel_cannot_read_discarded_color_dimensions(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(12, 36))
        altered = features.copy()
        altered[:, :30] = rng.normal(size=(12, 30))*100
        for original, changed in zip(feature_kernel(features, 'change-only'),
                                     feature_kernel(altered, 'change-only')):
            np.testing.assert_allclose(original, changed, rtol=0, atol=0)
        _, full_z, _, _ = feature_kernel(features)
        active, change_z, _, _ = feature_kernel(features, 'change-only')
        self.assertEqual(active.shape, (12, 6))
        np.testing.assert_allclose(change_z, full_z[:, 30:], rtol=0, atol=1e-12)

    def test_identical_features_reduce_visual_update_to_spatial_update(self):
        _, _, similarity, tau = feature_kernel(np.zeros((3, 36)), 'change-only')
        np.testing.assert_array_equal(similarity, np.ones((3, 3)))
        self.assertGreater(tau, 0)
        p0, xy = np.array([.8, .7, .6]), np.array([[0, 0], [1, 0], [2, 0]])
        scores = score_statistics(p0, 0, 0, xy, similarity)
        np.testing.assert_array_equal(scores['S1_spatial'], scores['S2_visual_spatial'])

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
