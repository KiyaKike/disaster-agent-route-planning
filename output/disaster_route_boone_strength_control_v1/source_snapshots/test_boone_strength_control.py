import unittest

import numpy as np

from run_boone_strength_control import matched_spatial


class StrengthControlTests(unittest.TestCase):
    def test_matches_actual_magnitude_when_some_targets_clip(self):
        p0 = np.array([.8, .01, .9, .5])
        spatial = np.array([1., .9, .8, .7])
        mask = np.array([False, True, True, True])
        gated = np.clip(p0 - .5 * spatial * .8 * np.array([1., 1., .2, .9]), 0, 1)
        scores, gain, desired = matched_spatial(p0, spatial, -.8, mask, gated)
        self.assertAlmostEqual(np.mean(np.abs(scores[mask]-p0[mask])), desired, places=13)
        self.assertGreater(gain, 0)
        self.assertLess(gain, .5)

    def test_unit_gate_recovers_spatial_update(self):
        p0, spatial = np.array([.5, .4, .3]), np.array([1., .8, .6])
        mask = np.array([False, True, True])
        gated = np.clip(p0 + .5 * spatial * .5, 0, 1)
        scores, gain, _ = matched_spatial(p0, spatial, .5, mask, gated)
        np.testing.assert_allclose(scores, gated, atol=1e-14)
        self.assertAlmostEqual(gain, .5)

    def test_zero_residual_requires_no_update(self):
        p0 = np.array([.5, .4, .3])
        scores, gain, desired = matched_spatial(p0, np.ones(3), 0., np.array([False, True, True]), p0)
        np.testing.assert_array_equal(scores, p0)
        self.assertEqual((gain, desired), (0., 0.))


if __name__ == '__main__':
    unittest.main()
