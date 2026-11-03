import unittest

import numpy as np

from run_boone_structure_check import pearson_change, top_metrics


class StructureCheckTests(unittest.TestCase):
    def test_channelwise_affine_intensity_does_not_change_structure(self):
        rng = np.random.default_rng(7)
        pre, post = rng.random((100, 3)), rng.random((100, 3))
        original, _ = pearson_change(pre, post)
        transformed, _ = pearson_change(pre*np.array([2, 3, .1])+[.2, .4, .8],
                                        post*np.array([.3, 5, 2])+[.7, .1, .2])
        self.assertAlmostEqual(original, transformed, places=13)

    def test_undefined_flat_correlations_are_not_fabricated(self):
        with self.assertRaises(ValueError):
            pearson_change(np.ones((10, 3)), np.ones((10, 3)))

    def test_ndcg_uses_unvisited_positive_count_and_rank(self):
        result = top_metrics({'a': .9, 'b': .8, 'c': .7, 'd': .6},
                             {'a': 0, 'b': 1, 'c': 1, 'd': 0})
        self.assertEqual(result['top3_positive'], 2)
        self.assertAlmostEqual(result['precision_at3'], 2/3)
        self.assertAlmostEqual(result['ndcg_at3'], (1/np.log2(3)+.5)/(1+1/np.log2(3)))


if __name__ == '__main__':
    unittest.main()
