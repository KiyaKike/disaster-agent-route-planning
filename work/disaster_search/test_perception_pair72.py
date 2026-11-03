import copy
import unittest

import numpy as np

from perception_pair72_core import rasterize_post, confusion, metrics, object_vote, prepare_r34, decode_output, TILES


def feature(grade, ring='1 1, 3 1, 3 3, 1 3, 1 1'):
    return {'wkt': 'POLYGON ((' + ring + '))', 'properties': {'subtype': grade}}


class TestPostReference(unittest.TestCase):
    def test_inclusive_rounding(self):
        d = {'features': {'xy': [feature('major-damage')]}}
        mask, audit = rasterize_post(d, (6, 6))
        self.assertEqual(int((mask == 3).sum()), 9)
        self.assertEqual(audit['ignored_union_pixels'], 0)

    def test_unknown_conflicts_and_order(self):
        d = {'features': {'xy': [feature('major-damage'), feature('no-damage'),
                                  feature('un-classified', '0 0, 1 0, 1 1, 0 1, 0 0')]}}
        a, audit = rasterize_post(d, (6, 6))
        e = copy.deepcopy(d); e['features']['xy'].reverse()
        b, _ = rasterize_post(e, (6, 6))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(audit['conflicting_known_pixels'], 9)
        self.assertEqual(audit['ignored_union_pixels'], 12)

    def test_boundary_and_invalid(self):
        d = {'features': {'xy': [feature('destroyed', '-0.5 0, 2 0, 2 2, -0.5 2, -0.5 0')]}}
        a, _ = rasterize_post(d, (6, 6)); self.assertEqual(int((a == 4).sum()), 9)
        d['features']['xy'][0] = feature('destroyed', '-0.6 0, 2 0, 2 2, -0.6 2, -0.6 0')
        with self.assertRaises(ValueError): rasterize_post(d, (6, 6))

    def test_no_image_geometry_resizing(self):
        coverage = np.zeros((1024, 1024), np.uint8)
        for x0, y0, x1, y1 in TILES: coverage[y0:y1, x0:x1] += 1
        self.assertTrue(np.all(coverage == 1))


class TestScoresAndAdapters(unittest.TestCase):
    def test_severe_collapses_3_and_4_and_ignores_unknown(self):
        truth = np.array([[3, 4, 0, 1, 255, 2]], np.uint8)
        pred = np.array([[4, 0, 3, 4, 3, 2]], np.uint8)
        result = metrics(confusion(truth, pred))
        self.assertEqual([result[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 2, 1, 1])
        self.assertEqual(result['precision'], 1/3)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['f1'], .4)
        self.assertEqual(result['evaluated_pixels'], 5)
        self.assertEqual(result['false_severe_background_pixels'], 1)

    def test_zero_reference_is_null_recall(self):
        a = np.zeros((2, 2), np.uint8)
        result = metrics(confusion(a, a))
        self.assertIsNone(result['recall']); self.assertIsNone(result['precision']); self.assertIsNone(result['f1'])
        result = metrics(confusion(a, np.full_like(a, 4)))
        self.assertEqual(result['f1'], 0); self.assertIsNone(result['recall'])

    def test_common_vote_8_connected_weighted_and_background(self):
        loc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], bool)
        classes = np.array([[4, 3, 3], [3, 2, 3], [3, 3, 3]], np.uint8)
        expected = loc.astype(np.uint8) * 2  # minor has 38 votes; destroyed 11
        np.testing.assert_array_equal(object_vote(loc, classes), expected)

    def test_normalization_and_decoders(self):
        import torch
        from PIL import Image
        from run_changeos_pilot import prepare_tensor
        rng = np.random.default_rng(17)
        a = Image.fromarray(rng.integers(0, 256, (32, 32, 3), np.uint8))
        b = Image.fromarray(rng.integers(0, 256, (32, 32, 3), np.uint8))
        torch.testing.assert_close(prepare_r34(a, b), prepare_tensor(a, b, 'native'), rtol=1e-5, atol=6e-7)
        damage = torch.zeros((1, 5, 2, 2)); damage[:, 3] = 1
        loc = torch.tensor([[[[.5, .6], [0., 1.]]]])
        building, classes = decode_output({'t1_semantic_prediction': loc, 'change_prediction': damage}, 'swint', (2, 2))
        np.testing.assert_array_equal(building, [[False, True], [False, True]])
        building2, classes2 = decode_output((loc - .5, damage * 5), 'r34', (2, 2))
        np.testing.assert_array_equal(building, building2); np.testing.assert_array_equal(classes, classes2)
        with self.assertRaises(ValueError): decode_output((loc, damage), 'r34', (4, 4))
        damage[:, 2] = float('nan')
        with self.assertRaises(ValueError): decode_output((loc, damage), 'r34', (2, 2))


if __name__ == '__main__': unittest.main()
