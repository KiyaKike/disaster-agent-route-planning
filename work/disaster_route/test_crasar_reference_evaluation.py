import unittest
from evaluate_crasar_routes import checked_reference_counts,detected_at_coverage

class ReferenceEvaluation(unittest.TestCase):
    def test_unknown_labels_are_not_negatives(self):
        r=checked_reference_counts([1,1,1],[1,0,0],[1,1,0])
        self.assertEqual(r,{'tp':1,'fp':1,'ignored_unknown':1,'precision':.5})
    def test_all_unknown_predictions_have_undefined_precision(self):
        r=checked_reference_counts([1,1],[0,0],[0,0])
        self.assertEqual(r['fp'],0);self.assertEqual(r['ignored_unknown'],2);self.assertIsNone(r['precision'])
    def test_invalid_positive_reference_rejected(self):
        with self.assertRaises(ValueError):checked_reference_counts([1],[1],[0])
    def test_full_coverage_is_not_necessarily_a_detection_ceiling(self):
        self.assertTrue(detected_at_coverage([.8],[.09])[0])
        self.assertFalse(detected_at_coverage([1.],[.09])[0])

if __name__=='__main__':unittest.main()
