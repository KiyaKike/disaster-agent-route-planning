import unittest
import numpy as np
from audit_feedback_increment import average_precision_tied


class AuditMetricsTests(unittest.TestCase):
    def test_constant_scores_equal_prevalence(self):
        self.assertEqual(average_precision_tied([1,0],[.5,.5]),.5)
        self.assertEqual(average_precision_tied([0,1],[.5,.5]),.5)

    def test_ties_invariant_to_row_order(self):
        y=np.array([1,0,0,1,1]);s=np.array([.5,.5,.5,.2,.2]);p=[4,2,0,3,1]
        self.assertEqual(average_precision_tied(y,s),average_precision_tied(y[p],s[p]))

    def test_agrees_with_explicit_threshold_enumeration(self):
        rng=np.random.default_rng(12)
        for _ in range(20):
            y=rng.integers(0,2,50);s=np.round(rng.random(50),1)
            previous=0;expected=0.
            for threshold in sorted(set(s),reverse=True):
                chosen=s>=threshold;tp=y[chosen].sum();recall=tp/y.sum()
                expected+=(recall-previous)*tp/chosen.sum();previous=recall
            self.assertAlmostEqual(average_precision_tied(y,s),expected)

    def test_no_positive_is_undefined(self):
        self.assertIsNone(average_precision_tied([0,0],[.2,.3]))


if __name__=='__main__':unittest.main()
