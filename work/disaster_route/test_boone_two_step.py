import unittest

import numpy as np

from run_boone_two_step import run_episode


class TwoStepTests(unittest.TestCase):
    def setUp(self):
        self.p0 = np.array([.4, .9, .8, .7])
        self.xy = np.array([[0., 0.], [10., 0.], [20., 0.], [30., 0.]])

    def test_labels_are_revealed_only_on_actual_path_and_visits_are_unique(self):
        accessed = []
        def reveal(i):
            self.assertNotIn(i, accessed)
            accessed.append(i)
            return int(i==1)
        row = run_episode(self.p0, self.xy, np.ones((4,4)), 0, 'pearson', 'score', reveal)
        self.assertEqual(accessed, row['route_indices'])
        self.assertEqual(len(accessed), 3)
        self.assertEqual(row['actions'][1]['observed_history_indices'], sorted(accessed[:2]))
        self.assertAlmostEqual(row['updates'][1]['residual'], 1-row['actions'][0]['score'])
        self.assertAlmostEqual(row['closed_distance_m'], 2*self.xy[max(accessed),0])

    def test_unit_feature_kernel_matches_sequential_spatial_policy(self):
        for method in ('spatial', 'matched_spatial'):
            first = run_episode(self.p0, self.xy, np.ones((4,4)), 0, 'pearson', 'score', lambda i:int(i==1))
            second = run_episode(self.p0, self.xy, np.ones((4,4)), 0, method, 'score', lambda i:int(i==1))
            self.assertEqual(first['route_indices'], second['route_indices'])
            for a,b in zip(first['updates'],second['updates']):
                np.testing.assert_allclose(a['scores_after'],b['scores_after'],atol=1e-14)

    def test_second_choice_is_unaffected_by_final_unobserved_labels(self):
        results = []
        for final_label in (0,1):
            labels = [0,1,final_label,final_label]
            results.append(run_episode(self.p0,self.xy,np.ones((4,4)),0,'pearson','score',lambda i:labels[i]))
        self.assertEqual(results[0]['route_indices'],results[1]['route_indices'])
        self.assertEqual(results[0]['actions'],results[1]['actions'])


if __name__ == '__main__':
    unittest.main()
