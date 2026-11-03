import unittest

import numpy as np

from crasar_pearson_budget_core import POLICIES, budget_episode, select_static
from route_headroom_core import exact_catalogue, brute_force


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[10.,0.],[40.,0.],[80.,0.],[0.,0.]])
        self.p0 = np.array([.9,.8,.7])
        self.sim = np.ones((3,3))
        self.cat = exact_catalogue(self.xy,3)

    def test_all_policies_pay_return_and_only_reveal_actual_visits(self):
        for policy in POLICIES:
            accessed=[]
            def reveal(i):
                self.assertNotIn(i,accessed)
                accessed.append(i)
                return int(i==1)
            row=budget_episode(self.p0,self.xy,self.sim,policy,30.,reveal,self.cat)
            self.assertEqual(accessed,row['route'])
            self.assertLessEqual(row['elapsed_proxy_s'],30.+1e-9)
            self.assertAlmostEqual(row['elapsed_proxy_s'],row['distance_m']/5+10*len(accessed)+row['planning_calls'])
            self.assertTrue(row['returned_home'])

    def test_infeasible_visit_reveals_no_label(self):
        for policy in POLICIES:
            row=budget_episode(self.p0,self.xy,self.sim,policy,12.,
                               lambda i:self.fail('Cannot pay observation and return'),self.cat)
            self.assertEqual(row['route'],[])

    def test_unknown_paid_labels_are_not_negative_feedback(self):
        row=budget_episode(self.p0,self.xy,self.sim,'pearson',100.,lambda i:None)
        self.assertEqual(len(row['route']),3)
        self.assertEqual(row['updates'],[])
        self.assertTrue(all(r['source_label'] is None for r in row['observations']))

    def test_exact_static_matches_independent_permutation_search(self):
        for budget in (20.,30.,60.):
            _,certificate=select_static(self.cat,self.p0,budget)
            expected=brute_force(self.xy,3,self.p0,budget,move=.2,observe=10.,plan=1.)
            self.assertAlmostEqual(certificate['public_objective'],expected)


if __name__=='__main__':
    unittest.main()
