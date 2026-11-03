import math
import unittest
from inspection_information_core import *


class InspectionInformationTests(unittest.TestCase):
    def test_mixture_keeps_identical_marginals(self):
        for rho in (0,.25,.5,.75,1):
            belief=world_distribution(rho)
            self.assertAlmostEqual(sum(belief),1)
            for i in range(3):
                self.assertAlmostEqual(sum(p*x[i] for p,x in zip(belief,WORLDS)),.5)

    def test_detection_is_paid_local_inspection(self):
        self.assertEqual(closed_cost((0,1)),6)
        self.assertEqual(closed_cost((0,2)),8)
        p=make_policies(0,1,7,1)["static"]
        self.assertAlmostEqual(exhaustive_evaluate(p,0,1,7)["reward"],1)

    def test_same_marginals_distinct_information_value(self):
        for rho in (0,.25,.5,.75,1):
            for q in (.5,.65,.8,1):
                pp=make_policies(rho,q,10,1)
                self.assertAlmostEqual(pp["static"].predicted_value,q)
                self.assertAlmostEqual(pp["forced_one_update"].predicted_value,q*(1+.5*rho*(2*q-1)))

    def test_no_information_and_same_fee_control(self):
        for rho,q in [(0,.8),(1,.5)]:
            for budget in range(6,15):
                for fee in range(4):
                    pp=make_policies(rho,q,budget,fee)
                    self.assertAlmostEqual(pp["static_with_same_extra_fee"].predicted_value,pp["forced_one_update"].predicted_value)
                    self.assertAlmostEqual(pp["static"].predicted_value,pp["best_static_or_one_update"].predicted_value)

    def test_replanning_can_displace_useful_inspections(self):
        pp=make_policies(0,1,12,1)
        self.assertAlmostEqual(pp["static"].predicted_value,1.5)
        self.assertAlmostEqual(pp["forced_one_update"].predicted_value,1)
        self.assertAlmostEqual(pp["best_static_or_one_update"].predicted_value,1.5)

    def test_closed_return_for_all_exact_branches(self):
        for rho in (0,.5,1):
            for q in (.5,.8,1):
                for budget in (6,10,12,14):
                    for fee in (0,1,3):
                        for p in make_policies(rho,q,budget,fee).values():
                            value=exhaustive_evaluate(p,rho,q,budget)
                            # Individual branches are checked with integer costs;
                            # the reported mean accumulates floating probabilities.
                            self.assertLessEqual(value["cost"],budget+1e-10)


if __name__ == "__main__":
    unittest.main()
