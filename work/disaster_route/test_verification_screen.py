import unittest
from fractions import Fraction as F
from verification_screen_core import solve,payoff
from run_verification_screen import replay,check_lp

class VerificationScreen(unittest.TestCase):
    def test_equal_first_pass_rates_but_different_posterior_action(self):
        r=solve(3,F(0),F(2),F(1))
        self.assertEqual(r['policies']['fixed_model0']['tails'][0][0],'B')
        self.assertEqual(r['policies']['fixed_model1']['tails'][0][0],'A')
    def test_uninformative_probe_never_adds_reward(self):
        r=solve(4,F(0),F(0),F(1,2))
        self.assertEqual(r['policies']['static_mixture']['values'],r['policies']['exact_bayes_reference']['values'])
    def test_persistent_miss_cannot_be_fixed_by_repeating(self):
        self.assertEqual(payoff(('A',4),1,F(1)),0)
    def test_primary_exact_branch_replay(self):
        r=solve(4,F(0),F(0),F(1));p=r['policies']['exact_bayes_reference']
        self.assertTrue(p['probe']);self.assertEqual(p['values'],(F(1,5),F(21,64)))
        for m in range(2):
            value,cost,_=replay(p,m,F(0),F(0),F(1));self.assertEqual(value,p['values'][m]);self.assertLessEqual(cost,4)
    def test_robust_reference_matches_known_witness_and_independent_lp(self):
        r=solve(3,F(0),F(2),F(1));self.assertEqual(r['robust_no_probe']['worst_regret'],F(41,525))
        check_lp(r['static_strategies'],r['oracle'],r['robust_no_probe'])
    def test_insufficient_budget_never_permits_probe_or_unpaid_return(self):
        r=solve(3,F(0),F(2),F(1))
        self.assertEqual(r['probe_strategies'],[])
        b=next(p for p in r['static_strategies'] if p['tails'][0][0]=='B')
        self.assertEqual(b['cost_cap'],3)

if __name__=='__main__':unittest.main()
