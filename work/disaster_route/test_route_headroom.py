import unittest

import numpy as np

from route_headroom_core import (exact_catalogue,reconstruct,subset_rewards,solve_catalogue,
                                 audit_bellman,brute_force)


class HeadroomTests(unittest.TestCase):
    def setUp(self):
        self.xy=np.array([[0,1],[1,1],[2,1],[0,2],[2,2],[1,0]],float)
        self.cat=exact_catalogue(self.xy,5)

    def test_matches_independent_permutation_enumeration(self):
        for weights in ([1,2,7,3,5],[0,0,0,0,0],[.2,.8,.3,.9,.1]):
            for b in (2,80,140,220):
                s=solve_catalogue(self.cat,weights,b)
                self.assertAlmostEqual(s['objective'],brute_force(self.xy,5,weights,b))

    def test_every_subset_reconstruction_matches_shortest_tour_length(self):
        for mask in range(32):
            route=reconstruct(self.cat,mask);nodes=[5,*route,5]
            length=sum(np.linalg.norm(self.xy[a]-self.xy[b]) for a,b in zip(nodes,nodes[1:]))
            self.assertAlmostEqual(length,self.cat['closed'][mask])
            self.assertEqual(set(route),{i for i in range(5) if mask&(1<<i)})

    def test_bellman_certificate_checks_all_valid_states(self):
        self.assertEqual(audit_bellman(self.cat),5*16)

    def test_return_and_each_paid_observation_are_required(self):
        c=exact_catalogue(np.array([[3,4],[0,0]]),1)
        self.assertEqual(solve_catalogue(c,[10],221)['route'],[])
        self.assertEqual(solve_catalogue(c,[10],222)['route'],[0])

    def test_more_budget_cannot_decrease_optimal_objective(self):
        scores=[solve_catalogue(self.cat,[1,2,3,4,5],b)['objective'] for b in (2,80,140,220)]
        self.assertEqual(scores,sorted(scores))

    def test_zero_targets_choose_empty_route_and_not_full_credit(self):
        s=solve_catalogue(self.cat,[0]*5,300)
        self.assertEqual(s['route'],[]);self.assertEqual(s['objective'],0)

    def test_public_objective_is_independent_of_reference_values(self):
        weights=[1,2,3,4,5];first=solve_catalogue(self.cat,weights,140)
        _=solve_catalogue(self.cat,[100,0,0,0,0],140)
        self.assertEqual(first,solve_catalogue(self.cat,weights,140))

    def test_subset_rewards_are_additive_without_repeated_tiles(self):
        weights=[3,5,7,11,13];values=subset_rewards(weights)
        for mask in range(32):self.assertEqual(values[mask],sum(w for i,w in enumerate(weights) if mask&(1<<i)))

    def test_invalid_reward_or_unaffordable_plan_is_rejected(self):
        with self.assertRaises(ValueError):solve_catalogue(self.cat,[1,2,-1,4,5],100)
        with self.assertRaises(ValueError):solve_catalogue(self.cat,[1,2,3,4,5],1)


if __name__=='__main__':unittest.main()
