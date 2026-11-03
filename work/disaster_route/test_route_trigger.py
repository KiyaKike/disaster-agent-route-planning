import copy
import itertools
import unittest

import numpy as np

from route_core import Config, Observer
from route_factor_core import Planner, execute as factor_execute
from route_snapshot_core import PrivateWorld, Session
from route_trigger_core import (POLICIES, accept_proposal, can_propose, execute,
                                feasible_old_suffix, gain_moments, paid_attempt, route_cost, score)
from test_route_factor import grid


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.g = grid(8)
        rng = np.random.default_rng(821)
        self.world = PrivateWorld('trigger-test', rng.random(64)<.2, rng.random(64)<.3, (.8, .1))

    def session(self, budget=1200):
        s = Session(self.g, self.world, budget)
        s.plan(Planner())
        for _ in range(5):
            s.step()
        return s

    def test_shared_cells_cancel_even_if_route_order_changes(self):
        p = Session(self.g, self.world).public()
        m = gain_moments(self.g, p, [0, 1], [1, 0], (.8, .1))
        self.assertEqual((m['mean_gain'], m['variance_gain'], m['shared_cells']), (0., 0., 2))

    def test_gain_moments_match_exact_latent_and_detection_enumeration(self):
        p = Session(self.g, self.world).public()
        # Three distinct cells; two may share a block. Enumerate all latent states.
        old, new = [0], [1, 8]
        indices = [0, 1, 8]
        blocks = sorted(set(int(self.g.blocks[i]) for i in indices))
        mean = square = 0.
        for states in itertools.product((0, 1), repeat=len(blocks)):
            prob = np.prod([p['belief_hot'][b] if z else 1-p['belief_hot'][b] for b, z in zip(blocks, states)])
            latent = dict(zip(blocks, states))
            rates = [(.6 if latent[int(self.g.blocks[i])] else .02)*.8 for i in indices]
            for y in itertools.product((0, 1), repeat=3):
                joint = prob*np.prod([r if bit else 1-r for bit, r in zip(y, rates)])
                diff = -y[0]+y[1]+y[2]
                mean += joint*diff
                square += joint*diff*diff
        m = gain_moments(self.g, p, old, new, (.8, .1))
        self.assertAlmostEqual(m['mean_gain'], mean)
        self.assertAlmostEqual(m['variance_gain'], square-mean*mean)

    def test_same_block_opposite_signs_cancel_latent_variance(self):
        p = Session(self.g, self.world).public()
        a, b = next((a, b) for a in range(64) for b in range(a+1, 64) if self.g.blocks[a] == self.g.blocks[b])
        q = p['belief_hot'][self.g.blocks[a]]
        expected = 2*((1-q)*.016*(1-.016)+q*.48*(1-.48))
        m = gain_moments(self.g, p, [a], [b], (.8, .1))
        self.assertAlmostEqual(m['variance_gain'], expected)

    def test_visited_or_duplicate_cells_cannot_be_future_reward(self):
        p = self.session().public()
        with self.assertRaises(ValueError):
            gain_moments(self.g, p, [], [p['visited_indices'][0]], (.8, .1))
        with self.assertRaises(ValueError):
            gain_moments(self.g, p, [63, 63], [], (.8, .1))

    def test_private_fields_rejected_by_forecast_interface(self):
        p = Session(self.g, self.world).public()
        p['targets'] = self.world.targets.tolist()
        with self.assertRaises(ValueError):
            gain_moments(self.g, p, [], [0], (.8, .1))

    def test_fixed_rules_use_strict_thresholds(self):
        self.assertFalse(accept_proposal('gain_5', {'mean_gain': 0, 'sd_gain': 0}))
        self.assertFalse(accept_proposal('one_sd_5', {'mean_gain': 1, 'sd_gain': 1}))
        self.assertTrue(accept_proposal('gain_5', {'mean_gain': .5, 'sd_gain': 1}))
        self.assertFalse(accept_proposal('one_sd_5', {'mean_gain': .5, 'sd_gain': 1}))

    def test_rejected_proposal_pays_and_does_not_observe(self):
        s = self.session()
        before = s.public()
        # A proposal with exactly the same future nodes has zero count gain.
        class SameRoute:
            def solve(inner, *args):
                return list(before['remaining_route']), {'wall_s': 0.}
        # Give enough slack so this controlled unchanged candidate remains legal.
        s.state['budget_s'] += 2
        decision = paid_attempt(s, SameRoute(), 'one_sd_5')
        self.assertFalse(decision['accepted'])
        self.assertEqual(s.state['elapsed_s'], before['elapsed_s']+1)
        self.assertEqual(s.state['history'], before['history'])
        self.assertEqual(len(s.state['planning_events']), len(before['planning_events'])+1)

    def test_rejection_tail_trim_preserves_return_and_does_not_refund(self):
        s = self.session()
        before = s.public()
        old = before['remaining_route']
        s.state['budget_s'] = before['elapsed_s']+route_cost(self.g, before, old)+.25
        class EmptyRoute:
            def solve(inner, *args):
                return [], {'wall_s': 0.}
        decision = paid_attempt(s, EmptyRoute(), 'gain_5')
        self.assertFalse(decision['accepted'])
        self.assertGreater(decision['dropped_tail_if_rejected'], 0)
        self.assertEqual(s.state['elapsed_s'], before['elapsed_s']+1)
        self.assertLessEqual(s.state['elapsed_s']+route_cost(self.g, s.public(), s.state['remaining_route']), s.state['budget_s']+1e-8)

    def test_no_affordable_proposal_leaves_state_unchanged(self):
        s = self.session()
        s.set_route([])
        s.state['budget_s'] = s.state['elapsed_s']+route_cost(self.g, s.public(), [])+.2
        before = s.public()
        self.assertFalse(can_propose(self.g, before))
        with self.assertRaises(ValueError):
            paid_attempt(s, Planner(), 'gain_5')
        self.assertEqual(s.public(), before)

    def test_all_policies_charge_every_call_and_close_within_budget(self):
        for policy in POLICIES:
            r = execute(self.g, self.world, policy, Planner(), 600)
            s = r['final_public']
            self.assertTrue(s['closed'])
            self.assertEqual(len(s['visited_indices']), len(set(s['visited_indices'])))
            self.assertAlmostEqual(s['elapsed_s'], s['distance_m']/5+20*len(s['history'])+len(s['planning_events']))
            self.assertLessEqual(s['elapsed_s'], 600+1e-8)
            self.assertEqual(len(s['planning_events']), 1+len(r['decisions']))
            if policy not in ('always', 'pre_static', 'post_static'):
                self.assertTrue(all(d['after_observations'] % 5 == 0 for d in r['decisions']))

    def test_post_static_exactly_matches_frozen_solver_baseline(self):
        a = execute(self.g, self.world, 'post_static', Planner())
        b = factor_execute(self.g, 'post_static', Config(max_visits=64), Observer(self.world.observations, self.world.likelihood), 'insertion_2opt', Planner())
        self.assertEqual(a['final_public']['visited_indices'], [h['cell_index'] for h in b['steps']])
        self.assertEqual(a['final_public']['elapsed_s'], b['total_time_s'])
        self.assertEqual(a['final_public']['distance_m'], b['total_distance_m'])

    def test_hidden_truth_cannot_change_any_policy_actions(self):
        altered = PrivateWorld('different-targets', ~self.world.targets, self.world.observations, self.world.likelihood)
        for policy in POLICIES:
            a = execute(self.g, self.world, policy, Planner())
            b = execute(self.g, altered, policy, Planner())
            self.assertEqual(a['final_public'], b['final_public'])
            self.assertEqual(a['decisions'], b['decisions'])

    def test_empty_targets_preserve_failure_null_and_zero_denominator(self):
        w = PrivateWorld('empty', np.zeros(64, bool), np.zeros(64, bool), (.8, .1))
        r = execute(self.g, w, 'one_sd_5', Planner())
        m = score(r, w)
        self.assertIsNone(m['first_correct_detection_s'])
        self.assertIsNone(m['detected_target_recall'])
        self.assertEqual(m['first_detection_or_budget_s'], 1200)
        self.assertTrue(m['no_correct_detection'])

    def test_accepted_and_rejected_session_states_restore_without_recharging(self):
        for policy in ('periodic_5', 'one_sd_5'):
            s = self.session()
            paid_attempt(s, Planner(), policy)
            snap = copy.deepcopy(s.snapshot())
            restored = Session.restore(self.g, snap, {self.world.token: self.world})
            self.assertEqual(restored.public(), s.public())
            self.assertEqual(restored.environment.seen, s.environment.seen)


if __name__ == '__main__':
    unittest.main()
