from copy import deepcopy
import itertools
import random
import unittest

from cross_cell_count_smoke import CELLS
from run_budget_planning import action_values, planning_inputs
from run_utility_tie_controls import expanded_values, tie_ranks, run_episode as frozen_episode
from serial_multifidelity_baseline import (TwoStageModel, compress_inputs,
    compressed_state, decrement, run_episode, choose)
from run_development_budget_controls import PaidBinaryEnvironment
import test_budget_planning


class SerialMultifidelityTests(unittest.TestCase):
    def model(self, cost=2, native=0.):
        return TwoStageModel([((.1, 0., 20.), .5), ((2., None, None), 0.)], native, cost)

    def test_costed_tail_matches_exhaustive_subsets(self):
        counts = (2, 1, 1, 2, 0, 0)
        for cost in [1, 2, 4]:
            model = self.model(cost)
            items = []
            for g, (rewards, _) in enumerate(model.classes):
                for status, reward in enumerate(rewards):
                    if reward is not None:
                        items.extend((1 if status == 0 else cost, reward) for _ in range(counts[3*g+status]))
            for budget in range(11):
                expected = max(sum(r for (c, r), yes in zip(items, mask) if yes)
                    for mask in itertools.product([False, True], repeat=len(items))
                    if sum(c for (c, r), yes in zip(items, mask) if yes) <= budget)
                value, selected = model.tail(counts, budget)
                self.assertAlmostEqual(value, expected)
                self.assertTrue(all(n <= available for n, available in zip(selected, counts)))

    def test_exact_two_exploration_outcomes_and_all_costs_charged(self):
        model = self.model()
        counts = (2, 0, 0, 5, 0, 0)
        value, batch = model.future(counts, 5)
        self.assertEqual(batch, (0, 0))
        # Two surveys cost2, leave3. At least one B+ (p=.75) permits one
        # reward20 upgrade at cost2 and one reward2 survey. Otherwise three
        # reward2 surveys. No free upgrade or simultaneous-budget shortcut.
        self.assertAlmostEqual(value, .2 + .75*22 + .25*6)
        self.assertAlmostEqual(model.after_surveys(counts, batch, 3), .75*22 + .25*6)

    def test_fantasy_batch_membership_is_not_adaptive(self):
        model = self.model()
        counts = (2, 0, 0, 5, 0, 0)
        self.assertEqual(model.exploration_order(counts, 5), (0, 0))
        explicit = 0.
        for statuses in itertools.product([1, 2], repeat=2):
            state = list(counts)
            for status in statuses:
                state[0] -= 1
                state[status] += 1
            explicit += .25 * model.tail(tuple(state), 3)[0]
        self.assertAlmostEqual(explicit, model.after_surveys(counts, (0, 0), 3))

    def test_exploration_excludes_current_final_batch_and_reserves_upgrade_cost(self):
        model = self.model()
        counts = (2, 0, 0, 5, 0, 0)
        _, selected = model.tail(counts, 5)
        batch = model.exploration_order(counts, 5)
        for group in set(batch):
            self.assertLessEqual(batch.count(group), counts[3*group]-selected[3*group])
        self.assertLessEqual(len(batch)+model.cost, 5)
        self.assertEqual(model.exploration_order(counts, 2), ())

    def test_empty_exploration_prefix_prevents_forced_surrogate_loss(self):
        model = self.model()
        for budget in range(9):
            for counts in [(1, 0, 0, 5, 0, 0), (0, 1, 1, 2, 0, 0), (3, 0, 0, 0, 0, 0)]:
                self.assertGreaterEqual(model.future(counts, budget)[0]+1e-12, model.tail(counts, budget)[0])

    def test_native_only_has_no_exploratory_upgrade_value(self):
        model = self.model(native=1.)
        counts = (2, 0, 0, 5, 0, 0)
        self.assertEqual(model.exploration_order(counts, 5), ())
        self.assertEqual(model.future(counts, 5)[0], model.tail(counts, 5)[0])

    def test_probabilities_and_states_do_not_gain_free_observations(self):
        model = self.model(native=.5)
        counts = (2, 0, 0, 5, 0, 0)
        branches = model.survey_branches(counts, 0)
        self.assertAlmostEqual(sum(p for p, _ in branches), 1.)
        self.assertEqual([p for p, _ in branches], [.5, .25, .25])
        self.assertEqual(counts, (2, 0, 0, 5, 0, 0))
        with self.assertRaises(ValueError):
            decrement(counts, 0, 1)

    def test_compression_one_stage_matches_frozen_full_cell_values(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = {c: {"change": i/64, "coarse_nonzero_fraction": 1.} for i, c in enumerate(CELLS)}
        rng = random.Random(1031)
        for scenario in ["native_survey", "mixed_survey", "blurred_survey"]:
            for cost in [1, 2, 4]:
                inputs = planning_inputs(features, models, scenario, cost)
                model, membership = compress_inputs(inputs, cost)
                state = [rng.randrange(4) for _ in CELLS]
                counts = compressed_state(state, membership, len(model.classes))
                for i, q, _ in expanded_values(state, inputs, 8, cost, "batch_tail"):
                    one, gain, _ = model.q(counts, 8, membership[i], state[i])
                    self.assertAlmostEqual(one, q, places=10)
                    self.assertGreaterEqual(gain, 0.)

    def test_one_stage_actions_and_utilities_reproduce_frozen_runner(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = {c: {"change": i/64, "coarse_nonzero_fraction": 1.} for i, c in enumerate(CELLS)}
        rows = [{"sample_id": "s", "cell_id": c, "mode": mode, "declared_target": i%5 == 0}
            for i,c in enumerate(CELLS) for mode in ["native", "blurred"]]
        for rule in ["row_major", "raw_change", "fixed_random"]:
            inputs = planning_inputs(features, models, "mixed_survey", 2)
            model, membership = compress_inputs(inputs, 2)
            ranks = tie_ranks(features, rule, 20261030)
            expected = frozen_episode(PaidBinaryEnvironment(rows, "s", "mixed_survey", 20261025, 2), inputs,
                "batch_tail", 8, 2, ranks)
            actual, stats = run_episode(PaidBinaryEnvironment(rows, "s", "mixed_survey", 20261025, 2), inputs,
                8, 2, ranks, "serial_one_stage", model, membership)
            self.assertEqual(actual, [{k:v for k,v in r.items() if k!='greedy_utility'} for r in expected])
            self.assertEqual(stats['sum_selected_surrogate_gain'], 0.)

    def test_private_feature_injection_cannot_affect_model(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = {c: {"change": i/64, "coarse_nonzero_fraction": 1.} for i,c in enumerate(CELLS)}
        poisoned = deepcopy(features)
        for row in poisoned.values():
            row.update(true_damage=True, future_quality="native", evaluator_matches=["secret"])
        a, membership_a = compress_inputs(planning_inputs(features, models, "mixed_survey", 2), 2)
        b, membership_b = compress_inputs(planning_inputs(poisoned, models, "mixed_survey", 2), 2)
        self.assertEqual(a.classes, b.classes)
        self.assertEqual(membership_a, membership_b)

    def test_lookahead_never_calls_environment_or_spends_fantasy_actions(self):
        class Env:
            def __init__(self): self.calls=[]
            def observe(self, cell, kind):
                self.calls.append((cell,kind))
                return {"mode": "native", "declared_target": False, "cost":1}
        inputs = {"reward": [[.1,0.,20.],[2.,None,None],[2.,None,None],[2.,None,None],[2.,None,None],[2.,None,None]],
            "greedy_utility": [[.1,0.,10.]]+[[2.,None,None]]*5,
            "blur_positive": [.5]+[0.]*5,"native_survey_probability":0.}
        model, membership = compress_inputs(inputs,2)
        env=Env()
        trace,_=run_episode(env,inputs,5,2,list(range(6)),"serial_two_stage",model,membership)
        self.assertEqual(len(env.calls),len(trace))
        self.assertEqual(sum(r['cost'] for r in trace),5)


if __name__ == "__main__":
    unittest.main()
