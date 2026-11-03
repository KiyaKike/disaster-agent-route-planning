import itertools
import math
import unittest

from run_cached_search import Action, CacheEnvironment, Evaluator, PublicCell, Planner
from run_coverage_controls import (CONTROL_POLICIES, CoveragePlanner, exact_random_discoveries,
                                   random_order, run_control, spatial_order)
from test_cached_search import RATES, observation, truth_row


class CoverageControlChecks(unittest.TestCase):
    def cells(self):
        return [PublicCell(f"r{y:02d}c{x:02d}", x, y, 0, .2) for y in range(4) for x in range(4)]

    def test_spatial_order_covers_grid_and_spreads_early_cells(self):
        order = spatial_order(self.cells())
        self.assertEqual(order[:2], ["r00c00", "r03c03"])
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(set(order), {c.cell_id for c in self.cells()})
        self.assertEqual(order, spatial_order(list(reversed(self.cells()))))

    def test_random_order_reproducible_and_not_quality_dependent(self):
        a = random_order(self.cells(), "s", 42)
        self.assertEqual(a, random_order(list(reversed(self.cells())), "s", 42))
        self.assertNotEqual(a, random_order(self.cells(), "s", 43))
        self.assertEqual(len(a), len(set(a)))

    def test_exact_random_expectation_deduplicates_cross_grid_buildings(self):
        # Building A visible in 2 of 4 cells, B in one. Enumerate budget-2 subsets.
        cell_buildings = [{"a"}, {"a"}, {"b"}, set()]
        empirical = sum(len(set().union(*(cell_buildings[i] for i in subset)))
                        for subset in itertools.combinations(range(4), 2)) / math.comb(4, 2)
        self.assertAlmostEqual(exact_random_discoveries(4, 2, [2, 1]), empirical)
        self.assertEqual(exact_random_discoveries(4, 0, [2, 1]), 0)
        self.assertEqual(exact_random_discoveries(4, 4, [2, 1]), 2)
        with self.assertRaises(ValueError):
            exact_random_discoveries(4, 5, [1])

    def test_uniform_gate_has_identical_actions_to_original_gate_with_flat_prior(self):
        state = CoveragePlanner(self.cells(), "uniform_gate", RATES, "mixed_survey", "scene")
        reference = Planner(self.cells(), "quality_gate", RATES, "mixed_survey")
        for _ in range(4):
            a, b = state.choose(6), reference.choose(6)
            self.assertEqual(a, b)
            obs = observation(a.cell_id, "blurred", False)
            state.observe(a, obs)
            reference.observe(b, obs)

    def test_gate_and_survey_controls_share_order_but_upgrade_costs_budget(self):
        cells = self.cells()
        for prefix in ("uniform", "spatial", "random"):
            gate = CoveragePlanner(cells, prefix + "_gate", RATES, "blurred_survey", "scene", 42)
            survey = CoveragePlanner(cells, prefix + "_survey", RATES, "blurred_survey", "scene", 42)
            first = gate.choose(5)
            self.assertEqual(first, survey.choose(5))
            gate.observe(first, observation(first.cell_id))
            survey.observe(first, observation(first.cell_id))
            self.assertEqual(gate.choose(4), Action(first.cell_id, "upgrade"))
            self.assertEqual(survey.choose(4).kind, "survey")

    def test_all_policies_respect_cost_and_keep_unknown_or_no_detection_censored(self):
        cells = self.cells()
        observations = [observation(c.cell_id, mode, False) for c in cells for mode in ("native", "blurred")]
        for policy in CONTROL_POLICIES:
            traces = []
            for target in (True, False):
                env = CacheEnvironment("scene", observations, "mixed_survey", 42)
                rows = [truth_row(c.cell_id, mode, target) for c in cells for mode in ("native", "blurred")]
                result, trace = run_control(cells, policy, RATES, "mixed_survey", 5, "scene", 42, env.observe, Evaluator(rows))
                self.assertLessEqual(result["cost"], 5)
                self.assertEqual(result["cost"], result["surveys"] + 2 * result["upgrades"])
                self.assertEqual(len(env.requests), len(set(env.requests)))
                self.assertTrue(result["no_true_discovery"])
                self.assertIsNone(result["first_discovery_cost"])
                traces.append(trace)
            self.assertEqual(*traces)


if __name__ == "__main__":
    unittest.main()
