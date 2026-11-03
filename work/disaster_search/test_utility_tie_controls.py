from copy import deepcopy
import itertools
import random
import unittest

from cross_cell_count_smoke import CELLS
from run_budget_planning import POLICIES, run_episode as old_episode
from run_development_budget_controls import PaidBinaryEnvironment
from run_utility_tie_controls import (TIES, aggregate_ties, choose, expanded_values,
    run_episode, tie_ranks)
from test_budget_planning import inputs
import test_budget_planning


class TieControlTests(unittest.TestCase):
    def features(self):
        return {c: {"change": i / 64, "coarse_nonzero_fraction": 1.} for i, c in enumerate(CELLS)}

    def test_ranks_are_fixed_permutations_and_ignore_private_fields(self):
        features = self.features()
        poisoned = deepcopy(features)
        for v in poisoned.values():
            v.update(true_damage=True, future_mode="native", matched_target_uids=["hidden"])
        for rule in TIES:
            rank = tie_ranks(features, rule, 20261030)
            self.assertEqual(sorted(rank), list(range(64)))
            self.assertEqual(rank, tie_ranks(poisoned, rule, 20261030))
        self.assertEqual(tie_ranks(features, "row_major", 0), list(range(64)))
        self.assertEqual(tie_ranks(features, "raw_change", 0), list(reversed(range(64))))
        self.assertNotEqual(tie_ranks(features, "fixed_random", 20261030), tie_ranks(features, "fixed_random", 20261031))

    def test_exact_raw_change_ties_fall_back_to_row_major(self):
        features = {c: {"change": 1.} for c in CELLS}
        self.assertEqual(tie_ranks(features, "raw_change", 0), list(range(64)))

    def test_group_expansion_restores_all_cells_and_can_select_last(self):
        model = inputs([[1., 0., 4.]] * 64, [.2] * 64)
        state = [0] * 64
        rank = tie_ranks(self.features(), "raw_change", 0)
        for policy in POLICIES:
            self.assertEqual(len(expanded_values(state, model, 8, 2, policy)), 64)
            self.assertEqual(choose(state, model, 8, 2, policy, rank)[0], 63)
            self.assertEqual(choose(state, model, 8, 2, policy, list(range(64)))[0], 0)

    def test_better_value_dominates_tie_rank_and_exact_greedy_is_retained(self):
        model = inputs([[1., 0., 0.], [1. + 1e-14, 0., 0.]])
        self.assertEqual(choose([0, 0], model, 1, 2, "proxy_greedy", [0, 1])[0], 1)
        for policy in ["two_step", "batch_tail"]:
            self.assertEqual(choose([0, 0], model, 1, 2, policy, [0, 1])[0], 1)
        model = inputs([[1., 0., 0.], [2., 0., 0.]])
        self.assertEqual(choose([0, 0], model, 1, 2, "batch_tail", [0, 1])[0], 1)

    def test_expanded_values_match_exhaustive_ungrouped_tail(self):
        rng = random.Random(67)
        for cost in [1, 2, 4]:
            rewards = [[1., 2., 5.], [1., 2., 5.], [3., 4., None], [2., 0., 7.]]
            model = inputs(rewards, [.3, .3, .8, .6], native=.5, cost=cost)
            for _ in range(12):
                state = [rng.randrange(4) for i in rewards]
                for budget in [1, 3, 6]:
                    for policy in ["two_step", "batch_tail"]:
                        actual = expanded_values(state, model, budget, cost, policy)
                        for i, q, _ in actual:
                            items = [(1 if s == 0 else cost, rewards[j][s]) for j, s in enumerate(state)
                                     if j != i and s != 3 and rewards[j][s] is not None]
                            def tail(b):
                                if b <= 0:
                                    return 0.
                                masks = itertools.product([False, True], repeat=len(items))
                                return max(sum(r for keep, (_, r) in zip(mask, items) if keep)
                                    for mask in masks if sum(c for keep, (c, _) in zip(mask, items) if keep) <= b
                                    and (policy != "two_step" or sum(mask) <= 1))
                            status = state[i]
                            b = budget - (1 if status == 0 else cost)
                            base = tail(b)
                            if status == 0:
                                branches = [max(base, rewards[i][s] + tail(b - cost))
                                    if b >= cost and rewards[i][s] is not None else base for s in [1, 2]]
                                future = .5 * base + .5 * ((1 - model["blur_positive"][i]) * branches[0] + model["blur_positive"][i] * branches[1])
                            else:
                                future = base
                            self.assertAlmostEqual(q, rewards[i][status] + future)

    def test_row_major_reproduces_old_planning_episodes(self):
        from run_budget_planning import planning_inputs
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = self.features()
        rows = [{"sample_id": "s", "cell_id": c, "mode": mode, "declared_target": i % 4 == 0}
                for i, c in enumerate(CELLS) for mode in ["native", "blurred"]]
        for scenario in ["native_survey", "mixed_survey", "blurred_survey"]:
            for cost in [1, 2, 4]:
                model = planning_inputs(features, models, scenario, cost)
                for policy in POLICIES:
                    expected = old_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, cost), model, policy, 32, cost)
                    actual = run_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, cost), model, policy, 32, cost, list(range(64)))
                    self.assertEqual(actual, expected)

    def test_native_all_policies_same_order_and_outcomes_do_not_change_ranks(self):
        rows = [{"sample_id": "s", "cell_id": c, "mode": "native", "declared_target": i == 63} for i, c in enumerate(CELLS)]
        model = inputs([[1., 0., 0.]] * 64, native=1.)
        for rule in TIES:
            rank = tie_ranks(self.features(), rule, 20261030)
            traces = [run_episode(PaidBinaryEnvironment(rows, "s", "native_survey", 0, 2), model, p, 8, 2, rank) for p in POLICIES]
            self.assertEqual([[a["cell_id"] for a in t] for t in traces], [[a["cell_id"] for a in traces[0]]] * 3)
            self.assertEqual([a["cell_id"] for a in traces[0]], [CELLS[i] for i in sorted(range(64), key=lambda i: rank[i])[:8]])
        raw = run_episode(PaidBinaryEnvironment(rows, "s", "native_survey", 0, 2), model, "proxy_greedy", 8, 2, tie_ranks(self.features(), "raw_change", 0))
        self.assertTrue(raw[0]["declared_target"])

    def test_no_repeat_no_native_shortcut_unsupported_upgrade_and_budget(self):
        rows = [{"sample_id": "s", "cell_id": c, "mode": mode, "declared_target": False}
                for c in CELLS for mode in ["native", "blurred"]]
        model = inputs([[1., None, None]] * 64)
        for rule in TIES:
            trace = run_episode(PaidBinaryEnvironment(rows, "s", "blurred_survey", 0, 4), model, "batch_tail", 32, 4, tie_ranks(self.features(), rule, 20261030))
            self.assertEqual(len(trace), 32)
            self.assertEqual(len({a["cell_id"] for a in trace}), 32)
            self.assertTrue(all(a["kind"] == "survey" and a["mode"] == "blurred" for a in trace))

    def test_scene_aggregation_does_not_pool_tie_rules_or_select_random_seed(self):
        from run_development_budget_controls import SUMMARY_FIELDS
        rows = []
        for rule, values in [("row_major", [1.]), ("raw_change", [3.]), ("fixed_random", [2., 8.])]:
            for value in values:
                rows.append({"tie_rule": rule, "fit_scope": "development_fit", "policy": "proxy_greedy",
                    "scenario": "mixed_survey", "upgrade_cost": 2, "budget": 32, "threshold": .1,
                    "visible_targets": 10, **{f: value for f in SUMMARY_FIELDS}})
        result = {r["tie_rule"]: r for r in aggregate_ties(rows)}
        self.assertEqual(result["fixed_random"]["distinct_found"], 5.)
        self.assertEqual(result["fixed_random"]["repeats"], 2)
        self.assertEqual(result["raw_change"]["distinct_found"], 3.)

    def test_tie_effects_are_event_equal_and_keep_unfavourable_events(self):
        from audit_utility_tie_controls import paired_effects
        def row(sid, event, rule, value):
            return {"sample_id": sid, "event": event, "policy": "proxy_greedy", "tie_rule": rule,
                "fit_scope": "development_fit", "scenario": "mixed_survey", "upgrade_cost": 2,
                "budget": 32, "threshold": .1, "distinct_found": value,
                "false_positive_cells": 0., "upgrades": 0., "positive_upgrades": 0.}
        rows = [row(s, e, rule, value) for s, e, v in [("a", "e1", 4.), ("b", "e1", 4.), ("c", "e2", -2.)]
                for rule, value in [("row_major", 3.), ("raw_change", 3. + v)]]
        effects = paired_effects(rows, [])
        macro = next(r for r in effects["event_macro_effects"] if r["comparison"] == "proxy_greedy/raw_change_vs_proxy_greedy/row_major")
        self.assertEqual(macro["delta_found"], 1.)
        self.assertEqual(macro["positive_events"], 1)

    def test_random_order_summary_averages_quality_and_retains_orders(self):
        from audit_utility_tie_controls import random_order_summary
        values = {("development_fit", "proxy_greedy", seed, event, str(i)): [v - 1, v, v, v, v + 1]
                  for seed, v in [(20261030, 2.), (20261031, 6.)]
                  for event in ["e1", "e2", "e3"] for i in range(24)}
        result = random_order_summary(values)
        row = result["primary_order_sensitivity"][0]
        self.assertEqual((row["minimum"], row["maximum"], row["mean"]), (2., 6., 4.))
        self.assertFalse(row["confidence_interval"])
        values.pop(next(iter(values)))
        with self.assertRaises(ValueError):
            random_order_summary(values)


if __name__ == "__main__":
    unittest.main()
