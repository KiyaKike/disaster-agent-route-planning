from copy import deepcopy
import unittest

from cross_cell_count_smoke import CELLS
from run_budget_planning import planning_inputs
from run_development_budget_controls import PaidBinaryEnvironment, utility_table, score_trace, run_episode as old_episode
from run_unified_revisit_controls import POLICIES, choose, import_count_trace, run_episode
from run_utility_tie_controls import TIES, tie_ranks, run_episode as old_tie_episode
import test_budget_planning


class UnifiedRevisitTests(unittest.TestCase):
    def features(self):
        return {c: {"change": i / 64, "coarse_nonzero_fraction": 1.} for i, c in enumerate(CELLS)}

    def observations(self):
        return [{"sample_id": "s", "cell_id": c, "mode": mode, "declared_target": i % 5 == 0}
                for i, c in enumerate(CELLS) for mode in ["native", "blurred"]]

    def test_all_six_row_controls_reproduce_original_actions_and_utilities(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        rows = self.observations()
        for scenario in ["native_survey", "mixed_survey", "blurred_survey"]:
            for cost in [1, 2, 4]:
                for policy in POLICIES:
                    table = utility_table(self.features(), models["count"], models["presence"], scenario, cost, policy)
                    expected = old_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, cost), table, policy, 32, cost, 0)
                    actual = run_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, cost), table, 32, cost, list(range(64)))
                    self.assertEqual(actual, expected)

    def test_count_all_matches_previous_tie_runner_all_rules_and_costs(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = self.features()
        rows = self.observations()
        for rule in TIES:
            rank = tie_ranks(features, rule, 20261030)
            for cost in [1, 2, 4]:
                model = planning_inputs(features, models, "mixed_survey", cost)
                expected = old_tie_episode(PaidBinaryEnvironment(rows, "s", "mixed_survey", 20261025, cost), model, "proxy_greedy", 32, cost, rank)
                table = utility_table(features, models["count"], models["presence"], "mixed_survey", cost, "count_all")
                actual = run_episode(PaidBinaryEnvironment(rows, "s", "mixed_survey", 20261025, cost), table, 32, cost, rank)
                self.assertEqual(actual, [{k: v for k, v in a.items() if k != "greedy_utility"} for a in expected])

    def test_ablation_changes_eligibility_only_within_each_utility_family(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        tables = {p: utility_table(self.features(), models["count"], models["presence"], "mixed_survey", 2, p) for p in POLICIES}
        for i in range(64):
            self.assertEqual(tables["count_survey"][i][0], tables["count_all"][i][0])
            self.assertEqual(tables["count_negative"][i][:2], tables["count_all"][i][:2])
            self.assertEqual(tables["count_survey"][i][1:], [None, None])
            self.assertIsNone(tables["count_negative"][i][2])
            self.assertEqual(tables["prior_survey"][i][0], tables["prior_gate"][i][0])
            self.assertIsNone(tables["prior_gate"][i][2])

    def test_true_value_wins_and_only_exact_ties_use_fixed_rank(self):
        self.assertEqual(choose([0, 0], [[1., None, None], [1. + 1e-14, None, None]], 1, 2, [0, 1])[0], 1)
        self.assertEqual(choose([0, 0], [[1., None, None], [1., None, None]], 1, 2, [1, 0])[0], 1)
        self.assertIsNone(choose([1, 3], [[1., None, None], [1., 4., 4.]], 32, 2, [0, 1]))

    def test_positive_revisit_must_pay_opportunity_cost_in_unique_score(self):
        class Env:
            def observe(self, cell, kind):
                return {"mode": "blurred" if cell == CELLS[0] and kind == "survey" else "native",
                        "declared_target": True, "cost": 2 if kind == "upgrade" else 1}
        all_trace = run_episode(Env(), [[4., 5., 5.], [3., None, None]], 3, 2, [0, 1])
        neg_trace = run_episode(Env(), [[4., 5., None], [3., None, None]], 3, 2, [0, 1])
        self.assertEqual([a["kind"] for a in all_trace], ["survey", "upgrade"])
        self.assertEqual([a["kind"] for a in neg_trace], ["survey", "survey"])
        evaluator = {(CELLS[0], mode): {"matches": [1] * 3, "truth": True} for mode in ["blurred", "native"]}
        evaluator[(CELLS[1], "native")] = {"matches": [2] * 3, "truth": True}
        all_score, neg_score = score_trace(all_trace, evaluator, 2)[1], score_trace(neg_trace, evaluator, 2)[1]
        self.assertEqual(all_score["distinct_found"], 1)
        self.assertEqual(neg_score["distinct_found"], 2)
        self.assertEqual(all_score["new_buildings_from_positive_upgrades"], 0)

    def test_native_terminal_and_insufficient_upgrade_budget(self):
        rows = self.observations()
        table = [[1., 10., 10.]] * 64
        trace = run_episode(PaidBinaryEnvironment(rows, "s", "native_survey", 0, 2), table, 32, 2, list(range(64)))
        self.assertEqual(len({r["cell_id"] for r in trace}), 32)
        self.assertTrue(all(r["kind"] == "survey" for r in trace))
        self.assertEqual(choose([1, 0], table[:2], 1, 2, [0, 1])[0], 1)

    def test_source_import_requires_exact_config_and_greedy_identity(self):
        config = {"sample_id": "s", "fit_scope": "development_fit", "scenario": "mixed_survey",
            "upgrade_cost": 2, "budget": 32, "quality_seed": 20261025, "order_seed": 20261030,
            "policy": "count_all", "tie_rule": "fixed_random"}
        row = {**config, "policy": "proxy_greedy", "trace": [{"utility": 1., "greedy_utility": 1., "cell_id": CELLS[0]}]}
        self.assertEqual(import_count_trace(iter([row]), config), [{"utility": 1., "cell_id": CELLS[0]}])
        bad = deepcopy(row)
        bad["order_seed"] += 1
        with self.assertRaises(ValueError):
            import_count_trace(iter([bad]), config)
        bad = deepcopy(row)
        bad["trace"][0]["utility"] = 2.
        with self.assertRaises(ValueError):
            import_count_trace(iter([bad]), config)

    def test_private_fields_do_not_change_public_utilities_or_ranks(self):
        models = test_budget_planning.PlannerBoundaryTests().model()
        features = self.features()
        poisoned = deepcopy(features)
        for v in poisoned.values():
            v.update(true_damage=True, future_quality="native", building_uid="hidden", evaluator_matches=["hidden"])
        for policy in POLICIES:
            self.assertEqual(utility_table(features, models["count"], models["presence"], "mixed_survey", 2, policy),
                             utility_table(poisoned, models["count"], models["presence"], "mixed_survey", 2, policy))

    def test_paired_net_effect_uses_unique_total_not_direct_upgrade_yield(self):
        from audit_unified_revisit_controls import paired_effects
        def row(policy, found):
            return {"sample_id": "s", "event": "e", "fit_scope": "development_fit", "scenario": "mixed_survey",
                "upgrade_cost": 2, "budget": 32, "threshold": .1, "tie_rule": "fixed_random", "policy": policy,
                "distinct_found": found, "false_positive_cells": 0., "upgrades": 1., "new_buildings_from_positive_upgrades": 2.}
        result = paired_effects([row("count_all", 4.), row("count_negative", 6.)])
        effect = result["event_macro_effects"][0]
        self.assertEqual(effect["comparison"], "count_all_vs_count_negative")
        self.assertEqual(effect["delta_found"], -2.)
        self.assertEqual(effect["candidate_new_from_positive_upgrades"], 2.)

    def test_paired_effects_keep_tie_rules_separate_and_weight_events_equally(self):
        from audit_unified_revisit_controls import paired_effects
        rows = []
        for sid, event, gain in [("a", "e1", 4.), ("b", "e1", 4.), ("c", "e2", -2.)]:
            for rule, multiplier in [("row_major", 1.), ("fixed_random", -1.)]:
                for policy, found in [("count_survey", 8.), ("count_negative", 8. + gain * multiplier)]:
                    rows.append({"sample_id": sid, "event": event, "fit_scope": "event_holdout", "scenario": "mixed_survey",
                        "upgrade_cost": 2, "budget": 32, "threshold": .1, "tie_rule": rule, "policy": policy,
                        "distinct_found": found, "false_positive_cells": 0., "upgrades": 0., "new_buildings_from_positive_upgrades": 0.})
        effects = {r["tie_rule"]: r for r in paired_effects(rows)["event_macro_effects"]}
        self.assertEqual(effects["row_major"]["delta_found"], 1.)
        self.assertEqual(effects["fixed_random"]["delta_found"], -1.)
        self.assertEqual(effects["row_major"]["positive_events"], 1)


if __name__ == "__main__":
    unittest.main()
