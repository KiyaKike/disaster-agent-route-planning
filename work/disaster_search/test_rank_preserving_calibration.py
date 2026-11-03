from copy import deepcopy
import math
import unittest

from audit_rank_calibration_control import independent_projection, paired_effects, without_utility
from cross_cell_count_smoke import CELLS
from rank_preserving_calibration import project_model, raw_order, policy_table, choose, run_episode, POLICIES
from run_development_budget_controls import PaidBinaryEnvironment, utility_table
from run_unified_revisit_controls import run_episode as original_episode
from run_utility_tie_controls import TIES, tie_ranks
from run_rank_calibration_control import assert_fit_ids


class RankCalibrationTests(unittest.TestCase):
    def model(self, values=(1., 4., 3., 0.)):
        return {"shrinkage_pseudo_cells": 8, "public_quantile_edges": [.25, .5, .75],
            "bins": [{"bin": i, "known_cells": [2, 12, 22, 32][i],
                "survey_mean_count": {"native": v, "blurred": v / 2},
                "upgrade": {"0": {"mean_new_within_cell": 1.}, "1": {"mean_new_within_cell": 100.}}}
                for i, v in enumerate(values)]}

    def features(self):
        return {c: {"change": (i // 4) / 16, "coarse_nonzero_fraction": 1.} for i, c in enumerate(CELLS)}

    def test_weighted_projection_has_independent_four_bin_solution(self):
        model = self.model()
        projected, details = project_model(model)
        # Last three pool: (20*4 + 30*3 + 40*0)/(20+30+40).
        self.assertAlmostEqual(projected["bins"][3]["survey_mean_count"]["native"], 170 / 90)
        for mode in ["native", "blurred"]:
            m = details["modes"][mode]
            for a, b in zip(m["projected"], independent_projection(m["original"], details["weights"])):
                self.assertAlmostEqual(a, b)

    def test_projection_preserves_source_and_every_upgrade_field(self):
        model = self.model()
        saved = deepcopy(model)
        projected, _ = project_model(model)
        self.assertEqual(model, saved)
        for a, b in zip(model["bins"], projected["bins"]):
            self.assertEqual(a["upgrade"], b["upgrade"])
        self.assertEqual(model["public_quantile_edges"], projected["public_quantile_edges"])

    def test_monotone_identity_and_nonnegative_mixed_scores(self):
        model = self.model((0., 1., 2., 3.))
        projected, _ = project_model(model)
        self.assertEqual(model, projected)
        for scenario in ["native_survey", "mixed_survey", "blurred_survey"]:
            table = policy_table(self.features(), model, projected, scenario, 2, "rank_negative_isotonic")
            self.assertEqual([r[0] for r in table], sorted(r[0] for r in table))
            self.assertTrue(all(r[2] is None for r in table))

    def test_invalid_source_values_rejected(self):
        for value in [math.nan, math.inf, -1.]:
            with self.assertRaises(ValueError):
                project_model(self.model((0., 1., 2., value)))

    def test_raw_order_wins_over_projected_plateau_and_nonmonotone_prices(self):
        order = [2, 1, 0]
        for table in [[[2., None, None]] * 3, [[100., None, None], [1., None, None], [0., None, None]]]:
            self.assertEqual(choose([0, 0, 0], table, 32, 2, [0, 1, 2], order)[0], 2)
            self.assertEqual(choose([0, 0, 3], table, 32, 2, [0, 1, 2], order)[0], 1)

    def test_only_negative_can_upgrade_and_budget_is_paid(self):
        table = [[1., 10., 999.], [.5, 5., 999.], [.2, None, None]]
        self.assertEqual(choose([2, 1, 0], table, 2, 2, [0, 1, 2], [0, 1, 2])[0], 1)
        self.assertEqual(choose([2, 1, 0], table, 1, 2, [0, 1, 2], [0, 1, 2])[0], 2)
        self.assertIsNone(choose([2, 3, 3], table, 32, 2, [0, 1, 2], [0, 1, 2]))

    def test_cross_action_exact_tie_and_tiny_value_difference(self):
        self.assertEqual(choose([1, 0], [[0., 1., None], [1., None, None]], 2, 2, [0, 1], [1, 0])[0], 0)
        self.assertEqual(choose([1, 0], [[0., 1., None], [1. + 1e-14, None, None]], 2, 2, [0, 1], [1, 0])[0], 1)

    def test_rank_survey_matches_original_raw_actions_all_ties_and_modes(self):
        features = self.features()
        model = self.model()
        projected, _ = project_model(model)
        rows = [{"sample_id": "s", "cell_id": c, "mode": mode, "declared_target": i % 5 == 0}
                for i, c in enumerate(CELLS) for mode in ["native", "blurred"]]
        for rule in TIES:
            ranks = tie_ranks(features, rule, 20261030)
            for scenario in ["native_survey", "mixed_survey", "blurred_survey"]:
                table = policy_table(features, model, projected, scenario, 2, "rank_survey")
                actual = run_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, 2), table, 32, 2, ranks, raw_order(features, ranks))
                expected = original_episode(PaidBinaryEnvironment(rows, "s", scenario, 20261025, 2),
                    utility_table(features, {}, {}, scenario, 2, "prior_survey"), 32, 2, ranks)
                self.assertEqual(without_utility(actual), without_utility(expected))

    def test_private_fields_cannot_change_projection_order_or_prices(self):
        features, model = self.features(), self.model()
        projected, _ = project_model(model)
        poisoned = deepcopy(features)
        for row in poisoned.values():
            row.update(true_damage=True, future_quality="native", evaluator_matches=["secret"])
        for policy in POLICIES:
            self.assertEqual(policy_table(features, model, projected, "mixed_survey", 2, policy),
                policy_table(poisoned, model, projected, "mixed_survey", 2, policy))
        self.assertEqual(raw_order(features, list(range(64))), raw_order(poisoned, list(range(64))))

    def test_held_fit_ids_must_exclude_whole_event(self):
        selected = [{"sample_id": str(i), "event": str(i // 24), "split": "development"} for i in range(72)]
        models = {"development_fit": {}}
        for event in ["0", "1", "2"]:
            models["held_" + event] = {"fit_ids": [r["sample_id"] for r in selected if r["event"] != event]}
        assert_fit_ids(models, selected)
        models["held_0"]["fit_ids"][0] = "0"
        with self.assertRaises(ValueError):
            assert_fit_ids(models, selected)

    def test_paired_projection_effect_averages_events_not_seed_counts(self):
        rows = []
        for sid, event, gain in [("a", "e1", 4.), ("b", "e1", 4.), ("c", "e2", -2.)]:
            for policy, found in [("rank_negative_unprojected", 5.), ("rank_negative_isotonic", 5. + gain)]:
                rows.append({"sample_id": sid, "event": event, "fit_scope": "event_holdout", "scenario": "mixed_survey",
                    "upgrade_cost": 2, "budget": 32, "threshold": .1, "tie_rule": "fixed_random", "policy": policy,
                    "distinct_found": found, "false_positive_cells": 0., "upgrades": 0.})
        row = paired_effects(rows)["event_macro_effects"][0]
        self.assertEqual(row["delta_found"], 1.)
        self.assertEqual(row["positive_events"], 1)


if __name__ == "__main__":
    unittest.main()
