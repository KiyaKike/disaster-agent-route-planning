import unittest

from run_cached_search import (Action, CacheEnvironment, Evaluator, Planner, PublicCell,
                               POLICIES, run_episode, survey_mode, update_probability)


RATES = {"native": {"tpr": .85, "fpr": .01}, "blurred": {"tpr": .15, "fpr": .005}}


def observation(cid="r00c00", mode="blurred", positive=False):
    group = "scene:" + cid
    return {"sample_id": "scene", "cell_id": cid, "mode": mode, "declared_target": positive,
            "evidence_group_id": group, "observation_id": group + ":" + mode,
            "uncalibrated_severe_score": .9 if positive else .1}


def truth_row(cid="r00c00", mode="blurred", target=True, matched=()):
    return {"cell_id": cid, "mode": mode, "target": target,
            "visible_target_uids": ["building"] if target else [], "matched_target_uids": list(matched)}


class CachedSearchChecks(unittest.TestCase):
    def cells(self):
        return [PublicCell(f"r00c{i:02d}", i, 0, 0 if i < 2 else 1, .3 if i < 2 else .1) for i in range(4)]

    def test_upgrade_replaces_likelihood_and_peer_contribution(self):
        cells = self.cells()
        state = Planner(cells, "adaptive_prior", RATES, "mixed_survey")
        state.observe(Action("r00c00", "survey"), observation())
        state.observe(Action("r00c00", "upgrade"), observation(mode="native"))
        direct = Planner(cells, "adaptive_prior", RATES, "mixed_survey")
        direct.observe(Action("r00c00", "survey"), observation(mode="native"))
        self.assertAlmostEqual(state.values["r00c00"], update_probability(.3, False, RATES["native"]))
        self.assertEqual(state.values, direct.values)
        self.assertEqual(state.shifts, direct.shifts)
        self.assertEqual(len(state.registry.representatives()), 1)

    def test_quality_negative_changes_upgrade_choice(self):
        cells = [PublicCell("r00c00", 0, 0, 0, .3), PublicCell("r00c01", 1, 0, 1, .1)]
        state = Planner(cells, "quality", RATES, "blurred_survey")
        state.observe(Action("r00c00", "survey"), observation())
        self.assertEqual(state.choose(2), Action("r00c00", "upgrade"))
        clear = Planner(cells, "quality", RATES, "blurred_survey")
        clear.observe(Action("r00c00", "survey"), observation(mode="native"))
        self.assertEqual(clear.choose(2), Action("r00c01", "survey"))

    def test_feedback_changes_unvisited_rank_using_only_public_groups(self):
        cells = [PublicCell("r00c00", 0, 0, 0, .3), PublicCell("r00c01", 1, 0, 0, .3),
                 PublicCell("r00c02", 2, 0, 1, .299)]
        state = Planner(cells, "adaptive_prior", RATES, "native_survey")
        audit = state.observe(Action("r00c00", "survey"), observation(mode="native"))
        self.assertTrue(audit["feedback_reordered_unvisited"])
        self.assertEqual(state.choose(1), Action("r00c02", "survey"))
        baseline = Planner(cells, "quality", RATES, "native_survey")
        baseline.observe(Action("r00c00", "survey"), observation(mode="native"))
        self.assertEqual(baseline.choose(1), Action("r00c01", "survey"))

    def test_duplicate_action_or_unsupported_upgrade_rejected(self):
        state = Planner(self.cells(), "quality", RATES, "mixed_survey")
        with self.assertRaises(ValueError):
            state.observe(Action("r00c00", "upgrade"), observation(mode="native"))
        state.observe(Action("r00c00", "survey"), observation(positive=True))
        with self.assertRaises(ValueError):
            state.observe(Action("r00c00", "survey"), observation())
        with self.assertRaises(ValueError):
            state.observe(Action("r00c00", "upgrade"), observation(mode="native"))

    def test_future_quality_not_available_before_selection(self):
        cells = self.cells()
        states = [Planner(cells, "adaptive_prior", RATES, "mixed_survey") for _ in range(2)]
        envs = [CacheEnvironment("scene", [], "mixed_survey", seed) for seed in (1, 1000)]
        self.assertEqual(states[0].choose(8), states[1].choose(8))
        self.assertTrue(all(not env.requests for env in envs))

    def test_schedule_paired_and_observation_fields_whitelisted(self):
        source = observation()
        source.update(target=True, matched_target_uids=["SECRET"], future_native_positive=True)
        env = CacheEnvironment("scene", [source], "blurred_survey", 1)
        returned = env.observe(Action("r00c00", "survey"))
        self.assertNotIn("target", returned)
        self.assertNotIn("matched_target_uids", returned)
        self.assertNotIn("future_native_positive", returned)
        self.assertEqual(survey_mode("mixed_survey", 42, "s", "c"), survey_mode("mixed_survey", 42, "s", "c"))

    def test_building_metric_deduplicates_cells_and_quality_modes(self):
        rows = [truth_row(cid, mode, True, ("building",)) for cid in ("r00c00", "r00c01") for mode in ("native", "blurred")]
        evaluator = Evaluator(rows)
        for i, row in enumerate(rows):
            evaluator.observe(observation(row["cell_id"], row["mode"], True), i + 1)
        result = evaluator.result()
        self.assertEqual(result["target_count"], 1)
        self.assertEqual(result["unique_buildings_found"], 1)
        self.assertEqual(result["first_discovery_cost"], 1)

    def test_unknown_and_no_target_do_not_become_false_discoveries(self):
        evaluator = Evaluator([truth_row(target=None)])
        evaluator.observe(observation(positive=True), 1)
        result = evaluator.result()
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["first_discovery_cost"])
        self.assertEqual(result["false_positive_cells"], 0)
        self.assertEqual(result["unknown_positive_cells"], 1)
        self.assertEqual(result["scene_type"], "indeterminate")

    def test_all_policies_respect_budget_and_unique_observation_identity(self):
        observations = [observation(f"r00c{i:02d}", mode, False) for i in range(4) for mode in ("native", "blurred")]
        rows = [truth_row(o["cell_id"], o["mode"], False) for o in observations]
        for policy in POLICIES:
            env = CacheEnvironment("scene", observations, "blurred_survey", 1)
            result, trace = run_episode(self.cells(), policy, RATES, "blurred_survey", 5, env.observe, Evaluator(rows))
            self.assertLessEqual(result["cost"], 5)
            self.assertEqual(result["cost"], result["surveys"] + 2 * result["upgrades"])
            self.assertEqual(len(env.requests), len(set(env.requests)))
            self.assertTrue(result["no_true_discovery"])
            self.assertIsNone(result["first_discovery_cost"])

    def test_evaluator_truth_cannot_change_action_trace(self):
        observations = [observation(f"r00c{i:02d}", mode, i == 1) for i in range(4) for mode in ("native", "blurred")]
        traces = []
        for present in (True, False):
            env = CacheEnvironment("scene", observations, "mixed_survey", 1)
            rows = [truth_row(o["cell_id"], o["mode"], present, ("building",) if present else ()) for o in observations]
            _, trace = run_episode(self.cells(), "adaptive_prior", RATES, "mixed_survey", 5, env.observe, Evaluator(rows))
            traces.append(trace)
        self.assertEqual(*traces)


if __name__ == "__main__":
    unittest.main()
