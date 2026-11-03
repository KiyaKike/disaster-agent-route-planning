"""Meaningful invariants for the feedback-search experiment, not efficacy tests."""
import copy
import unittest

from experiment import (Action, Belief, Cell, Observation, POLICIES, SCENARIOS,
                        SENSOR, posterior, run_episode, synthetic_scene)


class SearchChecks(unittest.TestCase):
    def test_clear_negative_is_stronger_than_poor_negative(self):
        clear = posterior(.7, False, SENSOR["clear"])
        poor = posterior(.7, False, SENSOR["poor"])
        self.assertLess(clear, poor)
        self.assertLess(poor, .7)

    def test_duplicate_evidence_and_unsupported_verify_are_rejected(self):
        state = Belief([Cell("a", 0, 0, "g", .6, False)], "quality")
        observation = Observation("image1", False, "poor")
        with self.assertRaises(ValueError):
            state.observe(Action("a", "verify"), observation)
        state.observe(Action("a", "survey"), observation)
        with self.assertRaises(ValueError):
            state.observe(Action("a", "verify"), observation)

    def test_budget_accounting_and_unique_opportunities_all_policies(self):
        for scenario in SCENARIOS:
            scene = synthetic_scene(1, scenario, side=4)
            for policy in POLICIES:
                for budget in (1, 2, 10, 40):
                    result, log = run_episode(scene, policy, budget)
                    self.assertLessEqual(result["cost"], budget)
                    self.assertEqual(result["cost"], sum(1 if s["action"]["slot"] == "survey" else 2 for s in log))
                    ids = [s["observation"]["observation_id"] for s in log]
                    self.assertEqual(len(ids), len(set(ids)))

    def test_ground_truth_is_not_used_by_planner(self):
        scene = synthetic_scene(5)
        altered = copy.deepcopy(scene)
        altered.truth = {cid: not value for cid, value in scene.truth.items()}
        for policy in POLICIES:
            _, original = run_episode(scene, policy, 20)
            _, changed = run_episode(altered, policy, 20)
            self.assertEqual(original, changed)

    def test_outcomes_are_paired_across_policies(self):
        scene = synthetic_scene(3)
        _, first = run_episode(scene, "quality", 40)
        _, second = run_episode(scene, "pooled", 40)
        a = {x["observation"]["observation_id"]: x["observation"] for x in first}
        b = {x["observation"]["observation_id"]: x["observation"] for x in second}
        self.assertTrue(a.keys() & b.keys())
        for key in a.keys() & b.keys():
            self.assertEqual(a[key], b[key])

    def test_adaptive_feedback_changes_unvisited_peer_belief(self):
        cells = [Cell("a", 0, 0, "g", .7, False), Cell("b", 1, 0, "g", .7, False)]
        state = Belief(cells, "adaptive_prior")
        before = state.current("b")
        state.observe(Action("a", "survey"), Observation("one", False, "clear"))
        self.assertLess(state.current("b"), before)

    def test_unobserved_quality_cannot_affect_first_action(self):
        scene = synthetic_scene(4)
        first = Belief(scene.cells, "quality").choose(10)
        for key in scene.observations:
            scene.observations[key] = Observation(str(key), False, "poor")
        self.assertEqual(first, Belief(scene.cells, "quality").choose(10))

    def test_revisit_can_be_chosen_and_costs_more(self):
        state = Belief([Cell("a", 0, 0, "g", .9, False)], "quality")
        state.observe(Action("a", "survey"), Observation("one", False, "poor"))
        self.assertIsNone(state.choose(1))
        self.assertEqual(state.choose(2), Action("a", "verify"))

    def test_all_negative_scene_and_empty_predictions_are_defined(self):
        scene = synthetic_scene(2, side=2)
        scene.truth = dict.fromkeys(scene.truth, False)
        scene.observations = {k: Observation(v.observation_id, False, v.quality) for k, v in scene.observations.items()}
        result, _ = run_episode(scene, "quality", 10)
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["precision"])
        self.assertTrue(result["no_true_discovery"])

    def test_repeatability(self):
        scene = synthetic_scene(8)
        self.assertEqual(run_episode(scene, "adaptive_prior", 20), run_episode(scene, "adaptive_prior", 20))


if __name__ == "__main__":
    unittest.main()
