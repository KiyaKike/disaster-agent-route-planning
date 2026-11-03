import unittest

from run_cached_search import Action, PublicCell, Evaluator
from run_local_feedback import LocalPlanner, LOCAL_POLICIES, run_local_episode
from test_cached_search import RATES, observation, truth_row


class LocalFeedbackChecks(unittest.TestCase):
    def cells(self):
        return [PublicCell("r00c00", 0, 0, 0, .3), PublicCell("r00c01", 1, 0, 0, .3),
                PublicCell("r00c02", 2, 0, 0, .3), PublicCell("r01c00", 0, 1, 1, .2)]

    def test_feedback_never_reaches_far_or_different_bin(self):
        for policy in LOCAL_POLICIES:
            state = LocalPlanner(self.cells(), policy, RATES, "native_survey")
            state.observe(Action("r00c00", "survey"), observation(mode="native", positive=True))
            self.assertGreater(state.current("r00c01"), .3)
            self.assertAlmostEqual(state.current("r00c02"), .3)
            self.assertAlmostEqual(state.current("r01c00"), .2)

    def test_positive_only_policies_ignore_neighbor_negative(self):
        for policy in ("local_positive", "neighbor_positive"):
            state = LocalPlanner(self.cells(), policy, RATES, "native_survey")
            state.observe(Action("r00c00", "survey"), observation(mode="native"))
            self.assertAlmostEqual(state.current("r00c01"), .3)
        state = LocalPlanner(self.cells(), "local_residual", RATES, "native_survey")
        state.observe(Action("r00c00", "survey"), observation(mode="native"))
        self.assertLess(state.current("r00c01"), .3)

    def test_upgrade_replaces_neighbor_residual_and_ignores_recursive_shift(self):
        for policy in LOCAL_POLICIES:
            state = LocalPlanner(self.cells(), policy, RATES, "mixed_survey")
            state.observe(Action("r00c00", "survey"), observation())
            state.observe(Action("r00c00", "upgrade"), observation(mode="native", positive=True))
            direct = LocalPlanner(self.cells(), policy, RATES, "mixed_survey")
            direct.observe(Action("r00c00", "survey"), observation(mode="native", positive=True))
            self.assertEqual(state.values, direct.values)
            self.assertEqual(state.cell_shifts, direct.cell_shifts)

    def test_evaluation_truth_never_changes_local_actions(self):
        for policy in LOCAL_POLICIES:
            traces = []
            for truth in (False, True):
                rows = [truth_row(c.cell_id, mode, truth, ("building",) if truth else ())
                        for c in self.cells() for mode in ("native", "blurred")]
                def provider(action):
                    return observation(action.cell_id, "native", action.cell_id == "r00c00")
                result, trace = run_local_episode(self.cells(), policy, RATES, "native_survey", 3, provider, Evaluator(rows))
                self.assertLessEqual(result["cost"], 3)
                traces.append(trace)
            self.assertEqual(*traces)


if __name__ == "__main__":
    unittest.main()
