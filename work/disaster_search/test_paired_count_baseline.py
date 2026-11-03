from copy import deepcopy
from pathlib import Path
import unittest

from paired_count_baseline import fit_joint, joint_posterior, native_positive_given_blur, fit_local_count, action_score
from run_hold_probe_perception import make_plan


def paired_rows():
    rows = []
    for i, (truth, blurred, native, bm, nm) in enumerate([
        (True, True, True, ["a"], ["a", "b"]),
        (True, False, True, [], ["c"]),
        (False, False, False, [], []),
        (None, False, False, [], []),
    ]):
        for mode, positive, matches in [("blurred", blurred, bm), ("native", native, nm)]:
            rows.append({"sample_id": "s", "cell_id": str(i), "mode": mode, "target": truth,
                         "predicted": positive, "matched_target_uids": matches,
                         "visible_target_uids": sorted(set(bm + nm))})
    return rows, {"s": {str(i): i / 10 for i in range(4)}}


class JointModelTests(unittest.TestCase):
    def test_joint_update_uses_correlated_pair_not_product_of_marginals(self):
        pairs = []
        for truth, counts in [(True, {"00": 2, "01": 1, "10": 0, "11": 7}),
                              (False, {"00": 8, "01": 1, "10": 1, "11": 0})]:
            for outcome, count in counts.items():
                pairs.extend([(truth, outcome[0] == "1", outcome[1] == "1")] * count)
        model = fit_joint(pairs)
        self.assertAlmostEqual(native_positive_given_blur(model, True, False), .375)
        self.assertAlmostEqual(joint_posterior(.2, model, blurred=False, native=True), .2)
        self.assertNotAlmostEqual(joint_posterior(.2, model, native=True), .2)
        self.assertAlmostEqual(joint_posterior(.2, model), .2)

    def test_zero_supported_class_is_explicitly_smoothed_not_a_zero_likelihood(self):
        model = fit_joint([])
        self.assertAlmostEqual(joint_posterior(.3, model, blurred=False, native=True), .3)
        self.assertEqual(sum(model["counts"]["1"].values()), 0)
        with self.assertRaises(ValueError):
            fit_joint([(None, False, False)])


class LocalCountTests(unittest.TestCase):
    def test_upgrade_removes_previously_found_same_cell_building(self):
        rows, scores = paired_rows()
        model = fit_local_count(rows, scores)
        self.assertEqual(model["paired_known_cells"], 3)
        self.assertEqual(model["unknown_pairs_excluded"], 1)
        self.assertAlmostEqual(model["global_upgrade"]["1"]["mean"], 1)
        self.assertAlmostEqual(model["global_survey_mean_count"]["native"], 1)
        self.assertAlmostEqual(action_score(model, 0., "upgrade", blurred_positive=True, cost=2), .5)
        self.assertAlmostEqual(action_score(model, 0., "upgrade", blurred_positive=True, cost=4), .25)

    def test_unknown_positive_condition_remains_unavailable_not_zero(self):
        rows, scores = paired_rows()
        for r in rows: r["predicted"] = False
        model = fit_local_count(rows, scores)
        self.assertIsNone(action_score(model, .1, "upgrade", blurred_positive=True, cost=2))

    def test_duplicate_observations_and_truth_changes_are_refused(self):
        rows, scores = paired_rows()
        with self.assertRaises(ValueError):
            fit_local_count(rows + [rows[0]], scores)
        changed = deepcopy(rows)
        changed[0]["target"] = False
        with self.assertRaises(ValueError):
            fit_local_count(changed, scores)

    def test_renaming_evaluator_ids_cannot_change_aggregate_action_parameters(self):
        rows, scores = paired_rows()
        expected = fit_local_count(rows, scores)
        for r in rows:
            for field in ["visible_target_uids", "matched_target_uids"]:
                r[field] = ["renamed-" + uid for uid in r[field]]
        self.assertEqual(fit_local_count(rows, scores), expected)


class RgbPlanTests(unittest.TestCase):
    def test_prediction_plan_ignores_all_annotation_paths_and_contents(self):
        selected = [{"sample_id": str(i), "event": "e", "split": "development"} for i in range(6)]
        manifest = {"samples": [{**s, "files": {"pre_image": f"{s['sample_id']}_pre.png", "post_image": f"{s['sample_id']}_post.png", "post_label": "unused.json"}} for s in selected]}
        first = make_plan(manifest, selected, Path("/tmp"))
        for sample in manifest["samples"]:
            sample["files"]["post_label"] = "/nonexistent/poison.json"
            sample["true_target_count"] = 999999
        self.assertEqual(make_plan(manifest, selected, Path("/tmp")), first)
        changed = deepcopy(selected)
        changed[0]["split"] = "confirmation"
        with self.assertRaises(ValueError):
            make_plan(manifest, changed, Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
