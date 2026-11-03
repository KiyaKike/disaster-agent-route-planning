import unittest
from pathlib import Path

from fit_real_calibration import fit_likelihood, fit_prior
from run_changeos_calibration import make_plan


class RealCalibrationChecks(unittest.TestCase):
    def test_unknown_truth_is_excluded_and_zero_counts_are_smoothed(self):
        rows = [{"target": True, "predicted": True}, {"target": False, "predicted": False},
                {"target": None, "predicted": True}]
        result = fit_likelihood(rows)
        self.assertEqual(result["unknown_excluded"], 1)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["smoothed_sensitivity"], .75)
        self.assertEqual(result["smoothed_false_positive_rate"], .25)

    def test_prior_uses_unique_cells_and_preserves_unknowns(self):
        result = fit_prior([(float(i), None if i == 0 else i % 2 == 0) for i in range(16)])
        self.assertEqual(sum(r["known_cells"] for r in result["bins"]), 15)
        self.assertEqual(sum(r["unknown_cells"] for r in result["bins"]), 1)
        self.assertTrue(all(0 < r["fitted_presence_probability"] < 1 for r in result["bins"]))
        # Image-only quantile edges must not depend on truth.
        other = fit_prior([(float(i), True) for i in range(16)])
        self.assertEqual(result["bin_edges"], other["bin_edges"])

    def test_plan_uses_only_calibration_without_reading_label_files(self):
        samples = [{"sample_id": f"s{i}", "event": f"e{i // 12}", "split": "calibration",
                    "files": {"pre_image": f"pre{i}.png", "post_image": f"post{i}.png"}} for i in range(24)]
        samples.append({"sample_id": "DO_NOT_READ", "split": "test"})
        plan = make_plan({"samples": samples}, Path("/nonexistent"))
        self.assertEqual(len(plan["samples"]), 24)
        self.assertEqual(len(plan["cells"]), 64)
        self.assertEqual(plan["observations"], 24 * 64 * 2)
        self.assertNotIn("DO_NOT_READ", str(plan))

    def test_validation_is_explicit_and_test_inference_disabled(self):
        samples = [{"sample_id": f"v{i}", "event": f"e{i // 12}", "split": "validation",
                    "files": {"pre_image": f"pre{i}.png", "post_image": f"post{i}.png"}} for i in range(24)]
        samples.append({"sample_id": "DO_NOT_READ", "split": "test"})
        plan = make_plan({"samples": samples}, Path("/nonexistent"), "validation")
        self.assertEqual(plan["stage"], "FROZEN_COMPLETE_VALIDATION_RGB_PASS")
        self.assertEqual({s["split"] for s in plan["samples"]}, {"validation"})
        self.assertNotIn("DO_NOT_READ", str(plan))
        with self.assertRaises(ValueError):
            make_plan({"samples": samples}, Path("/nonexistent"), "test")


if __name__ == "__main__":
    unittest.main()
