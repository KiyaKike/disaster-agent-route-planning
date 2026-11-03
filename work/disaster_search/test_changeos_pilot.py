import importlib.util
from pathlib import Path
import tempfile
import unittest

from run_changeos_pilot import CELLS, crop_pair, freeze_plan, make_plan, predict_summary, prepare_tensor
from evaluate_changeos_pilot import counts


def tiny_protocol():
    samples = []
    for split in ("test", "calibration", "validation"):
        for event in ("a", "b"):
            for i in range(3):
                samples.append({"sample_id": f"{split}_{event}_{i}", "event": event, "split": split,
                                "files": {"pre_image": "pre.png", "post_image": "post.png",
                                          "post_label": "MUST_NOT_BE_READ.json"}})
    return {"samples": samples}


class PilotPlanChecks(unittest.TestCase):
    def test_empty_predictions_do_not_imply_perfect_precision(self):
        self.assertIsNone(counts([])["precision"])
        self.assertIsNone(counts([])["recall"])
        result = counts([{"predicted": False, "target": True}])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)

    def test_confusion_counts_keep_false_positives(self):
        result = counts([{"predicted": True, "target": True}, {"predicted": True, "target": False},
                         {"predicted": False, "target": True}, {"predicted": False, "target": False}])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))
        self.assertEqual(result["precision"], .5)

    def test_only_first_two_calibration_scenes_without_labels(self):
        plan = make_plan(tiny_protocol(), Path("/nonexistent"))
        self.assertEqual(len(plan["samples"]), 4)
        self.assertEqual(plan["observations"], len(CELLS) * 4 * 2)
        self.assertTrue(all(s["sample_id"].startswith("calibration_") for s in plan["samples"]))
        self.assertFalse(any("label" in key for sample in plan["samples"] for key in sample))

    def test_changed_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            freeze_plan(path, {"a": 1})
            with self.assertRaises(ValueError):
                freeze_plan(path, {"a": 2})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "Inference environment required")
class PerceptionInputChecks(unittest.TestCase):
    def test_pre_post_channel_order_and_imagenet_normalization(self):
        from PIL import Image
        pre = Image.new("RGB", (128, 128), (255, 0, 0))
        post = Image.new("RGB", (128, 128), (0, 0, 255))
        x = prepare_tensor(pre, post, "native")
        self.assertEqual(tuple(x.shape), (1, 6, 128, 128))
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), (1 - .485) / .229, places=5)
        self.assertAlmostEqual(float(x[0, 3, 0, 0]), -.485 / .229, places=5)
        self.assertAlmostEqual(float(x[0, 5, 0, 0]), (1 - .406) / .225, places=5)

    def test_local_crop_does_not_include_neighboring_pixels(self):
        from PIL import Image
        image = Image.new("RGB", (1024, 1024), "black")
        image.paste((255, 255, 255), (128, 128, 256, 256))
        local, _ = crop_pair(image, image, "r01c01")
        neighbor, _ = crop_pair(image, image, "r01c02")
        self.assertEqual(local.size, (128, 128))
        self.assertEqual(local.getextrema(), ((255, 255),) * 3)
        self.assertEqual(neighbor.getextrema(), ((0, 0),) * 3)

    def test_background_and_minor_damage_do_not_count_as_severe(self):
        import torch
        d = torch.zeros(1, 5, 8, 8)
        d[:, 2] = 1
        l = torch.ones(1, 1, 8, 8)
        result, _ = predict_summary({"change_prediction": d, "t1_semantic_prediction": l})
        self.assertFalse(result["declared_target"])
        d.zero_()
        d[:, 3] = 1
        l.zero_()
        result, _ = predict_summary({"change_prediction": d, "t1_semantic_prediction": l})
        self.assertFalse(result["declared_target"])

    def test_declared_target_uses_both_localization_and_four_pixel_rule(self):
        import torch
        d = torch.zeros(1, 5, 8, 8)
        d[:, 4] = 1
        l = torch.zeros(1, 1, 8, 8)
        l[0, 0, 0, :3] = 1
        result, _ = predict_summary({"change_prediction": d, "t1_semantic_prediction": l})
        self.assertFalse(result["declared_target"])
        l[0, 0, 0, 3] = 1
        result, _ = predict_summary({"change_prediction": d, "t1_semantic_prediction": l})
        self.assertTrue(result["declared_target"])
        self.assertEqual(result["predicted_severe_pixels"], 4)

    def test_non_probability_output_and_batch_broadcast_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            predict_summary({"change_prediction": torch.ones(1, 5, 8, 8), "t1_semantic_prediction": torch.ones(1, 1, 8, 8)})
        with self.assertRaises(ValueError):
            predict_summary({"change_prediction": torch.ones(2, 5, 8, 8) / 5, "t1_semantic_prediction": torch.ones(2, 1, 8, 8)})


if __name__ == "__main__":
    unittest.main()
