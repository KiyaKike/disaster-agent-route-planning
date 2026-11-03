import unittest
import numpy as np
from PIL import Image

from inspect_external_rgb18 import phase_translation


class TranslationDiagnosticTests(unittest.TestCase):
    def test_identical_texture_has_zero_shift(self):
        rng = np.random.default_rng(53)
        image = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
        result = phase_translation(image, image)
        self.assertEqual(result["estimated_post_to_pre_translation_native_px"], [0, 0])

    def test_synthetic_translation_sign_and_scale(self):
        rng = np.random.default_rng(53)
        array = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        pre = Image.fromarray(array)
        post = Image.fromarray(np.roll(array, (10, -18), axis=(0, 1)))
        result = phase_translation(pre, post)
        self.assertEqual(result["estimated_post_to_pre_translation_native_px"], [18, -10])

    def test_no_texture_is_not_a_registration_acceptance(self):
        image = Image.new("RGB", (512, 512))
        result = phase_translation(image, image)
        self.assertEqual(result["phase_peak_not_probability"], 0.0)
        self.assertIn("Manual paired review", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
