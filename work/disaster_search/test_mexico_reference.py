import io
import unittest

import numpy as np
import pyarrow as pa
from PIL import Image

from mexico_reference_core import allowed_read, selected_rows, decode_mask, mask_correspondence


class ReferenceChecks(unittest.TestCase):
    def test_ranges_reject_forbidden_and_boundary_crossing(self):
        allowed_read(100, 10, [(100, 130)], [(0, 100), (130, 200)])
        for start, count in [(99, 11), (110, 21), (100, 0)]:
            with self.assertRaises(ValueError): allowed_read(start, count, [(100, 130)], [])
        with self.assertRaises(ValueError): allowed_read(100, 10, [(100, 130)], [(109, 120)])

    def test_only_selected_rows_leave_projection(self):
        table = pa.table({'image_name': ['unselected', 'chosen'], 't2_mask': [{'bytes': b'private'}, {'bytes': b'wanted'}]})
        self.assertEqual(selected_rows(table, ['chosen']), {'chosen': {'bytes': b'wanted'}})
        with self.assertRaises(ValueError): selected_rows(table, ['missing'])
        with self.assertRaises(ValueError): selected_rows(table, ['chosen', 'chosen'])
        with self.assertRaises(ValueError): selected_rows(table.append_column('t1_image', pa.array([1, 2])), ['chosen'])

    def test_known_domain_and_unknown_handling(self):
        a = np.array([[0, 3, 3, 255]], np.uint8); b = np.array([[0, 3, 0, 4]], np.uint8)
        result = mask_correspondence(a, b, np.ones(a.shape, bool))
        self.assertEqual(result['severe_iou'], .5)
        self.assertEqual(result['reference_severe_pixels_common_domain'], 2)
        self.assertEqual(result['published_severe_pixels_common_domain'], 1)
        self.assertEqual(result['footprint_common_valid_pixels'], 3)
        raw = io.BytesIO(); Image.fromarray(np.zeros((1024, 1024), np.uint8)).save(raw, format='PNG')
        self.assertEqual(decode_mask(raw.getvalue()).shape, (1024, 1024))
        raw = io.BytesIO(); Image.fromarray(np.full((1024, 1024), 7, np.uint8)).save(raw, format='PNG')
        with self.assertRaises(ValueError): decode_mask(raw.getvalue())


if __name__ == '__main__': unittest.main()
