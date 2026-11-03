"""Information-boundary checks for the development-only case builder."""
import unittest

from prepare_boone_context_pilot import binary_label, build_cases


class CaseBoundaryTests(unittest.TestCase):
    def test_anchor_never_appears_in_its_target_set(self):
        ids, cases = build_cases([{'building_id': str(i), 'eligible': True}
                                 for i in range(12)])
        self.assertEqual(len(ids), 12)
        self.assertEqual(sum(len(c['target_ids']) for c in cases), 132)
        for case in cases:
            self.assertNotIn(case['anchor_id'], case['target_ids'])
            self.assertEqual(set(case['target_ids']) | {case['anchor_id']}, set(ids))

    def test_label_changes_do_not_change_cohort_or_cases(self):
        geometry = [{'building_id': 'a', 'eligible': True, 'label': 'minor damage'},
                    {'building_id': 'b', 'eligible': True, 'label': 'no damage'},
                    {'building_id': 'c', 'eligible': False, 'label': 'destroyed'}]
        before = build_cases(geometry)
        for row in geometry:
            row['label'] = 'changed hidden label'
            row['future_prediction'] = 999
        self.assertEqual(before, build_cases(list(reversed(geometry))))
        self.assertEqual(before[0], ['a', 'b'])

    def test_unknown_does_not_become_negative(self):
        self.assertIsNone(binary_label('unclassified'))
        self.assertIsNone(binary_label('occluded'))
        self.assertEqual(binary_label('no damage'), 0)
        self.assertEqual(binary_label('minor damage'), 1)


if __name__ == '__main__':
    unittest.main()
