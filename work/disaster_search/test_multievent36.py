from copy import deepcopy
import unittest
import numpy as np
from multievent36_core import EVENTS,select,rgb_quality
from materialize_hold_probes import extract_allowed_rows,COLUMNS


def fixture():
    rows=[{'internal:current_id':len(EVENTS)*i+j,'ml:source_file':f'{e}_{i:08d}','ml:split_original':'tier3','ml:disaster':e} for j,e in enumerate(EVENTS) for i in range(15)]
    return rows,{'train':set(),'val':set(),'test':{r['ml:split_original']+'/images/'+r['ml:source_file'] for r in rows}}


class Multievent36Tests(unittest.TestCase):
    def test_selection_is_order_invariant_and_twelve_per_event(self):
        rows,splits=fixture();a=select(rows,splits,set(),set());b=select(list(reversed(rows)),splits,set(),set())
        self.assertEqual(a,b);self.assertEqual(len(a),36)
        for e in EVENTS:self.assertEqual(sum(r['event']==e for r in a),12)

    def test_damage_columns_are_rejected_before_selection(self):
        rows,splits=fixture();rows[0]['ml:damage']=[3]
        with self.assertRaises(ValueError):select(rows,splits,set(),set())

    def test_training_exposure_and_reserved_events_are_fatal(self):
        rows,splits=fixture();s=deepcopy(splits);s['train'].add(next(iter(s['test'])))
        with self.assertRaises(ValueError):select(rows,s,set(),set())
        with self.assertRaises(ValueError):select(rows,splits,set(),{EVENTS[0]})

    def test_duplicate_identity_or_insufficient_pool_cannot_trigger_substitution(self):
        rows,splits=fixture()
        with self.assertRaises(ValueError):select(rows+[rows[0]],splits,set(),set())
        blocked={r['ml:source_file'] for r in rows if r['ml:disaster']==EVENTS[0]}
        with self.assertRaises(ValueError):select(rows,splits,blocked,set())

    def test_quality_flags_do_not_relabel_black_pixels_or_drop_identity(self):
        image=np.zeros((1024,1024,3),np.uint8);copy=image.copy();q=rgb_quality(image)
        self.assertTrue(q['zero_RGB_at_least25pct_flag']);np.testing.assert_array_equal(image,copy)
        with self.assertRaises(ValueError):rgb_quality(image[:512])

    def test_shared_column_extractor_returns_only_new_locked_ids(self):
        import pyarrow as pa
        table=pa.Table.from_pylist([{'ml:damage':[i]} for i in range(4)])
        class Parquet:
            def read(self,*,columns,use_threads):
                assert columns==COLUMNS and use_threads is False
                return table
        ids=[{'ml:source_file':name} for name in ['old-reserved','new-a','new-b','unselected']]
        result=extract_allowed_rows(Parquet(),ids,['new-a','new-b'])
        self.assertEqual(list(result),['new-a','new-b']);self.assertEqual(result['new-b']['ml:damage'],[2])


if __name__=='__main__':unittest.main()
