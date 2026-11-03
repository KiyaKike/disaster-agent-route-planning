"""Contract tests for the distinct BGR/localization-plus-four-grade interface."""
import ast
from pathlib import Path
import unittest
import numpy as np
from strongbaseline24_core import prepare_pair,decode_probabilities,restore_probability,FLIPS,severe_score,source_namespace

SOURCE=Path(__file__).resolve().parent/'attachments/external/disasteradaptivenet_fada4e7'


class StrongBaselineContractTests(unittest.TestCase):
    def test_bgr_pre_post_order_and_exact_publisher_normalization(self):
        pre=np.array([[[0,127,255],[23,67,109]]],np.uint8);post=np.array([[[245,37,91],[56,78,200]]],np.uint8)
        tree=ast.parse((SOURCE/'utils/augmentations.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Normalize')
        ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_Normalize','exec'),ns)
        bgr=np.concatenate([pre[...,::-1],post[...,::-1]],axis=2)
        official,_=ns['Normalize']()((bgr,None))
        np.testing.assert_array_equal(prepare_pair(pre,post),official.transpose(2,0,1)[None])
        self.assertGreater(float(prepare_pair(pre,post).max()),1.0)

    def test_rgb_input_contract_rejects_silent_rescaling_or_mismatch(self):
        valid=np.zeros((32,32,3),np.uint8)
        for bad in [valid.astype(np.float32),valid[:,:,:2],valid[:16]]:
            with self.assertRaises(ValueError):prepare_pair(valid,bad)

    def test_first_channel_is_localization_and_threshold_is_strict(self):
        p=np.zeros((5,1,3),np.float32);p[0]=[0.38,0.39,0.9];p[3]=[0.6,0.7,0.8]
        loc,damage,pred=decode_probabilities(p)
        np.testing.assert_array_equal(loc,[[False,True,True]])
        np.testing.assert_array_equal(damage,[[3,3,3]])
        np.testing.assert_array_equal(pred,[[0,3,3]])

    def test_damage_ties_choose_first_damage_channel_and_ignore_localization_argmax(self):
        p=np.ones((5,1,1),np.float32)*0.5;p[0]=0.99
        _,damage,pred=decode_probabilities(p)
        self.assertEqual(int(damage[0,0]),1);self.assertEqual(int(pred[0,0]),1)

    def test_four_spatial_flips_restore_each_channel_without_channel_flip(self):
        p=np.arange(5*3*7,dtype=np.float32).reshape(5,3,7)
        for flip in FLIPS:
            forward=np.flip(p,axis=tuple(v-1 for v in flip)).copy() if flip else p
            np.testing.assert_array_equal(restore_probability(forward,flip),p)

    def test_invalid_probabilities_rejected(self):
        good=np.zeros((5,2,2),np.float32)
        bad=good.copy();bad[0,0,0]=np.nan
        for p in [bad,good.astype(np.float64),good[:4],good+2]:
            with self.assertRaises(ValueError):decode_probabilities(p)

    def test_unknown_pixels_and_empty_precision_do_not_count_as_success(self):
        truth=np.array([[3,4,255,0]],np.uint8);pred=np.array([[3,0,4,4]],np.uint8)
        s=severe_score(truth,pred);self.assertEqual((s['tp'],s['fp'],s['fn']),(1,1,1));self.assertEqual(s['evaluated_pixels'],3)
        empty=severe_score(truth,np.zeros_like(pred));self.assertIsNone(empty['precision']);self.assertEqual(empty['recall'],0)
        self.assertIsNone(severe_score(np.zeros_like(truth),np.zeros_like(truth))['f1'])

    def test_source_adapter_cannot_execute_changed_model_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'models.py';p.write_text('raise RuntimeError("must not execute")')
            with self.assertRaises(ValueError):source_namespace(p)


if __name__=='__main__':unittest.main()
