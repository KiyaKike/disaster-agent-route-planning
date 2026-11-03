import copy
import unittest

import numpy as np
from PIL import Image

from external_area_core import CELLS,box
from external_context_core import select_controls,stitch_native,overlap_metrics,paired_changes,summarize
from run_changeos_pilot import prepare_tensor


class ContextDiagnosticTests(unittest.TestCase):
    def test_positive_control_selection_is_order_independent_and_uses_known_outputs(self):
        rows=[]
        for event in ["hurricane-harvey","santa-rosa-wildfire"]:
            for i in range(4):rows.append({"event":event,"mode":"native","declared_target":True,"observation_id":f"{event}:{i}","private_truth_not_used":i})
        selected=[r["observation_id"] for r in select_controls(rows)]
        reversed_rows=copy.deepcopy(list(reversed(rows)))
        for row in reversed_rows:row["private_truth_not_used"]=100
        self.assertEqual([r["observation_id"] for r in select_controls(reversed_rows)],selected)
        with self.assertRaises(ValueError):select_controls([r for r in rows if r["event"]=="hurricane-harvey"])

    def test_stitch_keeps_cell_coordinates_and_does_not_resize(self):
        masks={c:np.zeros((128,128),dtype=np.bool_) for c in CELLS}
        for i,c in enumerate(CELLS):masks[c][i,127-i]=True
        result=stitch_native(masks);self.assertEqual(int(result.sum()),16)
        for i,c in enumerate(CELLS):
            x,y,_,_=box(c);self.assertTrue(result[y+i,x+127-i])

    def test_missing_or_noncanonical_native_cell_stops(self):
        masks={c:np.zeros((128,128),dtype=np.bool_) for c in CELLS}
        del masks[CELLS[0]]
        with self.assertRaises(ValueError):stitch_native(masks)
        masks[CELLS[0]]=np.zeros((128,128),dtype=np.uint8)
        with self.assertRaises(ValueError):stitch_native(masks)

    def test_whole_tensor_preserves512_pixels_and_normalization(self):
        a=Image.new("RGB",(512,512),"white");b=Image.new("RGB",(512,512))
        tensor=prepare_tensor(a,b,"native").numpy()
        self.assertEqual(tensor.shape,(1,6,512,512))
        self.assertAlmostEqual(float(tensor[0,0,0,0]),(1-.485)/.229,places=5)
        self.assertAlmostEqual(float(tensor[0,3,511,511]),-.485/.229,places=5)

    def test_overlap_zeros_are_null_and_not_perfect_recall(self):
        a=np.zeros((512,512),dtype=np.bool_);pred=a.copy();pred[1,1]=True
        result=overlap_metrics(pred,a,np.ones_like(a))
        self.assertIsNone(result["area_recall"]);self.assertEqual(result["reference_false_positive_pixels"],1)
        self.assertEqual(result["reference_precision"],0)

    def test_reference_gain_and_loss_conserve_coverage_change(self):
        reference=np.zeros((512,512),dtype=np.bool_);reference[0,:4]=True
        a=np.zeros_like(reference);a[0,:2]=True
        b=np.zeros_like(reference);b[0,1:4]=True
        domain=np.ones_like(reference);r=paired_changes(a,b,reference,domain)
        self.assertEqual(r["gained_reference_pixels"],2);self.assertEqual(r["lost_reference_pixels"],1)
        self.assertEqual(overlap_metrics(b,reference,domain)["covered_reference_pixels"]-overlap_metrics(a,reference,domain)["covered_reference_pixels"],r["gained_reference_pixels"]-r["lost_reference_pixels"])

    def test_diagnostic_nonzero_domain_does_not_mutate_primary_mask(self):
        reference=np.zeros((512,512),dtype=np.bool_);reference[0,:2]=True;pred=reference.copy();domain=np.ones_like(reference);domain[0,0]=False
        self.assertEqual(overlap_metrics(pred,reference,domain)["reference_pixels"],1)
        self.assertEqual(overlap_metrics(pred,reference,np.ones_like(reference))["reference_pixels"],2)

    def test_zero_reference_event_remains_in_event_count(self):
        reference=np.zeros((512,512),dtype=np.bool_);prediction=reference.copy();domain=np.ones_like(reference)
        row={"domain":"all_published_pixels","input_mode":"whole512_native","event":"zero","sample_id":"zero",**overlap_metrics(prediction,reference,domain)}
        result=summarize([row])["totals"][0]
        self.assertEqual(result["events"],1);self.assertEqual(result["reference_positive_events"],0)
        self.assertIsNone(result["pooled_area_recall"]);self.assertIsNone(result["event_macro_recall_target_events"])


if __name__=="__main__":unittest.main()
