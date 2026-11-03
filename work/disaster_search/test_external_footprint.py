import copy
import itertools
import random
import unittest

import numpy as np
from PIL import Image

from external_area_core import CELLS
from external_footprint_core import (NATIVE128,NATIVE256,CONTEXT256,FULL512,MODES,BLOCKS,units,block_bounds,compose_block,canonical_block,ranked_acquisitions,evaluate_acquisitions,exact_random,score_union,aggregate)


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.zero=np.zeros((512,512),dtype=np.bool_);self.domain=np.ones_like(self.zero)
        self.features={cell:{"change":0.2} for cell in CELLS}
        self.patch=(Image.new("RGB",(256,256),(10,20,30)),Image.new("RGB",(256,256),(40,50,60)))
        self.coarse=(Image.new("RGB",(64,64),(70,80,90)),Image.new("RGB",(64,64),(100,110,120)))

    def test_four_blocks_partition16_cells_without_overlap(self):
        cover=np.zeros((512,512),dtype=np.uint8);cells=[]
        for unit in units(NATIVE256):
            x,y,xx,yy=unit["bounds"];cover[y:yy,x:xx]+=1;cells.extend(unit["cells"])
        self.assertTrue(np.all(cover==1));self.assertEqual(sorted(cells),sorted(CELLS))

    def test_native256_preserves_pixels_and_does_not_resize(self):
        result=compose_block(NATIVE256,self.patch,"b11")
        self.assertEqual(result[0].size,(256,256));self.assertEqual(result[0].tobytes(),self.patch[0].tobytes())

    def test_selected256_is_pasted_at_each_correct_position(self):
        for block in BLOCKS:
            out=compose_block(CONTEXT256,self.patch,block,self.coarse)
            self.assertEqual(out[0].crop(block_bounds(block)).tobytes(),self.patch[0].tobytes())

    def test_outside_purchased_block_contains_only_free_coarse(self):
        out=compose_block(CONTEXT256,self.patch,"b10",self.coarse)
        self.assertEqual(out[0].getpixel((511,511)),(70,80,90));self.assertEqual(out[0].getpixel((0,511)),(10,20,30))

    def test_unseen_native512_rejected_as_context_or_selected_block(self):
        scene=tuple(Image.new("RGB",(512,512)) for _ in range(2))
        with self.assertRaises(ValueError):compose_block(CONTEXT256,self.patch,"b00",scene)
        with self.assertRaises(ValueError):compose_block(NATIVE256,scene,"b00")
        with self.assertRaises(ValueError):compose_block(NATIVE256,self.patch,"b00",self.coarse)

    def test_canonical256_excludes_all_predictions_outside_paid_roi(self):
        mask=np.ones((512,512),dtype=np.bool_);mask[:256,256:]=False
        result=canonical_block(mask,CONTEXT256,"b01");self.assertEqual(result.shape,(256,256));self.assertFalse(result.any())

    def test_wrong_mask_or_block_rejected(self):
        with self.assertRaises(ValueError):canonical_block(np.zeros((256,256),dtype=np.uint8),NATIVE256,"b00")
        with self.assertRaises(ValueError):canonical_block(self.zero,NATIVE256,"b00")
        with self.assertRaises(ValueError):block_bounds("b22")

    def test_cost_is_proportional_to_purchased_native_area(self):
        for mode in MODES:
            for u in units(mode):
                x,y,xx,yy=u["bounds"];self.assertEqual((xx-x)*(yy-y),128*128*u["cost"])

    def test_full_scene_cannot_be_acquired_at_budget8(self):
        self.assertEqual(ranked_acquisitions(self.features,FULL512,8,2),[])
        result=exact_random({"scene":self.domain},FULL512,8,self.domain,self.zero,self.domain)
        self.assertFalse(result["affordable"]);self.assertEqual(result["spent"],0);self.assertEqual(result["covered_reference_pixels"],0)

    def test_block_rank_uses_mean_not_maximum_member_cue(self):
        feats={cell:{"change":0.0} for cell in CELLS};feats['r00c00']["change"]=1.0
        for cell in units(NATIVE256)[1]["cells"]:feats[cell]["change"]=.4
        self.assertEqual(ranked_acquisitions(feats,NATIVE256,4,2)[0]["unit_id"],"b01")

    def test_block_tie_rank_is_minimum_member_cell_rank(self):
        order=list(CELLS);random.Random(6).shuffle(order);rank={c:i for i,c in enumerate(order)}
        expected=sorted(units(NATIVE256),key=lambda u:min(rank[c] for c in u["cells"]))
        actual=ranked_acquisitions(self.features,NATIVE256,8,6)
        self.assertEqual([r["unit_id"] for r in actual],[r["unit_id"] for r in expected[:2]])

    def test_ranking_does_not_read_private_truth_or_future_feedback(self):
        a=copy.deepcopy(self.features);b=copy.deepcopy(self.features)
        for r in a.values():r.update(reference=999,future_positive=True)
        for r in b.values():r.update(reference=0,future_positive=False)
        self.assertEqual(ranked_acquisitions(a,NATIVE128,8,6),ranked_acquisitions(b,NATIVE128,8,6))

    def test_overspend_duplicate_and_invalid_units_rejected(self):
        masks={b:np.zeros((256,256),dtype=np.bool_) for b in BLOCKS}
        for trace in [[{"unit_id":"b00","cost":1,"cumulative_cost":1}],
                      [{"unit_id":"b00","cost":4,"cumulative_cost":4},{"unit_id":"b00","cost":4,"cumulative_cost":8}],
                      [{"unit_id":"other","cost":4,"cumulative_cost":4}]]:
            with self.assertRaises(ValueError):evaluate_acquisitions(trace,masks,NATIVE256,4,self.zero,self.zero,self.domain)

    def test_exact_random_equals_enumeration_with_nonzero_free_baseline(self):
        reference=self.zero.copy();reference[0,:4]=True;reference[256,:2]=True
        free=self.zero.copy();free[0,0]=True;free[300,300]=True
        masks={b:np.zeros((256,256),dtype=np.bool_) for b in BLOCKS}
        masks['b00'][0,:4]=True;masks['b01'][0,0]=True;masks['b10'][0,:2]=True;masks['b11'][0,:3]=True
        values=[]
        for selected in itertools.combinations(BLOCKS,2):
            trace=[{"unit_id":uid,"cost":4,"cumulative_cost":4*(i+1)} for i,uid in enumerate(selected)]
            values.append(evaluate_acquisitions(trace,masks,NATIVE256,8,reference,free,self.domain))
        exact=exact_random(masks,NATIVE256,8,reference,free,self.domain)
        for key in ["covered_reference_pixels","reference_false_positive_pixels","incremental_covered_reference_pixels","incremental_false_positive_pixels","predicted_pixels"]:
            self.assertAlmostEqual(exact[key],sum(r[key] for r in values)/len(values))
        self.assertEqual(exact["reference_precision"],exact["covered_reference_pixels"]/exact["predicted_pixels"])

    def test_free_baseline_and_paid_overlap_count_each_pixel_once(self):
        free=self.zero.copy();free[0,:2]=True;paid=self.zero.copy();paid[0,1:4]=True;reference=self.zero.copy();reference[0,:3]=True
        result=score_union(reference,free,paid,self.domain)
        self.assertEqual(result["covered_reference_pixels"],3);self.assertEqual(result["incremental_covered_reference_pixels"],1)
        self.assertEqual(result["reference_false_positive_pixels"],1);self.assertEqual(result["predicted_pixels"],4)

    def test_zero_reference_recall_is_null(self):
        result=score_union(self.zero,self.zero,self.domain,self.domain)
        self.assertIsNone(result["area_recall"]);self.assertEqual(result["reference_precision"],0)

    def test_domain_exclusion_does_not_mutate_primary(self):
        reference=self.zero.copy();reference[0,:2]=True;domain=self.domain.copy();domain[0,0]=False
        self.assertEqual(score_union(reference,self.zero,reference,domain)["covered_reference_pixels"],1)
        self.assertEqual(score_union(reference,self.zero,reference,self.domain)["covered_reference_pixels"],2)

    def test_zero_reference_event_remains_in_macro(self):
        rows=[]
        for event in ['positive','zero']:
            ref=self.zero.copy()
            if event=='positive':ref[0,0]=True
            rows.append({"domain":"all","input_mode":NATIVE256,"budget":8,"policy":"random_exact","event":event,"sample_id":event,
                **score_union(ref,self.zero,ref,self.domain),"spent":8,"acquisitions":2,"affordable":True})
        result=aggregate(rows)["totals"][0]
        self.assertEqual(result["events"],2);self.assertEqual(result["event_macro_incremental_covered_pixels_per_scene"],.5)


if __name__=="__main__":unittest.main()
