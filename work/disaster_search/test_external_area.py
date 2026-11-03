import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from extract_external_area_masks import decode_reference,MaskRangeReader
from external_area_core import (CELLS,box,crop_cell,public_features,BinaryEnvironment,
    run_policy,evaluate_trace,exact_random_survey)
from cross_cell_count_smoke import quality


def png(array):
    stream=io.BytesIO();Image.fromarray(array).save(stream,format="PNG");return stream.getvalue()


class ReferenceSchemaTests(unittest.TestCase):
    def test_released_ids_and_zero_target_remain_valid(self):
        a=np.zeros((512,512),dtype=np.uint8);a[1,:5]=range(5)
        b,mode=decode_reference(png(a));self.assertTrue(np.array_equal(a,b));self.assertEqual(mode,"L")
        b,_=decode_reference(png(np.zeros_like(a)));self.assertEqual(int((b>=3).sum()),0)

    def test_unknown255_stops_without_becoming_background(self):
        with self.assertRaises(ValueError):decode_reference(png(np.full((512,512),255,dtype=np.uint8)))

    def test_color_values_cannot_be_guessed_as_classes(self):
        a=np.zeros((512,512,3),dtype=np.uint8);a[:,:,0]=3
        with self.assertRaises(ValueError):decode_reference(png(a))

    def test_replicated_integer_channels_are_lossless(self):
        a=np.full((512,512),4,dtype=np.uint8)
        b,mode=decode_reference(png(np.repeat(a[:,:,None],3,axis=2)))
        self.assertTrue(np.array_equal(a,b));self.assertIn("identical_channels",mode)

    def test_wrong_shape_stops(self):
        with self.assertRaises(ValueError):decode_reference(png(np.zeros((128,128),dtype=np.uint8)))

    def test_palette_indices_preserved_without_rgb_conversion(self):
        im=Image.new("P",(512,512),3);im.putpalette([v for i in range(256) for v in [i,i,i]])
        stream=io.BytesIO();im.save(stream,format="PNG",bits=8)
        b,mode=decode_reference(stream.getvalue());self.assertEqual(mode,"P");self.assertEqual(int(b.min()),3)


class MaskRangeBoundaryTests(unittest.TestCase):
    def member(self):
        return {"path":"E/masks/E_1_post_disaster.png","archive_url":"https://example.invalid/a.zip",
            "archive_size":1000,"local_header_offset":100,"next_header_or_directory":500,"compressed_bytes":60}

    def test_rgb_role_or_neighbor_access_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp,patch("extract_external_area_masks.urlopen") as network:
            reader=MaskRangeReader(Path(tmp))
            for offset,count,role in [(100,30,"selected_rgb_compressed"),(500,30,"reference_header30"),(101,30,"reference_header30")]:
                with self.assertRaises(ValueError):reader.read(self.member(),offset,count,role)
            network.assert_not_called()

    def test_full_response_body_is_never_read(self):
        with tempfile.TemporaryDirectory() as tmp,patch("extract_external_area_masks.urlopen") as network:
            response=network.return_value.__enter__.return_value;response.status=200;response.headers={}
            reader=MaskRangeReader(Path(tmp))
            with self.assertRaises(ValueError):reader.read(self.member(),100,30,"reference_header30")
            response.read.assert_not_called();self.assertEqual(reader.ledger[0]["received"],0)


class AreaDecisionTests(unittest.TestCase):
    def masks(self):return {(c,m):np.zeros((128,128),dtype=np.bool_) for c in CELLS for m in ["native","blurred"]}

    def test_four_by_four_crops_keep_native128px_context(self):
        a=np.zeros((512,512,3),dtype=np.uint8)
        for i,c in enumerate(CELLS):
            x0,y0,x1,y1=box(c);a[y0:y1,x0:x1]=i+1
        im=Image.fromarray(a)
        for i,c in enumerate(CELLS):
            pre,post=crop_cell(im,im,c);self.assertEqual(pre.size,(128,128));self.assertTrue(np.all(np.asarray(post)==i+1))
        with self.assertRaises(ValueError):box("r04c00")

    def test_public_feature_scale_and_blank_regions_are_not_filtered(self):
        features=public_features(Image.new("RGB",(512,512)),Image.new("RGB",(512,512),"white"))
        self.assertEqual(len(features),16)
        self.assertTrue(all(r["change"]==1 and r["coarse_nonzero_fraction"]==0 for r in features.values()))

    def test_pixel_union_does_not_double_count_upgrade_overlap(self):
        reference=np.zeros((512,512),dtype=np.bool_);reference[0,:3]=True;masks=self.masks()
        masks[CELLS[0],"blurred"][0,:3]=True;masks[CELLS[0],"native"][0,1:5]=True
        trace=[{"cell_id":CELLS[0],"mode":"blurred","kind":"survey","cumulative_cost":1},
               {"cell_id":CELLS[0],"mode":"native","kind":"upgrade","cumulative_cost":3}]
        row=evaluate_trace(trace,reference,masks)
        self.assertEqual(row["covered_reference_pixels"],3);self.assertEqual(row["reference_false_positive_pixels"],2)
        self.assertEqual(row["area_recall"],1)
        with self.assertRaises(ValueError):evaluate_trace(trace+[trace[-1]],reference,masks)

    def test_zero_target_recall_null_with_false_positive_area(self):
        reference=np.zeros((512,512),dtype=np.bool_);masks=self.masks();masks[CELLS[0],"native"][0,0]=True
        trace=[{"cell_id":CELLS[0],"mode":"native","kind":"survey","cumulative_cost":1}]
        row=evaluate_trace(trace,reference,masks);self.assertIsNone(row["area_recall"]);self.assertEqual(row["reference_false_positive_pixels"],1)

    def test_boolean_threshold_does_not_erase_subfour_pixel_area(self):
        reference=np.zeros((512,512),dtype=np.bool_);reference[0,0]=True;masks=self.masks();masks[CELLS[0],"native"][0,0]=True
        trace=[{"cell_id":CELLS[0],"mode":"native","kind":"survey","cumulative_cost":1,"declared_target":False}]
        self.assertEqual(evaluate_trace(trace,reference,masks)["covered_reference_pixels"],1)

    def test_negative_gate_spends_upgrade_cost_and_cannot_peek(self):
        sid="synthetic";seed=next(s for s in range(100) if quality("mixed_survey",s,sid,CELLS[0])=="blurred")
        features={c:{"change":1. if c==CELLS[0] else 0.} for c in CELLS}
        booleans={(c,m):False for c in CELLS for m in ["native","blurred"]}
        trace=run_policy(features,BinaryEnvironment(booleans,sid,seed),"prior_gate",3,1)
        self.assertEqual([a["kind"] for a in trace],["survey","upgrade"]);self.assertEqual(trace[-1]["cumulative_cost"],3)
        booleans[CELLS[0],"blurred"]=True
        trace=run_policy(features,BinaryEnvironment(booleans,sid,seed),"prior_gate",3,1)
        self.assertTrue(all(a["kind"]=="survey" for a in trace))

    def test_random_has_exactly_budget_distinct_surveys(self):
        booleans={(c,m):False for c in CELLS for m in ["native","blurred"]};features={c:{"change":0.} for c in CELLS}
        trace=run_policy(features,BinaryEnvironment(booleans,"s",1),"random_survey",8,3)
        self.assertEqual(len(trace),8);self.assertEqual(len({a["cell_id"] for a in trace}),8)

    def test_exact_random_expectation_matches_exhaustive_cell_sampling(self):
        from itertools import combinations
        ref=np.zeros((512,512),dtype=np.bool_);masks=self.masks();target_cells={CELLS[0],CELLS[5]}
        for c in target_cells:
            x,y,_,_=box(c);ref[y,x]=True
            for mode in ["native","blurred"]:masks[c,mode][0,0]=True
        exhaustive=sum(len(set(pair)&target_cells) for pair in combinations(CELLS,2))/120
        row=exact_random_survey(ref,masks,"s",1,2)
        self.assertEqual(row["expected_covered_reference_pixels"],exhaustive)


class AreaAggregationTests(unittest.TestCase):
    def test_target_free_event_stays_in_pixel_macro_but_recall_is_null(self):
        from run_external_area_controls import aggregate
        rows=[]
        for event,ref,found in [("A",10,5),("B",0,0)]:
            rows.append({"sample_id":event,"event":event,"domain":"all_published_pixels","budget":8,"policy":"p",
                "reference_pixels":ref,"covered_reference_pixels":found,"reference_false_positive_pixels":2,
                "surveys":8,"upgrades":0,"spent":8,"area_recall":found/ref if ref else None})
        summary=aggregate(rows);macro=summary["event_macro_rows"][0]
        self.assertEqual(macro["covered_reference_pixels"],2.5);self.assertEqual(macro["target_events"],1)
        self.assertEqual(macro["mean_pooled_recall_target_events"],.5)
        self.assertIsNone(next(r for r in summary["event_rows"] if r["event"]=="B")["pooled_area_recall"])

    def test_scene_pixel_denominator_includes_zero_target_scenes(self):
        from run_external_area_controls import aggregate
        rows=[]
        for sid,ref,found in [("a",10,5),("b",0,0)]:
            rows.append({"sample_id":sid,"event":"A","domain":"all_published_pixels","budget":8,"policy":"p",
                "reference_pixels":ref,"covered_reference_pixels":found,"reference_false_positive_pixels":0,
                "surveys":8,"upgrades":0,"spent":8,"area_recall":found/ref if ref else None})
        event=aggregate(rows)["event_rows"][0]
        self.assertEqual(event["scenes"],2);self.assertEqual(event["reference_positive_scenes"],1)
        self.assertEqual(event["covered_reference_pixels"],2.5);self.assertEqual(event["pooled_area_recall"],.5)


if __name__=="__main__":unittest.main()
