import unittest

import numpy as np
from PIL import Image,ImageFilter

from external_footprint_core import BLOCKS
from external_quality256_core import (QUALITIES,blurred_inputs,canonical_summary,mosaic,measure_pair,quality_summary,measurement_decision)
from run_changeos_pilot import degrade_image,prepare_tensor


class Quality256Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.pair=tuple(Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)) for _ in range(2))
        self.zero=np.zeros((512,512),dtype=np.bool_)

    def test_blur_rgb_bytes_equal_frozen_helper_and_explicit_recipe(self):
        out=blurred_inputs(self.pair)
        for source,result in zip(self.pair,out):
            expected=source.filter(ImageFilter.GaussianBlur(1.5)).resize((128,128),Image.Resampling.BOX).resize((256,256),Image.Resampling.BILINEAR)
            self.assertEqual(result.tobytes(),degrade_image(source,"blurred").tobytes())
            self.assertEqual(result.tobytes(),expected.tobytes())

    def test_transformed_native_tensor_equals_old_blurred_tensor_without_double_blur(self):
        a=prepare_tensor(*blurred_inputs(self.pair),"native").numpy();b=prepare_tensor(*self.pair,"blurred").numpy()
        self.assertEqual(a.shape,(1,6,256,256));self.assertTrue(np.array_equal(a,b))

    def test_blurred_inputs_remain256_rgb_for_both_dates(self):
        for image in blurred_inputs(self.pair):self.assertEqual((image.size,image.mode),((256,256),"RGB"))

    def test_full_scene_and_non_rgb_cannot_enter_selected_block_helper(self):
        with self.assertRaises(ValueError):blurred_inputs(tuple(Image.new("RGB",(512,512)) for _ in range(2)))
        with self.assertRaises(ValueError):blurred_inputs((Image.new("L",(256,256)),self.pair[1]))

    def test_transform_does_not_mutate_purchased_original_pixels(self):
        before=[im.tobytes() for im in self.pair];out=blurred_inputs(self.pair);out[0].putpixel((0,0),(0,0,0))
        self.assertEqual(before,[im.tobytes() for im in self.pair])

    def test_unpurchased_pixels_do_not_change_selected_blur(self):
        a=Image.new("RGB",(512,512));a.paste(self.pair[0],(0,0));b=a.copy()
        b.paste(Image.new("RGB",(256,512),"white"),(256,0))
        selected_a=(a.crop((0,0,256,256)),self.pair[1]);selected_b=(b.crop((0,0,256,256)),self.pair[1])
        self.assertEqual(blurred_inputs(selected_a)[0].tobytes(),blurred_inputs(selected_b)[0].tobytes())

    def test_canonical_declaration_threshold_and_dtype(self):
        mask=np.zeros((256,256),dtype=np.bool_);mask[0,:3]=True;self.assertFalse(canonical_summary(mask)["declared_target"])
        mask[0,3]=True;self.assertTrue(canonical_summary(mask)["declared_target"])
        with self.assertRaises(ValueError):canonical_summary(mask.astype(np.uint8))

    def test_missing_or_wrong_size_prediction_cannot_form_full_mosaic(self):
        masks={b:np.zeros((256,256),dtype=np.bool_) for b in BLOCKS};del masks["b11"]
        with self.assertRaises(ValueError):mosaic(masks)
        masks["b11"]=np.zeros((128,128),dtype=np.bool_)
        with self.assertRaises(ValueError):mosaic(masks)

    def test_mosaic_maps_each_block_without_resize_or_overlap(self):
        masks={b:np.zeros((256,256),dtype=np.bool_) for b in BLOCKS}
        for i,b in enumerate(BLOCKS):masks[b][i,255-i]=True
        out=mosaic(masks);self.assertEqual(int(out.sum()),4)
        for i,b in enumerate(BLOCKS):self.assertTrue(out[int(b[1])*256+i,int(b[2])*256+255-i])

    def test_blur_gate_requires_both_reference_positive_events(self):
        rows=[]
        for domain in ["all_published_pixels","both_dates_nonzero"]:
            for event,found in [("turkey",0),("pakistan",5),("texas",0)]:
                rows.append({"domain":domain,"input_mode":QUALITIES[1],"event":event,"reference_pixels":10 if event!="texas" else 0,"incremental_covered_reference_pixels":found})
        result=measurement_decision({"event_rows":rows});self.assertFalse(result["primary_screen_passed"])
        for r in rows:
            if r["event"]=="turkey":r["incremental_covered_reference_pixels"]=1
        self.assertTrue(measurement_decision({"event_rows":rows})["primary_screen_passed"])

    def test_zero_reference_event_keeps_null_recall_and_event_denominator(self):
        rows=[]
        for event in ["a","b","zero"]:
            metrics,_=measure_pair(self.zero,self.zero,self.zero,self.zero,np.ones_like(self.zero))
            for mode,score in metrics.items():rows.append({"sample_id":event,"event":event,"domain":"all_published_pixels","input_mode":mode,**score})
        summary=quality_summary(rows)
        self.assertEqual(summary["totals"][0]["events"],3);self.assertIsNone(summary["totals"][0]["pooled_area_recall"])

    def test_free_pixels_deduplicated_and_gain_loss_conserves_coverage(self):
        reference=self.zero.copy();reference[0,:4]=True;free=self.zero.copy();free[0,0]=True
        native=self.zero.copy();native[0,:3]=True;blur=self.zero.copy();blur[0,2:4]=True
        scores,changes=measure_pair(native,blur,reference,free,np.ones_like(reference))
        self.assertEqual(scores[QUALITIES[0]]["free_covered_reference_pixels"],1)
        self.assertEqual(scores[QUALITIES[0]]["incremental_covered_reference_pixels"],2)
        self.assertEqual(scores[QUALITIES[1]]["covered_reference_pixels"]-scores[QUALITIES[0]]["covered_reference_pixels"],changes["gained_reference_pixels"]-changes["lost_reference_pixels"])


if __name__=="__main__":unittest.main()
