import unittest

import numpy as np
from PIL import Image

from external_area_core import CELLS,box
from external_information_core import PATCH_MODES,COARSE_MODE,BLANK_RGB,compose_inputs,coarse_canvas,canonical_mask,local_summary


class InformationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.patch=(Image.new("RGB",(128,128),(21,56,203)),Image.new("RGB",(128,128),(224,34,17)))
        self.coarse=(Image.fromarray(np.random.default_rng(51).integers(0,256,(64,64,3),dtype=np.uint8)),Image.new("RGB",(64,64),(65,79,92)))

    def test_all16_roi_positions_are_exact_without_native_context(self):
        for cell in CELLS:
            out=compose_inputs(PATCH_MODES[2],self.patch,cell,self.coarse)
            for im,patch in zip(out,self.patch):self.assertTrue(np.array_equal(np.asarray(im.crop(box(cell))),np.asarray(patch)))

    def test_outside_paid_roi_is_identical_to_free_coarse_only(self):
        cell="r02c01";x,y,xx,yy=box(cell);outside=np.ones((512,512),dtype=np.bool_);outside[y:yy,x:xx]=False
        base=coarse_canvas(self.coarse)
        for patches in [self.patch,tuple(Image.new("RGB",(128,128),"white") for _ in range(2))]:
            result=compose_inputs(PATCH_MODES[2],patches,cell,self.coarse)
            for im,original in zip(result,base):self.assertTrue(np.array_equal(np.asarray(im)[outside],np.asarray(original)[outside]))

    def test_blank_frame_is_fixed_and_contains_only_selected_patch(self):
        cell="r03c03";out=compose_inputs(PATCH_MODES[1],self.patch,cell)
        self.assertEqual(out[0].getpixel((0,0)),BLANK_RGB)
        self.assertEqual(out[1].getpixel((511,511)),self.patch[1].getpixel((127,127)))

    def test_upscale_retains_only_patch_information_and_no_context_argument(self):
        result=compose_inputs(PATCH_MODES[0],self.patch,"r01c02")
        self.assertEqual(result[0].size,(512,512));self.assertEqual(result[0].getpixel((511,511)),(21,56,203))
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[0],self.patch,"r01c02",self.coarse)

    def test_coarse_only_has_expected_bilinear_pixels_and_no_native_roi(self):
        result=coarse_canvas(self.coarse)
        self.assertTrue(np.array_equal(np.asarray(result[0]),np.asarray(self.coarse[0].resize((512,512),Image.Resampling.BILINEAR))))

    def test_full_native_scene_cannot_be_supplied_as_coarse_context(self):
        pair=tuple(Image.new("RGB",(512,512)) for _ in range(2))
        with self.assertRaises(ValueError):coarse_canvas(pair)
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[2],self.patch,"r00c00",pair)

    def test_full_native_scene_cannot_be_supplied_as_patch(self):
        pair=tuple(Image.new("RGB",(512,512)) for _ in range(2))
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[0],pair,"r00c00")

    def test_wrong_rgb_mode_and_missing_pair_stop(self):
        with self.assertRaises(ValueError):coarse_canvas((Image.new("L",(64,64)),self.coarse[1]))
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[0],[self.patch[0]],"r00c00")

    def test_blank_control_rejects_coarse_or_unplanned_mode(self):
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[1],self.patch,"r00c00",self.coarse)
        with self.assertRaises(ValueError):compose_inputs("whole_native",self.patch,"r00c00")
        with self.assertRaises(ValueError):compose_inputs(PATCH_MODES[0],self.patch,"r04c00")

    def test_nearest_downscale_recovers_block_expanded128_mask(self):
        mask=np.random.default_rng(64).random((128,128))>.9
        expanded=np.repeat(np.repeat(mask,4,axis=0),4,axis=1)
        self.assertTrue(np.array_equal(canonical_mask(expanded,PATCH_MODES[0],"r03c03"),mask))

    def test_only_roi_prediction_can_enter_paid_observation(self):
        raw=np.ones((512,512),dtype=np.bool_);raw[128:256,256:384]=False
        for mode in [PATCH_MODES[1],PATCH_MODES[2],COARSE_MODE]:
            result=canonical_mask(raw,mode,"r01c02")
            self.assertEqual(result.shape,(128,128));self.assertEqual(int(result.sum()),0)

    def test_noncanonical_prediction_shape_or_dtype_stops(self):
        with self.assertRaises(ValueError):canonical_mask(np.zeros((512,512),dtype=np.uint8),PATCH_MODES[2],"r00c00")
        with self.assertRaises(ValueError):canonical_mask(np.zeros((128,128),dtype=np.bool_),PATCH_MODES[0],"r00c00")

    def test_declaration_uses_canonical_roi_with_fixed_four_pixel_threshold(self):
        mask=np.zeros((128,128),dtype=np.bool_);mask[0,:3]=True
        self.assertFalse(local_summary(mask)["declared_target"])
        mask[0,3]=True;self.assertTrue(local_summary(mask)["declared_target"])

    def test_transform_does_not_mutate_free_images_or_patch(self):
        before=[im.tobytes() for im in self.patch+self.coarse]
        out=compose_inputs(PATCH_MODES[2],self.patch,"r02c03",self.coarse)
        out[0].putpixel((0,0),(0,0,0))
        self.assertEqual(before,[im.tobytes() for im in self.patch+self.coarse])


if __name__=="__main__":unittest.main()
