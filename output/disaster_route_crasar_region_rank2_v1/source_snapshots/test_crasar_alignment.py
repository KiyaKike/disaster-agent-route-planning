import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gyirong_imagery_search/geo_libs'))
import numpy as np
from affine import Affine
from crasar_alignment_core import nearest_shift_indices,corrected_image_transform

class Alignment(unittest.TestCase):
    def test_polygon_to_image_direction(self):
        t=Affine(.5,0,100,0,-.5,500);shift=np.array([10.,-3.]);p=np.array([32.,77.])
        corrected=corrected_image_transform(t,t,shift)
        np.testing.assert_allclose(corrected*tuple(p+shift),t*tuple(p))
    def test_overview_anisotropic_rounding(self):
        t=Affine(.02,0,100,0,-.02,500);over=t*Affine.scale(7.999,8.001)
        p=np.array([323.,777.]);shift=np.array([-7.,72.])
        at_image=(~over)*(t*tuple(p+shift))
        np.testing.assert_allclose(corrected_image_transform(t,over,shift)*at_image,t*tuple(p))
    def test_local_vectors_follow_nearest_source_position(self):
        idx=nearest_shift_indices([[0,0],[100,0]],np.array([[10,90],[20,80]]),np.zeros((2,2)))
        np.testing.assert_array_equal(idx,[[0,1],[0,1]])

if __name__=='__main__':unittest.main()
