import unittest
import numpy as np
from crasar_feedback_diagnostic import assert_separate,feedback_features,ridge_fit_predict,box_distance

class Diagnostic(unittest.TestCase):
    def test_shared_building_is_not_unvisited_evidence(self):
        with self.assertRaises(ValueError):assert_separate(1,[0],[['shared'],['shared'],['other']])
        assert_separate(2,[0],[['shared'],['shared'],['other']])
    def test_future_or_missing_feedback_is_rejected(self):
        prior=np.zeros((3,4));xy=np.array([[0,0],[32,0],[64,0]])
        with self.assertRaises(ValueError):feedback_features(2,[0],prior,{0:.1,1:.9},xy)
        with self.assertRaises(ValueError):feedback_features(2,[0],prior,{},xy)
        with self.assertRaises(ValueError):feedback_features(0,[0],prior,{0:.1},xy)
    def test_features_do_not_consult_unvisited_values(self):
        prior=np.zeros((3,4));xy=np.array([[0,0],[32,0],[64,0]])
        actual=feedback_features(2,[0,1],prior,{0:.2,1:.8},xy)
        np.testing.assert_allclose(actual,[.5,.5,.56,.8])
    def test_test_batch_does_not_change_training_scaling(self):
        x=np.array([[0,0],[1,1],[2,0],[3,1]],float);y=np.array([0,.2,.8,1]);w=np.ones(4)
        first=ridge_fit_predict(x,y,np.array([[1.5,.5]]),w)
        together=ridge_fit_predict(x,y,np.array([[1.5,.5],[1000,-1000]]),w)
        self.assertAlmostEqual(first[0],together[0])
    def test_spatial_purge_distance_uses_rectangle_gap(self):
        self.assertEqual(box_distance([0,0,128,128],[64,64,192,192]),0)
        self.assertEqual(box_distance([0,0,128,128],[256,0,384,128]),128)
        self.assertAlmostEqual(box_distance([0,0,128,128],[131,132,140,140]),5)

if __name__=='__main__':unittest.main()
