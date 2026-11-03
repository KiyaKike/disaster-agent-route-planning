"""Meaningful geometry, paid-evidence, propagation and modeled-budget checks."""
import copy
import math
import unittest

import numpy as np

from structure_map_core import (EvidenceMemory, local_enu, ecef, ring, suffix_cost,
                               execute, HOME, RETURN_MARGIN, OBSERVE, POLICIES)


def fixture():
    # Building0 crosses tile0/1. Building1 is unrelated in tile2.
    a=np.zeros((1024,1024),bool);a[0:10,118:138]=True
    b=np.zeros_like(a);b[0:10,256:266]=True
    parts=np.zeros((2,64),int);parts[0,:2]=100;parts[1,2]=100
    xy=np.array([[j%8*10,j//8*10] for j in range(64)]+[[35,80]],float)
    p={'pre_building_tile_pixels':parts.tolist(),'pre_building_areas_px':[200,100],
       'building_mass':(parts/np.array([200,100])[:,None]).sum(0).tolist(),
       'post_rank_utility':[.75]*64,'xy_m_reconstructed':xy.tolist()}
    return p,[a,b]


def pay(mem,tile,mask):
    history=[{'tile_id':r['tile_id'],'observation_complete_s_model':r['observed_at_s_model'],'observed':True} for r in mem.records]
    history.append({'tile_id':tile,'observation_complete_s_model':20*(len(history)+1),'observed':False})
    mem.ingest(tile,mask,history[-1]['observation_complete_s_model'],history)


class StructureTests(unittest.TestCase):
    def test_wgs84_equator_anchor(self):
        np.testing.assert_allclose(ecef([[0,0]])[0],[6378137,0,0],atol=1e-8)

    def test_local_enu_meter_scale(self):
        pts=local_enu([[0,0],[.001,0],[0,.001]],[0,0])
        self.assertAlmostEqual(pts[1,0],111.31949,places=4)
        self.assertAlmostEqual(pts[2,1],110.57428,places=4)

    def test_polygon_parser_rejects_unclosed(self):
        with self.assertRaises(ValueError):ring('POLYGON ((0 0,1 0,1 1,0 1))')

    def test_fractional_building_mass_is_not_duplicated(self):
        p,_=fixture();self.assertAlmostEqual(sum(p['building_mass']),2)

    def test_initial_evidence_does_not_change_prior(self):
        p,m=fixture();mem=EvidenceMemory(p,m)
        for policy in POLICIES:
            np.testing.assert_allclose(mem.weights(policy,'pre'),p['building_mass'])
        np.testing.assert_allclose(mem.weights('online_structure','post'),mem.weights('online_frozen','post'))

    def test_unpaid_mask_is_rejected(self):
        p,m=fixture();mem=EvidenceMemory(p,m)
        with self.assertRaises(ValueError):mem.ingest(0,np.zeros((128,128),bool),20,[])

    def test_duplicate_cache_is_not_independent_evidence(self):
        p,m=fixture();mem=EvidenceMemory(p,m);mask=np.zeros((128,128),bool);pay(mem,0,mask)
        with self.assertRaises(ValueError):pay(mem,0,mask)

    def test_same_building_signal_does_not_spread_to_unrelated_building(self):
        p,m=fixture();mem=EvidenceMemory(p,m);before=mem.weights('online_structure','post');pay(mem,0,np.zeros((128,128),bool))
        after=mem.weights('online_structure','post')
        self.assertLess(after[1],before[1]);self.assertEqual(after[2],before[2])

    def test_background_false_positive_does_not_enter_structure_signal(self):
        p,m=fixture();mem=EvidenceMemory(p,m);mask=np.zeros((128,128),bool);mask[60:80,30:50]=True;pay(mem,0,mask)
        self.assertEqual(sum(mem.records[0]['severe_pixels_on_pre_buildings']),0)
        self.assertEqual(mem.records[0]['predicted_severe_pixels'],400)

    def test_guard_retains_half_original_utility(self):
        p,m=fixture();mem=EvidenceMemory(p,m);base=mem.weights('online_frozen','post');pay(mem,0,np.zeros((128,128),bool))
        guard=mem.weights('online_structure_guard','post')
        self.assertTrue(np.all(guard>=.5*base));self.assertGreater(guard[1],mem.weights('online_structure','post')[1])

    def test_observation_coverage_is_bounded_and_no_prior_mutation(self):
        p,m=fixture();saved=copy.deepcopy(p);mem=EvidenceMemory(p,m);pay(mem,0,np.zeros((128,128),bool));pay(mem,1,np.zeros((128,128),bool))
        self.assertEqual(p,saved);self.assertEqual(mem.weights('online_structure','post')[0],0)

    def test_yaw_turn_cost_is_charged(self):
        p,_=fixture();xy=np.array(p['xy_m_reconstructed']);x=suffix_cost(xy,HOME,math.pi/2,[0]);y=suffix_cost(xy,HOME,-math.pi/2,[0])
        self.assertGreater(x,y)

    def test_every_policy_pays_observation_and_returns_with_margin(self):
        p,m=fixture()
        for policy in POLICIES:
            calls=[]
            def observe(tile,end):calls.append((tile,end));return np.zeros((128,128),bool)
            r=execute(p,m,policy,150,observe);s=r['state']
            self.assertEqual(len(calls),len(s['visited']));self.assertEqual(s['position'],HOME)
            self.assertTrue(s['closed']);self.assertLessEqual(s['elapsed_s_model']+RETURN_MARGIN,150+1e-8)
            self.assertTrue(all(abs(h['observation_complete_s_model']-h['arrival_s_model']-OBSERVE)<1e-8 for h in s['history']))
            self.assertEqual(len(s['plans'])>=2,policy!='pre_static')


if __name__=='__main__':unittest.main()
