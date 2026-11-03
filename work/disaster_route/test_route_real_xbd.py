import unittest
import numpy as np
from real_xbd_route_core import POLICIES,BUDGETS,HOME,xy_grid,public_features,utility_from_evidence,execute,evaluate


class RealRouteTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(302);self.pre=rng.integers(0,256,(64,64,3),dtype=np.uint8)
        self.post=rng.integers(0,256,(64,64,3),dtype=np.uint8);self.f=public_features(self.pre,self.post)

    def test_only_coarse_public_input_is_accepted(self):
        with self.assertRaises(ValueError):public_features(np.zeros((1024,1024,3),np.uint8),self.post)
        self.assertTrue(all(.05<=x<=1 for x in self.f['pre_utility']+self.f['post_utility']))

    def test_post_pixels_cannot_change_pre_prior(self):
        other=public_features(self.pre,np.zeros_like(self.post))
        self.assertEqual(other['pre_utility'],self.f['pre_utility'])

    def test_neighbor_feedback_only_changes_adjacent_unvisited_scores(self):
        base=np.full(16,.5);changed=utility_from_evidence(base,[{'tile_id':0,'predicted_severe_pixels':1}])
        self.assertEqual(set(np.flatnonzero(changed!=base)),{1,4});self.assertEqual(changed[1],.75)

    def test_no_evidence_leaves_prior_intact(self):
        np.testing.assert_array_equal(utility_from_evidence(self.f['post_utility'],[]),self.f['post_utility'])

    def test_all_policies_include_travel_observation_planning_and_return(self):
        for budget in BUDGETS:
            for policy in POLICIES:
                seen=[]
                def observe(i,t):seen.append((i,t));return 0
                r=execute(self.f,policy,budget,observe);s=r['state']
                self.assertEqual([i for i,t in seen],s['visited']);self.assertEqual(len(seen),len(set(i for i,t in seen)))
                self.assertTrue(s['closed']);self.assertEqual(s['position_index'],HOME)
                self.assertAlmostEqual(s['elapsed_s_proxy'],s['distance_grid']*20+len(seen)*20+len(s['planning_events']))
                self.assertLessEqual(s['elapsed_s_proxy'],budget+1e-8)
                self.assertEqual([t for i,t in seen],[h['observation_complete_s_proxy'] for h in s['history']])

    def test_every_stored_suffix_returns_within_budget_without_revisits(self):
        r=execute(self.f,'online_neighbor',240,lambda i,t:i);xy=xy_grid();s=r['state']
        for e in s['planning_events']:
            old=s['visited'][:e['after_observations']];route=e['planned_route'];self.assertFalse(set(route)&set(old))
            nodes=[e['position_index'],*route,HOME];distance=sum(np.linalg.norm(xy[a]-xy[b]) for a,b in zip(nodes,nodes[1:]))
            self.assertLessEqual(e['elapsed_after_s_proxy']+distance*20+len(route)*20,240+1e-8)

    def test_static_and_no_feedback_paths_ignore_detector_outputs(self):
        for policy in ('pre_static','post_static','online_frozen'):
            a=execute(self.f,policy,240,lambda i,t:0);b=execute(self.f,policy,240,lambda i,t:65536)
            self.assertEqual(a['state']['visited'],b['state']['visited'])

    def test_first_actions_do_not_know_future_detector_outputs(self):
        a=execute(self.f,'online_neighbor',240,lambda i,t:0);b=execute(self.f,'online_neighbor',240,lambda i,t:65536)
        self.assertEqual(a['state']['history'][0]['tile_id'],b['state']['history'][0]['tile_id'])

    def test_reference_scores_do_not_change_routes_and_zero_denominators_remain_null(self):
        route=execute(self.f,'online_neighbor',240,lambda i,t:0);before=list(route['state']['visited'])
        score=evaluate(route,[{'reference':0,'tp':0,'fp':0}]*16)
        self.assertIsNone(score['recall']);self.assertIsNone(score['precision']);self.assertIsNone(score['first_correct_detection_s_proxy'])
        self.assertEqual(score['first_detection_or_budget_s_proxy'],240);self.assertEqual(before,route['state']['visited'])

    def test_unvisited_reference_is_still_in_recall_denominator(self):
        route=execute(self.f,'post_static',120,lambda i,t:0);counts=[{'reference':10,'tp':5,'fp':1} for i in range(16)]
        score=evaluate(route,counts)
        self.assertEqual(score['reference_severe_pixels'],160)
        self.assertEqual(score['true_positive_pixels'],5*len(route['state']['visited']))


if __name__=='__main__':unittest.main()
