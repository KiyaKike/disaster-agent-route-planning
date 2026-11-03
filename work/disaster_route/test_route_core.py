"""Meaningful invariant and information-boundary checks for closed route execution."""
import unittest
import numpy as np
from route_core import *


def grid(n=5):
    return public_map('test',[{'cell_id':f'c{r:02d}{c:02d}','area_m2':3600,'row':r,'col':c,
        'centroid_easting_m':c*60.,'centroid_northing_m':r*60.,'centroid_lon':c*.001,'centroid_lat':r*.001,
        'change_cue_score':(r+c)/(2*(n-1)) if n>1 else None,'true_damage':'poison'} for r in range(n) for c in range(n)])


class ClosedRouteTests(unittest.TestCase):
    def test_all_methods_charge_travel_observation_planning_and_return(self):
        g=grid();config=Config(budget_s=240,max_visits=30)
        for policy in POLICIES:
            o=Observer(np.zeros(len(g.ids),bool),(.8,.1));r=execute(g,policy,config,o)
            self.assertLessEqual(r['total_time_s'],240+1e-9)
            self.assertAlmostEqual(r['total_time_s'],r['movement_seconds']+r['observation_seconds']+r['simulated_planning_latency_s'])
            self.assertEqual(len(set(r['closed_route_ids'][1:-1])),r['visits'])
            self.assertEqual(r['closed_route_ids'][0],r['closed_route_ids'][-1]);self.assertEqual(o.accessed,[s['cell_index'] for s in r['steps']])

    def test_small_budget_yields_no_unpaid_observation(self):
        g=grid();o=Observer(np.ones(len(g.ids),bool),(1.,0.))
        r=execute(g,'online',Config(budget_s=20),o)
        self.assertEqual(r['visits'],0);self.assertEqual(o.accessed,[])

    def test_return_leg_can_make_inspection_infeasible(self):
        g=grid();visited=np.ones(len(g.ids),bool);visited[-1]=False
        route=greedy_route(g,g.xy[g.start_index],visited,0,g.prior_pre,Config(budget_s=75),0)
        self.assertEqual(route,[])

    def test_frozen_static_policies_ignore_observed_values(self):
        g=grid();config=Config(budget_s=300)
        for policy in ['pre_static','post_static','online_frozen']:
            a=execute(g,policy,config,Observer(np.zeros(len(g.ids),bool),(.8,.1)))
            b=execute(g,policy,config,Observer(np.ones(len(g.ids),bool),(.8,.1)))
            self.assertEqual(a['closed_route_ids'],b['closed_route_ids'])

    def test_first_action_does_not_access_future_signals(self):
        g=grid();a=execute(g,'online',Config(budget_s=240),Observer(np.zeros(len(g.ids),bool),(.8,.1)))
        b=execute(g,'online',Config(budget_s=240),Observer(np.ones(len(g.ids),bool),(.8,.1)))
        self.assertEqual(a['steps'][0]['cell_id'],b['steps'][0]['cell_id'])
        self.assertNotEqual(a['steps'][0]['block_belief_after'],b['steps'][0]['block_belief_after'])

    def test_bayes_positive_and_negative_updates_have_correct_direction(self):
        for noise in NOISE.values():
            self.assertGreater(update_belief(.2,True,*noise),.2)
            self.assertLess(update_belief(.2,False,*noise),.2)
        a=.2*.6/(.2*.6+.8*.02)
        self.assertAlmostEqual(update_belief(.2,True,1.,0.),a)

    def test_failed_detection_remains_censored_not_zero_time(self):
        g=grid();r=execute(g,'post_static',Config(budget_s=180),Observer(np.zeros(len(g.ids),bool),(.8,.1)))
        s=evaluate(r,np.ones(len(g.ids),bool))
        self.assertEqual(s['detected_target_cells'],0);self.assertIsNone(s['first_correct_detection_s'])
        self.assertEqual(s['first_detection_or_budget_s'],180);self.assertTrue(s['first_detection_censored'])

    def test_empty_world_recall_is_undefined(self):
        g=grid();r=execute(g,'pre_static',Config(budget_s=180),Observer(np.zeros(len(g.ids),bool),(1.,0.)))
        self.assertIsNone(evaluate(r,np.zeros(len(g.ids),bool))['detected_target_recall'])

    def test_worlds_and_observations_are_paired_reproducible(self):
        g=grid();a=simulate_targets(g,'reversed',123);b=simulate_targets(g,'reversed',123)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(signals(a,'perfect'),a['targets'])
        self.assertEqual(sorted(a['true_hot_probabilities']),sorted(g.prior_post))

    def test_public_map_projection_cannot_include_damage_truth(self):
        g=grid();self.assertFalse(hasattr(g,'targets'));self.assertFalse(hasattr(g,'true_damage'))
        self.assertFalse(g.xy.flags.writeable)

    def test_invalid_budget_likelihood_or_belief_rejected(self):
        for value in [0,float('nan'),-1]:
            with self.assertRaises(ValueError):Config(budget_s=value)
        with self.assertRaises(ValueError):update_belief(.2,True,.1,.2)
        g=grid()
        with self.assertRaises(ValueError):greedy_route(g,g.xy[0],np.zeros(len(g.ids),bool),0,np.ones(len(g.block_ids))*2,Config(),0)

    def test_no_region_crossing_and_visits_never_exceed_cap(self):
        g=grid();r=execute(g,'online',Config(max_visits=3),Observer(np.ones(len(g.ids),bool),(.8,.1)))
        self.assertLessEqual(r['visits'],3);self.assertTrue(set(r['closed_route_ids'])<=set(g.ids))


if __name__=='__main__':unittest.main()
