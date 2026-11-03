import copy
import json
import unittest
from dataclasses import replace

import numpy as np

from dual_map_core import DualMapAgent,DualMapMemory,MODES,POLICIES,planner_inputs
from route_factor_core import Planner
from route_snapshot_core import PrivateWorld
from test_route_factor import grid


class DualMapTests(unittest.TestCase):
    def setUp(self):
        self.g=grid(8);rng=np.random.default_rng(99)
        self.w=PrivateWorld('dual-test',rng.random(64)<.2,rng.random(64)<.25,(.8,.1))

    def agent(self,mode='prior_evidence',policy='online_5'):
        return DualMapAgent(self.g,self.w,mode,policy,Planner())

    def test_pre_phase_has_no_post_cues_and_no_paid_evidence(self):
        a=self.agent();v=a.view()
        self.assertEqual(v['phase'],'pre')
        self.assertTrue(all(x is None for x in v['layers']['prior_cues']['change_cue']))
        self.assertEqual(v['layers']['observation_evidence']['paid_records'],[])
        np.testing.assert_array_equal(v['fusion']['block_hot_posterior'],self.g.prior_pre)

    def test_post_cues_change_prior_without_manufacturing_observations(self):
        a=self.agent();a.prepare()
        self.assertEqual(a.maps.phase,'post');self.assertEqual(a.maps.records,[])
        np.testing.assert_array_equal(a.maps.prior(),self.g.prior_post)
        self.assertEqual(a.session.state['elapsed_s'],2)
        self.assertEqual(len(a.session.state['planning_events']),2)

    def test_online_observation_keeps_prior_layer_unchanged(self):
        a=self.agent();a.prepare();before=a.view()['layers']['prior_cues']
        for _ in range(5):a.advance()
        self.assertEqual(a.view()['layers']['prior_cues'],before)
        ev=a.view()['layers']['observation_evidence']
        self.assertEqual(sum(x is not None for x in ev['signals']),5)
        self.assertIsNone(ev['confirmed_targets'])
        np.testing.assert_allclose(a.maps.posterior(),a.session.state['belief_hot'],atol=1e-12,rtol=0)

    def test_repeated_ingest_is_idempotent_and_conflicting_history_rejected(self):
        a=self.agent();a.prepare();a.advance();before=a.maps.snapshot()
        a.maps.ingest(a.session.public());self.assertEqual(before,a.maps.snapshot())
        changed=a.session.public();changed['history'][0]['observed_positive']=not changed['history'][0]['observed_positive']
        with self.assertRaises(ValueError):a.maps.ingest(changed)
        self.assertEqual(before,a.maps.snapshot())

    def test_unpaid_or_misregistered_observation_cannot_enter_map(self):
        a=self.agent();a.prepare();a.advance()
        for key,value in [('observation_complete_s',9999),('cell_id','wrong-cell')]:
            state=a.session.public();state['history'][0][key]=value
            m=DualMapMemory(self.g,(.8,.1));m.receive_post_cues()
            with self.assertRaises(ValueError):m.ingest(state)
            self.assertEqual(m.records,[])

    def test_post_source_cannot_be_applied_twice_or_after_evidence(self):
        m=DualMapMemory(self.g,(.8,.1));m.receive_post_cues()
        with self.assertRaises(ValueError):m.receive_post_cues()
        a=self.agent();a.prepare();a.advance()
        with self.assertRaises(ValueError):a.maps.receive_post_cues()

    def test_two_interfaces_provide_identical_solver_inputs(self):
        a=self.agent();a.prepare()
        for _ in range(7):a.advance()
        x=[planner_inputs(self.g,a.maps.export(mode,a.session.public()),a.session.public()) for mode in MODES]
        for a_,b_ in zip(*x):np.testing.assert_array_equal(a_,b_)

    def test_geometry_view_does_not_claim_unknown_passability(self):
        a=self.agent('geometry_semantic');a.prepare()
        geo=a.view()['layers']['geometry_cost']
        self.assertIsNone(geo['road_passability']);self.assertIsNone(geo['terrain_obstacles'])
        for _ in range(2):a.advance()
        geo=a.view()['layers']['geometry_cost']
        self.assertTrue(all(not geo['budget_feasible_after_planning'][i] for i in a.session.state['visited_indices']))

    def test_all_full_policy_routes_equal_between_map_organizations(self):
        for policy in POLICIES:
            a=self.agent(MODES[0],policy).finish();b=self.agent(MODES[1],policy).finish()
            for key in ('final_public','maps','frames','skipped_unaffordable_replans'):self.assertEqual(a[key],b[key])

    def test_hidden_target_truth_cannot_change_maps_or_actions(self):
        changed=PrivateWorld('changed-truth',~self.w.targets,self.w.observations,self.w.likelihood)
        for mode in MODES:
            a=self.agent(mode).finish();b=DualMapAgent(self.g,changed,mode,'online_5',Planner()).finish()
            self.assertEqual(a,b)

    def test_snapshot_restores_layers_cursor_and_future_route_without_recharge(self):
        for mode in MODES:
            a=self.agent(mode);a.prepare()
            for _ in range(10):a.advance()
            snap=json.loads(json.dumps(a.snapshot()))
            b=DualMapAgent.restore(self.g,snap,{self.w.token:self.w},Planner())
            self.assertEqual(a.session.public(),b.session.public());self.assertEqual(a.maps.snapshot(),b.maps.snapshot())
            self.assertEqual(a.finish(),b.finish())

    def test_pre_snapshot_restore_and_corruption_or_changed_geometry_rejected(self):
        a=self.agent();snap=json.loads(json.dumps(a.snapshot()))
        b=DualMapAgent.restore(self.g,snap,{self.w.token:self.w},Planner());self.assertEqual(a.finish(),b.finish())
        corrupt=copy.deepcopy(snap);corrupt['payload']['maps']['phase']='post'
        with self.assertRaises(ValueError):DualMapAgent.restore(self.g,corrupt,{self.w.token:self.w},Planner())
        shifted=self.g.xy.copy();shifted[:,0]+=1
        with self.assertRaises(ValueError):DualMapAgent.restore(replace(self.g,xy=shifted),snap,{self.w.token:self.w},Planner())

    def test_every_policy_budget_includes_pre_post_planning_and_return(self):
        for mode in MODES:
            for policy in POLICIES:
                r=self.agent(mode,policy).finish();s=r['final_public']
                self.assertTrue(s['closed']);self.assertLessEqual(s['elapsed_s'],1200+1e-8)
                self.assertEqual(len(s['history']),len(r['maps']['records']))
                self.assertAlmostEqual(s['elapsed_s'],s['distance_m']/5+len(s['history'])*20+len(s['planning_events']))
                self.assertEqual(sum(e['reason']=='pre_plan' for e in s['planning_events']),1)
                self.assertEqual(sum(e['reason']=='post_plan' for e in s['planning_events']),int(policy!='pre_static'))

    def test_unvisited_map_has_no_future_bit_or_private_world_reference(self):
        a=self.agent();a.prepare();a.advance();v=a.view()
        text=json.dumps(v)
        for key in ('world_token','world_sha256','uniform_observation_draws','target_ids'):
            self.assertNotIn(key,text)
        signals=v['layers']['observation_evidence']['signals']
        seen=set(a.session.state['visited_indices'])
        self.assertTrue(all(signals[i] is None for i in range(64) if i not in seen))


if __name__=='__main__':unittest.main()
