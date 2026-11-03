import copy
import json
import unittest

import numpy as np

from route_factor_core import Planner,execute
from route_core import Config,Observer
from route_snapshot_core import Session,PrivateWorld,BRANCHES,pack,finish_branch,score_branch,PUBLIC_KEYS
from test_route_factor import grid


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.g=grid(8)
        self.world=PrivateWorld('private-test',np.arange(64)%5==0,np.arange(64)%4==0,(.8,.1))
        self.registry={self.world.token:self.world}

    def prefix(self,k=5):
        s=Session(self.g,self.world);s.plan(Planner(),'prior')
        for _ in range(k):s.step()
        return s

    def test_json_roundtrip_has_identical_remaining_execution(self):
        a=self.prefix();snap=json.loads(json.dumps(a.snapshot()))
        b=Session.restore(self.g,snap,self.registry)
        while a.state['remaining_route']:
            self.assertEqual(a.step(),b.step())
        a.close();b.close();self.assertEqual(a.public(),b.public())

    def test_restore_does_not_reobserve_or_recharge_prefix(self):
        a=self.prefix();b=Session.restore(self.g,a.snapshot(),self.registry)
        self.assertEqual(b.state['elapsed_s'],a.state['elapsed_s'])
        self.assertEqual(b.environment.seen,a.environment.seen)
        before=len(b.environment.seen);b.step();self.assertEqual(len(b.environment.seen),before+1)

    def test_snapshot_and_public_copies_cannot_mutate_live_session(self):
        s=self.prefix();before=s.public();snap=s.snapshot();public=s.public()
        snap['payload']['public']['belief_hot'][0]=.999;public['visited_indices'].clear()
        self.assertEqual(s.public(),before)

    def test_corrupt_snapshot_rejected(self):
        snap=self.prefix().snapshot();snap['payload']['public']['elapsed_s']+=2
        with self.assertRaises(ValueError):Session.restore(self.g,snap,self.registry)

    def test_rehashed_invalid_cost_ledger_rejected(self):
        snap=self.prefix().snapshot();p=snap['payload']['public'];p['elapsed_s']+=2
        with self.assertRaises(ValueError):Session.restore(self.g,pack(p,snap['payload']['environment']),self.registry)

    def test_changed_private_world_rejected(self):
        snap=self.prefix().snapshot();changed=PrivateWorld(self.world.token,~self.world.targets,self.world.observations,self.world.likelihood)
        with self.assertRaises(ValueError):Session.restore(self.g,snap,{changed.token:changed})

    def test_no_future_or_truth_fields_in_public_view_or_step(self):
        s=self.prefix();self.assertEqual(set(s.public()),PUBLIC_KEYS)
        for forbidden in ('targets','observations','world_token','world_sha256','seed','reward'):
            self.assertNotIn(forbidden,s.public());self.assertNotIn(forbidden,s.state['history'][-1])
        snap=s.snapshot();p=snap['payload']['public'];p['targets']=[False]*64
        with self.assertRaises(ValueError):Session.restore(self.g,pack(p,snap['payload']['environment']),self.registry)

    def test_private_forks_have_independent_seen_cursors(self):
        snap=self.prefix().snapshot();a=Session.restore(self.g,snap,self.registry);b=Session.restore(self.g,snap,self.registry)
        original=b.public();a.step();self.assertEqual(b.public(),original);self.assertNotEqual(a.environment.seen,b.environment.seen)

    def test_shared_future_cells_return_same_bit_in_different_branches(self):
        snap=self.prefix().snapshot();a=Session.restore(self.g,snap,self.registry);b=Session.restore(self.g,snap,self.registry)
        target=a.state['remaining_route'][0];a.set_route([target]);b.set_route([target])
        self.assertEqual(a.step()['observed_positive'],b.step()['observed_positive'])

    def test_continue_matches_original_static_motion_and_bits(self):
        r=execute(self.g,'post_static',Config(max_visits=64),Observer(self.world.observations,self.world.likelihood),'insertion_2opt',Planner())
        s=self.prefix();result=finish_branch(self.g,s.snapshot(),self.registry,'continue',Planner());final=result['final_public']
        self.assertEqual(final['visited_indices'],[h['cell_index'] for h in r['steps']])
        for a,b in zip(final['history'],r['steps']):
            for key in ('cell_id','elapsed_before_step_s','leg_m','arrival_s','observation_complete_s','observed_positive'):
                self.assertEqual(a[key],b[key])
        self.assertEqual(final['elapsed_s'],r['total_time_s']);self.assertEqual(final['distance_m'],r['total_distance_m'])

    def test_all_branches_preserve_exact_prefix_and_charge_planning(self):
        s=self.prefix();snap=s.snapshot()
        for branch in BRANCHES:
            r=finish_branch(self.g,snap,self.registry,branch,Planner());final=r['final_public']
            self.assertEqual(final['history'][:5],s.state['history'])
            self.assertEqual(final['planning_events'][:1],s.state['planning_events'])
            self.assertEqual(len(final['planning_events']),1+int(branch!='continue'))
            self.assertAlmostEqual(final['elapsed_s'],final['distance_m']/5+20*len(final['history'])+len(final['planning_events']))
            self.assertLessEqual(final['elapsed_s'],1200+1e-8)

    def test_resume_after_new_plan_and_after_return(self):
        s=self.prefix();s.plan(Planner(),'feedback');s.step()
        restored=Session.restore(self.g,s.snapshot(),self.registry)
        while restored.state['remaining_route']:restored.step()
        restored.close();again=Session.restore(self.g,restored.snapshot(),self.registry)
        self.assertEqual(again.public(),restored.public())
        with self.assertRaises(ValueError):again.close()
        with self.assertRaises(ValueError):again.step()

    def test_repeated_or_infeasible_route_rejected_without_observation(self):
        s=self.prefix();before=s.public();seen=s.environment.seen.copy()
        with self.assertRaises(ValueError):s.set_route([s.state['visited_indices'][0]])
        with self.assertRaises(ValueError):s.set_route(list(range(64)))
        self.assertEqual(before,s.public());self.assertEqual(seen,s.environment.seen)

    def test_future_only_metrics_do_not_count_prefix_detections(self):
        s=self.prefix();snap=s.snapshot();r=finish_branch(self.g,snap,self.registry,'continue',Planner());score=score_branch(r,self.world)
        prefix=sum(self.world.targets[h['cell_index']] and h['observed_positive'] for h in s.state['history'])
        self.assertEqual(score['total_detected_target_cells'],prefix+score['future_detected_target_cells'])
        self.assertAlmostEqual(score['suffix_elapsed_s'],r['final_public']['elapsed_s']-s.state['elapsed_s'])

    def test_no_discovery_retains_null_and_remaining_budget_penalty(self):
        world=PrivateWorld('empty',np.zeros(64,bool),np.zeros(64,bool),(.8,.1));s=Session(self.g,world);s.plan(Planner(),'prior')
        for _ in range(5):s.step()
        r=finish_branch(self.g,s.snapshot(),{world.token:world},'continue',Planner());score=score_branch(r,world)
        self.assertIsNone(score['next_detection_delay_s']);self.assertTrue(score['no_future_detection'])
        self.assertEqual(score['next_detection_or_remaining_budget_s'],1200-s.state['elapsed_s'])

    def test_same_paid_prefix_and_private_truth_change_cannot_change_plan(self):
        s=self.prefix();snap=s.snapshot();changed=self.world.targets.copy();changed[20:]=~changed[20:]
        world=PrivateWorld('different-truth',changed,self.world.observations,self.world.likelihood)
        alt=pack(snap['payload']['public'],{'world_token':world.token,'world_sha256':world.sha256,'seen':snap['payload']['environment']['seen']})
        a=finish_branch(self.g,snap,self.registry,'replan_feedback',Planner())
        b=finish_branch(self.g,alt,{world.token:world},'replan_feedback',Planner())
        self.assertEqual(a['planned_suffix_indices'],b['planned_suffix_indices'])


if __name__=='__main__':unittest.main()
