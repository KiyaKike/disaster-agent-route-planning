import unittest
import numpy as np

from route_core import Config, Observer, POLICIES, execute as old_execute, public_map, greedy_route
from route_factor_core import Planner, execute, path_distance, two_opt


def grid(n=7, region='test'):
    return public_map(region, [{'cell_id': f'c{r:02d}{c:02d}', 'area_m2': 3600, 'row': r, 'col': c,
        'centroid_easting_m': c*60., 'centroid_northing_m': r*60., 'centroid_lon': c*.001, 'centroid_lat': r*.001,
        'change_cue_score': (r+2*c)/(3*(n-1))} for r in range(n) for c in range(n)])


class RouteFactorTests(unittest.TestCase):
    def test_two_opt_shortens_crossing_without_changing_nodes(self):
        xy = np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]])
        d = np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        route = [1,2,3]
        improved = two_opt(d,0,0,route)
        self.assertCountEqual(improved,route)
        self.assertLess(path_distance(d,0,0,improved),path_distance(d,0,0,route))

    def test_new_executor_exactly_matches_old_greedy(self):
        g=grid(); bits=np.arange(len(g.ids))%3==0
        for policy in POLICIES:
            c=Config(budget_s=900)
            a=old_execute(g,policy,c,Observer(bits,(.8,.1)))
            b=execute(g,policy,c,Observer(bits,(.8,.1)),'greedy',Planner())
            for key in a:
                if key in ('measured_planning_wall_s','planning_events'):continue
                self.assertEqual(a[key],b[key],(policy,key))
            for x,y in zip(a['planning_events'],b['planning_events']):
                for key in x:self.assertEqual(x[key],y[key],(policy,key))

    def test_strong_planned_prior_reward_never_below_greedy_seed(self):
        g=grid();planner=Planner();visited=np.zeros(len(g.ids),bool)
        for budget in (100,300,700,1200):
            for limit in (1,6,30,49):
                c=Config(budget_s=budget,max_visits=limit)
                a=greedy_route(g,g.xy[g.start_index],visited,1.,g.prior_post,c,0.,limit)
                b,_=planner.solve(g,'insertion_2opt',g.xy[g.start_index],visited,1.,g.prior_post,c,0.,limit)
                reward=.02+.58*g.prior_post[g.blocks]
                self.assertGreaterEqual(float(reward[b].sum())+1e-12,float(reward[a].sum()))

    def test_time_only_setting_can_exceed_30_visits(self):
        g=grid(8)
        for solver in ('greedy','insertion_2opt'):
            r=execute(g,'post_static',Config(budget_s=3000,max_visits=len(g.ids)),Observer(np.zeros(len(g.ids),bool),(.8,.1)),solver,Planner())
            self.assertGreater(r['visits'],30)
            self.assertLessEqual(r['total_time_s'],3000+1e-8)

    def test_legacy_cap_remains_binding(self):
        g=grid(8)
        r=execute(g,'post_static',Config(budget_s=3000,max_visits=30),Observer(np.zeros(len(g.ids),bool),(.8,.1)),'insertion_2opt',Planner())
        self.assertEqual(r['visits'],30)

    def test_costs_closure_no_duplicates_and_paid_observation_access(self):
        g=grid();bits=np.arange(len(g.ids))%4==0
        for policy in POLICIES:
            o=Observer(bits,(.8,.1));c=Config(budget_s=800,max_visits=len(g.ids))
            r=execute(g,policy,c,o,'insertion_2opt',Planner())
            self.assertEqual(o.accessed,[s['cell_index'] for s in r['steps']])
            self.assertEqual(len(set(o.accessed)),r['visits'])
            self.assertEqual(r['closed_route_ids'][0],r['closed_route_ids'][-1])
            self.assertAlmostEqual(r['total_time_s'],r['movement_seconds']+r['observation_seconds']+r['simulated_planning_latency_s'])
            self.assertLessEqual(r['total_time_s'],800+1e-8)

    def test_every_planned_suffix_includes_return_and_excludes_visited(self):
        g=grid();r=execute(g,'online',Config(budget_s=900,max_visits=49),Observer(np.ones(49,bool),(.8,.1)),'insertion_2opt',Planner())
        used=set();position=g.start_index
        for event,step in zip(r['planning_events'],r['steps']):
            route=event['planned_suffix_indices']
            self.assertFalse(set(route)&used)
            self.assertEqual(len(set(route)),len(route))
            nodes=[position,*route,g.start_index]
            distance=sum(np.linalg.norm(g.xy[a]-g.xy[b]) for a,b in zip(nodes,nodes[1:]))
            self.assertLessEqual(step['elapsed_before_step_s']+len(route)*21+distance/5,900+1e-8)
            self.assertEqual(route[0],step['cell_index'])
            used.add(step['cell_index']);position=step['cell_index']

    def test_static_and_frozen_paths_ignore_feedback(self):
        g=grid()
        for policy in ('pre_static','post_static','online_frozen'):
            paths=[execute(g,policy,Config(budget_s=700),Observer(np.full(49,b,bool),(.8,.1)),'insertion_2opt',Planner())['closed_route_ids'] for b in (False,True)]
            self.assertEqual(*paths)

    def test_same_first_action_without_future_signal_access(self):
        g=grid()
        rs=[execute(g,'online',Config(budget_s=700),Observer(np.full(49,b,bool),(.8,.1)),'insertion_2opt',Planner()) for b in (False,True)]
        self.assertEqual(rs[0]['steps'][0]['cell_id'],rs[1]['steps'][0]['cell_id'])
        self.assertEqual(rs[0]['planning_events'][0]['planned_suffix_indices'],rs[1]['planning_events'][0]['planned_suffix_indices'])

    def test_cache_preserves_answer_and_keys_include_belief(self):
        g=grid();p=Planner();used=np.zeros(49,bool);c=Config(budget_s=700)
        args=(g,'insertion_2opt',g.xy[g.start_index],used,0.,g.prior_post,c,1.,30)
        a,sa=p.solve(*args);b,sb=p.solve(*args)
        self.assertEqual(a,b);self.assertFalse(sa['cache_hit']);self.assertTrue(sb['cache_hit'])
        changed=g.prior_post.copy();changed[0]=.9
        _,stats=p.solve(g,'insertion_2opt',g.xy[g.start_index],used,0.,changed,c,1.,30)
        self.assertFalse(stats['cache_hit'])

    def test_too_short_budget_never_observes(self):
        g=grid();o=Observer(np.zeros(49,bool),(.8,.1))
        r=execute(g,'online',Config(budget_s=10),o,'insertion_2opt',Planner())
        self.assertEqual(o.accessed,[]);self.assertEqual(r['total_time_s'],0.)


if __name__=='__main__':
    unittest.main()
