import itertools,unittest
import numpy as np
from crasar_route_controls import plan_route,replay

class Controls(unittest.TestCase):
    def setUp(self):
        self.xy=np.array([[0,1],[1,1],[2,1],[0,0]],float)
        self.c=np.array([[.5,0],[.5,.2],[0,.8]])
        self.d=np.array([.9,.2]);self.u=np.array([.6,1.])
        self.kw=dict(speed=1.,observe=1.,plan=.5,budget=7.)
    def test_exact_matches_independent_permutations(self):
        result=plan_route('static_exact',self.xy,self.c,self.d,self.u,**self.kw)
        best=0
        for k in range(4):
            for route in itertools.permutations(range(3),k):
                nodes=(3,)+route+(3,)
                dist=sum(np.linalg.norm(self.xy[a]-self.xy[b]) for a,b in zip(nodes,nodes[1:]))
                if dist+k+.5<=7.+1e-9:
                    fractions=self.c[list(route)].sum(axis=0) if route else np.zeros(2)
                    best=max(best,float(fractions@(self.d*self.u)))
        self.assertAlmostEqual(result['public_utility_proxy'],best)
    def test_all_controls_pay_return_and_observations(self):
        for name in ['distance','damage','uncertainty','static_exact','value_cost_proxy']:
            r=plan_route(name,self.xy,self.c,self.d,self.u,**self.kw)
            self.assertLessEqual(r['elapsed_proxy_s'],7+1e-9)
            self.assertAlmostEqual(r['elapsed_proxy_s'],.5+r['distance_m']+r['observation_count'])
            self.assertTrue(r['returned_home'])
    def test_return_infeasible_visit_rejected(self):
        r=plan_route('distance',np.array([[5.,0],[0,0]]),[[1]],[1],[1],speed=1,observe=1,plan=0,budget=10)
        self.assertEqual(r['route'],[])
    def test_double_counting_refused(self):
        with self.assertRaises(ValueError):plan_route('damage',self.xy,np.ones((3,2)),self.d,self.u,**self.kw)
    def test_private_reference_has_no_planner_argument(self):
        with self.assertRaises(TypeError):plan_route('distance',self.xy,self.c,self.d,self.u,private_labels=[1,1],**self.kw)
    def test_same_utility_greedy_never_beats_exact(self):
        rng=np.random.default_rng(883)
        for _ in range(12):
            d=rng.random(2);u=rng.random(2)
            exact=plan_route('static_exact',self.xy,self.c,d,u,**self.kw)
            greedy=plan_route('value_cost_proxy',self.xy,self.c,d,u,**self.kw)
            self.assertLessEqual(greedy['public_utility_proxy'],exact['public_utility_proxy']+1e-9)

if __name__=='__main__':unittest.main()
