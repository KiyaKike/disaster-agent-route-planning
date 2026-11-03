import copy
import unittest
import numpy as np
from feedback_increment_core import *


def fake_data():
    public=[];private={}
    for e in EVENTS:
        for c in range(3):
            sid=f'{e}_{c:08d}';rng=np.random.default_rng(len(public))
            public.append({'sample_id':sid,'event':e,'pre_texture':rng.uniform(.01,.1,16).tolist(),
                           'post_change':rng.uniform(.01,.2,16).tolist(),'post_utility':rng.uniform(0,1,16).tolist()})
            private[sid]=[{'predicted_severe_pixels':int(rng.integers(0,3))*10,'tp':int(rng.integers(0,2)),
                           'reference':int(rng.integers(0,2)),'fp':0} for _ in range(16)]
    return public,private


class FeedbackIncrementTests(unittest.TestCase):
    def test_unvisited_private_values_never_enter_features(self):
        public,private=fake_data();states=[{'scheme':'route','sample_id':public[0]['sample_id'],'event':EVENTS[0],'prefix':[3,7]}]
        a=build_rows(states,public,private)
        mutated=copy.deepcopy(private)
        for t in range(16):
            if t not in (3,7):mutated[public[0]['sample_id']][t]={'predicted_severe_pixels':999,'tp':1,'reference':1,'fp':0}
        b=build_rows(states,public,mutated)
        self.assertEqual([r['prior']+r['feedback']+[r['neighbor_rule']] for r in a],
                         [r['prior']+r['feedback']+[r['neighbor_rule']] for r in b])
        self.assertEqual(len(a),14);self.assertNotIn(3,[r['tile'] for r in a])

    def test_labels_change_does_not_change_features(self):
        public,private=fake_data();states=random_prefixes([p['sample_id'] for p in public],ks=(2,),draws=2)
        a=build_rows(states,public,private);mutated=copy.deepcopy(private)
        for v in mutated.values():
            for cell in v:cell['tp']=1-cell['tp'];cell['reference']=1-cell['reference']
        b=build_rows(states,public,mutated)
        self.assertEqual([r['prior']+r['feedback'] for r in a],[r['prior']+r['feedback'] for r in b])

    def test_route_prefixes_follow_history_order(self):
        route={'policy':'post_static','budget_s_proxy':240,'sample_id':'nepal-flooding_1','event':'nepal-flooding',
               'state':{'history':[{'tile_id':10},{'tile_id':9},{'tile_id':8}]}}
        self.assertEqual([s['prefix'] for s in route_prefixes([route])],[[10],[10,9],[10,9,8]])

    def test_permuted_donor_is_other_chip_same_k(self):
        public,_=fake_data();states=random_prefixes([p['sample_id'] for p in public],ks=(1,3),draws=2)
        for s,d in zip(states,permuted_donors(states)):
            self.assertNotEqual(s['sample_id'],d['sample_id']);self.assertEqual(len(s['prefix']),len(d['prefix']))

    def test_leave_one_event_out_never_trains_on_test_event(self):
        public,private=fake_data();states=random_prefixes([p['sample_id'] for p in public],ks=(1,),draws=3)
        rows=build_rows(states,public,private);seen=[]
        import feedback_increment_core as core
        original=core.fit_logistic
        def spy(X,y,l2=L2):
            seen.append(len(y));return original(X,y,l2)
        core.fit_logistic=spy
        try:out_of_fold(rows,'y_tp',lambda r:r['prior'],'event')
        finally:core.fit_logistic=original
        per_event={e:sum(r['event']==e for r in rows) for e in EVENTS}
        self.assertEqual(sorted(seen),sorted(len(rows)-n for n in per_event.values()))

    def test_metrics_on_known_cases(self):
        self.assertEqual(auroc([0,0,1,1],[.1,.2,.8,.9]),1.0)
        self.assertEqual(auroc([0,1,0,1],[.5,.5,.5,.5]),.5)
        self.assertIsNone(auroc([0,0],[.1,.2]))
        self.assertAlmostEqual(average_precision([1,0,1],[.9,.8,.1]),(1+2/3)/2)
        m=fit_logistic(np.array([[0.],[1.],[2.],[3.]]),np.array([0,0,1,1]))
        p=predict(m,np.array([[0.],[3.]]));self.assertLess(p[0],.5);self.assertGreater(p[1],.5)

    def test_next_pick_uses_top_score_and_skips_no_opportunity(self):
        rows=[{'state':0,'tile':1,'y_tp':0,'sample_id':'a'},{'state':0,'tile':2,'y_tp':1,'sample_id':'a'},
              {'state':1,'tile':1,'y_tp':0,'sample_id':'b'}]
        self.assertEqual(next_pick_hits(rows,[.2,.9,.5],'y_tp'),[('a',1)])


if __name__=='__main__':unittest.main()
