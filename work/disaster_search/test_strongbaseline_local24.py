import itertools
import unittest
import numpy as np
from strongbaseline_local24_core import units,crop_pair,coarse_pair,static_order,acquisitions,counts,policy_result,exact_random,oracle_order
from strongbaseline24_core import prepare_pair


class Local24ContractTests(unittest.TestCase):
    def test_partition_and_area_costs(self):
        cover=np.zeros((1024,1024),np.uint8)
        for u in units():
            x,y,xx,yy=u['bounds'];cover[y:yy,x:xx]+=1
            self.assertEqual((xx-x)*(yy-y),u['area_cost']*128**2)
        self.assertTrue(np.all(cover==1));self.assertEqual(sum(u['area_cost'] for u in units()),64)

    def test_outside_native_pixels_cannot_change_local_model_input(self):
        a=np.zeros((1024,1024,3),np.uint8);b=a.copy();b[256:]=255;b[:256,256:]=211
        np.testing.assert_array_equal(prepare_pair(*crop_pair(a,a,0)),prepare_pair(*crop_pair(b,b,0)))
        self.assertEqual(prepare_pair(*crop_pair(a,a,0)).shape,(1,6,256,256))

    def test_public_ranking_rejects_native_context(self):
        a=np.zeros((1024,1024,3),np.uint8)
        with self.assertRaises(ValueError):static_order(a,a)
        cp=coarse_pair(a,a);self.assertEqual(cp[0].shape,(64,64,3))
        self.assertEqual(static_order(*cp)['order'],list(range(16)))

    def test_coarse_change_is_mean_and_ties_are_row_major(self):
        pre=np.zeros((64,64,3),np.uint8);post=pre.copy();post[0,0]=255;post[:16,16:32]=32
        ranked=static_order(pre,post)
        self.assertEqual(ranked['order'][:2],[1,0]);self.assertEqual(ranked['order'][2:],list(range(2,16)))

    def test_payments_and_duplicate_invalid_budget_rejection(self):
        trace=acquisitions(list(range(16)),16);self.assertEqual(len(trace),4);self.assertEqual(trace[-1]['cumulative_cost'],16)
        self.assertEqual(acquisitions(list(range(16)),3),[])
        for order,budget in [(list(range(15))+[0],16),(list(range(16)),65),(list(range(16)),16.0)]:
            with self.assertRaises(ValueError):acquisitions(order,budget)

    def test_random_expected_counts_and_hit_probability_match_enumeration(self):
        rows=[{'tp':i if i<4 else 0,'fp':i%3,'reference':i+3} for i in range(16)]
        values=[]
        for chosen in itertools.combinations(range(16),2):
            order=list(chosen)+[i for i in range(16) if i not in chosen]
            values.append(policy_result(rows,order,8))
        random=exact_random(rows,8)
        for k in ['tp','fp','fn','recall']:
            self.assertAlmostEqual(random[k],sum(v[k] for v in values)/len(values))
        self.assertAlmostEqual(random['scene_any_tp_probability'],sum(v['tp']>0 for v in values)/len(values))

    def test_oracle_is_exhaustive_optimum_for_disjoint_equal_cost_actions(self):
        rows=[{'tp':(i*7)%17,'fp':i,'reference':20} for i in range(16)]
        chosen=oracle_order(rows);actual=policy_result(rows,chosen,16)['tp']
        expected=max(sum(rows[i]['tp'] for i in subset) for subset in itertools.combinations(range(16),4))
        self.assertEqual(actual,expected)

    def test_unknown_reference_ignored_and_unobserved_targets_remain_fn(self):
        truth=np.array([[3,4,255,0]],np.uint8);pred=np.array([[3,0,4,4]],np.uint8)
        row=counts(truth,pred);self.assertEqual(row,{'tp':1,'fp':1,'reference':2,'valid_pixels':3})
        rows=[row]+[{'tp':0,'fp':0,'reference':3} for _ in range(15)]
        result=policy_result(rows,list(range(16)),4)
        self.assertEqual(result['fn'],46);self.assertEqual(result['reference'],47)


if __name__=='__main__':unittest.main()
