import math
import random
import unittest

from aippms_control_core import (Layout,SENSORS,accuracy,actions,posterior,update,outcomes,
    ExactOneSense,Environment,expected_mode_gain,exact_evaluation,run_episode,POMCP)


class AIPPMSControlTests(unittest.TestCase):
    def test_pinned_sensor_formula_and_costs(self):
        l=Layout();self.assertEqual([s[1]/2 for s in SENSORS],[.5,2.])
        self.assertAlmostEqual(accuracy(l,l.home,0)[0],.5*(1+2**(-8/2.5)))

    def test_opposite_observation_produces_opposite_update(self):
        l=Layout();p=(.5,)*3
        plus=update(l,p,0,l.home,1,(True,True,True))
        minus=update(l,p,0,l.home,1,(False,False,False))
        for a,b in zip(plus,minus):self.assertGreater(a,.5);self.assertAlmostEqual(a+b,1)

    def test_bayes_matches_direct_joint_probability(self):
        p,q=.2,.8
        self.assertAlmostEqual(posterior(p,q,True),.16/(.16+.16))
        self.assertAlmostEqual(posterior(p,q,False),.04/(.04+.64))

    def test_zero_information_observation_changes_no_belief(self):
        l=Layout(sensor_model='uninformative');prior=(.2,.7,.4)
        self.assertEqual(update(l,prior,0,l.home,0,(True,False,True)),prior)
        self.assertAlmostEqual(expected_mode_gain(l,l.home,0,prior,1),0)

    def test_visited_rock_stays_consumed(self):
        l=Layout();self.assertEqual(update(l,(.5,.5,.5),1,l.home,1,(True,True,True))[0],0)

    def test_sensor_outcomes_normalize_and_preserve_prior_in_expectation(self):
        l=Layout();p=(.3,.6,.8);oo=list(outcomes(l,p,0,l.home,1))
        self.assertAlmostEqual(sum(a for a,b,c in oo),1)
        for j in range(3):self.assertAlmostEqual(sum(a*c[j] for a,b,c in oo),p[j])

    def test_sensing_available_only_at_beacon_with_return_reserve(self):
        l=Layout();self.assertFalse(any(a[0]=='sense' for a in actions(l,(1,0),20)))
        self.assertEqual(actions(l,(0,1),2),(('move',(0,0)),))

    def test_analytic_single_rock_information_value(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),));p=ExactOneSense(l)
        value=p.value(l.home,5,0,(.5,),1)[0]
        self.assertAlmostEqual(value,5*(2*accuracy(l,l.home,0)[0]-1))
        self.assertEqual(p.action(l.home,5,0,(.5,),1),('sense',0))
        exact=exact_evaluation(l,2.5,1,p);self.assertAlmostEqual(value,exact['expected_reward'])

    def test_no_information_exact_policy_does_not_pay_for_sensing(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),),sensor_model='uninformative');p=ExactOneSense(l)
        self.assertEqual(p.value(l.home,5,0,(.5,),1),(0.,('stop',None)))

    def test_exact_sensing_option_cannot_lower_optimal_value(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),));p=ExactOneSense(l)
        for ticks in (4,5,8,12):self.assertGreaterEqual(p.value(l.home,ticks,0,(.5,),1)[0],p.value(l.home,ticks,0,(.5,),0)[0]-1e-8)

    def test_collected_rock_cannot_be_rewarded_twice(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),));e=Environment(l,(True,),10,4)
        pos,left,mask,p,o,r=e.step(l.home,20,0,(.5,),('move',(0,1)));self.assertEqual(r,10)
        pos,left,mask,p,o,r=e.step(pos,left,mask,p,('move',l.home))
        pos,left,mask,p,o,r=e.step(pos,left,mask,p,('move',(0,1)));self.assertEqual(r,-10)

    def test_private_truth_does_not_enter_policy_interface(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),));p=ExactOneSense(l)
        before=p.action(l.home,5,0,(.5,),1)
        _=Environment(l,(False,),2.5,0);_=Environment(l,(True,),2.5,0)
        self.assertEqual(p.action(l.home,5,0,(.5,),1),before)

    def test_pomcp_uses_fixed_queries_and_returns(self):
        l=Layout(side=2,rocks=((0,1),),beacons=((0,0),))
        for policy in ('pomcp_random','pomcp_gcb'):
            r=run_episode(l,(True,),3,policy,3,5)
            self.assertTrue(r['returned']);self.assertLessEqual(r['energy_used'],3)
            self.assertEqual(r['planning_simulations']%100,0)

    def test_impossible_observation_is_rejected(self):
        with self.assertRaises(ValueError):posterior(0,1,True)
        with self.assertRaises(ValueError):update(Layout(),(.5,)*3,0,(1,0),0,(True,)*3)


if __name__=='__main__':unittest.main()
