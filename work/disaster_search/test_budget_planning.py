from copy import deepcopy
import itertools
import random
import unittest

from run_budget_planning import BatchTail,action_values,choose,planning_inputs,run_episode
from run_development_budget_controls import PaidBinaryEnvironment,run_episode as greedy_episode,utility_table
from cross_cell_count_smoke import CELLS
from paired_count_baseline import fit_joint


def inputs(rewards,probabilities=None,native=0.,cost=2):
    return {"reward":rewards,"greedy_utility":[[r[0]]+[None if x is None else x/cost for x in r[1:]] for r in rewards],
            "blur_positive":probabilities or [0.]*len(rewards),"native_survey_probability":native}


class PlanningValueTests(unittest.TestCase):
    def test_batch_solver_matches_exhaustive_subsets_for_all_costs_and_budgets(self):
        rng=random.Random(33)
        for cost in [1,2,4]:
            for _ in range(8):
                items=[(rng.choice([1,cost]),rng.random()*5) for _ in range(7)]
                solver=BatchTail(items,cost,"batch_tail")
                for budget in range(10):
                    expected=max(sum(r for include,(c,r) in zip(mask,items) if include)
                        for mask in itertools.product([0,1],repeat=7)
                        if sum(c for include,(c,r) in zip(mask,items) if include)<=budget)
                    self.assertAlmostEqual(solver.value(budget),expected)

    def test_remaining_budget_changes_opportunity_cost_choice(self):
        model=inputs([[0.,3.,3.],[2.,0.,0.]],cost=2);state=[1,0]
        self.assertEqual(choose(state,model,2,2,"proxy_greedy")[0],1)
        self.assertEqual(choose(state,model,2,2,"batch_tail")[0],0)
        self.assertEqual(choose(state,model,2,2,"two_step")[0],0)
        self.assertEqual(choose(state,model,1,2,"batch_tail")[0],1)

    def test_fantasy_branches_use_probabilities_and_raw_reward_not_reward_rate(self):
        model=inputs([[1.,2.,10.],[3.,0.,0.]],[.25,0.],native=0.,cost=2)
        for policy in ["two_step","batch_tail"]:
            q=dict((i,v) for i,v,_ in action_values([0,0],model,3,2,policy))
            self.assertAlmostEqual(q[0],1.+.75*3.+.25*10.)
        model["native_survey_probability"]=1.
        q=dict((i,v) for i,v,_ in action_values([0,0],model,3,2,"batch_tail"))
        self.assertAlmostEqual(q[0],4.)

    def test_batch_tail_uses_more_than_one_remaining_action(self):
        model=inputs([[1.,0.,0.],[3.,0.,0.],[2.,0.,0.]],native=1.)
        short=dict((i,v) for i,v,_ in action_values([0,0,0],model,3,2,"two_step"))
        full=dict((i,v) for i,v,_ in action_values([0,0,0],model,3,2,"batch_tail"))
        self.assertAlmostEqual(short[0],4.)
        self.assertAlmostEqual(full[0],6.)

    def test_equivalence_grouping_keeps_distinct_future_probabilities(self):
        model=inputs([[1.,0.,10.],[1.,0.,10.]],[.1,.9])
        values=action_values([0,0],model,3,2,"batch_tail")
        self.assertEqual(len(values),2)
        self.assertGreater(values[1][1],values[0][1])
        model["blur_positive"]=[.1,.1]
        self.assertEqual([v[0] for v in action_values([0,0],model,3,2,"batch_tail")],[0])

    def test_unsupported_upgrade_remains_absent_and_native_terminal(self):
        model=inputs([[1.,None,None],[2.,3.,4.]])
        self.assertIsNone(choose([1,3],model,32,2,"batch_tail"))
        self.assertEqual(choose([0,3],model,1,2,"batch_tail")[0],0)
        with self.assertRaises(ValueError):BatchTail([(1,-1)],2,"batch_tail")


class PlannerBoundaryTests(unittest.TestCase):
    def test_public_diagnostic_detects_ties_without_labels(self):
        from diagnose_utility_resolution import diagnose
        model=self.model();features={c:{"change":i/1000,"coarse_nonzero_fraction":1.} for i,c in enumerate(CELLS)}
        result=diagnose({"development_fit":model,"held_e":model},{"e_00000001":features})
        stats=result["by_scope"]["development_fit"]
        self.assertEqual(stats["mean_utility_distinct_values"],1.)
        self.assertEqual(stats["mean_top_tie_cells"],64.)
        self.assertEqual(stats["initial_choice_differs_from_raw_change_scenes"],1)
        self.assertFalse(result["labels_or_future_observations_read"])

    def test_paired_effects_average_scenes_within_events_before_event_macro(self):
        from audit_budget_planning import effects
        def row(sid,event,policy,value):
            return {"sample_id":sid,"event":event,"policy":policy,"fit_scope":"development_fit","scenario":"mixed_survey",
                "upgrade_cost":2,"budget":32,"threshold":.1,"distinct_found":value,"false_positive_cells":0.,"upgrades":0.,"positive_upgrades":0.}
        current=[row("a","e1","batch_tail",2.),row("b","e1","batch_tail",2.),row("c","e2","batch_tail",0.)]
        old=[row("a","e1","prior_survey",1.),row("b","e1","prior_survey",1.),row("c","e2","prior_survey",0.)]
        result=effects(current,old)
        self.assertEqual(len(result["scene_effects"]),3)
        self.assertAlmostEqual(result["event_macro_effects"][0]["delta_found"],.5)

    def model(self):
        count={"public_quantile_edges":[.25,.5,.75],"bins":[{"survey_mean_count":{"native":i+.2,"blurred":i*.2},
            "upgrade":{"0":{"mean_new_within_cell":i*.3},"1":{"mean_new_within_cell":i+1.}}} for i in range(4)],
            "paired_sensor":fit_joint([(True,False,True),(False,False,False),(True,True,True)])}
        presence={"public_quantile_edges":count["public_quantile_edges"],"paired_sensor":count["paired_sensor"],
            "bins":[{"prior":.1+.2*i} for i in range(4)]}
        return {"count":count,"presence":presence}

    def test_branch_inputs_use_same_paired_model_and_ignore_private_features(self):
        models=self.model();features={c:{"change":i/64,"coarse_nonzero_fraction":1.} for i,c in enumerate(CELLS)}
        first=planning_inputs(features,models,"mixed_survey",2)
        poisoned=deepcopy(features)
        for v in poisoned.values():v["true_damage"]=True;v["matched_target_uids"]=["private"];v["future_quality"]="native"
        self.assertEqual(planning_inputs(poisoned,models,"mixed_survey",2),first)
        prior=models["presence"]["bins"][0]["prior"];p=models["presence"]["paired_sensor"]["probabilities"]
        self.assertAlmostEqual(first["blur_positive"][0],prior*(p["1"]["10"]+p["1"]["11"])+(1-prior)*(p["0"]["10"]+p["0"]["11"]))

    def test_original_greedy_actions_reproduced_in_all_scenarios_and_costs(self):
        models=self.model();features={c:{"change":i/64,"coarse_nonzero_fraction":1.} for i,c in enumerate(CELLS)}
        rows=[{"sample_id":"s","cell_id":c,"mode":mode,"declared_target":i%5==0} for i,c in enumerate(CELLS) for mode in ["native","blurred"]]
        for scenario in ["native_survey","mixed_survey","blurred_survey"]:
            for cost in [1,2,4]:
                table=utility_table(features,models["count"],models["presence"],scenario,cost,"count_all")
                expected=greedy_episode(PaidBinaryEnvironment(rows,"s",scenario,20261025,cost),table,"count_all",32,cost,0)
                actual=run_episode(PaidBinaryEnvironment(rows,"s",scenario,20261025,cost),planning_inputs(features,models,scenario,cost),"proxy_greedy",32,cost)
                self.assertEqual([{k:v for k,v in a.items() if k!="greedy_utility"} for a in actual],expected)

    def test_all_native_independent_cells_batch_matches_greedy_with_tie_rule(self):
        models=self.model();features={c:{"change":i/64,"coarse_nonzero_fraction":1.} for i,c in enumerate(CELLS)}
        rows=[{"sample_id":"s","cell_id":c,"mode":"native","declared_target":False} for c in CELLS]
        model=planning_inputs(features,models,"native_survey",2)
        traces=[run_episode(PaidBinaryEnvironment(rows,"s","native_survey",0,2),model,p,16,2) for p in ["proxy_greedy","batch_tail","two_step"]]
        self.assertEqual([[a["cell_id"] for a in trace] for trace in traces],[[a["cell_id"] for a in traces[0]]]*3)


if __name__=="__main__":unittest.main()
