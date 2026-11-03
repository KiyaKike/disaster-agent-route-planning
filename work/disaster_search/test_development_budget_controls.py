from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image

from build_xbd_mini import sha
from paired_count_baseline import fit_joint
from run_development_budget_controls import (CELLS,PaidBinaryEnvironment,fit_presence,
    presence_utilities,utility_table,run_episode,private_evaluator,score_trace,public_rgb_features)


def count_model():
    return {"public_quantile_edges":[.25,.5,.75],"bins":[{
        "survey_mean_count":{"native":.8,"blurred":.2},
        "upgrade":{"0":{"mean_new_within_cell":.4},"1":{"mean_new_within_cell":3.}}} for _ in range(4)],
        "paired_sensor":fit_joint([(True,True,True),(True,False,True),(False,False,False)])}


def observation(cell,mode,positive):
    return {"sample_id":"s","cell_id":cell,"mode":mode,"declared_target":positive}


class ActionTests(unittest.TestCase):
    def test_count_all_can_upgrade_positive_and_negative_ablation_cannot(self):
        model=count_model();features={c:{"change":.5,"coarse_nonzero_fraction":1.} for c in CELLS}
        observations=[observation(c,m,c==CELLS[0]) for c in CELLS for m in ["native","blurred"]]
        traces={}
        for policy in ["count_all","count_negative","count_survey"]:
            table=utility_table(features,model,None,"blurred_survey",2,policy)
            env=PaidBinaryEnvironment(observations,"s","blurred_survey",0,2)
            traces[policy]=run_episode(env,table,policy,3,2,0)
        self.assertEqual([(r["cell_id"],r["kind"]) for r in traces["count_all"]],[(CELLS[0],"survey"),(CELLS[0],"upgrade")])
        self.assertTrue(all(r["kind"]=="survey" for r in traces["count_negative"]))
        self.assertTrue(all(r["kind"]=="survey" for r in traces["count_survey"]))

    def test_expensive_upgrade_is_unaffordable_and_native_is_terminal(self):
        table=[[.1,.2,.2] for _ in CELLS]
        obs=[observation(c,m,False) for c in CELLS for m in ["native","blurred"]]
        env=PaidBinaryEnvironment(obs,"s","blurred_survey",0,4)
        trace=run_episode(env,table,"count_all",3,4,0)
        self.assertEqual(len(trace),3);self.assertTrue(all(r["kind"]=="survey" for r in trace))
        env=PaidBinaryEnvironment(obs,"s","native_survey",0,2)
        env.observe(CELLS[0],"survey")
        with self.assertRaises(ValueError):env.observe(CELLS[0],"upgrade")

    def test_unpaid_future_and_evaluator_fields_cannot_change_action_selection(self):
        table=[[1.,2.,3.] for _ in CELLS]
        rows=[observation(c,m,False) for c in CELLS for m in ["native","blurred"]]
        first=PaidBinaryEnvironment(rows,"s","native_survey",0,2)
        trace=run_episode(first,table,"count_all",1,2,0)
        poisoned=deepcopy(rows)
        for r in poisoned:
            r["matched_target_uids"]=["private"];r["target"]=True;r["future_quality"]="blurred"
            if r["cell_id"]!=CELLS[0]:r["declared_target"]=True
        second=PaidBinaryEnvironment(poisoned,"s","native_survey",0,2)
        self.assertEqual(trace,run_episode(second,table,"count_all",1,2,0))
        paid=PaidBinaryEnvironment(poisoned,"s","blurred_survey",0,2).observe(CELLS[0],"survey")
        self.assertEqual(set(paid),{"cell_id","mode","declared_target","cost"})

    def test_coarse_usable_controls_rank_without_dropping_cells(self):
        model=count_model();features={c:{"change":1.,"coarse_nonzero_fraction":0.} for c in CELLS}
        features[CELLS[-1]]["coarse_nonzero_fraction"]=.5
        table=utility_table(features,model,None,"native_survey",2,"usable_coverage")
        rows=[observation(c,m,False) for c in CELLS for m in ["native","blurred"]]
        trace=run_episode(PaidBinaryEnvironment(rows,"s","native_survey",0,2),table,"usable_coverage",2,2,0)
        self.assertEqual([r["cell_id"] for r in trace],[CELLS[-1],CELLS[0]])
        self.assertEqual(len(table),64)

    def test_random_actions_replay_from_identical_seed(self):
        rows=[observation(c,m,False) for c in CELLS for m in ["native","blurred"]]
        table=[[1.,1.,1.] for _ in CELLS]
        def replay(seed):return run_episode(PaidBinaryEnvironment(rows,"s","blurred_survey",0,2),table,"random_upgrade",16,2,seed)
        self.assertEqual(replay(20261030),replay(20261030))
        self.assertNotEqual(replay(20261030),replay(20261031))


class ModelAndEvaluationTests(unittest.TestCase):
    def test_replay_audit_rejects_private_fields_shortcut_and_overspend(self):
        from audit_development_budget_controls import audit_actions
        action={"cell_id":CELLS[0],"kind":"survey","mode":"native","declared_target":False,
                "cost":1,"cumulative_cost":1,"utility":1.}
        record={"policy":"count_all","scenario":"native_survey","quality_seed":0,"sample_id":"s",
                "upgrade_cost":2,"budget":1,"trace":[action]}
        obs={(CELLS[0],"native"):False}
        self.assertEqual(audit_actions(record,obs)["actions"],1)
        for changed in [{**action,"building_uid":"secret"},{**action,"kind":"upgrade"},{**action,"cumulative_cost":2}]:
            with self.assertRaises(ValueError):audit_actions({**record,"trace":[changed]},obs)
        with self.assertRaises(ValueError):audit_actions({**record,"budget":0},obs)

    def test_replay_audit_checks_paid_boolean_and_negative_only_ablation(self):
        from audit_development_budget_controls import audit_actions
        trace=[{"cell_id":CELLS[0],"kind":"survey","mode":"blurred","declared_target":True,"cost":1,"cumulative_cost":1,"utility":1.},
            {"cell_id":CELLS[0],"kind":"upgrade","mode":"native","declared_target":True,"cost":2,"cumulative_cost":3,"utility":1.}]
        obs={(CELLS[0],"native"):True,(CELLS[0],"blurred"):True}
        record={"policy":"count_all","scenario":"blurred_survey","quality_seed":0,"sample_id":"s","upgrade_cost":2,"budget":3,"trace":trace}
        self.assertEqual(audit_actions(record,obs)["positive_upgrades"],1)
        with self.assertRaises(ValueError):audit_actions({**record,"policy":"count_negative"},obs)
        with self.assertRaises(ValueError):audit_actions(record,{**obs,(CELLS[0],"native"):False})

    def test_presence_uses_joint_conditional_and_positive_has_no_new_cell_reward(self):
        joint=fit_joint([(True,False,False)]*2+[(True,False,True)]+[(True,True,True)]*7+
                        [(False,False,False)]*8+[(False,False,True)]+[(False,True,False)])
        model={"public_quantile_edges":[.25,.5,.75],"bins":[{"prior":.2} for _ in range(4)],"paired_sensor":joint}
        values=presence_utilities(model,.5,"mixed_survey",2)
        # Exact P(T,B=0,N=1)/P(B=0), then divided by cost.
        p=joint["probabilities"]
        expected=(.2*p["1"]["01"])/(.2*(p["1"]["00"]+p["1"]["01"])+.8*(p["0"]["00"]+p["0"]["01"]))/2
        self.assertAlmostEqual(values[1],expected)
        self.assertEqual(values[2],0.)

    def test_presence_fit_ignores_unknown_truth_and_ids(self):
        rows=[{"sample_id":"s","cell_id":CELLS[i],"mode":"native","target":truth}
              for i,truth in enumerate([True,False,None])]
        scores={"s":{c:.4 for c in CELLS}}
        model=count_model();first=fit_presence(rows,scores,model)
        for r in rows:r["matched_target_uids"]=["private"];r["target_count"]=900
        self.assertEqual(first,fit_presence(rows,scores,model))
        self.assertEqual(sum(b["known_cells"] for b in first["bins"]),2)

    def test_true_ids_used_only_to_score_global_new_buildings_and_thresholds(self):
        def row(cell,mode,ids):
            return {"cell_id":cell,"mode":mode,"target":True,"visible_target_uids":["a","b"],
                "overlap_records":[{"uid":u,"visible_pixels":100,"severe_overlap_pixels":20,"visible_footprint_coverage":.2} for u in ids]}
        rows=[row(CELLS[0],"blurred",["a"]),row(CELLS[1],"native",["a"]),row(CELLS[0],"native",["a","b"])]
        trace=[{"cell_id":CELLS[0],"mode":"blurred","kind":"survey","declared_target":True,"cost":1,"cumulative_cost":1},
            {"cell_id":CELLS[1],"mode":"native","kind":"survey","declared_target":True,"cost":1,"cumulative_cost":2},
            {"cell_id":CELLS[0],"mode":"native","kind":"upgrade","declared_target":True,"cost":2,"cumulative_cost":4}]
        evaluator,targets=private_evaluator(rows);result=score_trace(trace,evaluator,targets)
        self.assertEqual(result[1]["distinct_found"],2)
        self.assertEqual(result[1]["cross_cell_duplicate_match_events"],1)
        self.assertEqual(result[1]["new_buildings_from_positive_upgrades"],1)
        self.assertEqual(result[2]["distinct_found"],0)
        self.assertTrue(result[2]["first_discovery_censored"])

    def test_public_rgb_features_do_not_read_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"rgb.png"
            Image.fromarray(np.zeros((1024,1024,3),dtype=np.uint8)).save(path)
            manifest={"samples":[{"sample_id":"s","files":{"pre_image":"rgb.png","post_image":"rgb.png","post_label":"/do-not-read"}}],
                      "files":[{"path":"rgb.png","sha256":sha(path.read_bytes())}]}
            features=public_rgb_features(root,manifest)
            self.assertEqual(features["s"][CELLS[0]],{"change":0.,"coarse_nonzero_fraction":0.})


if __name__=="__main__":unittest.main()
