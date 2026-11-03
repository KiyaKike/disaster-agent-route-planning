from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from build_xbd_mini import sha
from cross_cell_count_smoke import (CELLS, PaidEnvironment, PublicState, choose,
    evaluate, fit_discount, make_spatial_groups, mask_edges, quality, training_records)


def paid(cell, mode="native", positive=True, **edges):
    return {"cell_id":cell, "mode":mode, "declared_target":positive,
            "edges":{k:edges.get(k,False) for k in ["north","south","west","east"]},
            "kind":"survey", "cost":1, "cumulative_cost":1, "boundary_contact":False}


def evaluator_row(ids, target=True):
    return {"target":target,"predicted":True,"visible_target_uids":ids,
            "overlap_records":[{"uid":uid,"visible_pixels":100,"severe_overlap_pixels":20,
                                "visible_footprint_coverage":.2} for uid in ids]}


class OnlineBoundaryTests(unittest.TestCase):
    def test_boundary_uses_only_paid_neighbor_facing_edge(self):
        state = PublicState()
        state.observe(paid("r02c01",east=True))
        self.assertTrue(state.boundary_contact("r02c02"))
        self.assertFalse(state.boundary_contact("r02c00"))
        self.assertFalse(state.boundary_contact("r03c02"))
        state.observe(paid("r00c00",north=True,west=True))
        self.assertFalse(state.boundary_contact("r00c01"))

    def test_positive_blur_can_upgrade_but_native_cannot_repeat(self):
        state = PublicState()
        state.observe(paid("r00c00","blurred",True))
        self.assertIn(("r00c00","upgrade",2),state.candidates(2,2))
        self.assertNotIn(("r00c00","upgrade",2),state.candidates(1,2))
        state.observe(paid("r00c00","native",True))
        self.assertNotIn("r00c00",[x[0] for x in state.candidates(32,2)])
        with self.assertRaises(ValueError): state.observe(paid("r00c00"))

    def test_state_whitelist_ignores_poisoned_evaluator_fields(self):
        a,b = PublicState(),PublicState()
        clean = paid("r02c01",east=True)
        poisoned = {**clean,"matched_target_uids":["secret"],"future_actual_quality":"native","target":True}
        a.observe(clean); b.observe(poisoned)
        self.assertEqual(a.latest,b.latest)

    def test_four_pixel_strips_are_not_true_building_counts(self):
        mask = np.zeros((128,128),dtype=bool)
        mask[10:14,-1]=True
        self.assertEqual(mask_edges(mask),{"north":False,"south":False,"west":False,"east":True})
        with self.assertRaises(ValueError): mask_edges(mask.astype(np.uint8))

    def test_discount_changes_action_without_candidate_mask(self):
        state = PublicState(); state.observe(paid("r02c01",east=True))
        cues={cid:0. for cid in CELLS}; cues["r02c02"]=.9; cues["r06c06"]=.9
        model={"public_quantile_edges":[.25,.5,.75],"bins":[
            {"survey_mean_count":{"native":value,"blurred":0.},"upgrade":{}}
            for value in [.01,.01,.01,1.]]}
        discount=fit_discount([{"sample_id":"s","cell_id":"r00c00","mode":"native",
            "stratum":"survey","boundary_contact":True,"local_new":1,"global_new":0}])
        a=choose(state,cues,model,discount,"local_count","native_survey",2,10,random.Random(0))
        b=choose(state,cues,model,discount,"boundary_count","native_survey",2,10,random.Random(0))
        self.assertEqual(a["cell_id"],"r02c02")
        self.assertEqual(b["cell_id"],"r06c06")

    def test_environment_reads_only_paid_mask_and_whitelists_output(self):
        PaidEnvironment._verified_public_masks.clear()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"mask.npz"
            np.savez_compressed(path,severe_mask=np.zeros((128,128),dtype=bool))
            rows=[{"sample_id":"s","cell_id":"r00c00","mode":"blurred","cache_file":str(path),
                   "cache_sha256":sha(path.read_bytes()),"declared_target":False,"matched_target_uids":["secret"]},
                  {"sample_id":"s","cell_id":"r00c00","mode":"native","cache_file":"/must/not/be/read",
                   "cache_sha256":"unknown","declared_target":True}]
            env=PaidEnvironment(rows,"s","blurred_survey",1,2)
            result=env.observe("r00c00","survey")
            self.assertEqual(set(result),{"cell_id","mode","declared_target","edges","cost"})
            self.assertEqual(result["cost"],1)
            with self.assertRaises(ValueError): env.observe("r00c00","survey")


class OfflineAndEvaluationTests(unittest.TestCase):
    def test_saved_trajectory_audit_refuses_private_fields_and_overspend(self):
        from audit_cross_cell_smoke import audit_trace
        action={**paid("r00c00"),"base_utility":1.,"discount":1.,"utility":1.}
        record={"trace":[action],"budget":1,"scenario":"native_survey","quality_seed":0,
                "sample_id":"s","upgrade_cost":2}
        self.assertEqual(audit_trace(record)["actions"],1)
        with self.assertRaises(ValueError):
            audit_trace({**record,"trace":[{**action,"matched_target_uids":["secret"]}]})
        with self.assertRaises(ValueError): audit_trace({**record,"budget":0})

    def test_saved_trajectory_audit_refuses_edge_feature_from_unpaid_future(self):
        from audit_cross_cell_smoke import audit_trace
        action={**paid("r00c00"),"base_utility":1.,"discount":1.,"utility":1.,"boundary_contact":True}
        record={"trace":[action],"budget":1,"scenario":"native_survey","quality_seed":0,
                "sample_id":"s","upgrade_cost":2}
        with self.assertRaises(ValueError): audit_trace(record)

    def test_fit_averages_repeated_histories_and_has_no_support_fallback(self):
        record={"sample_id":"s","cell_id":"r00c00","mode":"native","stratum":"survey",
                "boundary_contact":True,"local_new":1,"global_new":0}
        a,b=fit_discount([record]),fit_discount([record]*30)
        self.assertEqual(a["parameters"],b["parameters"])
        self.assertEqual(a["parameters"]["survey"]["1"]["remaining_fraction"],.5)
        self.assertEqual(a["parameters"]["survey"]["0"]["remaining_fraction"],1.)
        self.assertFalse(a["parameters"]["upgrade_B1"]["1"]["reward_support"])
        self.assertNotIn("r00c00",str(a["parameters"]))
        with self.assertRaises(ValueError): fit_discount([{**record,"global_new":2}])

    def test_global_reward_subtracts_other_cell_and_same_cell_history(self):
        trace=[paid("r02c01","blurred",east=True),paid("r02c02"),paid("r02c01")]
        trace[2]["kind"]="upgrade"
        lookup={
            ("s","r02c01","blurred"):evaluator_row(["a"]),
            ("s","r02c02","native"):evaluator_row(["a","b"]),
            ("s","r02c01","native"):evaluator_row(["a","b","c"])}
        rows=training_records([("s",trace)],lookup)
        self.assertEqual([(r["local_new"],r["global_new"]) for r in rows],[(1,1),(2,1),(2,1)])
        self.assertEqual(rows[2]["stratum"],"upgrade_B1")
        self.assertTrue(rows[1]["boundary_contact"])
        result=evaluate(trace,lookup,"s",32,.1)
        self.assertEqual(result["distinct_found"],3)
        self.assertEqual(result["local_new_match_events"],5)
        self.assertEqual(result["cross_cell_duplicate_match_events"],2)
        # Fixed trajectory re-scoring must preserve threshold sensitivity.
        self.assertEqual(evaluate(trace,lookup,"s",32,.25)["distinct_found"],0)

    def test_cohort_geometry_missing_is_unverified_not_independent_singleton(self):
        selection=[{"sample_id":"a","event":"e","split":"development"},
                   {"sample_id":"b","event":"e","split":"development"}]
        audit={"candidate_envelope_components":[["a"],["b"]],"per_sample":[
            {"sample_id":sid,"annotation_envelope_wgs84":None,"training_annotation_envelope_overlap_ids":[]}
            for sid in ["a","b"]]}
        ledger=make_spatial_groups(selection,audit)
        self.assertEqual(ledger[0]["analysis_group"],ledger[1]["analysis_group"])
        self.assertIsNone(ledger[0]["annotation_envelope_component"])
        self.assertTrue(all(not r["full_image_independence_verified"] for r in ledger))
        self.assertEqual({r["sample_id"] for r in ledger},{"a","b"})

    def test_quality_assignment_matches_existing_environment(self):
        from run_cached_search import survey_mode
        for scenario in ["native_survey","mixed_survey","blurred_survey"]:
            for cid in CELLS:
                self.assertEqual(quality(scenario,20261025,"s",cid),survey_mode(scenario,20261025,"s",cid))


if __name__ == "__main__":
    unittest.main()
