import unittest

from cross_cell_count_smoke import CELLS, quality
from paired_count_baseline import fit_local_count
from run_utility_failure_diagnostic import (aggregate, component_calibration,
    levels_from_trace, selected_episode, trace_from_levels)


class UtilityFailureDiagnosticTests(unittest.TestCase):
    def selected_fixture(self):
        evaluator = {(cid, mode): {"matches": [0] * 3, "truth": False}
                     for cid in CELLS for mode in ["native", "blurred"]}
        masks = {0: (1, 3), 1: (4, 4), 2: (8, 8), 3: (0b1110000, 0b1110000), 4: (2, 2)}
        for i, (b, n) in masks.items():
            for mode, bits in [("blurred", b), ("native", n)]:
                evaluator[CELLS[i], mode] = {"matches": [bits] * 3, "truth": True}
        declarations = {(c, m): bool(v["matches"][1]) for (c, m), v in evaluator.items()}
        table = [[0., None, None] for c in CELLS]
        table[0] = [4., 3., 3.]
        table[1][0] = 2.
        table[2][0] = 1.5
        table[3][0] = .1
        table[4][0] = 4.5
        trace = []
        spent = 0
        for i, kind, mode, price, utility in [(4, "survey", "native", 1, 4.5),
                (0, "survey", "blurred", 1, 4.), (0, "upgrade", "native", 2, 3.)]:
            spent += price
            trace.append({"cell_id": CELLS[i], "kind": kind, "mode": mode, "cost": price,
                "cumulative_cost": spent, "utility": utility, "declared_target": declarations[CELLS[i], mode]})
        row = {"policy": "count_all", "quality_seed": 20261025, "sample_id": "s", "trace": trace}
        return row, table, evaluator, declarations

    def test_counterfactual_uses_public_rank_not_largest_private_count(self):
        row, table, evaluator, declarations = self.selected_fixture()
        fields, visits = selected_episode(row, table, list(range(64)), evaluator, declarations)
        self.assertEqual(visits[0]["publicly_ranked_equal_cost_alternative_cells"], [CELLS[1], CELLS[2]])
        self.assertNotIn(CELLS[3], visits[0]["publicly_ranked_equal_cost_alternative_cells"])
        self.assertEqual(visits[0]["actual_same_cell_new"], 1)
        self.assertEqual(visits[0]["actual_global_new"], 0)
        self.assertEqual(visits[0]["alternative_actual_global_new"], 2)
        self.assertEqual(visits[0]["one_step_actual_difference"], -2)
        self.assertEqual(fields["actual_local"], 3)
        self.assertEqual(fields["actual_global"], 2)
        self.assertEqual(fields["cross_cell_duplicates"], 1)
        self.assertAlmostEqual(fields["total_estimate_minus_global"], fields["local_estimation_excess"] + fields["known_duplicate_excess"])

    def test_noncount_utility_is_not_called_expected_buildings(self):
        row, table, evaluator, declarations = self.selected_fixture()
        row["policy"] = "presence"
        fields, visits = selected_episode(row, table, list(range(64)), evaluator, declarations)
        self.assertIsNone(fields["expected_known_local"])
        self.assertIsNone(fields["local_estimation_excess"])
        self.assertIsNone(visits[0]["expected_same_cell_new_count"])

    def test_unknown_selected_cells_not_mislabeled_known_zero_count(self):
        row, table, evaluator, declarations = self.selected_fixture()
        evaluator[CELLS[4], "native"] = {"matches": [0] * 3, "truth": None}
        declarations[CELLS[4], "native"] = False
        row["trace"][0]["declared_target"] = False
        fields, visits = selected_episode(row, table, list(range(64)), evaluator, declarations)
        self.assertEqual(fields["unknown_actions"], 1)
        self.assertEqual(fields["known_actions"], 2)
        self.assertEqual(fields["expected_known_local"], 10.)

    def test_component_reconstructs_frozen_source_means_and_retains_empty_bins(self):
        selected = [{"sample_id": "a", "event": "e1"}, {"sample_id": "b", "event": "e2"}, {"sample_id": "u", "event": "e1"}]
        features = {sid: {CELLS[0]: {"change": raw}} for sid, raw in [("a", .1), ("b", .8), ("u", .5)]}
        pairs = {}
        for sid, truth, native in [("a", True, 1), ("b", False, 0), ("u", None, 0)]:
            pair = {}
            for mode, bits in [("native", native), ("blurred", 0)]:
                pair[mode] = {"sample_id": sid, "cell_id": CELLS[0], "mode": mode, "target": truth,
                    "predicted": bool(bits), "matched_target_uids": ["target"] if bits else [],
                    "visible_target_uids": ["target"] if truth else [], "bits": bits}
            pairs[sid, CELLS[0]] = pair
        scores = {sid: {cid: v["change"] for cid, v in values.items()} for sid, values in features.items()}
        def fitted(ids):
            rows = [r for (sid, cid), p in pairs.items() if sid in ids for r in p.values()]
            return {"count": fit_local_count(rows, scores), "fit_ids": sorted(ids)}
        models = {"development_fit": fitted({"a", "b", "u"}), "held_e1": fitted({"b"}), "held_e2": fitted({"a", "u"})}
        rows = component_calibration(models, features, pairs, selected)
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["held_unknown_pairs_excluded"] for r in rows), 2)
        self.assertTrue(any(r["held_known_cells"] == 0 and r["held_observed_mixed"] is None for r in rows))

    def test_oracle_trace_roundtrip_uses_legal_paid_sequence(self):
        rows = [{"sample_id": "s", "cell_id": cid, "mode": mode, "declared_target": False}
                for cid in CELLS for mode in ["native", "blurred"]]
        levels = [0] * 64
        for i in range(4):
            levels[i] = 2 if quality("mixed_survey", 20261025, "s", CELLS[i]) == "blurred" else 1
        trace = trace_from_levels(levels, rows, "s", 20261025)
        self.assertEqual(levels_from_trace(trace), levels)
        self.assertEqual(trace[-1]["cumulative_cost"], sum(1 if k == 1 else 3 if k == 2 else 0 for k in levels))
        for i, action in enumerate(trace):
            if action["kind"] == "upgrade":
                self.assertTrue(any(t["cell_id"] == action["cell_id"] and t["mode"] == "blurred" for t in trace[:i]))

    def test_event_macro_not_pooled_across_scenes(self):
        rows = [{"event": "a", "value": 1.}, {"event": "a", "value": 3.}, {"event": "b", "value": 10.}]
        events = aggregate(rows, ["event"], ["value"])
        macro = aggregate(events, [], ["value"])
        self.assertEqual(macro[0]["value"], 6.)


if __name__ == "__main__":
    unittest.main()
