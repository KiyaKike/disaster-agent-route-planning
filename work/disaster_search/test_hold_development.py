from copy import deepcopy
from pathlib import Path
import unittest

import pyarrow as pa

from materialize_hold_development import validate_development, extract_development
from materialize_hold_probes import COLUMNS


def selection():
    rows=[{"sample_id":f"{event}_{i:08d}","event":event,"split":"development"}
          for event in ["hurricane-harvey","santa-rosa-wildfire","mexico-earthquake"] for i in range(24)]
    amendment={"allowed_returned_damage_rows":{"development":[r["sample_id"] for r in rows]}}
    return rows,rows+[{"sample_id":"reserved","event":"palu-tsunami","split":"confirmation"}],amendment


class DevelopmentAccessTests(unittest.TestCase):
    def test_count_diagnostic_reports_unsupported_upgrade_instead_of_zero_prediction(self):
        from fit_hold_development import count_prediction_diagnostic
        from paired_count_baseline import fit_local_count
        training=[]
        for mode in ["blurred","native"]:
            training.append({"sample_id":"train","cell_id":"r00c00","mode":mode,"target":True,
                "predicted":mode=="native","matched_target_uids":["a"] if mode=="native" else [],"visible_target_uids":["a"]})
        model=fit_local_count(training,{"train":{"r00c00":.2}})
        held=[{**row,"sample_id":"held","predicted":True} for row in training]
        result=count_prediction_diagnostic(model,held,{"held":{"r00c00":.9}})
        self.assertEqual(result["unsupported_upgrade_conditions"],1)
        self.assertNotIn("upgrade",result["count_prediction"])

    def test_rgb_plan_ignores_poison_labels_and_blocks_confirmation(self):
        from run_hold_development_perception import make_plan
        rows,_,_=selection()
        manifest={"samples":[{**s,"files":{"pre_image":s["sample_id"]+"_pre.png","post_image":s["sample_id"]+"_post.png","post_label":"unused.json"}} for s in rows]}
        expected=make_plan(manifest,rows,Path("/tmp"))
        for item in manifest["samples"]:
            item["files"]["post_label"]="/must-not-read/poison.json";item["true_targets"]=999
        self.assertEqual(make_plan(manifest,rows,Path("/tmp")),expected)
        changed=deepcopy(rows);changed[0]["split"]="confirmation"
        with self.assertRaises(ValueError):make_plan(manifest,changed,Path("/tmp"))

    def test_subpixel_geometry_exception_preserves_coordinates_and_rejects_larger_error(self):
        from hold_development_geometry import build_features
        from materialize_hold_probes import build_features as strict_features
        row={"ml:polygon_xy":[-.35,1.,9.,1.,9.,9.,-.35,1.],"ml:polygon_len":[4],
             "ml:damage":[2],"ml:width":32,"ml:height":32}
        result=build_features("s",row)
        self.assertIn("-0.349999",result[0]["wkt"])
        with self.assertRaises(ValueError):strict_features("s",row)
        row["ml:polygon_xy"][0]=row["ml:polygon_xy"][-2]=-.6
        with self.assertRaises(ValueError):build_features("s",row)

    def test_only_original_72_ids_and_order_are_accepted(self):
        rows,cohort,amendment=selection()
        validate_development(rows,cohort,amendment)
        for changed in [rows[:-1],list(reversed(rows)),rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):validate_development(changed,cohort,amendment)

    def test_reserve_row_or_wrong_event_cannot_enter_allowlist(self):
        rows,cohort,amendment=selection()
        changed=deepcopy(rows);changed[0]["split"]="confirmation"
        with self.assertRaises(ValueError):validate_development(changed,cohort,amendment)
        changed=deepcopy(rows);changed[0]["sample_id"]="reserved"
        with self.assertRaises(ValueError):validate_development(changed,cohort,amendment)
        changed=deepcopy(rows);changed[0]["event"]="palu-tsunami"
        with self.assertRaises(ValueError):validate_development(changed,changed,amendment)

    def test_trusted_extractor_returns_no_reserved_row(self):
        rows,cohort,amendment=selection()
        identities=[{"ml:source_file":"reserved-before"}]+[{"ml:source_file":r["sample_id"]} for r in rows]+[{"ml:source_file":"reserved-after"}]
        table=pa.Table.from_pylist([{"ml:damage":[i]} for i in range(74)])
        class Parquet:
            def read(self,*,columns,use_threads):
                assert columns==COLUMNS and use_threads is False
                return table
        result=extract_development(Parquet(),identities,rows,cohort,amendment)
        self.assertEqual(list(result),[r["sample_id"] for r in rows])
        self.assertEqual(result[rows[0]["sample_id"]]["ml:damage"],[1])
        self.assertNotIn("reserved-before",result)
        self.assertNotIn("reserved-after",result)


if __name__ == "__main__":unittest.main()
