"""Evaluate frozen calibration-pilot observations after inference; no tuning."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def counts(rows):
    tp = sum(r["predicted"] and r["target"] for r in rows)
    fp = sum(r["predicted"] and not r["target"] for r in rows)
    fn = sum(not r["predicted"] and r["target"] for r in rows)
    tn = sum(not r["predicted"] and not r["target"] for r in rows)
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--full-calibration", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    observations_file = args.pilot / "observations.json"
    observations = json.loads(observations_file.read_text())
    if args.validation and args.full_calibration:
        raise ValueError("Choose one evaluation split")
    selected_split = "validation" if args.validation else "calibration"
    if (args.full_calibration or args.validation) and (len(observations) != 3072 or len({o["sample_id"] for o in observations}) != 24):
        raise ValueError("Full calibration requires 24 scenes and 3072 observations")
    groups, unknowns, rows, truths = defaultdict(list), defaultdict(int), [], {}
    seen = set()
    for observation in observations:
        if observation["split"] != selected_split:
            raise ValueError("Observation/evaluator split mismatch")
        oid = observation["observation_id"]
        if oid in seen:
            raise ValueError("Duplicate observation ID")
        seen.add(oid)
        sid = observation["sample_id"]
        if sid not in truths:
            truths[sid] = json.loads((args.prepared_data / sid / "evaluation_truth.json").read_text())["cells"]
        target = truths[sid][observation["cell_id"]]["target_present"]
        if target is None:
            unknowns[observation["mode"]] += 1
            continue
        row = {"sample_id": sid, "cell_id": observation["cell_id"], "mode": observation["mode"],
               "target": target, "predicted": observation["declared_target"],
               "uncalibrated_score": observation["uncalibrated_severe_score"]}
        groups[observation["mode"]].append(row)
        rows.append(row)
    report = {"stage": "VALIDATION_CENTROID_DIAGNOSTICS_ONLY" if args.validation else ("CALIBRATION_CENTROID_DIAGNOSTICS_ONLY" if args.full_calibration else "CALIBRATION_PILOT_DIAGNOSTICS_ONLY"), "test_evaluation_run": False,
              "observation_sha256": hashlib.sha256(observations_file.read_bytes()).hexdigest(),
              "results_by_mode": {mode: {**counts(values), "unknown_excluded": unknowns[mode]} for mode, values in groups.items()},
              "rows": rows,
              "limitations": ["All 1536 cells from 24 validation scenes; fixed rules, no fitting." if args.validation else ("All 1536 cells from 24 calibration scenes; fitted diagnostics, not a model benchmark." if args.full_calibration else "Only 32 predetermined cells from four calibration scenes; not a model benchmark."),
                              "No threshold or calibration fitted here; rule was fixed before inference.",
                              "Grid truth is assigned by building centroid, while predictions are pixel based; buildings crossing grid edges can create association mismatches.",
                              "Paired degradation modes share imagery and are correlated.",
                              "Checkpoint training exposure is expected. Do not interpret these values as unseen-event accuracy."]}
    (args.pilot / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"stage": report["stage"], "results_by_mode": report["results_by_mode"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
