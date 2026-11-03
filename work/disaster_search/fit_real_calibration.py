"""Fit calibration-only sensor rates and a coarse-image presence prior.

Outputs aggregate fitted parameters, never evaluator building IDs. These are
in-sample estimates from two events, not validated probabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_changeos_pilot import counts
from run_changeos_pilot import freeze_plan


FIT_PROTOCOL = {
    "version": "real_calibration_fit_v1", "split": "calibration",
    "sensor": "Separate native/blurred likelihoods from fixed declared_target vs raster-visible presence; exclude null truth. Beta(0.5,0.5) posterior means for sensitivity and false-positive rate.",
    "prior": "Four equal-frequency raw-change bins, quantile edges computed from image-only scores for all calibration cells. Each bin presence frequency shrunk toward the global Jeffreys-smoothed prevalence with 8 pseudo-observations. Null truths excluded from rate fitting.",
    "correlation": "Each base cell occurs once per mode. Modes are not pooled as independent evidence; image registry keeps one representative per base cell. No binomial confidence intervals across correlated cells.",
    "quality": "Mode is a controlled image degradation, not a measured real-world image-quality estimator. Do not use unknown future observation quality to select actions.",
    "validation": "No threshold tuning, no validation/test labels. Event-specific diagnostics do not change chosen parameters. Calibrated parameter name means fitted on calibration split, not verified out of sample.",
    "boundary_protocol": "raster_visibility_v2, motivated posthoc by pilot. Cell presence includes visible boundary fragments of >=4 pixels; unique building discovery uses separate deduplicated evaluator.",
}


def fit_likelihood(rows):
    known = [r for r in rows if r["target"] is not None]
    result = counts(known)
    result["unknown_excluded"] = len(rows) - len(known)
    result["smoothed_sensitivity"] = (result["tp"] + .5) / (result["tp"] + result["fn"] + 1)
    result["smoothed_false_positive_rate"] = (result["fp"] + .5) / (result["fp"] + result["tn"] + 1)
    return result


def fit_prior(score_truth):
    edges = np.quantile([score for score, _ in score_truth], [.25, .5, .75]).tolist()
    known = [(score, truth) for score, truth in score_truth if truth is not None]
    prevalence = (sum(truth for _, truth in known) + .5) / (len(known) + 1)
    bins = [{"bin": i, "known_cells": 0, "positive_cells": 0, "unknown_cells": 0} for i in range(4)]
    for score, truth in score_truth:
        i = int(np.searchsorted(edges, score, side="right"))
        if truth is None:
            bins[i]["unknown_cells"] += 1
        else:
            bins[i]["known_cells"] += 1
            bins[i]["positive_cells"] += int(truth)
    for row in bins:
        row["fitted_presence_probability"] = (row["positive_cells"] + 8 * prevalence) / (row["known_cells"] + 8)
    return {"input": "raw_change_score", "bin_edges": edges, "edge_assignment": "searchsorted(side=right)",
            "global_smoothed_prevalence": prevalence, "shrinkage_pseudo_observations": 8, "bins": bins,
            "note": "Not forced monotone: RGB change may be unrelated to severe damage. Fitted calibration values only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    freeze_plan(args.out / "fit_protocol.json", FIT_PROTOCOL)
    if args.plan_only:
        print("Frozen calibration fit rules")
        return
    report = json.loads(args.association.read_text())
    observations = json.loads(args.observations.read_text())
    if report["stage"] != "FULL_CALIBRATION_BOUNDARY_DIAGNOSTICS":
        raise ValueError("Use complete calibration, not the inspected pilot")
    if hashlib.sha256(args.observations.read_bytes()).hexdigest() != report["observations_sha256"]:
        raise ValueError("Association and observations differ")
    if len(observations) != 3072 or any(o["split"] != "calibration" for o in observations):
        raise ValueError("Unexpected inference scope")
    rows = report["rows"]
    modes = sorted({r["mode"] for r in rows})
    scores, events, cue_hashes = {}, {}, {}
    for sid in sorted({r["sample_id"] for r in rows}):
        path = args.prepared_data / sid / "public_cues.json"
        cues = json.loads(path.read_text())
        if cues["split"] != "calibration":
            raise ValueError("Non-calibration cues")
        cue_hashes[sid] = hashlib.sha256(path.read_bytes()).hexdigest()
        events[sid] = cues["event"]
        scores.update({(sid, c["cell_id"]): c["raw_change_score"] for c in cues["cells"]})
    truth = {}
    for row in rows:
        key = row["sample_id"], row["cell_id"]
        if key in truth and truth[key] != row["target"]:
            raise ValueError("Quality variant changed truth")
        truth[key] = row["target"]
    if len(truth) != 1536:
        raise ValueError("Expected 1536 unique regions")
    rates = {mode: fit_likelihood([r for r in rows if r["mode"] == mode]) for mode in modes}
    prior = fit_prior([(scores[k], truth[k]) for k in sorted(truth)])
    per_event = {event: {mode: fit_likelihood([r for r in rows if r["mode"] == mode and events[r["sample_id"]] == event])
                         for mode in modes} for event in sorted(set(events.values()))}
    parameters = {"stage": "FITTED_ON_CALIBRATION_NOT_VALIDATED", "fit_protocol": FIT_PROTOCOL,
                  "likelihood_by_mode": rates, "initial_presence_prior": prior,
                  "fit_scenes": 24, "fit_events": sorted(set(events.values())), "base_regions": 1536,
                  "source_observations_sha256": report["observations_sha256"],
                  "source_association_sha256": hashlib.sha256(args.association.read_bytes()).hexdigest(),
                  "public_cue_hashes": cue_hashes, "test_used": False, "validation_used": False}
    (args.out / "calibration_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2))
    diagnostic = {"stage": parameters["stage"], "event_rates": per_event,
                  "per_scene_rates": {sid: {mode: fit_likelihood([r for r in rows if r["sample_id"] == sid and r["mode"] == mode])
                                            for mode in modes} for sid in sorted(events)},
                  "limitations": ["Strong within-scene spatial dependence; events, not cells, determine generalization.",
                      "Two fit events cannot establish reliability under new hazards or actual UAV imaging.",
                      "Checkpoint exposure expected; local context shift and artificial blur remain.",
                      "A quality-aware likelihood alone does not re-rank unvisited cells without a specified information-sharing mechanism."]}
    (args.out / "calibration_diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    print(json.dumps({"likelihood_by_mode": rates, "initial_prior": prior, "event_rates": per_event}, ensure_ascii=False))


if __name__ == "__main__":
    main()
