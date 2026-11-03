"""Evaluator-only calibration diagnosis of feedback neighborhoods.

Neighbor truth is never used to produce a feedback update. Shared target UIDs
are excluded in the primary analysis so boundary fragments cannot create a
spurious justification for spreading feedback to different buildings.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from run_cached_search import digest, update_probability
from run_changeos_pilot import freeze_plan


RELATIONS = ("same_bin", "neighbors", "neighbors_same_bin", "far_same_bin")
PROTOCOL = {
    "version": "feedback_scope_diagnostics_v1", "split": "calibration",
    "timing": "After inspecting the first cached search failure; exploratory mechanism diagnosis, not preregistered evidence or validation.",
    "relations": {"same_bin": "Same public RGB-change quartile, any distance",
                  "neighbors": "8-connected one-cell neighborhood, irrespective of change bin",
                  "neighbors_same_bin": "8-connected neighbor AND same public RGB-change quartile",
                  "far_same_bin": "Same public RGB-change quartile, Chebyshev distance >1"},
    "primary": "Ordered distinct known-truth source/candidate cells within each scene; exclude pairs sharing any visible severe-building UID. Also report inclusive diagnostic separately.",
    "conditional": "Candidate prevalence conditional on source truth, within each scene. Equal-scene mean difference only for scenes containing both source truth classes and eligible candidate pairs. Full per-scene values retained; no independence-based p-values.",
    "one_step_predictor": "From original source prior and its fixed detector observation, compute posterior with the already fitted mode likelihood. Shift candidate logit by clip(4*r*(source_posterior-source_prior)/(8+r),-1,1), r=(TPR-FPR)^2, unchanged synthetic constants. Candidate labels used only to score Brier loss.",
    "one_step_comparison": "After-minus-before Brier, averaged candidate pairs per source, then sources per scene, then scenes equally. Positive values mean harm. Report native/blurred, positive/negative observation, per-event and per-scene. Primary excludes shared target UIDs.",
    "limits": ["Labels serve evaluator diagnostics only, not an online neighborhood or known false-negative flag.",
        "Calibration prior and sensor rates were fitted using these scenes, so these scores are exploratory and in-sample.",
        "Conditional association can reflect scene density, layout or sampling; it is not a causal effect or proof of useful policy feedback.",
        "A fixed single-feedback test is different from an adaptive multi-step policy."]}


def related(a, b, relation):
    if a["cell_id"] == b["cell_id"]:
        return False
    near = max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"])) <= 1
    same = a["group"] == b["group"]
    return {"same_bin": same, "neighbors": near, "neighbors_same_bin": near and same,
            "far_same_bin": same and not near}[relation]


def eligible(a, b, relation, exclude_shared):
    return (a["target"] is not None and b["target"] is not None and related(a, b, relation)
            and not (exclude_shared and set(a["uids"]) & set(b["uids"])))


def one_step_candidate(source_prior, candidate_prior, positive, rates):
    post = update_probability(source_prior, positive, rates)
    r = (rates["tpr"] - rates["fpr"]) ** 2
    shift = max(-1., min(1., 4 * r * (post - source_prior) / (8 + r)))
    value = 1 / (1 + math.exp(-(math.log(candidate_prior / (1 - candidate_prior)) + shift)))
    return value, shift


def scene_diagnostic(sid, event, cells, relation, exclude_shared, rates):
    conditionals = {False: [], True: []}
    feedback = defaultdict(list)
    source_counts = defaultdict(int)
    excluded_shared_pairs = 0
    pair_count = 0
    for source in cells:
        candidates = []
        for target in cells:
            if not eligible(source, target, relation, False):
                continue
            if exclude_shared and set(source["uids"]) & set(target["uids"]):
                excluded_shared_pairs += 1
                continue
            candidates.append(target)
        if not candidates:
            continue
        pair_count += len(candidates)
        conditionals[source["target"]].append(statistics.mean(t["target"] for t in candidates))
        for mode in ("native", "blurred"):
            observation = source["observations"][mode]
            before, after = [], []
            for target in candidates:
                value, _ = one_step_candidate(source["prior"], target["prior"], observation, rates[mode])
                before.append((target["prior"] - target["target"]) ** 2)
                after.append((value - target["target"]) ** 2)
            key = f"{mode}:{'positive' if observation else 'negative'}"
            feedback[key].append(statistics.mean(after) - statistics.mean(before))
            source_counts[key] += 1
    positive = statistics.mean(conditionals[True]) if conditionals[True] else None
    negative = statistics.mean(conditionals[False]) if conditionals[False] else None
    return {"sample_id": sid, "event": event, "relation": relation, "exclude_shared_target_uid": exclude_shared,
            "ordered_pairs": pair_count, "shared_uid_pairs_excluded": excluded_shared_pairs,
            "candidate_rate_given_positive_source_truth": positive,
            "candidate_rate_given_negative_source_truth": negative,
            "within_scene_conditional_difference": positive - negative if positive is not None and negative is not None else None,
            "eligible_positive_truth_sources": len(conditionals[True]), "eligible_negative_truth_sources": len(conditionals[False]),
            "one_step_brier_delta": {k: statistics.mean(v) for k, v in feedback.items()},
            "one_step_source_counts": dict(source_counts)}


def aggregate(rows):
    contrasts = [r["within_scene_conditional_difference"] for r in rows if r["within_scene_conditional_difference"] is not None]
    keys = sorted({k for r in rows for k in r["one_step_brier_delta"]})
    return {"scenes": len(rows), "ordered_pairs": sum(r["ordered_pairs"] for r in rows),
            "shared_uid_pairs_excluded": sum(r["shared_uid_pairs_excluded"] for r in rows),
            "scenes_with_both_source_truth_classes": len(contrasts),
            "equal_scene_conditional_difference": statistics.mean(contrasts) if contrasts else None,
            "one_step_brier_delta": {k: {"equal_scene_mean": statistics.mean(r["one_step_brier_delta"][k] for r in rows if k in r["one_step_brier_delta"]),
                "scenes": sum(k in r["one_step_brier_delta"] for r in rows),
                "scenes_helped": sum(r["one_step_brier_delta"].get(k, 0) < 0 for r in rows),
                "scenes_harmed": sum(r["one_step_brier_delta"].get(k, 0) > 0 for r in rows),
                "sources": sum(r["one_step_source_counts"].get(k, 0) for r in rows)} for k in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "source_hashes": {k: digest(getattr(args, k)) for k in ("association", "observations", "parameters")}}
    freeze_plan(args.out / "diagnostic_protocol.json", protocol)
    if args.plan_only:
        print("Frozen feedback-scope diagnostic definitions")
        return
    params = json.loads(args.parameters.read_text())
    observations = json.loads(args.observations.read_text())
    association = json.loads(args.association.read_text())
    if len(observations) != 3072 or any(r["split"] != "calibration" for r in observations) or association["stage"] != "FULL_CALIBRATION_BOUNDARY_DIAGNOSTICS":
        raise ValueError("Only complete calibration inputs accepted")
    if params["source_observations_sha256"] != digest(args.observations) or params["source_association_sha256"] != digest(args.association):
        raise ValueError("Fitted parameters/source mismatch")
    observation_index = {(r["sample_id"], r["cell_id"], r["mode"]): r for r in observations}
    truth = {(r["sample_id"], r["cell_id"]): r for r in association["rows"] if r["mode"] == "native"}
    prior = params["initial_presence_prior"]
    rates = {m: {"tpr": r["smoothed_sensitivity"], "fpr": r["smoothed_false_positive_rate"]} for m, r in params["likelihood_by_mode"].items()}
    rows = []
    for sid, expected in sorted(params["public_cue_hashes"].items()):
        path = args.prepared_data / sid / "public_cues.json"
        if digest(path) != expected:
            raise ValueError("Public cues changed")
        cues = json.loads(path.read_text())
        cells = []
        for c in cues["cells"]:
            group = bisect_right(prior["bin_edges"], c["raw_change_score"])
            row = truth[(sid, c["cell_id"])]
            cells.append({"cell_id": c["cell_id"], "x": c["col"], "y": c["row"], "group": group,
                          "prior": prior["bins"][group]["fitted_presence_probability"],
                          "target": row["target"], "uids": row["visible_target_uids"],
                          "observations": {mode: observation_index[(sid, c["cell_id"], mode)]["declared_target"] for mode in ("native", "blurred")}})
        for relation in RELATIONS:
            for exclude in (False, True):
                rows.append(scene_diagnostic(sid, cues["event"], cells, relation, exclude, rates))
    summary = []
    for relation in RELATIONS:
        for exclude in (False, True):
            selected = [r for r in rows if r["relation"] == relation and r["exclude_shared_target_uid"] == exclude]
            summary.append({"relation": relation, "exclude_shared_target_uid": exclude, **aggregate(selected),
                            "by_event": {event: aggregate([r for r in selected if r["event"] == event]) for event in sorted({r["event"] for r in rows})}})
    report = {"stage": "EXPLORATORY_CALIBRATION_FEEDBACK_SCOPE", "protocol_sha256": digest(args.out / "diagnostic_protocol.json"),
              "runner_sha256": digest(Path(__file__)), "summary": summary, "scene_diagnostics": rows,
              "validation_used": False, "test_used": False, "limitations": PROTOCOL["limits"]}
    (args.out / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps([{k: r[k] for k in ("relation", "equal_scene_conditional_difference", "one_step_brier_delta", "shared_uid_pairs_excluded")}
                      for r in summary if r["exclude_shared_target_uid"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
