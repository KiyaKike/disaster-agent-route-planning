"""Development-only paired sensor model and local incremental-count utility.

The action utility predicts newly matched buildings within a cell, including
native-minus-blurred set difference for upgrades. It does not yet model a
building already discovered in a DIFFERENT cell. Do not call it an exact
global distinct-building planner or a validated new method.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


OUTCOMES = ["00", "01", "10", "11"]  # blurred, native


def fit_joint(pairs, alpha=.5):
    if not alpha > 0:
        raise ValueError("Joint pseudocount must be positive")
    counts = {str(t): {outcome: 0 for outcome in OUTCOMES} for t in [0, 1]}
    for truth, blurred, native in pairs:
        if any(type(value) is not bool for value in [truth, blurred, native]):
            raise ValueError("Joint observations must be known booleans")
        counts[str(int(truth))][f"{int(blurred)}{int(native)}"] += 1
    return {"outcome_order": "blurred,native", "alpha_per_outcome": alpha, "counts": counts,
            "probabilities": {t: {o: (n + alpha) / (sum(values.values()) + 4 * alpha)
                                  for o, n in values.items()} for t, values in counts.items()}}


def joint_posterior(prior, model, *, blurred=None, native=None):
    if not 0 < prior < 1 or any(v is not None and type(v) is not bool for v in [blurred, native]):
        raise ValueError("Bounded prior and optional boolean observations required")
    compatible = [o for o in OUTCOMES if (blurred is None or int(o[0]) == blurred) and (native is None or int(o[1]) == native)]
    likelihood = {t: sum(model["probabilities"][t][o] for o in compatible) for t in ["0", "1"]}
    return prior * likelihood["1"] / (prior * likelihood["1"] + (1 - prior) * likelihood["0"])


def native_positive_given_blur(model, truth, blurred):
    if type(truth) is not bool or type(blurred) is not bool:
        raise ValueError("Boolean truth class and observed blurred result required")
    values = model["probabilities"][str(int(truth))]
    prefix = str(int(blurred))
    return values[prefix + "1"] / (values[prefix + "0"] + values[prefix + "1"])


def fit_local_count(association_rows, public_scores, shrinkage=8):
    """Evaluator labels may fit aggregate parameters offline, never action inputs."""
    if not shrinkage > 0:
        raise ValueError("Count shrinkage must be positive")
    pairs = defaultdict(dict)
    for row in association_rows:
        key = (row["sample_id"], row["cell_id"])
        if row["mode"] not in {"native", "blurred"} or row["mode"] in pairs[key]:
            raise ValueError("Invalid or duplicated quality observation")
        pairs[key][row["mode"]] = row
    if not pairs or any(set(rows) != {"native", "blurred"} for rows in pairs.values()):
        raise ValueError("Every cell requires exactly one paired observation")
    all_scores = [public_scores[sid][cid] for sid, cid in pairs]
    if not all(math.isfinite(score) for score in all_scores):
        raise ValueError("Non-finite public cue")
    edges = np.quantile(all_scores, [.25, .5, .75]).tolist()
    samples, known, unknown = [], [], 0
    for (sid, cid), pair in pairs.items():
        b, n = pair["blurred"], pair["native"]
        if b["target"] is not n["target"]:
            raise ValueError("Paired quality cannot change truth")
        if n["target"] is None:
            unknown += 1
            continue
        known.append((n["target"], b["predicted"], n["predicted"]))
        bm, nm = set(b["matched_target_uids"]), set(n["matched_target_uids"])
        if not bm.issubset(set(b["visible_target_uids"])) or not nm.issubset(set(n["visible_target_uids"])):
            raise ValueError("Evaluator match is not a visible target")
        if n["target"] is False and (bm or nm):
            raise ValueError("Known-negative cell cannot have matched targets")
        bin_number = int(np.searchsorted(edges, public_scores[sid][cid], side="right"))
        samples.append({"bin": bin_number, "blurred_positive": b["predicted"],
                        "native": len(nm), "blurred": len(bm), "upgrade_increment": len(nm - bm)})
    if not samples:
        raise ValueError("No known cells to fit")
    global_mean = {mode: sum(r[mode] for r in samples) / len(samples) for mode in ["native", "blurred"]}
    global_upgrade = {}
    for positive in [False, True]:
        subset = [r for r in samples if r["blurred_positive"] == positive]
        global_upgrade[str(int(positive))] = {"cells": len(subset),
            "mean": sum(r["upgrade_increment"] for r in subset) / len(subset) if subset else None}
    bins = []
    for number in range(4):
        subset = [r for r in samples if r["bin"] == number]
        survey = {mode: (sum(r[mode] for r in subset) + shrinkage * global_mean[mode]) / (len(subset) + shrinkage)
                  for mode in ["native", "blurred"]}
        upgrades = {}
        for positive in [False, True]:
            conditioned = [r for r in subset if r["blurred_positive"] == positive]
            parent = global_upgrade[str(int(positive))]["mean"]
            upgrades[str(int(positive))] = {"cells": len(conditioned), "mean_new_within_cell":
                (sum(r["upgrade_increment"] for r in conditioned) + shrinkage * parent) / (len(conditioned) + shrinkage)
                if parent is not None else None}
        bins.append({"bin": number, "known_cells": len(subset), "survey_mean_count": survey, "upgrade": upgrades})
    return {"stage": "DEVELOPMENT_INTERFACE_FIT_NOT_VALIDATED", "paired_known_cells": len(samples),
            "unknown_pairs_excluded": unknown, "source_scenes": len({sid for sid, _ in pairs}),
            "public_quantile_edges": edges, "shrinkage_pseudo_cells": shrinkage,
            "paired_sensor": fit_joint(known), "global_survey_mean_count": global_mean,
            "global_upgrade": global_upgrade, "bins": bins,
            "limits": ["Only same-cell prior discovery is removed from upgrade rewards; cross-cell duplicate discovery is not modeled.",
                       "No reliable quality estimator or new viewpoint: fixed synthetic modes.",
                       "Fit on development data only; six interface scenes cannot establish transfer or policy improvement."]}


def action_score(model, raw_change_score, kind, *, scenario="mixed_survey", blurred_positive=None, cost=1):
    """Only public cue, observed outcome and aggregate fitted parameters enter."""
    if not math.isfinite(raw_change_score) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("Finite public score and positive cost required")
    number = int(np.searchsorted(model["public_quantile_edges"], raw_change_score, side="right"))
    params = model["bins"][number]
    if kind == "survey":
        native_weight = {"native_survey": 1, "mixed_survey": .5, "blurred_survey": 0}[scenario]
        return (native_weight * params["survey_mean_count"]["native"] +
                (1 - native_weight) * params["survey_mean_count"]["blurred"]) / cost
    if kind != "upgrade" or type(blurred_positive) is not bool:
        raise ValueError("Upgrade needs the actual earlier blurred boolean; no other kind supported")
    gain = params["upgrade"][str(int(blurred_positive))]["mean_new_within_cell"]
    return None if gain is None else gain / cost


def main():
    import argparse
    import json
    from pathlib import Path
    from build_xbd_mini import atomic_json, sha
    from plan_search_v2 import freeze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    association_path = args.perception / "association_report.json"
    scores_path = args.perception / "public_change_scores.json"
    association = json.loads(association_path.read_text())
    scores = json.loads(scores_path.read_text())
    expected = {r["sample_id"] for r in json.loads((args.design / "next_development_probes.json").read_text())}
    if association["stage"] != "DEVELOPMENT_PROBE_SENSOR_INTERFACE_DIAGNOSTIC" or set(scores) != expected:
        raise ValueError("This smoke fitter accepts only the six frozen development probes")
    if {r["sample_id"] for r in association["rows"]} != expected or len(association["rows"]) != 768:
        raise ValueError("Unexpected evaluator rows")
    freeze(args.out / "fit_protocol.json", {
        "stage": "SIX_DEVELOPMENT_COMPONENT_SMOKE_ONLY", "shrinkage_pseudo_cells": 8,
        "joint_pseudocount_per_outcome": .5,
        "association_sha256": sha(association_path.read_bytes()), "public_scores_sha256": sha(scores_path.read_bytes()),
        "method": "Joint paired B,N likelihoods conditioned on known cell truth; expected same-cell native-minus-blurred matched UID count, conditioned on observed B.",
        "no_policy_result": True, "cross_cell_duplicate_reward_model": "NOT_IMPLEMENTED",
        "runner_sha256": sha(Path(__file__).read_bytes())})
    model = fit_local_count(association["rows"], scores)
    model["source_hashes"] = {"association": sha(association_path.read_bytes()), "public_scores": sha(scores_path.read_bytes()),
                              "fit_protocol": sha((args.out / "fit_protocol.json").read_bytes())}
    atomic_json(args.out / "component_parameters.json", model)
    summary = {"stage": model["stage"], "source_scenes": model["source_scenes"],
        "paired_known_cells": model["paired_known_cells"], "unknown_pairs_excluded": model["unknown_pairs_excluded"],
        "global_survey_mean_count": model["global_survey_mean_count"], "global_upgrade": model["global_upgrade"],
        "native_positive_given_blurred_negative": {
            "target_cell": native_positive_given_blur(model["paired_sensor"], True, False),
            "negative_cell": native_positive_given_blur(model["paired_sensor"], False, False)},
        "unavailable_blur_outcome_conditions": [b for b, values in model["global_upgrade"].items() if values["mean"] is None],
        "search_policy_evaluated": False,
        "full_72_scene_fit_completed": False, "cross_cell_deduplication_model_completed": False}
    atomic_json(args.out / "report.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
