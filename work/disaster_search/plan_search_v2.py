"""Freeze a metadata-only candidate cohort before reading its damage outcomes.

This creates a design contract, not a runnable/approved benchmark. Annotation,
geographic, measurement and baseline implementation gates remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_xbd_mini import atomic_json, sha


SEED = 20261025
EVENTS = {
    "development": ["hurricane-harvey", "santa-rosa-wildfire", "mexico-earthquake"],
    "validation": ["hurricane-michael", "midwest-flooding", "socal-fire"],
    "confirmation": ["hurricane-florence", "hurricane-matthew", "palu-tsunami"],
}
PER_EVENT = 24


def select_cohort(pools):
    if set(pools) != {"train", "tier3", "test", "hold"}:
        raise ValueError("Expected the four audited author pools")
    for name, values in pools.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {name}")
    pool_sets = {name: set(values) for name, values in pools.items()}
    names = sorted(pool_sets)
    if any(pool_sets[a] & pool_sets[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError("Author pools overlap")
    rows = []
    for split, events in EVENTS.items():
        for event in events:
            candidates = [sid for sid in pools["hold"] if sid.rsplit("_", 1)[0] == event]
            if len(candidates) < PER_EVENT:
                raise ValueError(f"Insufficient metadata-only pool: {event}")
            def rank(sid):
                return hashlib.sha256(f"disaster-search-v2|{SEED}|{event}|{sid}".encode()).hexdigest()
            for number, sid in enumerate(sorted(candidates, key=lambda sid: (rank(sid), sid))[:PER_EVENT]):
                rows.append({"sample_id": sid, "event": event, "split": split,
                             "selection_rank": number + 1, "selection_hash": rank(sid),
                             "interface_probe": split == "development" and number < 2})
    if len({r["sample_id"] for r in rows}) != 216:
        raise ValueError("Unexpected cohort")
    return rows


def freeze(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Refusing to overwrite frozen design: {path}")
    else:
        atomic_json(path, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pools = json.loads(args.identities.read_text())
    rows = select_cohort(pools)
    protocol = {
        "version": "search_v2_design_20261025", "status": "DESIGN_LOCKED_DATA_NOT_ADMITTED",
        "objective": "Maximize expected NEW distinct major-damage/destroyed building discoveries under an observation budget; engineering image-search proxy.",
        "source": {"repo": "EVER-Z/torchange_xView2", "revision": "cda47e44377258222166dac63a85068e8bdca446",
                   "pool": "hold", "identities_sha256": sha(args.identities.read_bytes())},
        "sampling": {"seed": SEED, "per_event": PER_EVENT, "events": EVENTS,
                     "rank": "SHA256(disaster-search-v2|seed|event|sample_id), ascending; sample_id breaks ties",
                     "excluded_event": {"guatemala-volcano": "Only five hold identities; excluded before outcomes to keep equal 24-scene event cohorts."},
                     "no_replacement": "Missing files, invalid annotations, target sparsity or failed admission never trigger automatic swaps or outcome-based expansion.",
                     "interface_probes": "First two selected IDs in each development event only (six total). No confirmation damage payload access during admission."},
        "split_meaning": {
            "development": "All sensor/prior/count-utility fitting, interface QA and mechanism exploration; overlaps previously explored event types.",
            "validation": "One prespecified baseline screening run after implementations are frozen; prior Michael event exploration is disclosed.",
            "confirmation": "Florence/Matthew/Palu have not been used in mini72 policy experiments; use only after complete baseline/method lock.",
            "perception": "All nine events occur in the frozen detector training pool. ID-disjoint hold cannot establish unseen-event perception generalization.",
        },
        "measurement": {
            "unit": "Original (sample_id, building_uid), evaluator-only; never connected components substituted for original instances.",
            "primary": "For each event average distinct discovered buildings per scene, then average events equally at budget 32, mixed quality, native-upgrade cost 2.",
            "legacy_discovery_proxy": "raster_visibility_v2: >=4 overlapping severe pixels and >=10% visible-footprint coverage; explicit proxy, not verified real-world identification.",
            "required_measurement_gate": "Before policy scores, inspect the six development interface probes for UID, polygons, alignment and boundary behavior; re-score all fixed trajectories at 5/10/25% coverage. A changed measurement definition requires a new version before validation/confirmation outcomes.",
            "secondary": ["Event-wise and scene-wise building recall; no-target recall null", "Known-negative false-positive cells and unmatched positives", "Discovery-versus-cost curve and censored first-discovery cost", "Raw counts and event target composition; no target-free event counted as a positive replication"],
        },
        "observations": {
            "grid": "64 non-overlapping 128x128 cells in a 1024x1024 image; coarse RGB prior available initially; truth private.",
            "sensor": "Existing pinned ChangeOS model and frozen image preprocessing; no retraining in this stage.",
            "survey": "Cost 1; fixed hidden quality assignment per cell; future actual quality unavailable to planner.",
            "upgrade": "One native observation after any blurred survey, including a blurred positive. Upgrade is not an independent image. No repeated native action.",
            "positive_cells": "No blanket positive-cell terminal rule before the eligible native upgrade; expected additional building reward may be nonzero.",
            "quality_scenarios": ["native_survey", "mixed_survey", "blurred_survey"],
            "quality_seeds": [20261025, 20261026, 20261027, 20261028, 20261029],
            "budgets": [8, 16, 32], "upgrade_costs": [1, 2, 4],
            "cost_limits": "Declared observation costs only. No inference of actual flight/energy savings. No new UAV viewpoint or genuine recovery of lost image information.",
        },
        "acquisition": {
            "utility": "Expected incremental distinct-building discovery proxy / action cost, fitted on development; existence-only utility retained as an explicit ablation.",
            "correlation": "Joint native/blurred observation model or conditional native likelihood given observed blurred outcome; never multiply marginal likelihoods as independent and never call evidence replacement an exact joint posterior.",
            "count_proxy": "Any observable component/score feature is a prediction feature, not the evaluator building UID or a claimed true instance count.",
            "implementation_status": "NOT_IMPLEMENTED: count-utility model and correlated baseline must be specified and unit-checked before the validation run.",
        },
        "comparisons": {
            "required_simple": ["random survey", "spatial coverage survey", "prior survey", "random with upgrade", "prior with simple quality gate", "development-fitted count-greedy with correlated upgrade likelihood"],
            "required_literature": "Specify a budget-compatible ENS/MF-ENS or rollout adaptation before any new-method claim; do not present an adaptation as a faithful reproduction without verification.",
            "fairness": "Same allowed observations, action eligibility, costs, quality maps and fitting data; report actual upgrades and planning wall time.",
            "new_method": "NONE_SELECTED; local residual propagation and short-prefix switching are previous exploratory candidates, not established contributions.",
            "random_order_seeds": list(range(20261030, 20261040)),
        },
        "research_questions": [
            "Does the coarse image prior retain discovery value over coverage across new hold images and multiple events?",
            "What share of benefit is due to coverage/ranking versus the offered native upgrade, and is it robust to its declared cost?",
            "After count-objective alignment and correlated likelihoods, is there enough remaining gain to justify a new adaptive mechanism?",
        ],
        "admission_gates": [
            "Repackaged hold names map to the audited author hold pool; original UID and XY polygons are retained and traceable.",
            "RGB shape/orientation and canonical pixel hashes match overlapping original examples; masks/unknown semantics checked on development probes only.",
            "Audit exact image duplicates and available geographic overlap across policy splits, and disclose detector-training adjacency separately. Unavailable scene footprints are recorded as unverified.",
            "No damage outcomes determine cohort membership. At least three target-bearing confirmation events are required for a three-event positive replication claim; otherwise restrict conclusions and do not swap samples.",
            "Measurement and count/correlation baseline implementation gates complete before validation/confirmation policy scoring.",
        ],
        "decision_rule": {
            "current": "No new controller training or paid GPU until data and measurement admission passes.",
            "validation_go": "For a later prespecified candidate: >=10% relative primary gain over each required simple baseline, >=0.1 additional building/scene in event-macro mean, positive gains in >=2/3 target-bearing validation events, and <=0.05 extra false-positive cells/scene. These are engineering effect-size targets, not significance tests.",
            "failure": "Report failure and stop adding parameters to that candidate; a changed hypothesis requires a new version and unconsumed confirmation evidence.",
            "statistics": "Average repeated seeds within scenes first; report paired scene effects and separate events/spatial groups. Three events do not justify precise population-level confidence or treating seeds/cells as independent disasters.",
        },
        "planner_forbidden": ["building_uid", "true_damage", "future_observations", "evaluator_matches", "future_actual_quality"],
        "preservation": "All mini72 observations, fits, protocols, reports and failed variants remain unchanged.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    freeze(args.out / "experiment_contract.json", protocol)
    freeze(args.out / "cohort_selection.json", rows)
    freeze(args.out / "selection_receipt.json", {
        "stage": "METADATA_ONLY_BEFORE_NEW_COHORT_DAMAGE_ACCESS",
        "contract_sha256": sha((args.out / "experiment_contract.json").read_bytes()),
        "selection_sha256": sha((args.out / "cohort_selection.json").read_bytes()),
        "planner_source_sha256": sha(Path(__file__).read_bytes()),
        "pairs": len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
        "interface_probe_count": sum(r["interface_probe"] for r in rows),
        "new_cohort_damage_read": False, "data_admitted": False,
    })
    print(json.dumps({"stage": "DESIGN_LOCKED_DATA_NOT_ADMITTED", "pairs": len(rows),
                      "split_counts": dict(Counter(r["split"] for r in rows))}))


if __name__ == "__main__":
    main()
