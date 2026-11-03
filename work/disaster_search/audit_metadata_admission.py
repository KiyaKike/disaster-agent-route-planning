"""Replay geometry fits and overlap screens from the same offline metadata."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_hold_taco import MetadataView, read_index, column_ranges, assert_no_protected_reads
from audit_hold_spatial import components
from build_xbd_mini import atomic_json, sha
from inferred_scene_geometry import COLUMNS, LIMITS, OfflineGeometryRanges, fit_geometry, bounds
from run_metadata_admission_audit import overlap_rows, summarize_candidates
from run_utility_tie_controls import verify_receipt


def main():
    import argparse
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    report = json.loads((args.run / "report.json").read_text())
    paths = {k: Path(r["path"]) for k, r in protocol["sources"].items()}
    for k, path in paths.items():
        if sha(path.read_bytes()) != protocol["sources"][k]["sha256"]:
            raise ValueError("Frozen source changed: " + k)
    for name, digest in protocol["source_code_sha256"].items():
        if sha(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("Frozen code changed: " + name)
    verify_receipt(paths["prior_receipt"].parent)
    cohort = json.loads(paths["cohort"].read_text())
    identities = json.loads(paths["identities"].read_text())
    lookup = {r["ml:source_file"]: i for i, r in enumerate(identities)}
    author = json.loads(paths["author_ids"].read_text())
    training, selection = sorted(set(author["train"]) | set(author["tier3"])), sorted(author["test"])
    candidates = [r["sample_id"] for r in cohort]
    expected = set(training) | set(selection) | set(candidates)
    models = json.loads((args.run / "geometry_models.json").read_text())
    ledger = json.loads((args.run / "candidate_ledger.json").read_text())
    if set(models) != expected or len(expected) != 7326 or [r["sample_id"] for r in ledger] != candidates:
        raise ValueError("Scene membership differs")
    # Re-read only the protocol whitelist, protected BEFORE every request.
    ranges = OfflineGeometryRanges(Path(protocol["cache_root"]))
    index = read_index(ranges)
    parquet = pq.ParquetFile(MetadataView(ranges, index["METADATA/sample.parquet"]))
    protected = column_ranges(parquet, index["METADATA/sample.parquet"], {"ml:damage", "ml:n_buildings", "ml:n_buildings_pre"})
    assert_no_protected_reads(ranges.accesses, protected)
    ranges.protected = protected
    ranges.allowed = column_ranges(parquet, index["METADATA/sample.parquet"], set(COLUMNS)) + list(ranges.accesses)
    table = parquet.read(columns=COLUMNS, use_threads=False)
    values = {}
    for key in COLUMNS[:4]:
        c = table[key].combine_chunks()
        values[key] = c.values.to_numpy(zero_copy_only=False), c.offsets.to_numpy(zero_copy_only=False)
    dimensions = {k: table[k].to_pylist() for k in COLUMNS[4:]}
    def flat(key, row):
        data, offsets = values[key]
        return data[offsets[row]:offsets[row + 1]]
    counts = Counter()
    for number, sid in enumerate(sorted(expected), 1):
        i = lookup[sid]
        replay = fit_geometry(*(flat(k, i) for k in COLUMNS[:4]), dimensions["ml:width"][i], dimensions["ml:height"][i])
        if replay != models[sid]:
            raise ValueError("Geometric fit replay differs: " + sid)
        counts["geometric_fits_replayed"] += 1
        g = models[sid]
        if g["official_footprint_verified"]:
            raise ValueError("Inferred range called an official footprint")
        if g["usable_inferred_range"]:
            w, h = dimensions["ml:width"][i], dimensions["ml:height"][i]
            for prefix, pad in [("", 0.), ("padded_", LIMITS["padding_px"])]:
                # Independently reconstruct corners directly from saved coefficients.
                coeff = np.array(g["normalized_pixel_to_wgs84_coefficients"])
                origin = np.array(g["affine_origin_wgs84"])
                corners = []
                for x, y in [(-pad, -pad), (w + pad, -pad), (w + pad, h + pad), (-pad, h + pad)]:
                    corners.append((origin + (x / w - .5) * coeff[0] + (y / h - .5) * coeff[1] + coeff[2]).tolist())
                if not np.allclose(corners, g[prefix + "inferred_corners_wgs84"], rtol=0, atol=1e-12):
                    raise ValueError("Corner arithmetic differs")
                if bounds(g[prefix + "inferred_corners_wgs84"]) != g[prefix + "inferred_bounds_wgs84"]:
                    raise ValueError("Saved range bounds differ")
            counts["usable_ranges_corners_checked"] += 1
        if number % 1000 == 0:
            print(json.dumps({"stage": "GEOMETRY_REPLAY_PROGRESS", "processed": number}), flush=True)
    old = {r["sample_id"]: r for r in json.loads(paths["old_spatial"].read_text())["per_sample"]}
    for sample, saved in zip(cohort, ledger):
        sid = sample["sample_id"]
        if any(saved[k] != value for k, value in sample.items()):
            raise ValueError("Candidate fields changed")
        if saved["old_training_annotation_envelope_overlap_ids"] != old[sid]["training_annotation_envelope_overlap_ids"]:
            raise ValueError("Prior envelope evidence changed")
        for field, pool, padded in [("training_inferred_overlaps", training, False),
            ("training_padded_inferred_overlaps", training, True), ("selection_test_inferred_overlaps", selection, False)]:
            if overlap_rows(models[sid], pool, models, padded) != saved[field]:
                raise ValueError("Overlap replay differs")
            counts["candidate_pool_overlap_screens_replayed"] += 1
        if saved["independently_admitted"] or saved["official_footprint_verified"] or saved["pixel_independence_verified"] or not saved["candidate_preserved"]:
            raise ValueError("Unknown independence or membership misrepresented")
        counts["candidate_rows_checked"] += 1
    pairs, cross = [], []
    by_id = {r["sample_id"]: r for r in cohort}
    for i, sid in enumerate(candidates):
        for r in overlap_rows(models[sid], candidates[i + 1:], models) or []:
            other = r["sample_id"]
            pair = {"sample_a": sid, "split_a": by_id[sid]["split"], "sample_b": other,
                "split_b": by_id[other]["split"], "fraction_of_smaller_inferred_post_range": r["fraction_of_smaller_inferred_post_range"]}
            pairs.append(pair)
            if pair["split_a"] != pair["split_b"]:
                cross.append(pair)
    if pairs != report["inferred_candidate_overlap_pairs"] or cross != report["inferred_cross_policy_split_overlaps"]:
        raise ValueError("Candidate graph differs")
    if components(candidates, [(r["sample_a"], r["sample_b"]) for r in pairs]) != report["inferred_candidate_components"]:
        raise ValueError("Candidate components differ")
    if summarize_candidates(ledger) != report["by_split_and_event"] or dict(Counter(g["status"] for g in models.values())) != report["geometry_status_counts"]:
        raise ValueError("Summary aggregation differs")
    assert_no_protected_reads(ranges.accesses, protected)
    if ranges.remote.budget.received or ranges.remote.budget.reserved or not all(r["cache_hit"] for r in ranges.accesses):
        raise ValueError("Offline boundary violated")
    if report["network_bytes_received"] or report["validation_or_confirmation_scores_run"] or report["whole_216_cohort_independently_admitted"]:
        raise ValueError("Unauthorized scoring or independence claim")
    audit = {"stage": "OFFLINE_METADATA_ADMISSION_FULL_REPLAY_PASSED", "counts": dict(counts),
        "all_geometry_fits_replayed": True, "all_overlap_screens_replayed": True,
        "canonical_cohort_unchanged": True, "prior_sources_preserved": True,
        "protected_columns_blocked_before_cache_read": True, "no_protected_payload_intersections": True,
        "network_bytes_received": 0, "new_damage_reads": 0, "new_inference_calls": 0,
        "validation_or_confirmation_scores_run": False, "whole_216_cohort_independently_admitted": False}
    atomic_json(args.run / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
