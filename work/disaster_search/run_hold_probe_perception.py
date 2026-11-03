"""Separate RGB-only prediction and evaluator-only checks on six held-out IDs.

These are development interface probes, not policy validation. Source events
were used by the detector. The old mini72 runner and all its outputs stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze
from run_changeos_pilot import (MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, CONFIG_SHA256,
    CODE_COMMIT, MODES, file_hash, crop_pair, prepare_tensor, predict_summary, sync)

CELLS = [f"r{r:02d}c{c:02d}" for r in range(8) for c in range(8)]


def make_plan(manifest, selected, root):
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    if len(by_id) != 6 or set(by_id) != {s["sample_id"] for s in selected}:
        raise ValueError("Prediction accepts only the six selected probes")
    if any(s["split"] != "development" for s in selected):
        raise ValueError("Validation/confirmation prediction is not enabled")
    samples = [{"sample_id": s["sample_id"], "event": s["event"],
                **{mode + "_image": str((root / by_id[s["sample_id"]]["files"][mode + "_image"]).resolve()) for mode in ["pre", "post"]}}
               for s in selected]
    return {"version": "six_hold_probe_perception_smoke_v1", "samples": samples,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
        "code_commit": CODE_COMMIT, "cells": CELLS, "modes": MODES, "observations": 768,
        "prediction": "RGB-only selected-cell inference, unchanged mini72 tensor/degradation/declaration rules; canonical boolean severe masks cached.",
        "evaluation": "Separate command, unchanged raster_visibility_v2; 5/10/25% coverage diagnostics, no tuning.",
        "planned_component_fit": "Development smoke only: Jeffreys(0.5 per joint outcome) 2x2 paired sensor tables; four public RGB-change quantile bins, count means shrunk with 8 pseudo-cells. No search policy scores or generalization claim.",
        "scope": "Six development scenes only; full 216-scene admission and 72-scene development fitting remain incomplete; no new controller training."}


def predict(args, manifest, plan):
    for key in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        os.environ[key] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import numpy as np
    from PIL import Image
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model

    expected = {str((args.data / f["path"]).resolve()): f["sha256"] for f in manifest["files"] if f["path"].endswith(".png")}
    for sample in plan["samples"]:
        for mode in ["pre", "post"]:
            path = Path(sample[mode + "_image"])
            if file_hash(path) != expected[str(path)]:
                raise ValueError("Input PNG differs from frozen manifest")
    if file_hash(args.model / "model.safetensors") != WEIGHTS_SHA256 or file_hash(args.model / "config.json") != CONFIG_SHA256:
        raise ValueError("Frozen model/config differs")
    direct_url = importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if not direct_url or CODE_COMMIT not in direct_url:
        raise ValueError("Installed model code is not pinned")
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    config = json.loads((args.model / "config.json").read_text())["config"]
    config["encoder"]["params"]["weights"] = None
    model = ChangeOS(config).eval()
    missing, unexpected = load_model(model, str(args.model / "model.safetensors"), strict=True)
    if missing or unexpected:
        raise ValueError("Checkpoint does not exactly match architecture")
    model.requires_grad_(False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    sync(device)
    (args.out / "cache").mkdir(exist_ok=True)
    (args.out / "scene_records").mkdir(exist_ok=True)
    observations = []
    started = time.perf_counter()
    for sample in plan["samples"]:
        sid = sample["sample_id"]
        record_path = args.out / "scene_records" / f"{sid}.json"
        if record_path.exists():
            records = json.loads(record_path.read_text())
            expected_ids = {f"{sid}:{cell}:{mode}" for cell in CELLS for mode in MODES}
            if len(records) != 128 or {r["observation_id"] for r in records} != expected_ids:
                raise ValueError("Incomplete resume records")
            if any(file_hash(Path(r["cache_file"])) != r["cache_sha256"] for r in records):
                raise ValueError("Resume mask differs")
            observations.extend(records)
            continue
        with Image.open(sample["pre_image"]) as im:
            pre = im.convert("RGB")
        with Image.open(sample["post_image"]) as im:
            post = im.convert("RGB")
        records = []
        for cell in CELLS:
            pair = crop_pair(pre, post, cell)
            for mode in MODES:
                start = time.perf_counter()
                tensor = prepare_tensor(*pair, mode).to(device)
                with torch.inference_mode():
                    output = model(tensor)
                    sync(device)
                    summary, mask = predict_summary(output)
                path = args.out / "cache" / f"{sid}_{cell}_{mode}.npz"
                np.savez_compressed(path, severe_mask=mask)
                group = f"{sid}:{cell}"
                records.append({"sample_id": sid, "event": sample["event"], "split": "development",
                    "cell_id": cell, "mode": mode, "observation_id": f"{group}:{mode}", "evidence_group_id": group,
                    **summary, "end_to_end_s": time.perf_counter() - start,
                    "cache_file": str(path.resolve()), "cache_sha256": file_hash(path)})
                del output, tensor
        atomic_json(record_path, records)
        observations.extend(records)
        print(json.dumps({"sample_id": sid, "observations_done": len(observations), "device": str(device)}), flush=True)
    if len(observations) != 768:
        raise ValueError("Incomplete probe prediction pass")
    atomic_json(args.out / "observations.json", observations)
    atomic_json(args.out / "inference_report.json", {
        "stage": "DEVELOPMENT_PROBE_RGB_INFERENCE", "observations": len(observations), "scenes": 6,
        "device": str(device), "loop_seconds_this_invocation": time.perf_counter() - started,
        "median_observation_seconds": statistics.median(o["end_to_end_s"] for o in observations),
        "label_contents_read_by_predict": False, "paid_gpu_used": False, "policy_evaluation_run": False,
        "protocol_sha256": file_hash(args.out / "probe_protocol.json"),
        "runner_sha256": file_hash(Path(__file__)), "shared_rgb_helpers_sha256": file_hash(Path(__file__).with_name("run_changeos_pilot.py")),
        "observations_sha256": file_hash(args.out / "observations.json")})


def evaluate(args, manifest):
    import numpy as np
    from PIL import Image
    from associate_buildings import visible_objects, associate, discovery_summary
    from evaluate_changeos_pilot import counts
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}
    observations = json.loads((args.out / "observations.json").read_text())
    by_scene = {s["sample_id"]: s for s in manifest["samples"]}
    objects, scores, labels_by_scene = {}, {}, {}
    for sid, sample in by_scene.items():
        label_path = sample["files"]["post_label"]
        if file_hash(args.data / label_path) != expected[label_path]:
            raise ValueError("Evaluation label differs from manifest")
        label = json.loads((args.data / label_path).read_text())
        width, height = label["metadata"]["width"], label["metadata"]["height"]
        cells = {f"r{r:02d}c{c:02d}": [c * width // 8, r * height // 8, (c + 1) * width // 8, (r + 1) * height // 8]
                 for r in range(8) for c in range(8)}
        labels_by_scene[sid] = label
        objects[sid] = visible_objects(label["features"]["xy"], (width, height), cells)
        with Image.open(args.data / sample["files"]["pre_image"]) as im:
            pre = np.asarray(im.resize((128, 128), Image.Resampling.BOX), dtype=float)
        with Image.open(args.data / sample["files"]["post_image"]) as im:
            post = np.asarray(im.resize((128, 128), Image.Resampling.BOX), dtype=float)
        delta = np.abs(pre - post).mean(axis=2) / 255
        scores[sid] = {f"r{r:02d}c{c:02d}": float(np.percentile(delta[16*r:16*(r+1),16*c:16*(c+1)],75))
                       for r in range(8) for c in range(8)}
    expected_ids = {f"{sid}:{cell}:{mode}" for sid in by_scene for cell in CELLS for mode in MODES}
    if len(observations) != 768 or {o["observation_id"] for o in observations} != expected_ids:
        raise ValueError("Unexpected observation identities")
    rows = []
    for observation in observations:
        sid, cell = observation["sample_id"], observation["cell_id"]
        path = Path(observation["cache_file"])
        if file_hash(path) != observation["cache_sha256"]:
            raise ValueError("Prediction mask differs")
        with np.load(path, allow_pickle=False) as cache:
            mask = cache["severe_mask"]
        if mask.dtype != np.bool_ or int(mask.sum()) != observation["predicted_severe_pixels"]:
            raise ValueError("Canonical mask/declaration differs")
        if bool(mask.sum() >= 4) != observation["declared_target"]:
            raise ValueError("Frozen four-pixel rule differs")
        row = {k: observation[k] for k in ["sample_id", "cell_id", "mode", "observation_id", "evidence_group_id"]}
        row.update(associate(mask, objects[sid][cell]))
        row["predicted"] = observation["declared_target"]
        rows.append(row)
    results = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        known = [r for r in subset if r["target"] is not None]
        results[mode] = {"visible_cell_confusion": {**counts(known), "unknown_excluded": len(subset)-len(known)},
                         "building_proxy": discovery_summary(subset)}
    sensitivity = {}
    for threshold in [.05, .1, .25]:
        changed = [{**r, "matched_target_uids": [v["uid"] for v in r["overlap_records"]
            if v["visible_pixels"] >= 4 and v["severe_overlap_pixels"] >= 4 and v["visible_footprint_coverage"] >= threshold]} for r in rows]
        sensitivity[str(threshold)] = {mode: discovery_summary([r for r in changed if r["mode"] == mode]) for mode in MODES}
    atomic_json(args.out / "public_change_scores.json", scores)
    atomic_json(args.out / "association_report.json", {
        "stage": "DEVELOPMENT_PROBE_SENSOR_INTERFACE_DIAGNOSTIC", "results_by_mode": results,
        "sensitivity_not_for_threshold_selection": sensitivity, "rows": rows,
        "per_scene": {sid: {mode: discovery_summary([r for r in rows if r["sample_id"] == sid and r["mode"] == mode]) for mode in MODES} for sid in by_scene},
        "policy_evaluation_run": False, "independent_efficacy_claim": False,
        "observations_sha256": file_hash(args.out / "observations.json"),
        "labels_sha256": {s["files"]["post_label"]: expected[s["files"]["post_label"]] for s in manifest["samples"]}})
    print(json.dumps({"stage": "PROBE_SENSOR_CHECK_COMPLETE", "results_by_mode": results}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "design", "model", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=["predict", "evaluate"], required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    gate = json.loads((args.data / "interface_gate.json").read_text())
    if not gate["data_interface_passed"] or file_hash(args.data / "manifest.json") != gate["manifest_sha256"]:
        raise ValueError("Probe data has not passed its interface gate")
    selected = json.loads((args.design / "next_development_probes.json").read_text())
    plan = make_plan(manifest, selected, args.data)
    plan["manifest_sha256"] = file_hash(args.data / "manifest.json")
    plan["runner_sha256"] = file_hash(Path(__file__))
    freeze(args.out / "probe_protocol.json", plan)
    if args.stage == "predict":
        predict(args, manifest, plan)
    else:
        evaluate(args, manifest)


if __name__ == "__main__":
    main()
