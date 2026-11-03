"""RGB-only complete calibration pass; reuses the frozen pilot model and rules."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

from run_changeos_pilot import (MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, CONFIG_SHA256,
    CODE_COMMIT, MODES, file_hash, freeze_plan, crop_pair, prepare_tensor, predict_summary, sync)


def make_plan(protocol, raw_root, split="calibration"):
    if split not in {"calibration", "validation"}:
        raise ValueError("Test inference is not enabled")
    samples = [{"sample_id": s["sample_id"], "event": s["event"], "split": split,
                **{k: str((raw_root / s["files"][k]).resolve()) for k in ("pre_image", "post_image")}}
               for s in protocol["samples"] if s["split"] == split]
    if len(samples) != 24 or len({s["event"] for s in samples}) != 2:
        raise ValueError("Expected the frozen 24-scene, two-event calibration split")
    plan = {"stage": "FROZEN_COMPLETE_CALIBRATION_RGB_PASS", "samples": samples,
            "cells": [f"r{r:02d}c{c:02d}" for r in range(8) for c in range(8)],
            "modes": MODES, "observations": 3072, "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "code_commit": CODE_COMMIT,
            "selection": "All calibration scenes and all 64 cells. No label filtering or substitutions.",
            "rules": "Unchanged pilot RGB preprocessing, synthetic degradation, four severe pixel declaration and top-16 score.",
            "independence": "Modes share base imagery; 3072 records are 1536 paired regions in 24 scenes and two events.",
            "scope": "Calibration fitting only, not validation/test, no unseen-event model claim. Boundary v2 is motivated posthoc by pilot.",
            "planned_calibration": "Report per-mode and per-event cell TPR/FPR plus Beta(0.5,0.5) smoothed rates; no independence-based confidence intervals. Fixed four equal-frequency raw-change bins for initial presence prior, shrunk toward global prevalence with 8 pseudo-observations. No score threshold tuning. Validation needed before use as reliable probabilities."}
    if split == "validation":
        plan.update(stage="FROZEN_COMPLETE_VALIDATION_RGB_PASS",
                    selection="All 24 frozen validation scenes and all 64 cells. No label filtering or substitutions.",
                    scope="Validation of frozen calibration-fitted parameters and search rules. No fitting, threshold adjustment or test inference. Checkpoint exposure still expected.",
                    planned_calibration="No refitting: apply original calibration parameters unchanged. Evaluate all previously frozen local variants and controls; preserve validation failures.")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--split", choices=["calibration", "validation"], default="calibration")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_protocol = args.data / "protocol.json"
    plan = make_plan(json.loads(source_protocol.read_text()), args.data, args.split)
    plan["mini72_protocol_sha256"] = file_hash(source_protocol)
    plan_path = args.out / (args.split + "_protocol.json")
    freeze_plan(plan_path, plan)
    if args.plan_only:
        print(json.dumps({"frozen": str(plan_path), "observations": plan["observations"]}))
        return
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import numpy as np
    from PIL import Image
    import torch
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    manifest = json.loads((args.data / "manifest.json").read_text())
    expected = {str((args.data / f["path"]).resolve()): f["sha256"]
                for f in manifest["files"] if f["path"].endswith(".png")}
    for sample in plan["samples"]:
        for key in ("pre_image", "post_image"):
            if file_hash(Path(sample[key])) != expected[sample[key]]:
                raise ValueError("Input RGB hash mismatch")
    if file_hash(args.model / "model.safetensors") != WEIGHTS_SHA256 or file_hash(args.model / "config.json") != CONFIG_SHA256:
        raise ValueError("Model checksum mismatch")
    direct_url = importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if direct_url is None or CODE_COMMIT not in direct_url:
        raise ValueError("Unpinned torchange installation")
    config = json.loads((args.model / "config.json").read_text())["config"]
    config["encoder"]["params"]["weights"] = None
    model = ChangeOS(config).eval()
    missing, unexpected = load_model(model, str(args.model / "model.safetensors"), strict=True)
    if missing or unexpected:
        raise ValueError("Checkpoint architecture mismatch")
    model.requires_grad_(False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    sync(device)
    cache_root = args.out / "cache"
    cache_root.mkdir(exist_ok=True)
    records_root = args.out / "scene_records"
    records_root.mkdir(exist_ok=True)
    observations = []
    started = time.perf_counter()
    for sample in plan["samples"]:
        records_path = records_root / (sample["sample_id"] + ".json")
        if records_path.exists():
            records = json.loads(records_path.read_text())
            expected_ids = {f"{sample['sample_id']}:{cell}:{mode}" for cell in plan["cells"] for mode in MODES}
            if len(records) != 128 or {r["observation_id"] for r in records} != expected_ids or any(r["split"] != args.split for r in records):
                raise ValueError("Incomplete or mismatched resume records")
            if any(file_hash(Path(r["cache_file"])) != r["cache_sha256"] for r in records):
                raise ValueError("Resume cache hash mismatch")
            observations.extend(records)
            continue
        with Image.open(sample["pre_image"]) as im:
            pre = im.convert("RGB")
        with Image.open(sample["post_image"]) as im:
            post = im.convert("RGB")
        records = []
        for cell in plan["cells"]:
            pair = crop_pair(pre, post, cell)
            for mode in MODES:
                start = time.perf_counter()
                tensor = prepare_tensor(*pair, mode).to(device)
                with torch.inference_mode():
                    output = model(tensor)
                    sync(device)
                    summary, mask = predict_summary(output)
                cache_file = cache_root / f"{sample['sample_id']}_{cell}_{mode}.npz"
                np.savez_compressed(cache_file, severe_mask=mask)
                group = f"{sample['sample_id']}:{cell}"
                records.append({"sample_id": sample["sample_id"], "cell_id": cell, "split": args.split,
                    "event": sample["event"], "mode": mode, "evidence_group_id": group,
                    "observation_id": f"{group}:{mode}", "synthetic_degradation": mode != "native",
                    "input_shape": list(tensor.shape), **summary,
                    "end_to_end_s": time.perf_counter() - start,
                    "cache_file": str(cache_file.resolve()), "cache_sha256": file_hash(cache_file)})
                del output, tensor
        records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        observations.extend(records)
        print(json.dumps({"scene": sample["sample_id"], "observations_done": len(observations), "device": str(device)}), flush=True)
    if len(observations) != plan["observations"]:
        raise ValueError("Incomplete calibration pass")
    (args.out / "observations.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2))
    report = {"stage": "COMPLETE_" + args.split.upper() + "_INFERENCE", "observations": len(observations), "scenes": 24,
              "base_regions": 1536, "device": str(device), "paid_gpu_used": False,
              "label_contents_read_by_prediction": False, "test_inference_run": False, "validation_inference_run": args.split == "validation",
              "protocol_sha256": file_hash(plan_path),
              "runner_sha256": file_hash(Path(__file__)), "shared_pilot_runner_sha256": file_hash(Path(__file__).with_name("run_changeos_pilot.py")),
              "observations_sha256": file_hash(args.out / "observations.json"),
              "input_image_sha256": {s[k]: expected[s[k]] for s in plan["samples"] for k in ("pre_image", "post_image")},
              "this_run_loop_seconds": time.perf_counter() - started,
              "median_observation_seconds": statistics.median(r["end_to_end_s"] for r in observations),
              "cache_note": "Only canonical boolean severe masks cached; float32-derived summaries in JSON. Full probabilities not needed for fixed-rule likelihood calibration.",
              "limitations": [plan["scope"], plan["independence"], "128x128 local crops differ from training context."]}
    (args.out / "inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("stage", "observations", "device", "this_run_loop_seconds")}))


if __name__ == "__main__":
    main()
