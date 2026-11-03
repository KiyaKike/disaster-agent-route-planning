"""Frozen ChangeOS local-crop inference on a prespecified calibration-only pilot.

The prediction path reads RGB inputs only. Evaluation labels are a separate step.
No whole-scene inference hidden behind per-cell observations; no test inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time


MODEL_ID = "EVER-Z/torchange_example_changeos_swint_on_xview2_best42k"
MODEL_REVISION = "4007dd10c5eaee67161e2d4a45a43da3ead6c722"
WEIGHTS_SHA256 = "81ae6386ceebe225ab6c072cb987b57312fdcfdb2a7f72ab9b30e5e8f1041cc4"
CONFIG_SHA256 = "dd452bc568ff31a4706687b83d2ce0712b136f23d22705834311592d98a31d23"
CODE_COMMIT = "190b07a2eb4c389a5dc3f72802c67f428cdfd6b7"
CELLS = [f"r{row:02d}c{col:02d}" for row in (1, 3, 5, 7) for col in (1, 5)]
MODES = ["native", "blurred"]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(dataset_protocol, raw_root):
    counts = Counter()
    chosen = []
    for sample in dataset_protocol["samples"]:
        if sample["split"] != "calibration" or counts[sample["event"]] >= 2:
            continue
        counts[sample["event"]] += 1
        chosen.append({"sample_id": sample["sample_id"], "event": sample["event"], "split": "calibration",
                       "pre_image": str((raw_root / sample["files"]["pre_image"]).resolve()),
                       "post_image": str((raw_root / sample["files"]["post_image"]).resolve())})
    if len(chosen) != 4 or len(counts) != 2:
        raise ValueError("Pilot requires exactly two calibration events with two scenes each")
    return {"stage": "FROZEN_RGB_ONLY_PERCEPTION_PILOT", "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "code_commit": CODE_COMMIT, "weights_sha256": WEIGHTS_SHA256,
            "samples": chosen, "cells": CELLS, "modes": MODES, "observations": 64,
            "selection": "First two scenes per calibration event in frozen mini72 order; fixed eight cells, independent of labels.",
            "input": "Pre/post RGB within selected cell only, native 128x128, six channels, ImageNet normalization.",
            "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
            "degradation": {"blurred": "GaussianBlur(radius=1.5), BOX downsample by 2, BILINEAR upsample to original crop; applied to both timepoints"},
            "declaration_rule": {"predicted_building_threshold": .5, "damage_argmax_classes": [3, 4],
                                 "min_severe_pixels": 4, "soft_score_top_pixels": 16},
            "evidence": "Two modes share a base RGB pair and evidence_group_id; they must not be multiplied as independent likelihoods.",
            "training_overlap": "Checkpoint trained on upstream train+tier3. Mini72 comes from train: assume exposure; no unseen-disaster perception claim.",
            "scope": "Engineering and calibration diagnostics only; no tuning or test-set evaluation."}


def freeze_plan(path, plan):
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Refusing to overwrite a different frozen pilot plan")
    if not path.exists():
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))


def crop_pair(pre, post, cell_id):
    row, col = int(cell_id[1:3]), int(cell_id[4:6])
    if not 0 <= row < 8 or not 0 <= col < 8 or pre.size != post.size:
        raise ValueError("Invalid selected cell or mismatched RGB sizes")
    width, height = pre.size
    box = (col * width // 8, row * height // 8, (col + 1) * width // 8, (row + 1) * height // 8)
    return pre.crop(box), post.crop(box)


def degrade_image(image, mode):
    from PIL import Image, ImageFilter
    if mode not in MODES:
        raise ValueError("Unknown observation mode")
    if mode == "blurred":
        width, height = image.size
        image = image.filter(ImageFilter.GaussianBlur(1.5)).resize((width // 2, height // 2), Image.Resampling.BOX)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def prepare_tensor(pre, post, mode):
    import numpy as np
    import torch

    images = []
    for image in (pre, post):
        image = degrade_image(image, mode)
        array = np.asarray(image, dtype=np.float32) / 255.
        array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        images.append(array.transpose(2, 0, 1))
    return torch.from_numpy(np.concatenate(images, axis=0)[None].copy())


def predict_summary(output):
    import torch

    damage = output["change_prediction"]
    loc = output["t1_semantic_prediction"]
    if damage.ndim != 4 or damage.shape[0:2] != (1, 5) or loc.shape != (1, 1, *damage.shape[2:]):
        raise ValueError("Unexpected output shapes")
    if not torch.isfinite(damage).all() or not torch.isfinite(loc).all():
        raise ValueError("Non-finite model output")
    if not torch.allclose(damage.sum(1), torch.ones_like(damage[:, 0]), atol=1e-4):
        raise ValueError("Expected damage softmax probabilities, not logits")
    classes = damage.argmax(1)[0]
    building = loc[0, 0] > .5
    severe = building & ((classes == 3) | (classes == 4))
    joint_score = loc[0, 0] * (damage[0, 3] + damage[0, 4])
    score = torch.topk(joint_score.flatten(), k=16).values.mean().item()
    severe_count = int(severe.sum().item())
    return {"declared_target": severe_count >= 4, "predicted_building_pixels": int(building.sum().item()),
            "predicted_severe_pixels": severe_count, "uncalibrated_severe_score": score,
            "mean_max_damage_probability": damage.max(1).values.mean().item()}, severe.cpu().numpy()


def sync(device):
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.data / "protocol.json"
    plan = make_plan(json.loads(protocol_path.read_text()), args.data)
    freeze_plan(args.out / "pilot_protocol.json", plan)
    if args.plan_only:
        print(json.dumps({"frozen": str(args.out / "pilot_protocol.json"), "observations": plan["observations"]}))
        return

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import numpy as np
    from PIL import Image
    import torch
    import torchange
    from torchange.models.changeos import ChangeOS
    from safetensors.torch import load_model

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    manifest = json.loads((args.data / "manifest.json").read_text())
    expected_images = {str((args.data / item["path"]).resolve()): item["sha256"]
                       for item in manifest["files"] if item["path"].endswith(".png")}
    for sample in plan["samples"]:
        for key in ("pre_image", "post_image"):
            if file_hash(Path(sample[key])) != expected_images[sample[key]]:
                raise ValueError("Frozen input RGB hash mismatch")
    model_file = args.model / "model.safetensors"
    if file_hash(model_file) != WEIGHTS_SHA256:
        raise ValueError("Frozen model weight hash mismatch")
    if file_hash(args.model / "config.json") != CONFIG_SHA256:
        raise ValueError("Frozen architecture config hash mismatch")
    config = json.loads((args.model / "config.json").read_text())["config"]
    config["encoder"]["params"]["weights"] = None  # All encoder weights are in the checked local checkpoint.
    start = time.perf_counter()
    model = ChangeOS(config).eval()
    missing, unexpected = load_model(model, str(model_file), strict=True)
    if missing or unexpected:
        raise ValueError("Checkpoint does not exactly match the architecture")
    model.requires_grad_(False)
    load_s = time.perf_counter() - start
    selected_device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    if selected_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS is unavailable")
    device = torch.device(selected_device)
    model.to(device)
    sync(device)
    first = plan["samples"][0]
    with Image.open(first["pre_image"]) as a, Image.open(first["post_image"]) as b:
        pair = crop_pair(a.convert("RGB"), b.convert("RGB"), CELLS[0])
    with torch.inference_mode():
        warmup = model(prepare_tensor(*pair, "native").to(device))
        sync(device)
        predict_summary(warmup)
    del warmup

    observations, timings, input_hashes = [], [], {}
    cache = args.out / "cache"
    cache.mkdir(exist_ok=True)
    for sample in plan["samples"]:
        with Image.open(sample["pre_image"]) as image:
            pre = image.convert("RGB")
        with Image.open(sample["post_image"]) as image:
            post = image.convert("RGB")
        for key in ("pre_image", "post_image"):
            input_hashes[sample[key]] = file_hash(Path(sample[key]))
        for cell_id in CELLS:
            a, b = crop_pair(pre, post, cell_id)
            for mode in MODES:
                total_start = time.perf_counter()
                tensor = prepare_tensor(a, b, mode).to(device)
                sync(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    output = model(tensor)
                    sync(device)
                    forward_s = time.perf_counter() - start
                    summary, mask = predict_summary(output)
                timings.append(forward_s)
                group_id = f"{sample['sample_id']}:{cell_id}"
                cache_file = cache / f"{sample['sample_id']}_{cell_id}_{mode}.npz"
                np.savez_compressed(cache_file,
                                    damage_probabilities=output["change_prediction"][0].cpu().numpy().astype(np.float16),
                                    building_probability=output["t1_semantic_prediction"][0, 0].cpu().numpy().astype(np.float16),
                                    severe_mask=mask)
                observations.append({"sample_id": sample["sample_id"], "cell_id": cell_id, "split": "calibration",
                                     "mode": mode, "evidence_group_id": group_id,
                                     "observation_id": f"{group_id}:{mode}", "synthetic_degradation": mode != "native",
                                     "input_shape": list(tensor.shape), **summary,
                                     "forward_s": forward_s, "end_to_end_s": time.perf_counter() - total_start,
                                     "cache_file": str(cache_file.resolve()), "cache_sha256": file_hash(cache_file)})
                del output, tensor
        print(json.dumps({"sample": sample["sample_id"], "observations_done": len(observations), "device": str(device)}), flush=True)
    (args.out / "observations.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2))
    freeze = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "torchange", "ever-beta", "numpy", "Pillow", "safetensors"]}
    direct_url = importlib.metadata.distribution("torchange").read_text("direct_url.json")
    if direct_url is None or CODE_COMMIT not in direct_url:
        raise ValueError("Installed torchange does not identify the pinned code commit")
    report = {"stage": "FROZEN_PERCEPTION_PILOT_COMPLETED", "device": str(device), "paid_gpu_used": False,
              "observations": len(observations), "scenes": len(plan["samples"]), "selected_cells": 32,
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
              "protocol_sha256": file_hash(args.out / "pilot_protocol.json"), "mini72_protocol_sha256": file_hash(protocol_path),
              "runner_sha256": file_hash(Path(__file__)), "input_image_sha256": input_hashes,
              "load_seconds": load_s, "median_forward_seconds": statistics.median(timings),
              "cache_probability_dtype": "float16; JSON declarations and scores are canonical float32-derived outputs",
              "mean_forward_seconds": statistics.mean(timings),
              "mean_end_to_end_seconds": statistics.mean(o["end_to_end_s"] for o in observations),
              "estimated_calibration_3072_observation_minutes": statistics.mean(timings) * 3072 / 60,
              "estimated_all72_9216_observation_minutes": statistics.mean(timings) * 9216 / 60,
              "estimate_note": "Linear single-crop extrapolation on this device, excludes model load and scene I/O; not a rented-GPU quote or a completed full pass.",
              "packages": freeze, "torchange_direct_url": json.loads(direct_url),
              "labels_read_by_prediction_path": False, "test_inference_run": False,
              "limitations": ["Native crops are 128x128; the checkpoint was trained on 512x512 patches. Local context/boundary shift must be evaluated.",
                              "Native and blurred modes are derived from the same satellite image; not independent UAV viewpoints.",
                              "Raw score and the four-pixel declaration rule are not calibrated target probabilities.",
                              "Model training exposure is expected for this mini dataset. This pilot does not establish unseen-event generalization."]}
    (args.out / "inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("stage", "device", "observations", "median_forward_seconds", "estimated_calibration_3072_observation_minutes")}), flush=True)


if __name__ == "__main__":
    main()
