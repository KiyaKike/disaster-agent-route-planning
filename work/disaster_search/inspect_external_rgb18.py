"""RGB-only quality diagnostics and prediction-blind review packet, never gold."""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_xbd_mini import atomic_json, sha
from plan_search_v2 import freeze


def phase_translation(pre, post):
    """Global gradient phase-correlation hint, not a registration acceptance test."""
    signals = []
    for image in [pre, post]:
        gray = np.asarray(image.convert("L").resize((256, 256), Image.Resampling.BOX), dtype=float)
        dy, dx = np.gradient(gray)
        signal = np.hypot(dx, dy)
        signal -= signal.mean()
        signal *= np.outer(np.hanning(256), np.hanning(256))
        signals.append(np.fft.fft2(signal))
    product = signals[0] * np.conj(signals[1])
    normalized = product / np.maximum(np.abs(product), 1e-12)
    correlation = np.fft.ifft2(normalized).real
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = (y if y <= 128 else y-256), (x if x <= 128 else x-256)
    return {"estimated_post_to_pre_translation_native_px": [int(dx*2), int(dy*2)],
        "phase_peak_not_probability": float(correlation[y, x]),
        "interpretation": "Global diagnostic only; low texture, clouds, damage or repeated roofs can invalidate this estimate. Manual paired review remains required."}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root/"rgb_manifest.json").read_text())
    cohort = json.loads((args.parent/"tentative_cohort.json").read_text())
    freeze(root/"rgb_inspection_protocol.json", {"stage": "RGB_ONLY_QA_NOT_GOLD_OR_PERCEPTION",
        "manifest_sha256": sha((root/"rgb_manifest.json").read_bytes()),
        "cohort_sha256": sha((args.parent/"tentative_cohort.json").read_bytes()),
        "inspector_sha256": sha(Path(__file__).read_bytes()),
        "predefined_diagnostics": ["shape and mode", "pixel hashes and exact duplicates", "all-channel-zero fraction", "global translation hint at256px gradient phase correlation", "PNG metadata keys"],
        "not_measured": ["building or damage labels", "detector predictions", "policy reward", "official geolocation", "registration accuracy"],
        "no_scene_substitution": True, "inference_enabled": False})
    by_scene, digests, files = defaultdict(dict), defaultdict(list), []
    for row in manifest:
        path = Path(row["path"])
        data = path.read_bytes()
        if sha(data) != row["sha256"]:
            raise ValueError("RGB bytes changed")
        with Image.open(path) as image:
            image.load()
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError("Unexpected RGB shape")
            array = np.asarray(image)
            pixel_sha = sha(array.tobytes())
            digests[pixel_sha].append(row["sample_id"]+":"+row["mode"])
            files.append({"sample_id": row["sample_id"], "mode": row["mode"],
                "pixel_sha256": pixel_sha, "all_channel_zero_fraction": float(np.all(array == 0, axis=2).mean()),
                "gray_std": float(np.asarray(image.convert("L"), dtype=float).std()),
                "png_metadata_keys": sorted(image.info), "sha256": row["sha256"]})
            by_scene[row["sample_id"]][row["mode"]] = (row, image.copy())
    expected = {(r["sample_id"], mode) for r in cohort for mode in ["pre", "post"]}
    if {(r["sample_id"], r["mode"]) for r in manifest} != expected or len(manifest) != 36:
        raise ValueError("Fixed complete18 pair cohort required; no dropped scenes")
    scenes, review_rows = [], []
    # Three event sheets with all6 pre/post pairs, fixed hash rank; no target labels.
    for event in dict.fromkeys(r["event"] for r in cohort):
        part = [r for r in cohort if r["event"] == event]
        sheet = Image.new("RGB", (548, len(part)*290+70), "#f1f2f0")
        draw = ImageDraw.Draw(sheet)
        draw.text((14, 10), event+"  |  PRE / POST  |  RGB review only", fill="#172522")
        for index, selected in enumerate(part):
            sid = selected["sample_id"]
            pre_row, pre = by_scene[sid]["pre"]
            post_row, post = by_scene[sid]["post"]
            phase = phase_translation(pre, post)
            a, b = np.asarray(pre, dtype=float), np.asarray(post, dtype=float)
            scenes.append({"sample_id": sid, "event": event, "rank": selected["rank_within_event"],
                **phase, "rgb_mean_absolute_difference": float(np.abs(a-b).mean()),
                "registration_review": "PENDING_HUMAN_REVIEW",
                "geographic_independence": "UNKNOWN", "instance_gold": "NOT_STARTED"})
            top = 45+index*290
            draw.text((14, top), sid+"  PRE", fill="#172522")
            draw.text((280, top), sid+"  POST", fill="#172522")
            sheet.paste(pre.resize((256, 256), Image.Resampling.BOX), (14, top+20))
            sheet.paste(post.resize((256, 256), Image.Resampling.BOX), (280, top+20))
            review_rows.append({"sample_id": sid, "event": event, "rank": selected["rank_within_event"],
                "pre": os.path.relpath(pre_row["path"], root), "post": os.path.relpath(post_row["path"], root),
                "image_sha256": {"pre": pre_row["sha256"], "post": post_row["sha256"]}})
        target = root/"previews"/(event+".jpg")
        target.parent.mkdir(exist_ok=True)
        sheet.save(target, quality=92)
    atomic_json(root/"rgb_quality_report.json", {"stage": "STRUCTURAL_RGB_CHECKS_COMPLETE_HUMAN_REVIEW_PENDING",
        "files": files, "scenes": scenes,
        "exact_pixel_duplicate_groups_within36": [v for v in digests.values() if len(v)>1],
        "training_pixel_or_geographic_independence_checked": False,
        "human_alignment_review_complete": False, "gold_complete": False,
        "new_inference": 0, "policy_scores": 0})
    atomic_json(root/"review_packet_index.json", review_rows)
    print(json.dumps({"rgb_files_checked": len(files), "pairs_kept": len(scenes),
        "pixel_duplicate_groups": sum(len(v)>1 for v in digests.values()),
        "max_absolute_phase_shift_px": max(max(abs(x) for x in r["estimated_post_to_pre_translation_native_px"]) for r in scenes),
        "manual_review_pending": True}))


if __name__ == "__main__":
    main()
