"""Scientific QA previews of unchanged RGB data and evaluator-only polygons."""
import argparse
import json
from pathlib import Path

COLORS = {"no-damage": "#4caf50", "minor-damage": "#fbc02d", "major-damage": "#ef5350",
          "destroyed": "#ab47bc", "un-classified": "#00acc1"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    from verify_taco_polygon_bridge import wkt_points

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    samples = manifest["samples"]
    fig, axes = plt.subplots(6, 2, figsize=(12, 30))
    details = []
    for i, sample in enumerate(samples):
        label = json.loads((args.data / sample["files"]["post_label"]).read_text())
        features = label["features"]["xy"]
        target_points = []
        for j, mode in enumerate(["pre", "post"]):
            ax = axes[i, j]
            with Image.open(args.data / sample["files"][mode + "_image"]) as im:
                ax.imshow(im)
            if mode == "post":
                for feature in features:
                    subtype = feature["properties"]["subtype"]
                    ring = wkt_points(feature["wkt"])
                    ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor=COLORS[subtype], linewidth=.8,
                                         linestyle="--" if subtype == "un-classified" else "-"))
                    if subtype in {"major-damage", "destroyed"}:
                        target_points.extend(ring)
            ax.set_title(f"{sample['sample_id']} | {mode}" + (" + post annotations" if mode == "post" else ""), fontsize=9)
            ax.set_xlim(0, label["metadata"]["width"])
            ax.set_ylim(label["metadata"]["height"], 0)
            ax.axis("off")
        if target_points:
            xs, ys = zip(*target_points)
            details.append((sample, features, [max(0, min(xs)-60), max(0, min(ys)-60),
                                              min(1024, max(xs)+60), min(1024, max(ys)+60)]))
    fig.suptitle("Six frozen development probes | unchanged RGB; evaluator polygons on post only", fontsize=13)
    fig.text(.5, .008, "Green: no damage | Yellow: minor | Red: major | Purple: destroyed | Dashed cyan: unknown\nxBD / Maxar Open Data; TACO repackaging by ISP, Universitat de Valencia. CC BY-NC-SA 4.0.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.98))
    fig.savefig(args.out / "annotation_overview.png", dpi=140)
    plt.close(fig)
    if details:
        fig, axes = plt.subplots(len(details), 2, figsize=(12, 6 * len(details)), squeeze=False)
        for i, (sample, features, box) in enumerate(details):
            for j, mode in enumerate(["pre", "post"]):
                ax = axes[i, j]
                with Image.open(args.data / sample["files"][mode + "_image"]) as im:
                    ax.imshow(im)
                if mode == "post":
                    for feature in features:
                        subtype = feature["properties"]["subtype"]
                        ax.add_patch(Polygon(wkt_points(feature["wkt"]), closed=True, fill=False,
                            edgecolor=COLORS[subtype], linewidth=1.4, linestyle="--" if subtype == "un-classified" else "-"))
                ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1])
                ax.set_title(f"{sample['sample_id']} | {mode}", fontsize=10)
                ax.axis("off")
        fig.suptitle("Evaluator-only target-area zooms for annotation QA; selected from the same six fixed probes", fontsize=11)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(args.out / "annotation_details.png", dpi=140)
        plt.close(fig)
    print(json.dumps({"overview_scenes": len(samples), "evaluator_target_zoom_scenes": len(details),
                      "cohort_selection_changed": False}))


if __name__ == "__main__":
    main()
