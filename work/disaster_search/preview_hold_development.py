"""Annotation/RGB alignment QA on predetermined additional development scenes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

from build_xbd_mini import atomic_json,sha
from verify_taco_polygon_bridge import wkt_points


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.data/"manifest.json").read_text())
    by_id={r["sample_id"]:r for r in manifest["samples"]}
    # Third frozen ID per event, plus the sole >0.0001px overhang case.
    selected=["hurricane-harvey_00000403","santa-rosa-wildfire_00000278","mexico-earthquake_00000036","hurricane-harvey_00000125"]
    directory=args.data/"qa";directory.mkdir(exist_ok=True)
    figure,axes=plt.subplots(4,3,figsize=(12,15),layout="constrained")
    for index,sid in enumerate(selected):
        sample=by_id[sid]
        label=json.loads((args.data/sample["files"]["post_label"]).read_text())
        for j,mode in enumerate(["pre","post","post"]):
            ax=axes[index,j]
            with Image.open(args.data/sample["files"][mode+"_image"]) as im:ax.imshow(im)
            if j==2:
                for feature in label["features"]["xy"]:
                    grade=feature["properties"]["subtype"]
                    color="#ef4444" if grade in {"major-damage","destroyed"} else "#9ca3af" if grade=="un-classified" else "#22c55e"
                    ax.add_patch(Polygon(wkt_points(feature["wkt"]),fill=False,edgecolor=color,linewidth=.6))
            ax.set_xlim(0,1024);ax.set_ylim(1024,0);ax.axis("off")
            ax.set_title(sid+"\n"+["Before","After","Source polygons (red: target)"][j],fontsize=9)
    figure.savefig(directory/"extended_annotation_checks.png",dpi=140);plt.close(figure)
    sid="hurricane-harvey_00000125";sample=by_id[sid]
    label=json.loads((args.data/sample["files"]["post_label"]).read_text())
    polygon=wkt_points(label["features"]["xy"][3]["wkt"])
    xs,ys=zip(*polygon)
    figure,axes=plt.subplots(1,2,figsize=(10,5),layout="constrained")
    for ax in axes:
        with Image.open(args.data/sample["files"]["post_image"]) as im:ax.imshow(im)
        ax.add_patch(Polygon(polygon,fill=False,edgecolor="#ef4444",linewidth=1.5))
        ax.axvline(0,color="#2563eb",linestyle="--",linewidth=1)
    axes[0].set_xlim(0,1024);axes[0].set_ylim(1024,0);axes[0].set_title("Source object 3, full image")
    axes[1].set_xlim(-8,max(xs)+16);axes[1].set_ylim(min(1024,max(ys)+16),max(0,min(ys)-16))
    axes[1].set_title("Original coordinates retained\nMaximum overhang: 0.35144 pixels")
    figure.savefig(directory/"subpixel_boundary.png",dpi=150);plt.close(figure)
    atomic_json(directory/"qa_receipt.json",{"selected":selected,"selection":"Third frozen scene per event plus geometry-only maximum-overhang case; not selected using detector results.",
        "scope":"Technical alignment QA only; no independent expert damage relabeling.","runner_sha256":sha(Path(__file__).read_bytes()),
        "figures":{p.name:sha(p.read_bytes()) for p in directory.glob("*.png")}})


if __name__=="__main__":main()
