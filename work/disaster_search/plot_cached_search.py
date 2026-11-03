"""Static scientific plot of frozen calibration diagnostics; no error-bar inference."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((args.run / "report.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), sharey=True)
    styles = {"prior": ("Static prior", "#667085", "o", "-"),
              "pooled": ("Mode-agnostic update", "#0077B6", "s", ":"),
              "quality_gate": ("Simple quality gate", "#9B5DE5", "D", "--"),
              "quality": ("Quality likelihood", "#07845C", "o", "-"),
              "adaptive_prior": ("+ Peer-prior correction", "#D55E00", "^", "-")}
    scenarios = [("native_survey", "Native surveys"), ("mixed_survey", "50% degraded surveys"), ("blurred_survey", "Degraded surveys")]
    for ax, (scenario, title) in zip(axes, scenarios):
        for policy, (label, color, marker, linestyle) in styles.items():
            rows = sorted((r for r in report["summary"] if r["scenario"] == scenario and r["policy"] == policy), key=lambda r: r["budget"])
            ax.plot([r["budget"] for r in rows], [r["sum_buildings_found"] for r in rows],
                    label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.9, markersize=5)
        ax.set_title(title, fontsize=11, fontweight="semibold", pad=10)
        ax.set_xticks([8, 16, 32])
        ax.set_xlim(6.7, 33.3)
        ax.set_ylim(0, 340)
        ax.set_xlabel("Observation-cost budget per scene")
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Unique target buildings found across 24 scenes")
    figure.suptitle("Peer-prior correction changes actions, without consistent discovery gains", fontsize=13, fontweight="semibold", y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .08), ncol=3, frameon=False, fontsize=9)
    figure.text(.5, .025, "Calibration only: 24 scenes / 2 events. Mixed results average 5 paired quality maps. Survey cost 1; upgrade cost 2.\nOverlapping curves reflect equal outcomes; quality gate and quality likelihood have identical action traces in this run.",
                ha="center", fontsize=8.2, color="#475467", linespacing=1.5)
    figure.subplots_adjust(left=.07, right=.985, top=.84, bottom=.30, wspace=.12)
    figure.savefig(args.run / "discovery_curves.png", dpi=180)
    figure.savefig(args.run / "discovery_curves.pdf")
    plt.close(figure)
    print(json.dumps({"png": str(args.run / "discovery_curves.png"), "pdf": str(args.run / "discovery_curves.pdf")}))


if __name__ == "__main__":
    main()
