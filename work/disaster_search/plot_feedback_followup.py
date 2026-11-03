"""Export calibrated/validation follow-up curves, all frozen local variants."""
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
    validation = report["stage"] == "FROZEN_VALIDATION_SEARCH"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    styles = {"quality_gate": ("Quality gate / likelihood", "#475467", "o", "-"),
              "adaptive_prior": ("Global-bin residual (v1)", "#D55E00", "x", ":"),
              "local_residual": ("Local signed residual", "#0072B2", "s", "-"),
              "local_positive": ("Local positive residual", "#009E73", "^", "--"),
              "neighbor_positive": ("Simple positive neighbor", "#CC79A7", "D", "--")}
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), sharey=True)
    for ax, (scenario, title) in zip(axes, [("native_survey", "Native surveys"), ("mixed_survey", "50% degraded surveys"), ("blurred_survey", "Degraded surveys")]):
        for policy, (label, color, marker, line) in styles.items():
            rows = sorted([r for r in report["summary"] if r["scenario"] == scenario and r["policy"] == policy], key=lambda r: r["budget"])
            ax.plot([r["budget"] for r in rows], [r["sum_buildings_found"] for r in rows], label=label,
                    color=color, marker=marker, linestyle=line, linewidth=1.7, markersize=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([8, 16, 32])
        ax.set_xlabel("Observation-cost budget per scene")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linewidth=.7, color="#E5E7EB")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Unique target buildings found across 24 scenes")
    figure.suptitle(("Frozen validation" if validation else "Exploratory calibration") + ": bounded local feedback versus fixed controls", fontweight="bold", fontsize=13, y=.97)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, frameon=False, fontsize=9)
    figure.text(.5, .025, "24 scenes / 2 events. Mixed condition averages 5 paired quality maps. Survey cost 1; quality upgrade cost 2.\nAll methods and budgets retained. No confidence intervals based on treating cells or quality seeds as independent events.",
                ha="center", fontsize=8, color="#475467", linespacing=1.5)
    figure.subplots_adjust(left=.07, right=.985, top=.84, bottom=.30, wspace=.12)
    figure.savefig(args.run / "feedback_curves.png", dpi=180, facecolor="white")
    figure.savefig(args.run / "feedback_curves.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
