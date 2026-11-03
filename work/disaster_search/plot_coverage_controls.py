"""Static coverage-control comparison with descriptive random-order ranges."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    styles = {"quality_gate": ("Fitted prior + quality gate", "#7B3294", "o", "-"),
              "local_residual": ("+ Local residual feedback", "#0072B2", "s", "-"),
              "uniform_gate": ("Flat prior, original nearest tie", "#667085", "v", ":"),
              "spatial_gate": ("Spatial coverage + gate", "#D55E00", "^", "--"),
              "random_gate": ("Random + gate, 10-order mean", "#009E73", "D", "-")}
    for ax, split in zip(axes, ("calibration", "validation")):
        for policy, (label, color, marker, line) in styles.items():
            rows = sorted([r for r in report["summary"] if r["split"] == split and r["scenario"] == "mixed_survey" and r["policy"] == policy], key=lambda r: r["budget"])
            x, y = [r["budget"] for r in rows], [r["sum_buildings_found"] for r in rows]
            if policy == "random_gate":
                ax.fill_between(x, [r["random_order_minimum_found"] for r in rows], [r["random_order_maximum_found"] for r in rows], color=color, alpha=.13, linewidth=0)
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=line, linewidth=1.8, markersize=5)
        exact = sorted([r for r in report["exact_random_survey_expectation"] if r["split"] == split and r["scenario"] == "mixed_survey"], key=lambda r: r["budget"])
        ax.plot([r["budget"] for r in exact], [r["sum_expected_buildings_found"] for r in exact],
                color="#111827", linestyle="--", linewidth=1.2, label="Random survey, exact expectation")
        ax.set_title(("Calibration" if split == "calibration" else "Validation") + " (24 scenes)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Observation-cost budget per scene")
        ax.set_ylabel("Unique target buildings found")
        ax.set_xticks([8, 16, 32])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7)
        ax.set_axisbelow(True)
    figure.suptitle("Useful calibration ranking does not establish transferable search gains", fontsize=13, fontweight="bold", y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085), ncol=2, frameon=False, fontsize=9)
    figure.text(.5, .025, "Posthoc controls, 50% degraded survey condition. Curves average 5 fixed quality maps. Panel y-scales differ.\nShading: range of 10 prespecified random orders, not a confidence interval. Validation targets occur in one event only.",
                ha="center", fontsize=8, color="#475467", linespacing=1.5)
    figure.subplots_adjust(left=.075, right=.98, top=.84, bottom=.34, wspace=.25)
    figure.savefig(args.run / "coverage_comparison.png", dpi=180, facecolor="white")
    figure.savefig(args.run / "coverage_comparison.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
