"""Export short-probe sparsity and optimistic switching headroom diagnostics."""
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
    selected = {(r["split"], r["prefix_length"]): r for r in report["summary"] if r["scenario"] == "mixed_survey" and r["budget"] == 32}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    colors = {"calibration": "#0072B2", "validation": "#D55E00"}
    for i, split in enumerate(("calibration", "validation")):
        xpos = [j + (i - .5) * .32 for j in range(2)]
        negative = [100 * selected[(split, prefix)]["all_negative_probe_fraction_with_opportunity"] for prefix in (4, 8)]
        headroom = [selected[(split, prefix)]["oracle_minus_no_probe_prior"] for prefix in (4, 8)]
        axes[0].bar(xpos, negative, width=.30, label=split.capitalize(), color=colors[split])
        axes[1].bar(xpos, headroom, width=.30, label=split.capitalize(), color=colors[split])
        for x, v in zip(xpos, negative):
            axes[0].text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        for x, v in zip(xpos, headroom):
            axes[1].text(x, v + .10, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_title("Negative probes with remaining targets", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("All-negative probe cases (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Oracle switching gain after paying for probe", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Expected additional unique buildings")
    axes[1].set_ylim(0, 6.4)
    for ax in axes:
        ax.set_xticks([0, 1], ["4-cell probe", "8-cell probe"])
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7)
        ax.set_axisbelow(True)
    figure.suptitle("Short probes are sparse; a reliable switching signal is not established", fontsize=12.5, fontweight="bold", y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .09), ncol=2, frameon=False)
    figure.text(.5, .022, "Exploratory, mixed survey quality, total cost budget 32; probe costs included. No learned switcher.\nRight: evaluator oracle chooses prior or random expected continuation after the SAME prefix, minus no-probe prior baseline.",
                ha="center", fontsize=8, color="#475467", linespacing=1.5)
    figure.subplots_adjust(left=.075, right=.985, top=.80, bottom=.29, wspace=.30)
    figure.savefig(args.run / "early_evidence_diagnostics.png", dpi=180, facecolor="white")
    figure.savefig(args.run / "early_evidence_diagnostics.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
