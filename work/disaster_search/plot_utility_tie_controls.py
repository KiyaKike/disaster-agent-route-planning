"""Development-only tie sensitivity; no independent efficacy inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_xbd_mini import atomic_json, sha
from run_utility_tie_controls import POLICIES, TIES, is_primary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run / "summary.json", args.run / "random_order_sensitivity.json", args.previous / "summary.json"]
    rows = json.loads(paths[0].read_text())["event_macro_means"]
    orders = json.loads(paths[1].read_text())["primary_order_sensitivity"]
    old = json.loads(paths[2].read_text())["event_macro_means"]
    def value(policy, rule, scope="development_fit", cost=2):
        return next(r["distinct_found"] for r in rows if r["policy"] == policy and r["tie_rule"] == rule
            and r["fit_scope"] == scope and r["scenario"] == "mixed_survey" and r["budget"] == 32
            and r["upgrade_cost"] == cost and r["threshold"] == .1)
    baseline = next(r["distinct_found"] for r in old if r["policy"] == "prior_survey" and r["fit_scope"] == "development_fit" and is_primary(r))
    colors = ["#94a3b8", "#0f766e", "#d97706"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for ax, scope, title in zip(axes[:2], ["development_fit", "event_holdout"],
            ["A  All 72 development scenes fit", "B  Other two development events fit"]):
        for k, (rule, label, color) in enumerate(zip(TIES, ["Row-major tie", "Raw-change tie", "Fixed-random tie"], colors)):
            positions = [i + (k - 1) * .25 for i in range(3)]
            heights = [value(p, rule, scope) for p in POLICIES]
            ax.bar(positions, heights, .24, color=color, label=label)
            if rule == "fixed_random":
                ranges = [next(r for r in orders if r["fit_scope"] == scope and r["policy"] == p) for p in POLICIES]
                ax.errorbar(positions, heights, yerr=[[v - r["minimum"] for v, r in zip(heights, ranges)],
                    [r["maximum"] - v for v, r in zip(heights, ranges)]], fmt="none", ecolor="#78350f", capsize=3, lw=1.)
        ax.axhline(baseline, color="#334155", linestyle="--", label=f"Raw-change survey ({baseline:.3f})")
        ax.set(xticks=range(3), xticklabels=["Greedy", "Two-step", "Batch tail"], title=title,
               ylabel="Distinct buildings / scene (event macro)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for rule, label, color in zip(TIES, ["Row-major tie", "Raw-change tie", "Fixed-random tie"], colors):
        axes[2].plot([1, 2, 4], [value("proxy_greedy", rule, cost=c) for c in [1, 2, 4]],
            color=color, marker="o", label=label, linewidth=2)
    axes[2].axhline(baseline, color="#334155", linestyle="--", label="Raw-change survey")
    axes[2].set(xticks=[1, 2, 4], xlabel="Upgrade cost", title="C  Greedy cost sensitivity (all dev fit)",
                ylabel="Distinct buildings / scene")
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Tie handling with unchanged four-bin utility, actions and cached observations", fontsize=13)
    fig.supxlabel("Mixed quality, budget32, matching threshold10%. Random whiskers: ten-order range, not confidence intervals. Development only.", fontsize=9)
    directory = args.run / "figures"
    directory.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(directory / ("tie_controls." + ext), dpi=180)
    plt.close(fig)
    atomic_json(directory / "figure_receipt.json", {"source_sha256": sha(Path(__file__).read_bytes()),
        "inputs_sha256": {str(p): sha(p.read_bytes()) for p in paths},
        "scope": "Development only; original primary setting retained. Order range is algorithm sensitivity, not a confidence interval.",
        "files": {p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.suffix in {".png", ".pdf"}}})


if __name__ == "__main__":
    main()
