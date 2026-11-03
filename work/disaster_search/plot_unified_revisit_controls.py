"""Matched-tie net revisit effects; development only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_xbd_mini import atomic_json, sha
from run_utility_tie_controls import TIES, is_primary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    source = args.run / "paired_effects.json"
    rows = json.loads(source.read_text())["event_macro_effects"]
    comparisons = ["prior_gate_vs_prior_survey", "count_negative_vs_count_survey",
                   "count_all_vs_count_negative", "count_all_vs_count_survey"]
    labels = ["Raw score\nNeg vs survey", "Count\nNeg vs survey", "Count\nAll vs neg", "Count\nAll vs survey"]
    colors = ["#94a3b8", "#0f766e", "#d97706"]
    def value(scope, rule, comparison, cost=2):
        return next(r["delta_found"] for r in rows if r["fit_scope"] == scope and r["tie_rule"] == rule
            and r["comparison"] == comparison and r["scenario"] == "mixed_survey" and r["budget"] == 32
            and r["upgrade_cost"] == cost and r["threshold"] == .1)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), layout="constrained")
    for ax, scope, title in zip(axes[:2], ["development_fit", "event_holdout"],
            ["A  All development scenes fit", "B  Other two development events fit"]):
        for k, (rule, color, label) in enumerate(zip(TIES, colors, ["Row-major tie", "Raw-change tie", "Fixed-random tie"])):
            ax.bar([i + (k - 1) * .25 for i in range(4)], [value(scope, rule, c) for c in comparisons],
                   .24, color=color, label=label)
        ax.axhline(0., color="#334155", linewidth=1)
        ax.set(xticks=range(4), xticklabels=labels, title=title,
               ylabel="Net extra distinct buildings / scene", ylim=(-.52, .24))
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for comparison, label, color in zip(comparisons,
            ["Raw: negative vs survey", "Count: negative vs survey", "Count: all vs negative", "Count: all vs survey"],
            ["#334155", "#0f766e", "#dc2626", "#2563eb"]):
        axes[2].plot([1, 2, 4], [value("development_fit", "fixed_random", comparison, cost) for cost in [1, 2, 4]],
            marker="o", color=color, label=label, linewidth=2)
    axes[2].axhline(0., color="#334155", linewidth=1)
    axes[2].axvline(2., color="#94a3b8", linestyle=":", linewidth=1)
    axes[2].set(xticks=[1, 2, 4], xlabel="Upgrade cost (2 is primary)",
                ylabel="Net extra distinct buildings / scene", title="C  Cost sensitivity, fixed-random ties")
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Revisit benefit depends on utility family, cost and development event", fontsize=13)
    fig.supxlabel("Mixed quality, budget32, matching10%. Ten random orders averaged. Mexico retained. Previously inspected development data; no independent validation.", fontsize=9)
    directory = args.run / "figures"
    directory.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(directory / ("revisit_net_effects." + ext), dpi=180)
    plt.close(fig)
    atomic_json(directory / "figure_receipt.json", {"source_sha256": sha(Path(__file__).read_bytes()),
        "input_sha256": sha(source.read_bytes()), "primary": "mixed_survey/budget32/cost2/threshold10%",
        "scope": "Development only. Raw policies do not use fitted models, so their identical results across fit scopes are not independent validation.",
        "files": {p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.suffix in {".pdf", ".png"}}})


if __name__ == "__main__":
    main()
