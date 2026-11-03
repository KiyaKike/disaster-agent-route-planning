"""Frozen-bin transfer and perfect-information cache ceilings; no efficacy claim."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_xbd_mini import atomic_json, sha


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run / name for name in ["component_calibration.json", "oracle_summary.json", "ranking_summary.json"]]
    components = json.loads(paths[0].read_text())["rows"]
    oracle = json.loads(paths[1].read_text())["event_macro_means"]
    ranking = json.loads(paths[2].read_text())["event_macro_means"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), layout="constrained")
    for ax, event, title in zip(axes[:2], ["hurricane-harvey", "santa-rosa-wildfire"],
            ["A  Harvey: intermediate bins dominate", "B  Santa Rosa: held ordering reverses"]):
        rows = [r for r in components if r["event"] == event and r["fit_scope"] == "event_holdout"]
        ax.plot(range(1, 5), [r["predicted_mixed"] for r in rows], marker="o", color="#2563eb", label="Frozen source prediction", linewidth=2)
        ax.plot(range(1, 5), [r["held_observed_mixed"] for r in rows], marker="s", color="#0f766e", label="Held-event observed mean", linewidth=2)
        ax.set(xticks=range(1, 5), xlabel="Frozen raw-change bin (low to high)", ylabel="Local matches / known cell", title=title, ylim=(0, .6))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    budgets = [8, 16, 32]
    for kind, label, color in [("survey_and_upgrade", "Perfect-information full oracle", "#2563eb"),
            ("survey_only", "Perfect-information survey oracle", "#0f766e")]:
        axes[2].plot(budgets, [next(r["lower"] for r in oracle if r["budget"] == b and r["oracle_kind"] == kind) for b in budgets],
            marker="o", linewidth=2, color=color, label=label)
    raw = [next(r["distinct_found"] for r in ranking if r["fit_scope"] == "development_fit" and r["policy"] == "prior_survey"
        and r["tie_rule"] == "fixed_random" and r["threshold"] == .1 and r["budget"] == b) for b in budgets]
    axes[2].plot(budgets, raw, marker="^", color="#334155", linestyle="--", linewidth=2, label="Executable raw survey reference")
    axes[2].set(xticks=budgets, xlabel="Observation budget", ylabel="Unique buildings / scene (event macro)",
                title="C  Offline cache ceilings, mixed quality", ylim=(0, 15))
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Event-dependent utility mismatch and perfect-information discovery ceilings", fontsize=13)
    fig.supxlabel("Cost2, matching10%. Oracles know future outcomes and true matches; their gains are not executable-policy evidence. Mexico retained, zero cache detections.", fontsize=9)
    directory = args.run / "figures"
    directory.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(directory / ("utility_transfer_and_cache_ceiling." + ext), dpi=180)
    plt.close(fig)
    atomic_json(directory / "figure_receipt.json", {"source_sha256": sha(Path(__file__).read_bytes()),
        "inputs_sha256": {p.name: sha(p.read_bytes()) for p in paths},
        "scope": "Development only. A/B local known-cell means, C event-equal unique buildings. No independent validation or policy efficacy claim from oracle.",
        "files": {p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.suffix in {".pdf", ".png"}}})


if __name__ == "__main__":
    main()
