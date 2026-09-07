"""
Regenerates Figure 5.3 (fig_pooled_comparison.png): full 20-seed reserve error
distributions, single-peril vs. pooled architecture, both perils, against
each peril's ICL point error (dashed line), individual seeds overlaid as
points. Built directly from outputs/multiseed_theft.json,
outputs/multiseed_windscreen.json and outputs/pooled_multiseed_results.json -
no source script previously existed for this figure (a reproducibility gap;
see README/dissertation Chapter 8), and the earlier image was undersized at
print width with near-illegible tick/legend text.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ICL_ERROR = {"theft": 1.3, "windscreen": -2.8}


def main(out_path: str) -> None:
    theft_single = json.load(open("outputs/multiseed_theft.json"))["per_seed_error_pct"]
    ws_single = json.load(open("outputs/multiseed_windscreen.json"))["per_seed_error_pct"]
    pooled = json.load(open("outputs/pooled_multiseed_results.json"))
    theft_pooled = pooled["theft_error_pct"]
    ws_pooled = pooled["windscreen_error_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, single, pooled_errs, peril, label in (
        (axes[0], theft_single, theft_pooled, "theft", "Theft"),
        (axes[1], ws_single, ws_pooled, "windscreen", "Windscreen"),
    ):
        data = [single, pooled_errs]
        bp = ax.boxplot(data, tick_labels=["Single-Peril", "Pooled"], widths=0.5,
                         patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], ["tab:blue", "tab:orange"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        for i, seeds in enumerate(data, start=1):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(seeds))
            ax.scatter(np.full(len(seeds), i) + jitter, seeds, color="black", s=28, zorder=3, alpha=0.7)
        ax.axhline(ICL_ERROR[peril], color="red", linestyle="--", linewidth=2,
                   label=f"ICL point error ({ICL_ERROR[peril]:+.1f}\\%)".replace("\\%", "%"))
        ax.set_title(f"{label}: 20-Seed Reserve Error", fontsize=17)
        ax.set_ylabel("Holdout Reserve Error (%)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=13, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("written:", out_path)


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else "outputs/figures/fig_pooled_comparison.png"
    Path(out_arg).parent.mkdir(parents=True, exist_ok=True)
    main(out_arg)
