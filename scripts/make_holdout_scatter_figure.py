"""
Regenerates the per-model actual-vs-predicted holdout scatter figure (Figure 5.1)
directly from outputs/results.json's per-cell arrays (the same 90 active Theft
cells Table 5.2's error percentages are computed over). Replaces an earlier
version of this figure whose baked-in matplotlib font sizes were illegible once
laid out at print size in a four-panel row; this version uses a 2x2 grid with
larger fonts so each panel is wide enough to stay legible after LaTeX scales it
to the page's text width.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ("mack", "Mack Paid CL", "tab:red"),
    ("icl", "Incurred CL", "tab:green"),
    ("gbm", "Joint GBM", "tab:purple"),
    ("lstm", "Single LSTM", "tab:blue"),
]


def main(results_path: str, out_path: str) -> None:
    with open(results_path) as f:
        results = json.load(f)
    theft = results["theft"]
    cells = theft["cells"]
    actual = np.array(cells["actual"])
    error_pct = theft["error_pct"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 5))
    fig.suptitle("Actual vs Predicted Holdout Paid Losses Across Reserving Models (Theft)", fontsize=17)

    for ax, (key, label, color) in zip(axes.flat, MODELS):
        pred = np.array(cells[key])
        ax.scatter(actual, pred, color=color, edgecolor="black", linewidth=0.4, s=45, alpha=0.85)
        lo, hi = 0, max(actual.max(), pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{label} ({error_pct[key]:+.1f}\\%)".replace("\\%", "%"), fontsize=15)
        ax.set_xlabel("Actual Holdout Paid (RMk)", fontsize=13)
        ax.set_ylabel("Predicted Holdout Paid (RMk)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("written:", out_path)


if __name__ == "__main__":
    results_arg = sys.argv[1] if len(sys.argv) > 1 else "outputs/results.json"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/figures/fig5_1_scatter_plots.png"
    Path(out_arg).parent.mkdir(parents=True, exist_ok=True)
    main(results_arg, out_arg)
