"""
Regenerates Figure 5.2 (fig5_2_residual_dev.png): mean prediction residual
(predicted minus actual) by position within the 2024 Q1-2026 Q1 holdout
window, Theft, for Mack Paid CL, ICL and the single-peril LSTM. Built
directly from outputs/results.json's 'cells_all' arrays (every holdout cell,
not only the active subset) - no source script previously existed for this
figure, and the earlier image had a hardcoded "Figure 5.2:" title baked into
the pixels, duplicating the LaTeX caption and going stale if the figure's
auto-number ever changes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VAL_END = 55
MODELS = [
    ("mack", "Mack Paid CL", "red", "x"),
    ("icl", "Incurred CL", "green", "s"),
    ("lstm", "Single LSTM", "blue", "o"),
]


def main(results_path: str, out_path: str) -> None:
    with open(results_path) as f:
        results = json.load(f)
    cells = results["theft"]["cells_all"]
    cal = np.array(cells["cal_idx"])
    actual = np.array(cells["actual"])
    positions = cal - VAL_END

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    for key, label, color, marker in MODELS:
        model_vals = np.array(cells[key])
        means = []
        for pos in range(1, 10):
            mask = positions == pos
            means.append((model_vals[mask] - actual[mask]).mean() / 1000.0)
        ax.plot(range(1, 10), means, color=color, marker=marker, markersize=9,
                linewidth=2.2, label=label)

    ax.set_xlabel("Holdout Calendar Quarter Index (1 = 2024 Q1, ..., 9 = 2026 Q1)", fontsize=12)
    ax.set_ylabel("Mean Cell Residual (RM Thousands)", fontsize=12)
    ax.set_xticks(range(1, 10))
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("written:", out_path)


if __name__ == "__main__":
    results_arg = sys.argv[1] if len(sys.argv) > 1 else "outputs/results.json"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "outputs/figures/fig5_2_residual_dev.png"
    Path(out_arg).parent.mkdir(parents=True, exist_ok=True)
    main(results_arg, out_arg)
