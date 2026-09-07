"""
Regenerates a cumulative-by-calendar-quarter holdout run-off figure (used for
both Theft and Windscreen) from the real per-cell model outputs saved in
outputs/results.json (the 'cells_all' arrays: actual/mack/icl/munich/gbm/lstm
per cell, each tagged with its calendar index). No figure number is drawn on
the chart -- see make_exposure_series_figure.py for why.
"""
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    ("actual", "Actual Holdout Payments", "black", "--", None),
    ("icl", "Incurred Chain Ladder", "green", "-", "o"),
    ("mack", "Mack Paid Chain Ladder", "red", "-", "o"),
    ("munich", "Munich Chain Ladder", "cyan", "-", "o"),
    ("gbm", "Gradient Boosting (GBM)", "purple", "-", "o"),
    ("lstm", "Single-Peril LSTM", "blue", "-", "o"),
]


def main(results_json: str, peril: str, out_path: str, title: str) -> None:
    r = json.load(open(results_json))
    cells = r[peril]["cells_all"]
    cal = np.array(cells["cal_idx"])
    cal_labels = {}
    start_year, start_q = 2010, 1
    for k in sorted(set(cal.tolist())):
        yq = k
        cal_labels[k] = yq

    quarters = sorted(set(cal.tolist()))
    q_labels = []
    base_year, base_q = 2024, 1
    for i, k in enumerate(quarters):
        yr = base_year + (base_q - 1 + i) // 4
        q = (base_q - 1 + i) % 4 + 1
        q_labels.append(f"{yr} Q{q}")

    fig, ax = plt.subplots(figsize=(12, 6))
    reserve_totals = {}
    for key, label, color, ls, marker in MODELS:
        vals = np.array(cells[key])
        cum = [vals[cal <= k].sum() / 1000.0 for k in quarters]
        reserve_totals[key] = cum[-1]
        err = r[peril]["error_pct"].get(key)
        lab = f"{label} (RM {cum[-1]:,.0f}k" + (f", {err:+.1f}%)" if err is not None else ")")
        ax.plot(range(len(quarters)), cum, marker=marker, linestyle=ls, color=color, label=lab,
                linewidth=2.4, markersize=7)

    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(q_labels, rotation=30, ha="right", fontsize=13)
    ax.set_xlabel("Holdout Calendar Quarter", fontsize=16)
    ax.set_ylabel("Cumulative Holdout Reserve (RM Thousands)", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    # No in-figure title: the caption in main.tex already states this,
    # duplicating it in the chart itself was flagged as bad practice.
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(reserve_totals)
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
