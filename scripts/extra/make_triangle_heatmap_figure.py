"""
Regenerates the triangle heatmap figure (Figure 3.1) from the real triangles:
the three input surfaces the models are built from, for the 16 most recent
accident quarters at development ages 0-15. Cells outside the observed triangle
are masked rather than plotted as zero, so the calibration boundary is visible.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import load_and_preprocess_triangles  # noqa: E402

FIRST_AQ, N_AQ, N_DEV = 49, 16, 16


def main(data_dir: str, out_path: str) -> None:
    inc, balos, _, _ = load_and_preprocess_triangles(
        f"{data_dir}/paid_qtr_triangle.csv", f"{data_dir}/outstanding_qtr_triangle.csv"
    )
    cnt = pd.read_csv(f"{data_dir}/claims_count_qtr_triangle.csv")
    dev_cols = [c for c in cnt.columns if c.startswith("Dev_")]
    counts = cnt[dev_cols].values.astype(float)

    cum = np.nancumsum(np.nan_to_num(inc), axis=1)
    cum[np.isnan(inc)] = np.nan

    sl = (slice(FIRST_AQ, FIRST_AQ + N_AQ), slice(0, N_DEV))
    panels = [
        (np.ma.masked_invalid(cum[sl]) / 1000.0, r"Cumulative Paid Losses $C_{ij}$ (RM thousands)", "viridis"),
        (np.ma.masked_invalid(balos[sl]) / 1000.0, r"Outstanding Case Reserves $B_{ij}$ (RM thousands)", "plasma"),
        (np.ma.masked_invalid(counts[sl]), r"Cumulative Reported Claim Counts $N_{ij}$", "cividis"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, (M, title, cmap) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        im = ax.imshow(M, aspect="auto", cmap=cm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Development Quarter (Dev 0..15)", fontsize=11)
        ax.set_ylabel(f"Accident Quarter (AQ {FIRST_AQ}..{FIRST_AQ + N_AQ - 1})", fontsize=11)
        ax.set_xticks(range(0, N_DEV, 2))
        ax.set_yticks(range(0, N_AQ, 2))
        ax.set_yticklabels(range(FIRST_AQ, FIRST_AQ + N_AQ, 2))

    fig.suptitle("Theft Input Surfaces, 16 Most Recent Accident Quarters "
                 "(white = outside the observed triangle)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    for M, t, _ in panels:
        print(f"  {t}: min {M.min():,.1f}  max {M.max():,.1f}  observed cells {M.count()}")
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
