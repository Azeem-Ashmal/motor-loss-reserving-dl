"""
Regenerates the dissertation's pre-2020 vs post-2020 paid link-ratio comparison
figure (Figure 5.4) directly from the real triangles via
src.diagnostics.link_ratios_by_period, so that every plotted value and every
annotated percentage shift is computed from the data rather than typed in.

Calibration window: k <= VAL_END (55), matching the pipeline's own cutoff.
Period split: k <= 39 (through 2019 Q4) vs 40..55 (2020 Q1 onward).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import load_and_preprocess_triangles  # noqa: E402
from src.diagnostics import link_ratios_by_period  # noqa: E402

SPLIT_CAL_IDX = 39   # end of 2019 Q4
CUTOFF_CAL_IDX = 55  # VAL_END: last calibration diagonal


def main(data_dir: str, out_path: str) -> None:
    inc, balos, _, _ = load_and_preprocess_triangles(
        f"{data_dir}/paid_qtr_triangle.csv", f"{data_dir}/outstanding_qtr_triangle.csv"
    )
    cum = np.nancumsum(np.nan_to_num(inc), axis=1)
    cum[np.isnan(inc)] = np.nan

    r = link_ratios_by_period(cum, split_cal_idx=SPLIT_CAL_IDX, cutoff_cal_idx=CUTOFF_CAL_IDX)
    early, late = r["early"], r["late"]

    js = np.arange(1, 9)
    pre = np.array([early[j] for j in js])
    post = np.array([late[j] for j in js])
    rel = (post - pre) / pre * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    panels = [
        (axes[0], slice(0, 3), "Dev 1-3: Large, Volatile Factors",
         "Pre-2020 Training Diagonals", "Post-2020 Training Diagonals"),
        (axes[1], slice(3, 8), "Dev 4-8: Small, Near-Identical Factors",
         "Pre-2020 Training Diagonals (Normal Settlement)",
         "Post-2020 Training Diagonals (MCO Window)"),
    ]
    for ax, sl, title, lab_pre, lab_post in panels:
        x, p0, p1, rr = js[sl], pre[sl], post[sl], rel[sl]
        ax.plot(x, p0, "s-", color="green", label=lab_pre, linewidth=2, markersize=8)
        ax.plot(x, p1, "o-", color="red", label=lab_post, linewidth=2, markersize=8)
        for xi, yi, ri in zip(x, np.maximum(p0, p1), rr):
            ax.annotate(f"{ri:+.2f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontweight="bold", fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Development Quarter Index (j)", fontsize=12)
        ax.set_ylabel(r"Paid Link Ratio ($f_j^P$)", fontsize=12)
        ax.set_xticks(x)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)

    fig.suptitle("Paid Link Ratios Pre- vs Post-2020, Full Dev 1-8 Range", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    for j, a, b, c in zip(js, pre, post, rel):
        print(f"  Dev {j}: {a:.4f} -> {b:.4f}  ({c:+.2f}%)")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
