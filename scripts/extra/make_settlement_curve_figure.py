"""
Regenerates the cumulative paid settlement pattern figure (Figure 3.2) directly
from the real triangles. Profile definition, stated so it is reproducible:
exposure-weighted mean cumulative paid as a percentage of each cohort's own
latest observed cumulative paid, over cohorts with at least 16 observed
development quarters (i.e. effectively run off).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import load_and_preprocess_triangles  # noqa: E402

MIN_DEVS = 16
N_SHOW = 16


def profile(data_dir: str) -> np.ndarray:
    inc, _, _, _ = load_and_preprocess_triangles(
        f"{data_dir}/paid_qtr_triangle.csv", f"{data_dir}/outstanding_qtr_triangle.csv"
    )
    num = np.zeros(N_SHOW)
    den = 0.0
    for i in range(inc.shape[0]):
        row = inc[i]
        v = np.where(~np.isnan(row))[0]
        if len(v) < MIN_DEVS:
            continue
        c = np.nancumsum(np.nan_to_num(row))
        total = c[v[-1]]
        if total > 0:
            num += c[:N_SHOW]
            den += total
    return num / den * 100.0


def main(theft_dir: str, ws_dir: str, out_path: str) -> None:
    t, w = profile(theft_dir), profile(ws_dir)
    x = np.arange(N_SHOW)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, t, "o-", color="blue", linewidth=2, markersize=7,
            label=f"Theft (Medium-Tail: {t[3]:.1f}% by Dev 3, {t[8]:.1f}% by Dev 8)")
    ax.plot(x, w, "s-", color="green", linewidth=2, markersize=7,
            label=f"Windscreen (Short-Tail: {w[1]:.1f}% by Dev 1, {w[3]:.1f}% by Dev 3)")
    ax.axvline(3, color="green", linestyle="--", alpha=0.7, label=f"Dev 3 (Windscreen {w[3]:.1f}% Mark)")
    ax.axvline(8, color="blue", linestyle="--", alpha=0.7, label=f"Dev 8 (Theft {t[8]:.1f}% Mark)")
    ax.set_title("Cumulative Paid Settlement Patterns by Peril", fontsize=15)
    ax.set_xlabel("Development Quarter (j)", fontsize=12)
    ax.set_ylabel("Cumulative Paid Settlement (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(x[::2])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    for j in [0, 1, 2, 3, 4, 8]:
        print(f"  Dev {j}: Theft {t[j]:5.1f}%   Windscreen {w[j]:5.1f}%")
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
