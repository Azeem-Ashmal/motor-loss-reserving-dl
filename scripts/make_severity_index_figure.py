"""
Regenerates the Theft as-reported severity index figure (Figure 3.4) from the
real triangles. Severity for accident quarter i is that cohort's latest observed
cumulative paid divided by its latest observed cumulative reported count, indexed
to 100 at 2010 Q1. The final quarters fall steeply by construction: those cohorts
are observed only at very early development ages, where little of the eventual
large-claim settlement has emerged. The same artefact is visible in Figure 5.7
and is why the last two quarters are excluded from the log-linear fits.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(data_dir: str, out_path: str) -> None:
    paid = pd.read_csv(f"{data_dir}/paid_qtr_triangle.csv")
    cnt = pd.read_csv(f"{data_dir}/claims_count_qtr_triangle.csv")
    dev = [c for c in paid.columns if c.startswith("Dev_")]
    P, C = paid[dev].values.astype(float), cnt[dev].values.astype(float)

    n = P.shape[0]
    sev = np.full(n, np.nan)
    for i in range(n):
        vp = np.where(~np.isnan(P[i]))[0]
        vc = np.where(~np.isnan(C[i]))[0]
        if len(vp) and len(vc) and C[i][vc[-1]] > 0:
            sev[i] = P[i][vp[-1]] / C[i][vc[-1]]
    idx = sev / sev[0] * 100.0
    x = 2010 + np.arange(n) / 4.0

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axvspan(2017.5, 2017.75, color="orange", alpha=0.35, label="Motor De-tariffication (July 2017)")
    ax.axvspan(2020.0, 2021.75, color="grey", alpha=0.30, label="COVID-19 MCO Lockdowns (2020-2021)")
    ax.plot(x, idx, "o-", color="firebrick", linewidth=1.8, markersize=4)
    ax.set_title("Theft As-Reported Severity Index by Accident Quarter (65 Quarterly Observations)", fontsize=14)
    ax.set_xlabel("Accident Quarter", fontsize=12)
    ax.set_ylabel("As-Reported Severity Index (Base = 100 at 2010 Q1)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("  2010 Q1:", round(idx[0], 1), "| peak:", round(np.nanmax(idx), 1),
          "at", round(x[int(np.nanargmax(idx))], 2), "| 2026 Q1:", round(idx[-1], 1))
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
