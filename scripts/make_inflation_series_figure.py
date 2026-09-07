"""
Regenerates the full 65-quarter frequency/severity/burning-cost series for both
perils directly from the real triangles. As-reported basis: each accident
quarter's latest observed cumulative paid and count on the diagonal, divided by
that quarter's exposure. No figure number is drawn on the chart -- see the note
in make_exposure_series_figure.py for why.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(data_dir: str):
    paid = pd.read_csv(f"{data_dir}/paid_qtr_triangle.csv")
    cnt = pd.read_csv(f"{data_dir}/claims_count_qtr_triangle.csv")
    exp = pd.read_csv(f"{data_dir}/exposure_qtr.csv")
    dev = [c for c in paid.columns if c.startswith("Dev_")]
    P, C = paid[dev].values.astype(float), cnt[dev].values.astype(float)
    E = exp["Exposure"].values.astype(float)
    n = P.shape[0]
    ult_p = np.full(n, np.nan)
    ult_c = np.full(n, np.nan)
    for i in range(n):
        vp = np.where(~np.isnan(P[i]))[0]
        vc = np.where(~np.isnan(C[i]))[0]
        if len(vp):
            ult_p[i] = P[i][vp[-1]]
        if len(vc):
            ult_c[i] = C[i][vc[-1]]
    freq = ult_c / E
    sev = ult_p / ult_c
    bc = ult_p / E
    return freq, sev, bc


def main(theft_dir: str, ws_dir: str, out_path: str) -> None:
    tf, ts, tb = series(theft_dir)
    wf, ws_, wb = series(ws_dir)
    n = len(tf)
    x = 2010 + np.arange(n) / 4.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    ax = axes[0]
    ax2 = ax.twinx()
    ax.axvspan(2017.5, 2018.5, color="orange", alpha=0.25)
    ax.axvspan(2020.0, 2021.75, color="grey", alpha=0.30)
    l1, = ax.plot(x, tf, color="firebrick", label="Theft (left axis)")
    l2, = ax2.plot(x, wf, color="steelblue", label="Windscreen (right axis)")
    ax.set_ylabel("Theft: Claims / Vehicle-Year", color="firebrick")
    ax2.set_ylabel("Windscreen: Claims / Vehicle-Year", color="steelblue")
    ax.tick_params(axis="y", colors="firebrick")
    ax2.tick_params(axis="y", colors="steelblue")
    ax.set_title("Claim Frequency (note: separate axes)")
    ax.legend(handles=[l1, l2], loc="upper left")

    ax = axes[1]
    ax.axvspan(2017.5, 2018.5, color="orange", alpha=0.25)
    ax.axvspan(2020.0, 2021.75, color="grey", alpha=0.30)
    ax.plot(x, ts, color="firebrick", label="Theft")
    ax.plot(x, ws_, color="steelblue", label="Windscreen")
    ax.set_yscale("log")
    ax.set_ylabel("RM / Claim")
    ax.set_title("Claim Severity")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.axvspan(2017.5, 2018.5, color="orange", alpha=0.25)
    ax.axvspan(2020.0, 2021.75, color="grey", alpha=0.30)
    ax.plot(x, tb, color="firebrick", label="Theft")
    ax.plot(x, wb, color="steelblue", label="Windscreen")
    ax.set_ylabel("RM / Vehicle-Year")
    ax.set_title("Burning Cost")
    ax.set_xlabel("Accident Quarter")
    ax.legend(loc="upper left")

    for a in axes:
        a.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("As-Reported Frequency, Severity and Burning Cost, 2010 Q1 - 2026 Q1", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Theft 2016/2024/2025 BC:", round(tb[24], 2), round(tb[56], 2), round(tb[60], 2))
    print("WS 2010/2024/2025 BC:", round(wb[0], 2), round(wb[56], 2), round(wb[60], 2))
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
