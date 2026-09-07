"""
Regenerates the annual earned vehicle-years exposure figure from the real
exposure series (identical across both perils, same Comprehensive policy base).
No figure number is drawn on the chart itself -- LaTeX's \\caption assigns and
owns that number, and hardcoding it here has previously gone stale when the
figure's position in the document changed.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(exposure_csv: str, out_path: str) -> None:
    df = pd.read_csv(exposure_csv)
    df["Year"] = df["Cohort"].str.slice(0, 4).astype(int)
    annual = df.groupby("Year")["Exposure"].sum() / 1_000_000.0
    annual = annual[annual.index <= 2025]  # 2026 Q1 alone is a partial year

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(annual.index, annual.values, "D-", color="#1f77b4", linewidth=2, markersize=6)
    for yr in [2010, 2012, 2017, 2020, 2025]:
        if yr in annual.index:
            ax.annotate(f"{annual[yr]:.2f}M", (yr, annual[yr]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontweight="bold")
    ax.set_title("Annual Earned Vehicle-Years Exposure (Comprehensive Private Car)", fontsize=14)
    ax.set_xlabel("Accident Year", fontsize=12)
    ax.set_ylabel("Earned Exposure (Millions of Vehicle-Years)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    for yr, val in annual.items():
        print(f"  {yr}: {val:.2f}M")
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
