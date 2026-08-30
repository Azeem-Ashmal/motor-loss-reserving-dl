"""
Figure Generation Script.
Regenerates figures directly from outputs/results.json (produced by
scripts/run_all.py) - no figure here is drawn from a value that isn't in that
file. Run `python scripts/run_all.py` first.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fig_reserve_comparison(results, peril, out_path):
    data = results[peril]
    models = [m for m in data["reserves_rm_k"] if m != "actual"]
    reserves = [data["reserves_rm_k"][m] for m in models]
    actual = data["reserves_rm_k"]["actual"]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if r > actual else "#2980b9" for r in reserves]
    ax.bar(models, reserves, color=colors)
    ax.axhline(actual, color="black", linestyle="--", label=f"Actual = RM {actual:,.0f}k")
    ax.set_ylabel("Projected Reserve (RM k)")
    ax.set_title(f"{peril.title()}: Projected Reserve by Model (Holdout)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_dev_error_decomposition(results, peril, out_path):
    dev_err = results[peril]["dev_error_mack_rm"]
    devs = sorted(int(k) for k in dev_err.keys())
    values = [dev_err[str(d)] / 1000.0 for d in devs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(d) for d in devs], values, color="#8e44ad")
    ax.set_xlabel("Development Quarter")
    ax.set_ylabel("Mack Signed Error (RM k)")
    ax.set_title(f"{peril.title()}: Mack Paid CL Error by Development Quarter")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_link_ratio_comparison(results, peril, out_path):
    f_p = results[peril]["mack_link_ratios"]
    f_i = results[peril]["icl_link_ratios"]
    n_show = min(15, len(f_p))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(n_show), f_p[:n_show], marker="o", label="Paid link ratio $f_j^P$")
    ax.plot(range(n_show), f_i[:n_show], marker="s", label="Incurred link ratio $f_j^I$")
    ax.set_xlabel("Development Quarter j")
    ax.set_ylabel("Link Ratio")
    ax.set_title(f"{peril.title()}: Paid vs Incurred Link Ratios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    results_path = Path(args.results) if args.results else base_dir / "outputs" / "results.json"
    if not results_path.exists():
        print(f"[ERROR] {results_path} not found. Run scripts/run_all.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    fig_dir = base_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for peril in results:
        fig_reserve_comparison(results, peril, fig_dir / f"{peril}_reserve_comparison.png")
        fig_dev_error_decomposition(results, peril, fig_dir / f"{peril}_dev_error_decomposition.png")
        fig_link_ratio_comparison(results, peril, fig_dir / f"{peril}_link_ratio_comparison.png")

    print(f"[SUCCESS] Wrote figures for {list(results.keys())} to {fig_dir}")


if __name__ == "__main__":
    main()
