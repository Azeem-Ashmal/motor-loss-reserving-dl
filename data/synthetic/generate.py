"""
Synthetic Loss Triangle Generator.
Produces mock 65x65 cumulative-paid and outstanding-case-reserve (BALOS)
quarterly triangles for two perils, with broadly Theft-like (low frequency,
high severity, slow settlement) and Windscreen-like (high frequency, low
severity, fast settlement) shapes. Fixed seed, round base numbers, and this
header comment make clear the data is synthetic: it exercises the pipeline
end to end but does not reproduce the dissertation's results, which were
computed on a proprietary industry claims database (see data/README.md).

Index convention: 0-indexed i (accident quarter), j (development quarter),
k = i + j (calendar quarter); the upper-right (k > 64) cells are unobserved
and left as NaN, matching the real triangle layout.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROFILES = {
    "theft": {
        "base_claim_rate": 0.00015,   # low frequency
        "avg_severity": 75000.0,      # high severity
        "decay_rate": 0.04,           # slow settlement (medium tail)
        "balos_horizon": 40.0,
    },
    "windscreen": {
        "base_claim_rate": 0.045,     # high frequency
        "avg_severity": 1500.0,       # low severity
        "decay_rate": 0.35,           # fast settlement (short tail)
        "balos_horizon": 4.0,
    },
}


def generate_peril_triangles(profile, n_quarters=65, seed=42):
    rng = np.random.default_rng(seed)
    cohorts = [f"{2010 + i // 4} Q{i % 4 + 1}" for i in range(n_quarters)]
    exposure = 7_000_000 + np.linspace(0, 3_500_000, n_quarters)

    dev_factors = np.exp(-profile["decay_rate"] * np.arange(n_quarters))
    dev_factors /= dev_factors.sum()

    paid_matrix = np.full((n_quarters, n_quarters), np.nan)
    balos_matrix = np.full((n_quarters, n_quarters), np.nan)
    count_matrix = np.full((n_quarters, n_quarters), np.nan)

    for i in range(n_quarters):
        base_claim_cnt = max(1, int(exposure[i] * profile["base_claim_rate"] * rng.uniform(0.9, 1.1)))
        base_loss = base_claim_cnt * profile["avg_severity"] * rng.uniform(0.95, 1.05)

        cum_paid = 0.0
        cum_count = 0.0
        for j in range(n_quarters - i):
            inc_paid = base_loss * dev_factors[j] * rng.uniform(0.92, 1.08)
            cum_paid += inc_paid
            paid_matrix[i, j] = round(cum_paid, 2)

            rem_pct = max(0.0, 1.0 - (j / profile["balos_horizon"]))
            balos_matrix[i, j] = round(base_loss * rem_pct * 0.5 * rng.uniform(0.85, 1.15), 2)

            # Cumulative reported claim count, same reporting-pattern shape as
            # paid development (dev_factors), rounded to whole claims.
            inc_count = base_claim_cnt * dev_factors[j] * rng.uniform(0.92, 1.08)
            cum_count += inc_count
            count_matrix[i, j] = round(cum_count)

    dev_cols = [f"Dev_{j}" for j in range(n_quarters)]
    df_paid = pd.DataFrame(paid_matrix, columns=dev_cols)
    df_paid.insert(0, "Cohort", cohorts)
    df_balos = pd.DataFrame(balos_matrix, columns=dev_cols)
    df_balos.insert(0, "Cohort", cohorts)
    df_count = pd.DataFrame(count_matrix, columns=dev_cols)
    df_count.insert(0, "Cohort", cohorts)
    return df_paid, df_balos, df_count, cohorts


def generate_all(out_dir=None, n_quarters=65, seed=42):
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent
    for peril_name, profile in PROFILES.items():
        peril_dir = out_dir / peril_name
        peril_dir.mkdir(parents=True, exist_ok=True)
        df_paid, df_balos, df_count, cohorts = generate_peril_triangles(profile, n_quarters=n_quarters, seed=seed)
        df_paid.to_csv(peril_dir / "paid_qtr_triangle.csv", index=False)
        df_balos.to_csv(peril_dir / "outstanding_qtr_triangle.csv", index=False)
        df_count.to_csv(peril_dir / "claims_count_qtr_triangle.csv", index=False)

        exposure = 7_000_000 + np.linspace(0, 3_500_000, n_quarters)
        pd.DataFrame({"Cohort": cohorts, "Exposure": np.round(exposure, 0)}).to_csv(
            peril_dir / "exposure_qtr.csv", index=False
        )
        print(f"[SUCCESS] Wrote synthetic {peril_name} triangles to {peril_dir}")


if __name__ == "__main__":
    generate_all()
