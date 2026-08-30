"""
Tests for MackPaidChainLadder.mack_windowed_bootstrap - the Mack-model-
consistent simulation of the *windowed* holdout reserve (see its docstring
for why this exists: mack_se_full_runoff() only covers the full run-off to
ultimate, not the specific calendar-quarter window this pipeline's reserve
figures actually use).
"""

import unittest

import numpy as np
import pandas as pd

from src.classical.mack import MackPaidChainLadder

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
SYN_PAID = BASE_DIR / "data" / "synthetic" / "theft" / "paid_qtr_triangle.csv"

VAL_END = 55
HOLDOUT_END = 64


def _load_cum_paid():
    df = pd.read_csv(SYN_PAID)
    dev_cols = [c for c in df.columns if c.startswith("Dev_")]
    return df[dev_cols].values.astype(float)


class TestMackWindowedBootstrap(unittest.TestCase):
    def test_deterministic_across_two_runs_with_same_seed(self):
        cum_paid = _load_cum_paid()
        mack = MackPaidChainLadder(cum_paid, cutoff_cal_idx=VAL_END)
        mack.fit_predict()
        r1 = mack.mack_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=200, seed=7)
        r2 = mack.mack_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=200, seed=7)
        self.assertEqual(r1, r2)

    def test_bootstrap_mean_reconciles_to_the_deterministic_point_estimate(self):
        """The simulation's per-step noise has mean zero and f_sim is centred
        on the fitted f, so by the tower property the simulated windowed
        reserve must be unbiased for the deterministic chain-ladder point
        estimate for the same window - check the Monte Carlo mean lands
        within a generous tolerance of it (large n_sims to keep MC error
        small without making the test slow)."""
        cum_paid = _load_cum_paid()
        mack = MackPaidChainLadder(cum_paid, cutoff_cal_idx=VAL_END)
        f, C_full = mack.fit_predict()

        inc_full = np.zeros_like(C_full)
        inc_full[:, 0] = C_full[:, 0]
        inc_full[:, 1:] = C_full[:, 1:] - C_full[:, :-1]
        I, J = inc_full.shape
        point_estimate_rm_k = 0.0
        for i in range(VAL_END + 1):  # cohorts existing as of the cutoff only, matches pipeline.py
            for j in range(J):
                k = i + j
                if VAL_END + 1 <= k <= HOLDOUT_END and not np.isnan(inc_full[i, j]):
                    point_estimate_rm_k += inc_full[i, j]
        point_estimate_rm_k /= 1000.0

        result = mack.mack_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=4000, seed=42)
        rel_diff = abs(result["mean_rm_k"] - point_estimate_rm_k) / abs(point_estimate_rm_k)
        self.assertLess(rel_diff, 0.15,
                         f"bootstrap mean {result['mean_rm_k']} vs deterministic {point_estimate_rm_k}")
        self.assertGreater(result["se_rm_k"], 0.0)
        self.assertLess(result["p5_rm_k"], result["mean_rm_k"])
        self.assertLess(result["mean_rm_k"], result["p95_rm_k"])


if __name__ == "__main__":
    unittest.main()
