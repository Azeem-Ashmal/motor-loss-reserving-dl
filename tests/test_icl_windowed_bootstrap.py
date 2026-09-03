"""
Tests for IncurredChainLadder.icl_windowed_bootstrap - the windowed
stochastic uncertainty of the ICL reserve this pipeline actually reports
(paid-equivalent, allocated via the fixed Mack paid pattern), not the raw
incurred triangle's own uncertainty. Mirrors
test_mack_windowed_bootstrap.py's checks: determinism under a fixed seed,
and unbiasedness of the simulated mean against the deterministic point
estimate for the same window.
"""

import unittest

import numpy as np
import pandas as pd

from src.classical.mack import MackPaidChainLadder
from src.classical.icl import IncurredChainLadder

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent
SYN_PAID = BASE_DIR / "data" / "synthetic" / "theft" / "paid_qtr_triangle.csv"
SYN_OUTSTANDING = BASE_DIR / "data" / "synthetic" / "theft" / "outstanding_qtr_triangle.csv"

VAL_END = 55
HOLDOUT_END = 64


def _load_dev_cols(path):
    df = pd.read_csv(path)
    dev_cols = [c for c in df.columns if c.startswith("Dev_")]
    return df[dev_cols].values.astype(float)


class TestICLWindowedBootstrap(unittest.TestCase):
    def _fit(self):
        cum_paid = _load_dev_cols(SYN_PAID)
        outstanding = _load_dev_cols(SYN_OUTSTANDING)
        mack = MackPaidChainLadder(cum_paid, cutoff_cal_idx=VAL_END)
        f_paid, _ = mack.fit_predict()
        icl = IncurredChainLadder(cum_paid, outstanding, cutoff_cal_idx=VAL_END)
        icl.fit_predict(f_paid)
        return icl

    def test_deterministic_across_two_runs_with_same_seed(self):
        icl = self._fit()
        r1 = icl.icl_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=200, seed=7)
        r2 = icl.icl_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=200, seed=7)
        self.assertEqual(r1, r2)

    def test_bootstrap_mean_reconciles_to_the_deterministic_point_estimate(self):
        icl = self._fit()

        # Deterministic point estimate: same paid_allocation the point
        # estimate uses, windowed and restricted to cohorts existing as of
        # the cutoff, exactly as run_peril_evaluation does elsewhere.
        I_full = icl.I_full_
        inc_pattern = icl.inc_pattern_
        I = I_full.shape[0]
        point_estimate_rm_k = 0.0
        for i in range(VAL_END + 1):
            ultimate_incurred = I_full[i, -1]
            if np.isnan(ultimate_incurred):
                continue
            paid_alloc_i = ultimate_incurred * inc_pattern
            for j in range(I):
                k = i + j
                if VAL_END + 1 <= k <= HOLDOUT_END:
                    point_estimate_rm_k += paid_alloc_i[j]
        point_estimate_rm_k /= 1000.0

        result = icl.icl_windowed_bootstrap(VAL_END + 1, HOLDOUT_END, n_sims=4000, seed=42)
        rel_diff = abs(result["mean_rm_k"] - point_estimate_rm_k) / abs(point_estimate_rm_k)
        self.assertLess(rel_diff, 0.15,
                         f"bootstrap mean {result['mean_rm_k']} vs deterministic {point_estimate_rm_k}")
        self.assertGreater(result["se_rm_k"], 0.0)
        self.assertLess(result["p5_rm_k"], result["mean_rm_k"])
        self.assertLess(result["mean_rm_k"], result["p95_rm_k"])


if __name__ == "__main__":
    unittest.main()
