"""
Unit tests for the classical reserving models against a hand-computable
example, independent of any real claims data or the integration-level reconciliation
in test_reconciliation.py.

The triangle below is the dissertation's own worked example (Table 1,
Section on Main Reserving Performance): a 4x4 synthetic triangle whose link
ratios (1.500, 1.200, 1.111) were verified by hand in the text. If this test
fails, either the dissertation's worked example or this code disagrees with
basic chain-ladder arithmetic - a five-minute check that should never
regress silently.
"""

import unittest

import numpy as np

from src.classical.mack import MackPaidChainLadder
from src.classical.icl import IncurredChainLadder
from src.seeding import set_deterministic_seed

# fmt: off
WORKED_EXAMPLE_PAID = np.array([
    [1000., 1500., 1800., 2000.],
    [1100., 1650., 1980., np.nan],
    [1200., 1800., np.nan, np.nan],
    [1300., np.nan, np.nan, np.nan],
])
# fmt: on


class TestMackWorkedExample(unittest.TestCase):
    def test_link_ratios_match_hand_calculation(self):
        mack = MackPaidChainLadder(WORKED_EXAMPLE_PAID, cutoff_cal_idx=3)
        f, _ = mack.fit_predict()
        np.testing.assert_allclose(f, [1.5, 1.2, 1.0 + 2.0 / 18.0], rtol=1e-9)

    def test_completed_triangle_matches_hand_calculation(self):
        mack = MackPaidChainLadder(WORKED_EXAMPLE_PAID, cutoff_cal_idx=3)
        _, C_full = mack.fit_predict()
        # AQ2 (row 1) dev 3: 1980 * f_2 = 1980 * 20/18 = 2200, per Table 1's [2,200]
        self.assertAlmostEqual(C_full[1, 3], 2200.0, places=6)
        # AQ3 (row 2) dev 2: 1800 * f_1 = 1800 * 1.2 = 2160, per Table 1's [2,160]
        self.assertAlmostEqual(C_full[2, 2], 2160.0, places=6)
        # AQ4 (row 3) dev 1: 1300 * f_0 = 1300 * 1.5 = 1950, per Table 1's [1,950]
        self.assertAlmostEqual(C_full[3, 1], 1950.0, places=6)

    def test_no_future_leakage_into_link_ratios(self):
        """A link ratio fit with a cutoff must not change if cells beyond the
        cutoff are altered - proves the calibration mask actually excludes them."""
        leaked = WORKED_EXAMPLE_PAID.copy()
        leaked[1, 3] = 999999.0  # a cell beyond cutoff_cal_idx=3 that should never be read
        f_clean, _ = MackPaidChainLadder(WORKED_EXAMPLE_PAID, cutoff_cal_idx=3).fit_predict()
        f_leaked, _ = MackPaidChainLadder(leaked, cutoff_cal_idx=3).fit_predict()
        np.testing.assert_array_equal(f_clean, f_leaked)


class TestICLWorkedExample(unittest.TestCase):
    def test_icl_allocation_sums_to_ultimate_incurred(self):
        balos = np.zeros_like(WORKED_EXAMPLE_PAID)  # BALOS=0 => incurred == paid on this toy example
        mack = MackPaidChainLadder(WORKED_EXAMPLE_PAID, cutoff_cal_idx=3)
        f_P, _ = mack.fit_predict()
        icl = IncurredChainLadder(WORKED_EXAMPLE_PAID, balos, cutoff_cal_idx=3)
        f_I, I_full, paid_allocation = icl.fit_predict(f_P)
        # With BALOS=0, incurred link ratios must equal the paid ones exactly
        np.testing.assert_allclose(f_I, f_P, rtol=1e-9)
        for i in range(4):
            self.assertAlmostEqual(paid_allocation[i, :].sum(), I_full[i, -1], places=4)


class TestDeterminism(unittest.TestCase):
    def test_set_deterministic_seed_is_repeatable(self):
        """The same seed must produce identical draws across independent calls -
        the actual property the reproducibility claim in Chapter 8 depends on."""
        import torch

        set_deterministic_seed(123)
        a = torch.rand(10).numpy().copy()
        set_deterministic_seed(123)
        b = torch.rand(10).numpy().copy()
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
