"""
Paid Chain Ladder diagnostic utilities.
Supports the "why does Mack over/under-project" investigation: link ratios
estimated on two disjoint calendar sub-periods, and the development-quarter /
accident-quarter decomposition of the total holdout error. Every number here is
computed from the same cumulative-paid triangle passed to MackPaidChainLadder -
nothing is asserted in advance.
"""

from typing import Dict

import numpy as np


def link_ratios_by_period(cum_paid: np.ndarray, split_cal_idx: int, cutoff_cal_idx: int) -> Dict[str, np.ndarray]:
    """
    Link factors f_j estimated separately on calendar cells k <= split_cal_idx
    ("early period") and split_cal_idx < k <= cutoff_cal_idx ("late period").
    Both estimated only from the calibration window (k <= cutoff_cal_idx), so
    this never touches holdout cells.
    """
    n = cum_paid.shape[0]

    def _factors(k_lo, k_hi):
        f = np.ones(n - 1)
        for j in range(n - 1):
            valid = [i for i in range(n) if k_lo <= i + j <= k_hi and i + j + 1 <= k_hi
                     and not np.isnan(cum_paid[i, j]) and not np.isnan(cum_paid[i, j + 1]) and cum_paid[i, j] > 0]
            if valid:
                num = sum(cum_paid[i, j + 1] for i in valid)
                den = sum(cum_paid[i, j] for i in valid)
                if den > 0:
                    f[j] = num / den
        return f

    return {
        "early": _factors(0, split_cal_idx),
        "late": _factors(split_cal_idx + 1, cutoff_cal_idx),
    }


def reserve_impact_of_factor_vector(cum_paid: np.ndarray, f: np.ndarray, cutoff_cal_idx: int, holdout_end_idx: int) -> float:
    """
    Total RM projected into the holdout window (cutoff_cal_idx < k <= holdout_end_idx)
    if the *entire* triangle were projected using factor vector f from the last
    observed diagonal. Used to isolate how much reserve a given link-ratio
    vector, on its own, would produce - the check the previous review round
    asked for before attributing +56% error to a specific development range.
    """
    n = cum_paid.shape[0]
    total = 0.0
    for i in range(n):
        last_obs_j = cutoff_cal_idx - i
        if last_obs_j < 0 or last_obs_j >= n - 1:
            continue
        level = cum_paid[i, last_obs_j]
        if np.isnan(level):
            continue
        prev_cum = level
        for j in range(last_obs_j, n - 1):
            k = i + j + 1
            proj = prev_cum * f[j]
            if cutoff_cal_idx < k <= holdout_end_idx:
                total += proj - prev_cum
            prev_cum = proj
    return total
