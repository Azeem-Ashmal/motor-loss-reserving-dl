"""
Bornhuetter-Ferguson (1972) Reserving Module.
Anchors immature accident cohorts to an ex-ante exposure-based burning-cost
prior, rather than multiplying volatile early paid link ratios:
R_i^BF = E_i * BC_0 * (1 - 1/prod(f_k^P))

BC_0 is never a hardcoded constant here: compute_prior_burning_cost derives it
from a window of mature cohorts in whichever triangle this is applied to, so a
value fitted on one dataset can never silently leak into a run on another.
"""

import numpy as np


class BornhuetterFerguson:
    def __init__(self, prior_bc: float):
        if prior_bc <= 0:
            raise ValueError("prior_bc must be a positive burning cost per exposure unit")
        self.prior_bc = prior_bc

    @staticmethod
    def compute_prior_burning_cost(cum_paid: np.ndarray, exposure: np.ndarray, mature_cohort_range) -> float:
        """
        cum_paid: (n_cohorts, n_devs) cumulative paid triangle.
        exposure: (n_cohorts,) exposure aligned to cum_paid rows, same units as
                  the reserve this prior will be applied to.
        mature_cohort_range: iterable of cohort indices treated as fully (or
                              near-fully) developed, e.g. range(24, 36) for
                              AY2016-2018 on a 2010 Q1-origin quarterly triangle.
        Returns the average burning cost (ultimate paid / exposure) across
        those cohorts.
        """
        burning_costs = []
        for i in mature_cohort_range:
            row = cum_paid[i]
            valid = row[~np.isnan(row)]
            if len(valid) == 0 or exposure[i] <= 0:
                continue
            burning_costs.append(valid[-1] / exposure[i])
        if not burning_costs:
            raise ValueError("No valid mature cohorts found to compute a burning-cost prior from")
        return float(np.mean(burning_costs))

    def allocate_holdout_reserve(
        self,
        exposure: np.ndarray,
        paid_link_ratios: np.ndarray,
        cutoff_cal_idx: int,
        holdout_end_idx: int,
        last_dev_by_cohort,
    ) -> float:
        """
        For each cohort i with last observed development last_dev_by_cohort[i]
        (None if the cohort has no holdout exposure), projects the BF ultimate
        (E_i * prior_bc) forward using the cumulative settlement pattern implied
        by paid_link_ratios, and sums the incremental reserve that falls inside
        the holdout window (cutoff_cal_idx, holdout_end_idx].
        """
        n = len(paid_link_ratios) + 1
        cum_pattern = np.ones(n)
        for j in range(n - 2, -1, -1):
            cum_pattern[j] = cum_pattern[j + 1] / paid_link_ratios[j]
        cum_pattern = cum_pattern / cum_pattern[-1]
        inc_pattern = np.zeros(n)
        inc_pattern[0] = cum_pattern[0]
        inc_pattern[1:] = cum_pattern[1:] - cum_pattern[:-1]

        total = 0.0
        for i, last_dev in enumerate(last_dev_by_cohort):
            if last_dev is None:
                continue
            prior_ultimate = exposure[i] * self.prior_bc
            for j in range(last_dev + 1, n):
                k = i + j
                if cutoff_cal_idx < k <= holdout_end_idx:
                    total += prior_ultimate * inc_pattern[j]
        return total
