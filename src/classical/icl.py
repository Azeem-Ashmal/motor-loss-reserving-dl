"""
Incurred Chain Ladder (ICL) Module.
Models cumulative incurred losses I_{i,j} = C_{i,j} + B_{i,j}, projected on link
ratios calibrated only from cells with calendar index k <= cutoff_cal_idx.
Ultimate incurred is then re-expressed as expected *paid* cashflow by applying the
paid settlement pattern implied by the Mack link ratios (allocation, not
projection: ICL sets the ultimate level, Mack's f_j only spreads it across time).
"""

import numpy as np


class IncurredChainLadder:
    def __init__(self, paid_triangle, balos_triangle, cutoff_cal_idx=None):
        self.C = np.array(paid_triangle, dtype=float)
        self.B = np.array(balos_triangle, dtype=float)
        self.I_mat = self.C + self.B
        self.I = self.C.shape[0]
        self.cutoff_cal_idx = self.I - 2 if cutoff_cal_idx is None else cutoff_cal_idx

    def fit_predict(self, paid_link_ratios):
        """
        paid_link_ratios: f_P from MackPaidChainLadder.fit_predict(), used only to
        allocate the ICL ultimate across development quarters as expected paid cash.

        Returns:
        - f_I: incurred link factors
        - I_full: completed cumulative incurred triangle
        - paid_allocation: expected incremental paid triangle implied by the ICL
          ultimate incurred and the Mack paid settlement pattern
        """
        train = np.full_like(self.I_mat, np.nan)
        for i in range(self.I):
            for j in range(self.I):
                if i + j <= self.cutoff_cal_idx:
                    train[i, j] = self.I_mat[i, j]

        f_I = np.zeros(self.I - 1)
        for j in range(self.I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            num = train[valid, j + 1].sum()
            den = train[valid, j].sum()
            f_I[j] = num / den if valid.sum() > 0 and den > 0 else 1.0

        I_full = train.copy()
        for i in range(self.cutoff_cal_idx + 1, self.I):
            if not np.isnan(self.I_mat[i, 0]):
                I_full[i, 0] = self.I_mat[i, 0]
        for i in range(self.I):
            for j in range(self.I - 1):
                if np.isnan(I_full[i, j + 1]) and not np.isnan(I_full[i, j]):
                    I_full[i, j + 1] = I_full[i, j] * f_I[j]

        # Cumulative paid settlement pattern normalised to the ultimate, from the
        # Mack paid link ratios. Allocation only: ICL still sets the ultimate level.
        f_P = np.asarray(paid_link_ratios, dtype=float)
        cum_pattern = np.ones(self.I)
        for j in range(self.I - 2, -1, -1):
            cum_pattern[j] = cum_pattern[j + 1] / f_P[j]
        cum_pattern = cum_pattern / cum_pattern[-1]
        inc_pattern = np.zeros(self.I)
        inc_pattern[0] = cum_pattern[0]
        inc_pattern[1:] = cum_pattern[1:] - cum_pattern[:-1]

        paid_allocation = np.zeros_like(I_full)
        for i in range(self.I):
            paid_allocation[i, :] = I_full[i, -1] * inc_pattern

        self.f_I_, self.I_full_ = f_I, I_full
        return f_I, I_full, paid_allocation
