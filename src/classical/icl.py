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
        self.inc_pattern_ = inc_pattern
        return f_I, I_full, paid_allocation

    def icl_windowed_bootstrap(
        self, holdout_start_cal_idx: int, holdout_end_cal_idx: int,
        n_sims: int = 8000, seed: int = 42,
    ):
        """
        Windowed stochastic uncertainty for the ICL reserve this pipeline
        actually reports, not the raw incurred triangle. fit_predict() must
        be called first (this reuses its fitted f_I_, I_full_, and
        inc_pattern_ - the deterministic paid-settlement pattern derived from
        Mack's own paid link ratios, unchanged from the point estimate).

        Applies exactly the same Mack-model simulation mechanics as
        MackPaidChainLadder.mack_windowed_bootstrap (parameter variance
        se(f_j)^2 = sigma_j^2 / sum_i I_ij, one shared f_sim draw per
        (simulation, development period) rather than independently per
        cohort, process variance sigma_j^2 * I_ij per cell), applied to the
        incurred triangle I_ij = C_ij + B_ij instead of the paid triangle.
        Each simulated draw's completed incurred triangle gives a simulated
        ultimate incurred per cohort, I_sim[i, -1]; that ultimate is then
        spread across development quarters using the SAME fixed inc_pattern_
        the deterministic point estimate uses (an allocation device, not a
        re-simulated quantity - see fit_predict()'s docstring on why the paid
        pattern is held fixed rather than re-derived per draw), and only the
        incremental "paid-equivalent" amounts falling inside
        [holdout_start_cal_idx, holdout_end_cal_idx], for cohorts that already
        existed as of cutoff_cal_idx, are summed - matching exactly what
        Table 3/6's ICL row and run_peril_evaluation's cohort population use
        elsewhere in this pipeline.

        Caveat this function does not resolve on its own: Mack's independence
        and proportionality assumptions were derived for paid losses, and are
        a stronger assumption to apply to case-reserve-inclusive incurred
        data, where increments partly reflect adjuster judgement rather than
        pure random development. This measures ICL's uncertainty under a
        Mack-consistent model of the incurred triangle; it does not newly
        validate that model's assumptions for incurred data specifically.

        Returns dict with mean/se/p5/p95 of the simulated windowed
        paid-equivalent reserve, in RM thousands.
        """
        rng = np.random.default_rng(seed)
        I = self.I
        I_mat = self.I_mat
        f_I = self.f_I_
        inc_pattern = self.inc_pattern_

        train = np.full_like(I_mat, np.nan)
        for i in range(I):
            for j in range(I):
                if i + j <= self.cutoff_cal_idx:
                    train[i, j] = I_mat[i, j]

        sigma2 = np.zeros(I - 1)
        for j in range(I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            n_j = int(valid.sum())
            if n_j > 1:
                resid = train[valid, j + 1] / train[valid, j] - f_I[j]
                sigma2[j] = float(np.sum(train[valid, j] * resid**2) / (n_j - 1))
            elif j >= 2 and sigma2[j - 1] > 0 and sigma2[j - 2] > 0:
                sigma2[j] = min(sigma2[j - 1], sigma2[j - 2], sigma2[j - 1] ** 2 / sigma2[j - 2])
            elif j >= 1:
                sigma2[j] = sigma2[j - 1]

        se_f2 = np.zeros(I - 1)
        for j in range(I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            denom = train[valid, j].sum()
            se_f2[j] = sigma2[j] / denom if denom > 0 else 0.0

        windowed_sums = np.zeros(n_sims)
        for s in range(n_sims):
            I_sim = train.copy()
            for i in range(self.cutoff_cal_idx + 1, I):
                if not np.isnan(I_mat[i, 0]):
                    I_sim[i, 0] = I_mat[i, 0]
            for j in range(I - 1):
                # Shared per-(sim, dev-period) parameter draw, not independent
                # per cohort - see MackPaidChainLadder.mack_windowed_bootstrap
                # for why an independent-per-cohort draw would understate the
                # windowed reserve's true aggregate uncertainty.
                f_sim = f_I[j] + rng.normal(0.0, np.sqrt(max(se_f2[j], 0.0)))
                for i in range(I):
                    if i + j + 1 <= self.cutoff_cal_idx:
                        continue
                    if np.isnan(I_sim[i, j]):
                        continue
                    mean_next = I_sim[i, j] * f_sim
                    process_sd = np.sqrt(max(sigma2[j], 0.0) * max(I_sim[i, j], 0.0))
                    I_sim[i, j + 1] = mean_next + rng.normal(0.0, process_sd)

            window_total = 0.0
            for i in range(self.cutoff_cal_idx + 1):
                ultimate_incurred_sim = I_sim[i, -1]
                if np.isnan(ultimate_incurred_sim):
                    continue
                paid_alloc_sim_i = ultimate_incurred_sim * inc_pattern
                for j in range(I):
                    k = i + j
                    if holdout_start_cal_idx <= k <= holdout_end_cal_idx:
                        window_total += paid_alloc_sim_i[j]
            windowed_sums[s] = window_total

        return {
            "n_sims": n_sims,
            "mean_rm_k": float(np.mean(windowed_sums)) / 1000.0,
            "se_rm_k": float(np.std(windowed_sums, ddof=1)) / 1000.0,
            "p5_rm_k": float(np.percentile(windowed_sums, 5)) / 1000.0,
            "p95_rm_k": float(np.percentile(windowed_sums, 95)) / 1000.0,
        }
