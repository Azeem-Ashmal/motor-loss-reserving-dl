"""
Mack (1993) Paid Chain Ladder Module.
Computes link ratios f_j from cells observed at or before a calendar cutoff
only (no future-diagonal leakage into calibration) and projects the lower
triangle. Standard error estimation delegates to the `chainladder` package
(chainladder-python, MackChainladder) rather than a hand-rolled formula: an
earlier in-house implementation of Mack's variance/covariance terms was
cross-checked against chainladder on the published RAA benchmark triangle and
found to under-count the aggregate standard error by roughly 1.85x, while the
link-ratio and ultimate-loss logic in this module matched chainladder exactly
on the same data. Reusing a maintained, independently-tested implementation
for the one component that was actually wrong is safer than patching a subtle
covariance formula under time pressure.

Indexing: i = accident quarter, j = development quarter, k = i + j = calendar
quarter. cutoff_cal_idx is the last calendar index used to fit link ratios
(k <= cutoff_cal_idx); everything with k > cutoff_cal_idx is projected.
"""

import numpy as np


class MackPaidChainLadder:
    def __init__(self, paid_triangle, cutoff_cal_idx=None):
        self.C = np.array(paid_triangle, dtype=float)
        self.I = self.C.shape[0]
        self.cutoff_cal_idx = self.I - 2 if cutoff_cal_idx is None else cutoff_cal_idx

    def _train_matrix(self):
        train = np.full_like(self.C, np.nan)
        for i in range(self.I):
            for j in range(self.I):
                if i + j <= self.cutoff_cal_idx:
                    train[i, j] = self.C[i, j]
        return train

    def fit_predict(self):
        """
        Returns:
        - f: link factors f_0..f_{I-2}, fit only on k <= cutoff_cal_idx
        - C_full: completed cumulative triangle (observed cells preserved,
          k > cutoff_cal_idx cells replaced by the chain-ladder projection)
        """
        train = self._train_matrix()
        f = np.zeros(self.I - 1)
        for j in range(self.I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            num = train[valid, j + 1].sum()
            den = train[valid, j].sum()
            f[j] = num / den if valid.sum() > 0 and den > 0 else 1.0

        C_full = train.copy()
        for i in range(self.cutoff_cal_idx + 1, self.I):
            if not np.isnan(self.C[i, 0]):
                C_full[i, 0] = self.C[i, 0]
        for i in range(self.I):
            for j in range(self.I - 1):
                if np.isnan(C_full[i, j + 1]) and not np.isnan(C_full[i, j]):
                    C_full[i, j + 1] = C_full[i, j] * f[j]

        self.f_, self.C_full_, self.train_ = f, C_full, train
        return f, C_full

    def mack_se_full_runoff(self):
        """
        Aggregate Mack (1993) standard error of the FULL future run-off (every
        development period beyond the calibration cutoff, not only a specific
        holdout window), via chainladder.MackChainladder fit on the same
        calibration-window triangle used by fit_predict(). Must be called
        after fit_predict(). Returns None if chainladder cannot be imported.

        Note: this is the standard, textbook quantity (total prediction error
        of the full reserve to ultimate). It is NOT restricted to a specific
        calendar-quarter holdout window the way this module's own reserve
        figures are (see pipeline.py) - reconciling a *windowed* stochastic
        interval against a *full-run-off* one is a modelling decision, not
        something this function can resolve on its own.
        """
        try:
            import chainladder as cl
            import pandas as pd
        except ImportError:
            return None

        origins = pd.period_range("2010Q1", periods=self.I, freq="Q")
        rows = []
        for i in range(self.I):
            for j in range(self.I):
                if not np.isnan(self.train_[i, j]):
                    dev_date = (origins[i] + j).end_time.date()
                    rows.append({"origin": origins[i].start_time.date(), "development": dev_date,
                                 "value": self.train_[i, j]})
        long_df = pd.DataFrame(rows)
        tri = cl.Triangle(long_df, origin="origin", development="development", columns="value", cumulative=True)
        mack_cl = cl.MackChainladder().fit(tri)
        se_obj = mack_cl.total_mack_std_err_
        se_df = se_obj.to_frame() if hasattr(se_obj, "to_frame") else se_obj
        se_value = float(se_df.values.flatten()[-1])
        return {
            "ultimate_sum": float(mack_cl.ultimate_.sum()),
            "se_aggregate_full_runoff": se_value,
        }

    def _calibration_sigma2(self):
        """
        Mack (1993)'s own per-development-period process variance estimator,
        sigma_j^2 = (1/(n_j-1)) * sum_i C_ij * (C_i,j+1/C_ij - f_j)^2, computed
        strictly from calibration-window cells (k <= cutoff_cal_idx) - the same
        data fit_predict() used for f itself. Mack's own extrapolation rule is
        applied when a late development period has fewer than 2 calibration
        pairs (n_j <= 1): sigma_j^2 = min(sigma_{j-1}^2, sigma_{j-2}^2,
        sigma_{j-1}^4 / sigma_{j-2}^2).
        """
        train = self.train_
        I = self.I
        sigma2 = np.zeros(I - 1)
        for j in range(I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            n_j = int(valid.sum())
            if n_j > 1:
                resid = train[valid, j + 1] / train[valid, j] - self.f_[j]
                sigma2[j] = float(np.sum(train[valid, j] * resid**2) / (n_j - 1))
            elif j >= 2 and sigma2[j - 1] > 0 and sigma2[j - 2] > 0:
                sigma2[j] = min(sigma2[j - 1], sigma2[j - 2], sigma2[j - 1] ** 2 / sigma2[j - 2])
            elif j >= 1:
                sigma2[j] = sigma2[j - 1]
        return sigma2

    def mack_windowed_bootstrap(
        self, holdout_start_cal_idx: int, holdout_end_cal_idx: int,
        n_sims: int = 5000, seed: int = 42,
    ):
        """
        Mack-model-consistent parametric simulation of the *windowed* future
        reserve (only the calendar quarters in
        [holdout_start_cal_idx, holdout_end_cal_idx], not the full run-off to
        ultimate that mack_se_full_runoff() reports). This exists because the
        reserve figures this pipeline actually publishes are windowed holdout
        sums (see pipeline.py), and chainladder's total_mack_std_err_ has no
        direct equivalent for an arbitrary partial window - reconciling a
        windowed interval against a full-run-off one is not something that
        function can resolve, as already noted in mack_se_full_runoff()'s
        docstring. Must be called after fit_predict().

        Mirrors fit_predict()'s cell-by-cell recursive completion exactly
        (real dev-0 values re-seeded for post-cutoff cohorts, everything
        after propagated only from fitted link ratios), except at each
        simulated cell C_i,j+1 the link ratio itself is perturbed by parameter
        uncertainty (se(f_j)^2 = sigma_j^2 / sum_i C_ij over calibration i's)
        and the resulting mean is perturbed again by process variance
        (sigma_j^2 * C_ij), the same two variance components Mack's own
        closed-form formula decomposes the total MSEP into - this is a
        simulation of that same model, not a different one.

        Returns dict with mean/se/p5/p95 of the simulated windowed reserve
        (sum of incremental paid over the window, across all cohorts).
        """
        rng = np.random.default_rng(seed)
        I = self.I
        f, train = self.f_, self.train_
        sigma2 = self._calibration_sigma2()

        se_f2 = np.zeros(I - 1)
        for j in range(I - 1):
            valid = ~np.isnan(train[:, j]) & ~np.isnan(train[:, j + 1]) & (train[:, j] > 0)
            denom = train[valid, j].sum()
            se_f2[j] = sigma2[j] / denom if denom > 0 else 0.0

        windowed_sums = np.zeros(n_sims)
        for s in range(n_sims):
            C_sim = train.copy()
            for i in range(self.cutoff_cal_idx + 1, I):
                if not np.isnan(self.C[i, 0]):
                    C_sim[i, 0] = self.C[i, 0]
            for j in range(I - 1):
                # Parameter uncertainty in f_j is ONE shared unknown quantity
                # per development period, not independent per cohort - Mack's
                # own aggregate SE formula has a cross-accident-year covariance
                # term specifically because every cohort's projection beyond
                # the cutoff uses the *same* uncertain f_j. Drawing f_sim once
                # per (simulation, j) here, applied to every cohort i at that
                # j within this draw, is what makes that shared uncertainty
                # actually correlate across cohorts in the simulated sum -
                # drawing it independently per cohort would incorrectly
                # diversify it away, understating the windowed reserve's true
                # aggregate uncertainty. Process variance (below) remains an
                # independent per-cell draw, matching Mack's own assumption.
                f_sim = f[j] + rng.normal(0.0, np.sqrt(max(se_f2[j], 0.0)))
                for i in range(I):
                    if i + j + 1 <= self.cutoff_cal_idx:
                        continue  # target cell is itself calibration data - keep the observed value
                    if np.isnan(C_sim[i, j]):
                        continue  # source cell not yet available (future accident quarter)
                    mean_next = C_sim[i, j] * f_sim
                    process_sd = np.sqrt(max(sigma2[j], 0.0) * max(C_sim[i, j], 0.0))
                    C_sim[i, j + 1] = mean_next + rng.normal(0.0, process_sd)

            inc_sim = np.zeros_like(C_sim)
            inc_sim[:, 0] = C_sim[:, 0]
            inc_sim[:, 1:] = C_sim[:, 1:] - C_sim[:, :-1]
            window_total = 0.0
            for i in range(self.cutoff_cal_idx + 1):
                # Restricted to cohorts that already exist as of cutoff_cal_idx,
                # matching pipeline.py's run_peril_evaluation loop (which only
                # rolls forward cohorts with last_dev = cutoff_cal_idx - i >= 0
                # - i.e. never invents a "reserve" for accident quarters that
                # only begin inside the holdout window itself).
                for j in range(I - 1):
                    k = i + j
                    if holdout_start_cal_idx <= k <= holdout_end_cal_idx and not np.isnan(inc_sim[i, j]):
                        window_total += inc_sim[i, j]
            windowed_sums[s] = window_total

        return {
            "n_sims": n_sims,
            "mean_rm_k": float(np.mean(windowed_sums)) / 1000.0,
            "se_rm_k": float(np.std(windowed_sums, ddof=1)) / 1000.0,
            "p5_rm_k": float(np.percentile(windowed_sums, 5)) / 1000.0,
            "p95_rm_k": float(np.percentile(windowed_sums, 95)) / 1000.0,
        }
