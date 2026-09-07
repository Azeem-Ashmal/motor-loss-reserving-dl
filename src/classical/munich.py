"""
Munich Chain Ladder (Quarg & Mack, 2004) Module.
Adjusts the Mack paid link factor at each cell using the standardized residual of
that cohort's incurred-to-paid ratio, regressed against the standardized paid
development residual (linear regression through the origin). Calibrated strictly
on cells with calendar index k <= cutoff_cal_idx.
"""

import numpy as np


class MunichChainLadder:
    def __init__(self, paid_triangle, balos_triangle, cutoff_cal_idx=None):
        self.C = np.array(paid_triangle, dtype=float)
        self.B = np.array(balos_triangle, dtype=float)
        self.I_mat = self.C + self.B
        self.I = self.C.shape[0]
        self.cutoff_cal_idx = self.I - 2 if cutoff_cal_idx is None else cutoff_cal_idx

    def fit_predict(self):
        num_cohorts, num_devs = self.I, self.I
        P_train = np.full_like(self.C, np.nan)
        I_train = np.full_like(self.I_mat, np.nan)
        for i in range(num_cohorts):
            for j in range(num_devs):
                if i + j <= self.cutoff_cal_idx:
                    P_train[i, j] = self.C[i, j]
                    I_train[i, j] = self.I_mat[i, j]

        f_P, f_I, sigma_f_P, sigma_f_I = [], [], [], []
        for j in range(num_devs - 1):
            v_P = ~np.isnan(P_train[:, j]) & ~np.isnan(P_train[:, j + 1]) & (P_train[:, j] > 0)
            if v_P.sum() > 0:
                f_p_val = P_train[v_P, j + 1].sum() / P_train[v_P, j].sum()
                factors = P_train[v_P, j + 1] / P_train[v_P, j]
                s_p = float(np.std(factors)) if v_P.sum() > 1 else 0.0
            else:
                f_p_val, s_p = 1.0, 0.0
            f_P.append(f_p_val); sigma_f_P.append(s_p)

            v_I = ~np.isnan(I_train[:, j]) & ~np.isnan(I_train[:, j + 1]) & (I_train[:, j] > 0)
            if v_I.sum() > 0:
                f_i_val = I_train[v_I, j + 1].sum() / I_train[v_I, j].sum()
                factors_i = I_train[v_I, j + 1] / I_train[v_I, j]
                s_i = float(np.std(factors_i)) if v_I.sum() > 1 else 0.0
            else:
                f_i_val, s_i = 1.0, 0.0
            f_I.append(f_i_val); sigma_f_I.append(s_i)

        f_P, f_I = np.array(f_P), np.array(f_I)
        sigma_f_P, sigma_f_I = np.array(sigma_f_P), np.array(sigma_f_I)

        q_I_list, dev_res_P_list = [], []
        for j in range(num_devs - 1):
            valid = (~np.isnan(P_train[:, j]) & ~np.isnan(I_train[:, j])
                     & ~np.isnan(P_train[:, j + 1]) & (P_train[:, j] > 0))
            if valid.sum() > 3:
                Q_j = I_train[valid, j] / P_train[valid, j]
                q_I = (Q_j - np.mean(Q_j)) / (np.std(Q_j) + 1e-6)
                dev_res_P = (P_train[valid, j + 1] / P_train[valid, j] - f_P[j]) / (sigma_f_P[j] + 1e-6)
                q_I_list.extend(q_I); dev_res_P_list.extend(dev_res_P)

        lambda_P = (float(np.cov(q_I_list, dev_res_P_list)[0, 1] / (np.var(q_I_list, ddof=1) + 1e-6))
                    if len(q_I_list) > 0 else 0.0)

        P_mcl, I_mcl = P_train.copy(), I_train.copy()
        for i in range(self.cutoff_cal_idx + 1, num_cohorts):
            if not np.isnan(self.C[i, 0]):
                P_mcl[i, 0] = self.C[i, 0]
                I_mcl[i, 0] = self.I_mat[i, 0]

        for i in range(num_cohorts):
            for j in range(num_devs - 1):
                if np.isnan(P_mcl[i, j + 1]) and not np.isnan(P_mcl[i, j]):
                    valid_train = ~np.isnan(P_train[:, j]) & ~np.isnan(I_train[:, j]) & (P_train[:, j] > 0)
                    if valid_train.sum() > 0 and P_mcl[i, j] > 0:
                        Q_series = I_train[valid_train, j] / P_train[valid_train, j]
                        Q_ij = I_mcl[i, j] / P_mcl[i, j]
                        q_I_ij = (Q_ij - np.mean(Q_series)) / (np.std(Q_series) + 1e-6)
                    else:
                        q_I_ij = 0.0
                    adj_f_P = f_P[j] + lambda_P * sigma_f_P[j] * q_I_ij
                    P_mcl[i, j + 1] = P_mcl[i, j] * max(adj_f_P, 1.0)
                    I_mcl[i, j + 1] = I_mcl[i, j] * max(f_I[j], 1.0)

        inc_P_mcl = np.zeros_like(P_mcl)
        inc_P_mcl[:, 0] = P_mcl[:, 0]
        inc_P_mcl[:, 1:] = P_mcl[:, 1:] - P_mcl[:, :-1]

        return {"lambda_P": lambda_P, "cl_cum_paid": P_mcl, "cl_inc_paid": inc_P_mcl}
