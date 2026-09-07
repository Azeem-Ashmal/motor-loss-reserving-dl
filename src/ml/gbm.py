"""
Gradient Boosting Machine (GBM) Non-Neural Control.
Two independent GradientBoostingRegressors (IncPaid, BALOS), trained on cells
with calendar index k <= cutoff_cal_idx, using features [dev_idx, IncPaid_t,
BALOS_t, CumPaid_t]. Predictions cannot extrapolate beyond
[min(y_train), max(y_train)] because tree leaves output bounded constants -
this is the documented mechanism behind the Windscreen GBM failure.

paid_only=True drops BALOS_t from the feature vector entirely and fits only
the paid head, for the paid-only-information-set control completing RQ1's
model-class grid (Table 5.2's note; §1.3, §8.1).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GBMReservingControl:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, paid_only=False):
        self.paid_only = paid_only
        self.gbm_paid = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
        )
        self.gbm_balos = None if paid_only else GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state,
        )

    def _features(self, dev_idx, inc_t, balos_t, cum_t):
        return [dev_idx, inc_t, cum_t] if self.paid_only else [dev_idx, inc_t, balos_t, cum_t]

    def fit(self, inc_matrix, balos_matrix, cutoff_cal_idx):
        num_cohorts, num_devs = inc_matrix.shape
        cum_paid = np.cumsum(inc_matrix, axis=1)
        X, y_paid, y_balos = [], [], []
        for i in range(num_cohorts):
            for t in range(num_devs - 1):
                k = i + t + 1
                if k <= cutoff_cal_idx and not np.isnan(inc_matrix[i, t]) and not np.isnan(inc_matrix[i, t + 1]):
                    X.append(self._features(t, inc_matrix[i, t], balos_matrix[i, t], cum_paid[i, t]))
                    y_paid.append(inc_matrix[i, t + 1])
                    y_balos.append(balos_matrix[i, t + 1])
        self.gbm_paid.fit(X, y_paid)
        if not self.paid_only:
            self.gbm_balos.fit(X, y_balos)
        return self

    def predict_step(self, dev_idx, inc_t, balos_t, cum_t):
        feat = [self._features(dev_idx, inc_t, balos_t, cum_t)]
        pred_inc = max(float(self.gbm_paid.predict(feat)[0]), 0.0)
        pred_balos = 0.0 if self.paid_only else max(float(self.gbm_balos.predict(feat)[0]), 0.0)
        return pred_inc, pred_balos

    def rollout(self, start_dev_idx, inc_start, balos_start, cum_start, n_steps):
        """Autoregressive multi-step forecast, feeding predictions back as input."""
        curr_inc, curr_bal, curr_cum = inc_start, balos_start, cum_start
        forecasts = []
        for step in range(n_steps):
            pred_inc, pred_bal = self.predict_step(start_dev_idx + step, curr_inc, curr_bal, curr_cum)
            forecasts.append(pred_inc)
            curr_inc, curr_cum = pred_inc, curr_cum + pred_inc
            if not self.paid_only:
                curr_bal = pred_bal
        return np.array(forecasts)
