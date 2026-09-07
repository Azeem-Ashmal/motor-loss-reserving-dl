"""
Tests for the Duan (1983) smearing correction (Section 4.1.3):
MinMaxSequenceScaler.inverse_transform_targets(..., smearing_psi=...) and
src.evaluate.compute_duan_smearing_psi.
"""
import unittest

import numpy as np
import torch

from src.data_pipeline import MinMaxSequenceScaler
from src.evaluate import compute_duan_smearing_psi
from src.ml.lstm import DeepTriangleLSTM


class TestSmearingRetransformation(unittest.TestCase):
    def _fitted_scaler(self):
        seqs = [np.array([[100.0, 50.0], [200.0, 40.0], [150.0, 30.0]])]
        tgts = [np.array([120.0, 25.0])]
        return MinMaxSequenceScaler(use_log_transform=True).fit(seqs, tgts)

    def test_psi_one_reproduces_plain_expm1(self):
        scaler = self._fitted_scaler()
        scaled = scaler.transform_feature_array(np.array([[300.0, 60.0]]))
        plain = scaler.inverse_transform_targets(scaled)
        corrected = scaler.inverse_transform_targets(scaled, smearing_psi=(1.0, 1.0))
        np.testing.assert_allclose(plain, corrected, rtol=1e-5)

    def test_psi_greater_than_one_increases_prediction(self):
        scaler = self._fitted_scaler()
        scaled = scaler.transform_feature_array(np.array([[300.0, 60.0]]))
        plain = scaler.inverse_transform_targets(scaled)
        corrected = scaler.inverse_transform_targets(scaled, smearing_psi=(1.2, 1.2))
        self.assertGreater(corrected[0, 0], plain[0, 0])
        self.assertGreater(corrected[0, 1], plain[0, 1])

    def test_offset_stays_outside_psi_multiplication(self):
        # Y = psi * exp(log_val) - 1, not expm1(log_val) * psi.
        scaler = self._fitted_scaler()
        scaled = scaler.transform_feature_array(np.array([[300.0, 60.0]]))
        log_vals = scaler.unscale_to_log1p(scaled)
        psi = (1.15, 1.05)
        corrected = scaler.inverse_transform_targets(scaled, smearing_psi=psi)
        expected_inc = psi[0] * np.exp(log_vals[0, 0]) - 1.0
        expected_bal = psi[1] * np.exp(log_vals[0, 1]) - 1.0
        self.assertAlmostEqual(float(corrected[0, 0]), float(expected_inc), places=3)
        self.assertAlmostEqual(float(corrected[0, 1]), float(expected_bal), places=3)

    def test_psi_from_perfect_predictions_is_one(self):
        """If predictions exactly match targets, residuals are 0 and psi = exp(0) = 1."""
        scaler = self._fitted_scaler()
        n = 8
        X = torch.zeros(n, 65, 2)
        mask = torch.zeros(n, 65)
        mask[:, :3] = 1.0
        peril_ids = torch.zeros(n, dtype=torch.long)
        Y = torch.rand(n, 2) * 0.5 + 0.25  # arbitrary valid scaled targets

        class PerfectModel:
            def eval(self):
                pass

            def __call__(self, X_, mask_, peril_):
                return Y

        psi_inc, psi_balos = compute_duan_smearing_psi(PerfectModel(), scaler, {
            "X": X, "Y": Y, "mask": mask, "peril_ids": peril_ids,
        })
        self.assertAlmostEqual(psi_inc, 1.0, places=5)
        self.assertAlmostEqual(psi_balos, 1.0, places=5)

    def test_psi_positive_for_a_real_trained_model(self):
        """Sanity check against an actual (untrained, random-weight) model: psi
        must be a finite positive number, since it's a mean of exp(.) terms."""
        scaler = self._fitted_scaler()
        model = DeepTriangleLSTM(input_dim=2, hidden_dim=8, dropout=0.0, output_activation="softplus")
        n = 6
        X = torch.rand(n, 65, 2) * 0.3
        mask = torch.zeros(n, 65)
        mask[:, :4] = 1.0
        peril_ids = torch.zeros(n, dtype=torch.long)
        Y = torch.rand(n, 2) * 0.5

        psi_inc, psi_balos = compute_duan_smearing_psi(model, scaler, {
            "X": X, "Y": Y, "mask": mask, "peril_ids": peril_ids,
        })
        self.assertTrue(np.isfinite(psi_inc) and psi_inc > 0)
        self.assertTrue(np.isfinite(psi_balos) and psi_balos > 0)


if __name__ == "__main__":
    unittest.main()
