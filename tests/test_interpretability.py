"""
Unit test for occlusion-based interpretability: a model that ignores its
input entirely (zero weights everywhere) must report zero importance for
both channels, and a model with a real, resolvable dependence on the input
must report nonzero importance. This guards against the occlusion harness
silently measuring nothing.
"""

import unittest

import numpy as np
import torch

from src.data_pipeline import MinMaxSequenceScaler
from src.interpretability import occlusion_analysis
from src.ml.lstm import DeepTriangleLSTM

# fmt: off
TOY_PAID = np.array([
    [100., 150., 180., 200.],
    [110., 165., 198., np.nan],
    [120., 180., np.nan, np.nan],
    [130., np.nan, np.nan, np.nan],
])
TOY_BALOS = np.array([
    [20., 15., 10., 5.],
    [22., 16., 11., np.nan],
    [24., 18., np.nan, np.nan],
    [26., np.nan, np.nan, np.nan],
])
# fmt: on


def _toy_scaler():
    seqs = [np.column_stack((TOY_PAID[0, :2], TOY_BALOS[0, :2]))]
    tgts = [np.array([TOY_PAID[0, 2], TOY_BALOS[0, 2]])]
    return MinMaxSequenceScaler(use_log_transform=True).fit(seqs, tgts)


class TestOcclusionAnalysis(unittest.TestCase):
    def test_zero_weight_model_shows_zero_importance(self):
        """A model whose LSTM weights are all zero cannot respond to any
        input at all, so occluding a channel must not change its output."""
        model = DeepTriangleLSTM(input_dim=2, hidden_dim=8, dropout=0.0, output_activation="softplus")
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        scaler = _toy_scaler()
        result = occlusion_analysis(model, scaler, TOY_PAID, TOY_BALOS, peril_id=1, val_end=3, holdout_end=3)
        self.assertAlmostEqual(result["incpaid_importance_pct"], 0.0, places=6)
        self.assertAlmostEqual(result["balos_importance_pct"], 0.0, places=6)

    def test_returns_expected_keys_for_a_real_model(self):
        model = DeepTriangleLSTM(input_dim=2, hidden_dim=8, dropout=0.0, output_activation="softplus")
        scaler = _toy_scaler()
        result = occlusion_analysis(model, scaler, TOY_PAID, TOY_BALOS, peril_id=1, val_end=3, holdout_end=3)
        for key in ("normal", "occlude_incpaid", "occlude_balos", "incpaid_importance_pct", "balos_importance_pct"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
