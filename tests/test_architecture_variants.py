"""
Smoke, determinism, and parameter-count tests for the two architecture/feature
variants added during the harsh-critic model audit: the GRU cell option
(rnn_type="gru", tested because Kuo (2019)'s original DeepTriangle used a GRU,
not an LSTM) and the claim-count input channel (count_path, tested because the
occlusion analysis diagnosed feature poverty as a candidate cause of the
single-peril model's underperformance). Both were rejected as fixes by the
real 20-seed sweeps on proprietary claims data (see README's "Verified findings"), but the
code paths themselves need the same test coverage as everything else, run
here against the bundled synthetic data.
"""

import unittest
from pathlib import Path

from src.pipeline import run_peril_evaluation

BASE_DIR = Path(__file__).resolve().parent.parent
SYN_DIR = BASE_DIR / "data" / "synthetic"
THEFT_PAID = str(SYN_DIR / "theft" / "paid_qtr_triangle.csv")
THEFT_OUT = str(SYN_DIR / "theft" / "outstanding_qtr_triangle.csv")
THEFT_COUNT = str(SYN_DIR / "theft" / "claims_count_qtr_triangle.csv")


class TestGRUVariant(unittest.TestCase):
    def test_runs_end_to_end_with_expected_param_count(self):
        result = run_peril_evaluation(
            THEFT_PAID, THEFT_OUT, peril_id=1, seed=42, epochs=3,
            checkpoint_path="/tmp/test_gru_smoke.pt", rnn_type="gru",
        )
        self.assertIn("lstm", result["error_pct"])
        self.assertGreater(result["n_active"], 0)
        # GRU has 3 gates per layer vs LSTM's 4: 3 * hidden * (input + hidden + 2).
        hidden, input_dim = 64, 2
        expected_gru_params = 3 * hidden * (input_dim + hidden + 2)
        head_params = (hidden * 32 + 32) + (32 * 1 + 1)
        expected = expected_gru_params + 2 * head_params
        self.assertEqual(expected, result["lstm_param_count"])

    def test_deterministic_across_two_runs(self):
        kwargs = dict(paid_path=THEFT_PAID, out_path=THEFT_OUT, peril_id=1,
                      seed=99, epochs=3, rnn_type="gru")
        r1 = run_peril_evaluation(**kwargs, checkpoint_path="/tmp/test_gru_det_a.pt")
        r2 = run_peril_evaluation(**kwargs, checkpoint_path="/tmp/test_gru_det_b.pt")
        self.assertEqual(r1["error_pct"]["lstm"], r2["error_pct"]["lstm"])


class TestClaimCountFeatureVariant(unittest.TestCase):
    def test_runs_end_to_end_with_expected_param_count(self):
        result = run_peril_evaluation(
            THEFT_PAID, THEFT_OUT, peril_id=1, seed=42, epochs=3,
            checkpoint_path="/tmp/test_counts_smoke.pt", count_path=THEFT_COUNT,
        )
        self.assertIn("lstm", result["error_pct"])
        self.assertGreater(result["n_active"], 0)
        # One extra input channel adds 4*hidden gate weights (LSTM, input_dim=3).
        hidden, input_dim = 64, 3
        expected_lstm_params = 4 * hidden * (input_dim + hidden + 2)
        head_params = (hidden * 32 + 32) + (32 * 1 + 1)
        expected = expected_lstm_params + 2 * head_params
        self.assertEqual(expected, result["lstm_param_count"])

    def test_deterministic_across_two_runs(self):
        kwargs = dict(paid_path=THEFT_PAID, out_path=THEFT_OUT, peril_id=1,
                      seed=99, epochs=3, count_path=THEFT_COUNT)
        r1 = run_peril_evaluation(**kwargs, checkpoint_path="/tmp/test_counts_det_a.pt")
        r2 = run_peril_evaluation(**kwargs, checkpoint_path="/tmp/test_counts_det_b.pt")
        self.assertEqual(r1["error_pct"]["lstm"], r2["error_pct"]["lstm"])


if __name__ == "__main__":
    unittest.main()
