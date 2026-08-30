"""
Smoke and determinism tests for the pooled multi-peril LSTM pipeline
(src/pipeline.py's run_pooled_lstm_evaluation). Uses the bundled synthetic
data and a small epoch budget so this runs in CI-reasonable time; the
dissertation's own reported numbers come from a full 300-epoch run on real
real claims data, not from this test.
"""

import unittest
from pathlib import Path

from src.pipeline import run_pooled_lstm_evaluation

BASE_DIR = Path(__file__).resolve().parent.parent
SYN_DIR = BASE_DIR / "data" / "synthetic"


class TestPooledPipeline(unittest.TestCase):
    def test_runs_end_to_end_and_returns_expected_shape(self):
        result = run_pooled_lstm_evaluation(
            str(SYN_DIR / "theft" / "paid_qtr_triangle.csv"),
            str(SYN_DIR / "theft" / "outstanding_qtr_triangle.csv"),
            str(SYN_DIR / "windscreen" / "paid_qtr_triangle.csv"),
            str(SYN_DIR / "windscreen" / "outstanding_qtr_triangle.csv"),
            seed=42, epochs=3, checkpoint_path="/tmp/test_pooled_smoke.pt",
        )
        for peril in ("theft", "windscreen"):
            self.assertIn(peril, result)
            self.assertIn("error_pct", result[peril])
            self.assertIn("reserve_rm_k", result[peril])
            self.assertGreater(result[peril]["n_active"], 0)
        # Verified architecture fact (Table 2): pooled = 23,698 parameters.
        self.assertEqual(result["param_count"], 23698)

    def test_deterministic_across_two_runs(self):
        kwargs = dict(
            theft_paid_path=str(SYN_DIR / "theft" / "paid_qtr_triangle.csv"),
            theft_out_path=str(SYN_DIR / "theft" / "outstanding_qtr_triangle.csv"),
            ws_paid_path=str(SYN_DIR / "windscreen" / "paid_qtr_triangle.csv"),
            ws_out_path=str(SYN_DIR / "windscreen" / "outstanding_qtr_triangle.csv"),
            seed=123, epochs=3,
        )
        r1 = run_pooled_lstm_evaluation(**kwargs, checkpoint_path="/tmp/test_pooled_det_a.pt")
        r2 = run_pooled_lstm_evaluation(**kwargs, checkpoint_path="/tmp/test_pooled_det_b.pt")
        self.assertEqual(r1["theft"]["error_pct"], r2["theft"]["error_pct"])
        self.assertEqual(r1["windscreen"]["error_pct"], r2["windscreen"]["error_pct"])


if __name__ == "__main__":
    unittest.main()
