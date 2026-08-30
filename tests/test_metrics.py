"""Unit tests for src/metrics.py, the per-unit Freq x Sev = BC identity."""

import unittest

import numpy as np

from src.metrics import ActuarialMetrics


class TestPerUnitMetrics(unittest.TestCase):
    def test_frequency_times_severity_equals_burning_cost(self):
        result = ActuarialMetrics.compute_per_unit(
            ultimate_count=1401, ultimate_loss=83247.44 * 1401, exposure=10142700
        )
        self.assertAlmostEqual(result['frequency'] * result['severity'], result['burning_cost'], places=9)

    def test_vectorised_over_multiple_accident_years(self):
        counts = np.array([1401, 1120])
        losses = np.array([83247.44 * 1401, 62563.52 * 1120])
        exposure = np.array([10142700, 10460659])
        result = ActuarialMetrics.compute_per_unit(counts, losses, exposure)
        np.testing.assert_allclose(result['frequency'] * result['severity'], result['burning_cost'], rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
