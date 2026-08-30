"""
Per-Unit Actuarial Metrics Module.
Frequency, severity, and burning cost from ultimate claim counts, ultimate
losses, and exposure - the identities behind Table 5 (Freq x Sev = BC). Not
yet wired into scripts/run_all.py, which currently only reports triangle-level
reserve figures; per-unit metrics need claim-count triangles and an exposure
series as additional inputs (see data/README.md's optional exposure_qtr.csv).
"""

import numpy as np


class ActuarialMetrics:
    @staticmethod
    def compute_per_unit(ultimate_count, ultimate_loss, exposure):
        """
        ultimate_count, ultimate_loss, exposure: scalars or same-shaped arrays,
        one entry per accident period.
        Returns frequency (claims/exposure), severity (loss/claim), and
        burning cost (loss/exposure). By construction, frequency * severity
        always equals burning cost - see tests/test_metrics.py.
        """
        ultimate_count = np.asarray(ultimate_count, dtype=float)
        ultimate_loss = np.asarray(ultimate_loss, dtype=float)
        exposure = np.asarray(exposure, dtype=float)

        freq = ultimate_count / exposure
        sev = ultimate_loss / ultimate_count
        bc = ultimate_loss / exposure
        return {'frequency': freq, 'severity': sev, 'burning_cost': bc}
