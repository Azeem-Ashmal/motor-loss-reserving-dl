"""
Automated Actuarial Reconciliation Test Suite.
Asserts internal mathematical identities against outputs/results.json, which is
itself produced by scripts/run_all.py calling src/ directly - nothing here is
checked against a hand-typed constant.

Run scripts/run_all.py before this suite (see README quickstart).
"""

import json
import os
import unittest


class TestActuarialReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.base_dir = base_dir
        results_path = os.path.join(base_dir, "outputs", "results.json")
        if not os.path.exists(results_path):
            raise unittest.SkipTest(
                "outputs/results.json not found - run `python scripts/run_all.py` first."
            )
        with open(results_path, "r") as f:
            cls.results = json.load(f)

    def test_reserve_error_identity(self):
        """error_pct must equal (reserve - actual) / actual * 100 for every model."""
        for peril, data in self.results.items():
            actual = data["reserves_rm_k"]["actual"]
            for model, reserve in data["reserves_rm_k"].items():
                if model == "actual":
                    continue
                expected_pct = (reserve - actual) / actual * 100.0
                self.assertAlmostEqual(
                    expected_pct, data["error_pct"][model], places=1,
                    msg=f"{peril}/{model}: error_pct does not match reserve identity",
                )

    def test_dev_error_decomposition_sums_to_total(self):
        """Per-development-quarter Mack error must sum to the total Mack error."""
        for peril, data in self.results.items():
            dev_errors = data["dev_error_mack_rm"]
            total_from_decomp = sum(dev_errors.values()) / 1000.0
            actual = data["reserves_rm_k"]["actual"]
            mack = data["reserves_rm_k"]["mack"]
            self.assertAlmostEqual(total_from_decomp, mack - actual, delta=max(1.0, abs(mack - actual) * 0.01))

    def test_ay_error_decomposition_sums_to_total(self):
        """Per-accident-year Mack error must sum to the total Mack error."""
        for peril, data in self.results.items():
            ay_errors = data["ay_error_mack_rm"]
            total_from_decomp = sum(ay_errors.values()) / 1000.0
            actual = data["reserves_rm_k"]["actual"]
            mack = data["reserves_rm_k"]["mack"]
            self.assertAlmostEqual(total_from_decomp, mack - actual, delta=max(1.0, abs(mack - actual) * 0.01))

    def test_lstm_param_count_matches_architecture_formula(self):
        """
        4 * hidden * (input + hidden + 1) counts the LSTM gates; each dense head
        is Linear(hidden,32)+Linear(32,1) = hidden*32+32 + 32+1.
        """
        hidden = 64
        for peril, data in self.results.items():
            input_dim = 2  # IncPaid, BALOS
            lstm_params = 4 * hidden * (input_dim + hidden + 2)  # PyTorch LSTM has separate input-hidden and hidden-hidden biases
            head_params = (hidden * 32 + 32) + (32 * 1 + 1)
            expected = lstm_params + 2 * head_params
            self.assertEqual(expected, data["lstm_param_count"])

    def test_mack_full_runoff_se_is_sane(self):
        """Mack's full run-off SE (via chainladder) must be positive and small
        relative to the ultimate it describes - a sign the calculation ran on
        the right scale, not a mis-parsed triangle."""
        for peril, data in self.results.items():
            se_block = data.get("mack_se_full_runoff")
            if not se_block:
                continue
            self.assertGreater(se_block["se_aggregate_full_runoff"], 0)
            self.assertGreater(se_block["ultimate_sum"], 0)
            cv = se_block["se_aggregate_full_runoff"] / se_block["ultimate_sum"]
            self.assertLess(cv, 0.5, "SE exceeding 50% of ultimate suggests a unit or scope mismatch")

    def test_env_block_matches_lockfile(self):
        import re

        lockfile_path = os.path.join(self.base_dir, "requirements-lock.txt")
        env_block_path = os.path.join(self.base_dir, "env_block.tex")
        self.assertTrue(os.path.exists(lockfile_path), "requirements-lock.txt missing!")
        self.assertTrue(os.path.exists(env_block_path), "env_block.tex missing!")

        with open(lockfile_path, "r", encoding="utf-8") as f:
            lock_content = f.read()
        with open(env_block_path, "r", encoding="utf-8") as f:
            env_content = f.read()

        lock_pkgs = {}
        for line in lock_content.splitlines():
            line = line.strip()
            if line.startswith("Python "):
                lock_pkgs["python"] = line.split("Python ")[1]
            elif "==" in line:
                k, v = line.split("==", 1)
                lock_pkgs[k.lower()] = v

        label_to_pkg = {"python": "python", "pytorch": "torch", "scikit-learn": "scikit-learn",
                         "numpy": "numpy", "pandas": "pandas"}
        env_matches = re.findall(r"\\textbf\{(.*?)\}:\s*\\texttt\{(.*?)\}", env_content)
        env_pkgs = {}
        for label, ver in env_matches:
            key = label_to_pkg.get(label.lower())
            if key:
                env_pkgs[key] = ver.replace("\\_", "_")

        for pkg_key in ["python", "torch", "scikit-learn", "numpy", "pandas"]:
            self.assertIn(pkg_key, lock_pkgs, f"Package {pkg_key} missing in lockfile!")
            self.assertIn(pkg_key, env_pkgs, f"Package {pkg_key} missing in env_block.tex!")
            self.assertEqual(env_pkgs[pkg_key], lock_pkgs[pkg_key],
                              f"Version mismatch for {pkg_key}: {env_pkgs[pkg_key]} vs {lock_pkgs[pkg_key]}!")


if __name__ == "__main__":
    unittest.main()
