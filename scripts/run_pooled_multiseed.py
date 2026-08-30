"""
20-seed variance sweep for the pooled multi-peril DeepTriangle LSTM, mirroring
run_multiseed.py's rigor but for the pooled architecture (src/pipeline.py's
run_pooled_lstm_evaluation). Reports both perils' error distributions from one
shared model per seed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.pipeline import run_pooled_lstm_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / "config" / "seeds.yaml") as f:
        seeds = yaml.safe_load(f)["seeds"]

    data_dir = Path(args.data_dir)
    theft_paid = data_dir / "theft" / "paid_qtr_triangle.csv"
    theft_out = data_dir / "theft" / "outstanding_qtr_triangle.csv"
    ws_paid = data_dir / "windscreen" / "paid_qtr_triangle.csv"
    ws_out = data_dir / "windscreen" / "outstanding_qtr_triangle.csv"

    theft_errors, ws_errors = [], []
    for seed in seeds:
        print(f"\n=== Pooled model, seed {seed} ===")
        r = run_pooled_lstm_evaluation(
            str(theft_paid), str(theft_out), str(ws_paid), str(ws_out),
            seed=seed, epochs=args.epochs,
            checkpoint_path=f"outputs/checkpoints/pooled_multiseed/seed_{seed}.pt",
        )
        theft_errors.append(r["theft"]["error_pct"])
        ws_errors.append(r["windscreen"]["error_pct"])
        print(f"seed {seed}: theft={r['theft']['error_pct']:.2f}%  windscreen={r['windscreen']['error_pct']:.2f}%")

    import numpy as np
    out = {
        "seeds": seeds,
        "theft_error_pct": theft_errors,
        "windscreen_error_pct": ws_errors,
        "theft_mean": float(np.mean(theft_errors)),
        "theft_std": float(np.std(theft_errors, ddof=1)),
        "windscreen_mean": float(np.mean(ws_errors)),
        "windscreen_std": float(np.std(ws_errors, ddof=1)),
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/pooled_multiseed_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[SUCCESS] Pooled multi-seed sweep complete.")
    print(f"Theft: mean={out['theft_mean']:.2f}%  std={out['theft_std']:.2f}pp")
    print(f"Windscreen: mean={out['windscreen_mean']:.2f}%  std={out['windscreen_std']:.2f}pp")


if __name__ == "__main__":
    main()
