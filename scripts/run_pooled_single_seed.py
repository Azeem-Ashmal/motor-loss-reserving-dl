"""
Single-seed pooled multi-peril evaluation, matching run_all.py's role for the
single-peril baseline. Writes outputs/pooled_results.json, the seed-42 figure
Table 5.6's pooled row and the occlusion baseline in Section 4.7 are drawn
from - previously produced by an ad-hoc, unsaved call to
run_pooled_lstm_evaluation, in violation of this project's own
every-number-traceable-to-code standard, and predating the active-cell
threshold fix (commit d45aabf).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pooled_lstm_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    theft_paid = data_dir / "theft" / "paid_qtr_triangle.csv"
    theft_out = data_dir / "theft" / "outstanding_qtr_triangle.csv"
    ws_paid = data_dir / "windscreen" / "paid_qtr_triangle.csv"
    ws_out = data_dir / "windscreen" / "outstanding_qtr_triangle.csv"

    result = run_pooled_lstm_evaluation(
        str(theft_paid), str(theft_out), str(ws_paid), str(ws_out),
        seed=args.seed, epochs=args.epochs,
        checkpoint_path=f"outputs/checkpoints/pooled_single_seed_{args.seed}.pt",
    )

    print(f"Theft:      error={result['theft']['error_pct']:+.2f}%  "
          f"n_active={result['theft']['n_active']}  R2={result['theft']['metrics']['r2']:.4f}")
    print(f"Windscreen: error={result['windscreen']['error_pct']:+.2f}%  "
          f"n_active={result['windscreen']['n_active']}  R2={result['windscreen']['metrics']['r2']:.4f}")

    out = {
        "theft": result["theft"],
        "windscreen": result["windscreen"],
        "param_count": result["param_count"],
        "seed": result["seed"],
    }
    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / "pooled_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SUCCESS] Wrote {out_path}")


if __name__ == "__main__":
    main()
