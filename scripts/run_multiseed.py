"""
Multi-Seed Neural Variance Experiment.
Re-runs the DeepTriangle LSTM across the explicit 20-seed list in
config/seeds.yaml (classical benchmarks are deterministic and unaffected by
seed, so only the LSTM reserve varies run to run). Reports the genuine mean,
standard deviation, min/max and percentiles of the resulting reserve
distribution - no constant is asserted in advance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_peril_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peril", choices=["theft", "windscreen"], default="theft")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--use-counts", action="store_true",
                         help="Add incremental claim counts as a third LSTM input channel, "
                              "from <peril_dir>/claims_count_qtr_triangle.csv")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    seeds_path = base_dir / "config" / "seeds.yaml"
    with open(seeds_path, "r") as f:
        seeds = yaml.safe_load(f)["seeds"]

    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "synthetic"
    peril_dir = data_dir / args.peril
    paid_path, out_path = peril_dir / "paid_qtr_triangle.csv", peril_dir / "outstanding_qtr_triangle.csv"
    peril_id = 1 if args.peril == "theft" else 0
    count_path = None
    if args.use_counts:
        count_path = peril_dir / "claims_count_qtr_triangle.csv"
        if not count_path.exists():
            raise FileNotFoundError(f"--use-counts given but {count_path} does not exist")

    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint dir is peril- and architecture-specific: an earlier version of
    # this script used a single outputs/checkpoints/multiseed/seed_{s}.pt path
    # regardless of --peril, so a Windscreen sweep silently overwrote a prior
    # Theft sweep's checkpoints. Keyed by peril, rnn-type and feature set.
    feat_tag = "counts" if args.use_counts else "base"
    ckpt_dir = output_dir / "checkpoints" / "multiseed" / args.peril / args.rnn_type / feat_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(seeds)}-seed experiment on {args.peril} ({args.rnn_type}, {feat_tag})...")
    reserves, errors_pct = [], []
    actual_rm_k = None
    for s in seeds:
        res = run_peril_evaluation(
            str(paid_path), str(out_path), peril_id=peril_id, seed=s,
            epochs=args.epochs, checkpoint_path=str(ckpt_dir / f"seed_{s}.pt"),
            rnn_type=args.rnn_type,
            count_path=str(count_path) if count_path is not None else None,
        )
        reserves.append(res["reserves_rm_k"]["lstm"])
        errors_pct.append(res["error_pct"]["lstm"])
        actual_rm_k = res["reserves_rm_k"]["actual"]
        print(f"  seed {s:>5d}: reserve = RM {res['reserves_rm_k']['lstm']:,.0f}k ({res['error_pct']['lstm']:+.2f}%)")

    reserves = np.array(reserves)
    errors_pct = np.array(errors_pct)
    summary = {
        "peril": args.peril,
        "rnn_type": args.rnn_type,
        "use_counts": args.use_counts,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "actual_reserve_rm_k": actual_rm_k,
        "mean_reserve_rm_k": float(np.mean(reserves)),
        "std_reserve_rm_k": float(np.std(reserves, ddof=1)),
        "mean_error_pct": float(np.mean(errors_pct)),
        "std_error_pct": float(np.std(errors_pct, ddof=1)),
        "min_reserve_rm_k": float(np.min(reserves)),
        "max_reserve_rm_k": float(np.max(reserves)),
        "per_seed_reserve_rm_k": reserves.tolist(),
        "per_seed_error_pct": errors_pct.tolist(),
    }

    suffix = "" if args.rnn_type == "lstm" else f"_{args.rnn_type}"
    suffix += "_counts" if args.use_counts else ""
    out_file = output_dir / f"multiseed_{args.peril}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Mean = RM {summary['mean_reserve_rm_k']:,.0f}k ({summary['mean_error_pct']:+.2f}%), "
          f"Std Dev = RM {summary['std_reserve_rm_k']:,.0f}k ({summary['std_error_pct']:.2f} pp)")
    print(f"Wrote {out_file}.")


if __name__ == "__main__":
    main()
