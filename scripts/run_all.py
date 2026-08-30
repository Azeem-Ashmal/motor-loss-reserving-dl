"""
Master Pipeline Script.
Runs the full reserving evaluation (Mack, ICL, Munich, GBM, DeepTriangle LSTM,
seed 42) for both perils and writes outputs/results.json. Every number in that
file is computed here from src/ - nothing is hand-typed.

Data source: by default reads data/synthetic (safe to run and to publish). To
reproduce the dissertation's numbers on the real underlying triangles, pass --data-dir
pointing at a local (never-committed) copy of the Theft/Windscreen triangles,
laid out as <data-dir>/theft/{paid,outstanding}_qtr_triangle.csv and
<data-dir>/windscreen/{paid,outstanding}_qtr_triangle.csv.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_peril_evaluation
from src.seeding import set_deterministic_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None,
                         help="Directory containing theft/ and windscreen/ subfolders "
                              "with paid_qtr_triangle.csv and outstanding_qtr_triangle.csv. "
                              "Defaults to the bundled synthetic data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = base_dir / "data" / "synthetic"

    perils = {
        "theft": {"peril_id": 1, "dirname": "theft"},
        "windscreen": {"peril_id": 0, "dirname": "windscreen"},
    }

    results = {}
    for peril_name, cfg in perils.items():
        peril_dir = data_dir / cfg["dirname"]
        paid_path = peril_dir / "paid_qtr_triangle.csv"
        out_path = peril_dir / "outstanding_qtr_triangle.csv"
        if not paid_path.exists() or not out_path.exists():
            print(f"[SKIP] {peril_name}: expected triangles not found under {peril_dir}")
            continue

        exposure_path = peril_dir / "exposure_qtr.csv"
        print(f"\n=== Running {peril_name} (seed {args.seed}) ===")
        res = run_peril_evaluation(
            str(paid_path), str(out_path), peril_id=cfg["peril_id"], seed=args.seed,
            epochs=args.epochs, checkpoint_path=str(output_dir / "checkpoints" / f"{peril_name}_lstm.pt"),
            exposure_path=str(exposure_path) if exposure_path.exists() else None,
        )
        results[peril_name] = res

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n[SUCCESS] Master pipeline executed. Wrote {results_path}.")


if __name__ == "__main__":
    main()
