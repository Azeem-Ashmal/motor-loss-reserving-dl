"""
Applies the Duan (1983) smearing correction (Section 4.1.3) to the single-
peril seed-42 LSTM baseline and reports the corrected reserve/error, isolating
retransformation bias from the seed-to-seed instability documented in
Chapter 6 - previously Limitation 5 stated this separation "cannot be done
without re-running the pipeline with a smearing correction". This does that
re-run.

psi is estimated from the trained model's own one-step-ahead training
residuals (src.evaluate.compute_duan_smearing_psi), then applied during the
autoregressive rollout's final retransformation
(predict_auto_regressive(..., smearing_psi=(psi_inc, psi_balos))).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import MinMaxSequenceScaler, load_and_preprocess_triangles, pad_and_tensorize
from src.evaluate import compute_duan_smearing_psi, compute_metrics, predict_auto_regressive
from src.ml.lstm import DeepTriangleLSTM
from src.pipeline import ACTIVE_CELL_THRESHOLD_RM, VAL_END, HOLDOUT_END, _build_lstm_splits
from src.seeding import set_deterministic_seed
from src.train import train_deeptriangle_model


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _evaluate(model, scaler, inc, balos, peril_id, smearing_psi=None):
    cum_paid = np.cumsum(inc, axis=1)
    num_cohorts, num_devs = inc.shape
    actual_sum = dt_sum = 0.0
    actual_cells, dt_cells = [], []
    for i in range(num_cohorts):
        last_dev = VAL_END - i
        if last_dev < 0 or last_dev >= num_devs - 1:
            continue
        obs_seq_raw = np.column_stack((inc[i, : last_dev + 1], balos[i, : last_dev + 1]))
        obs_seq_scaled = scaler.transform_feature_array(obs_seq_raw)
        forecast = predict_auto_regressive(model, scaler, obs_seq_scaled, peril_id=peril_id,
                                            max_total_len=num_devs, smearing_psi=smearing_psi)
        for step_idx in range(len(forecast)):
            target_dev = last_dev + 1 + step_idx
            target_cal = i + target_dev
            if VAL_END < target_cal <= HOLDOUT_END and not np.isnan(inc[i, target_dev]):
                act_val = inc[i, target_dev]
                dt_val = forecast[step_idx, 0]
                actual_sum += act_val
                dt_sum += dt_val
                if act_val > ACTIVE_CELL_THRESHOLD_RM:
                    actual_cells.append(act_val)
                    dt_cells.append(dt_val)
    actual_arr, dt_arr = np.array(actual_cells), np.array(dt_cells)
    metrics = compute_metrics(dt_arr, actual_arr)
    error_pct = float((dt_sum - actual_sum) / actual_sum * 100.0) if actual_sum else 0.0
    return {
        "reserve_rm_k": float(dt_sum) / 1000.0, "actual_rm_k": float(actual_sum) / 1000.0,
        "error_pct": error_pct, "metrics": metrics, "n_active": len(actual_cells),
    }


def _train_and_correct(paid_path, out_path, peril_id, seed, epochs, ckpt_path):
    set_deterministic_seed(seed)
    inc, balos, _, _ = load_and_preprocess_triangles(paid_path, out_path)
    train_s, train_t, train_m, val_s, val_t, val_m = _build_lstm_splits(inc, balos, peril_id=peril_id)
    scaler = MinMaxSequenceScaler(use_log_transform=True).fit(train_s, train_t)
    model = DeepTriangleLSTM(input_dim=2, hidden_dim=64, dropout=0.2, output_activation="softplus")
    train_tensors = pad_and_tensorize(train_s, train_t, train_m, scaler)
    val_tensors = pad_and_tensorize(val_s, val_t, val_m, scaler)
    train_deeptriangle_model(model, train_tensors, val_tensors, epochs=epochs,
                              learning_rate=1e-3, patience=25, checkpoint_path=ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    psi_inc, psi_balos = compute_duan_smearing_psi(model, scaler, train_tensors)

    naive = _evaluate(model, scaler, inc, balos, peril_id, smearing_psi=None)
    corrected = _evaluate(model, scaler, inc, balos, peril_id, smearing_psi=(psi_inc, psi_balos))
    return {
        "psi_inc": psi_inc, "psi_balos": psi_balos,
        "naive": naive, "smearing_corrected": corrected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = {}
    for peril_name, peril_id in (("theft", 1), ("windscreen", 0)):
        print(f"=== {peril_name}: seed {args.seed} ===")
        paid_path = str(data_dir / peril_name / "paid_qtr_triangle.csv")
        out_path = str(data_dir / peril_name / "outstanding_qtr_triangle.csv")
        ckpt = f"outputs/checkpoints/smearing_{peril_name}.pt"
        Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
        result = _train_and_correct(paid_path, out_path, peril_id, args.seed, args.epochs, ckpt)
        print(f"  psi_inc={result['psi_inc']:.4f}  psi_balos={result['psi_balos']:.4f}")
        print(f"  naive error:     {result['naive']['error_pct']:+.2f}%")
        print(f"  corrected error: {result['smearing_corrected']['error_pct']:+.2f}%")
        out[peril_name] = result

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/smearing_correction_results.json", "w") as f:
        json.dump(_to_native(out), f, indent=2)
    print("\n[SUCCESS] Wrote outputs/smearing_correction_results.json")


if __name__ == "__main__":
    main()
