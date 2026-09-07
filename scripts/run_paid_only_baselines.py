"""
Paid-only GBM and paid-only LSTM controls, completing RQ1's information-set x
model-class grid (named as a missing cell in Table 5.2's note, and in
Sections 1.3 and 8.1 three separate times).

Paid-only GBM: GBMReservingControl(paid_only=True) - drops BALOS_t from the
feature vector entirely (src/ml/gbm.py).

Paid-only LSTM: input_dim=1 (IncPaid history only, BALOS never seen as
input), keeping the same two-head architecture and dual-task loss as the
joint model so the two-headed 21,378-parameter configuration this module's
own docstring already anticipates is what actually gets trained. The
distinction from an "occluded" LSTM (Section 4.7) matters: occlusion trains
normally then neutralises BALOS only at inference, so the network's weights
still reflect having seen real BALOS during training. This model never sees
real BALOS at all, in training or rollout - the correct analogue to Mack
Paid CL never touching case reserves anywhere in its calibration.

Both are calibrated on k <= TRAIN_CUTOFF (matching the GBM's existing,
disclosed calibration window - see the Section 3.3 asymmetry note), evaluated
on the standard 504-cell holdout population, at the corrected RM 1,000
active-cell threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import MinMaxSequenceScaler, load_and_preprocess_triangles
from src.evaluate import compute_metrics
from src.ml.gbm import GBMReservingControl
from src.ml.lstm import DeepTriangleLSTM
from src.pipeline import (
    ACTIVE_CELL_THRESHOLD_RM, TRAIN_CUTOFF, VAL_END, HOLDOUT_END, _build_lstm_splits,
)
from src.seeding import set_deterministic_seed
from src.train import train_deeptriangle_model


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _run_paid_only_gbm(inc, balos, peril_id):
    gbm = GBMReservingControl(paid_only=True).fit(inc, balos, cutoff_cal_idx=TRAIN_CUTOFF)
    cum_paid = np.cumsum(inc, axis=1)
    num_cohorts, num_devs = inc.shape
    actual_sum = gbm_sum = 0.0
    actual_cells, gbm_cells = [], []
    for i in range(num_cohorts):
        last_dev = VAL_END - i
        if last_dev < 0 or last_dev >= num_devs - 1:
            continue
        curr_inc, curr_bal, curr_cum = inc[i, last_dev], balos[i, last_dev], cum_paid[i, last_dev]
        forecast = gbm.rollout(last_dev, curr_inc, curr_bal, curr_cum, num_devs - 1 - last_dev)
        for step_idx in range(len(forecast)):
            target_dev = last_dev + 1 + step_idx
            target_cal = i + target_dev
            if VAL_END < target_cal <= HOLDOUT_END and not np.isnan(inc[i, target_dev]):
                act_val = inc[i, target_dev]
                gbm_val = forecast[step_idx]
                actual_sum += act_val
                gbm_sum += gbm_val
                if act_val > ACTIVE_CELL_THRESHOLD_RM:
                    actual_cells.append(act_val)
                    gbm_cells.append(gbm_val)
    actual_arr, gbm_arr = np.array(actual_cells), np.array(gbm_cells)
    metrics = compute_metrics(gbm_arr, actual_arr)
    error_pct = float((gbm_sum - actual_sum) / actual_sum * 100.0) if actual_sum else 0.0
    return {
        "reserve_rm_k": float(gbm_sum) / 1000.0,
        "actual_rm_k": float(actual_sum) / 1000.0,
        "error_pct": error_pct,
        "metrics": metrics,
        "n_active": len(actual_cells),
    }


def _pad_paid_only(sequences, targets, scaler, max_len=65):
    n = len(sequences)
    X = np.zeros((n, max_len, 1), dtype=np.float32)
    Y = np.zeros((n, 2), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    peril_ids = np.zeros((n,), dtype=np.int64)
    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        seq_scaled = scaler.transform_feature_array(seq)  # 2-col: uses real BALOS only to get IncPaid's own min-max right
        seq_len = len(seq)
        X[i, :seq_len, 0] = seq_scaled[:, 0]  # BALOS column (index 1) dropped - never fed to the model
        mask[i, :seq_len] = 1.0
        Y[i, :] = scaler.transform_feature_array(tgt.reshape(1, 2)).squeeze(0)
    return {
        "X": torch.tensor(X), "Y": torch.tensor(Y),
        "mask": torch.tensor(mask), "peril_ids": torch.tensor(peril_ids),
    }


def _run_paid_only_lstm(inc, balos, peril_id, seed, epochs, ckpt_path):
    set_deterministic_seed(seed)
    train_seqs, train_tgts, train_meta, val_seqs, val_tgts, val_meta = _build_lstm_splits(inc, balos, peril_id=peril_id)
    # Scaler fit on the real 2-column data purely to get correct log1p/min-max
    # constants per channel (a scaling constant is not itself "information");
    # the BALOS column these constants imply is never handed to the model below.
    scaler = MinMaxSequenceScaler(use_log_transform=True).fit(train_seqs, train_tgts)
    model = DeepTriangleLSTM(input_dim=1, hidden_dim=64, dropout=0.2, output_activation="softplus")
    assert model.count_parameters() == 21378, f"unexpected paid-only LSTM param count: {model.count_parameters()}"

    train_tensors = _pad_paid_only(train_seqs, train_tgts, scaler)
    val_tensors = _pad_paid_only(val_seqs, val_tgts, scaler)
    train_deeptriangle_model(model, train_tensors, val_tensors, epochs=epochs,
                              learning_rate=1e-3, patience=25, checkpoint_path=ckpt_path)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    cum_paid = np.cumsum(inc, axis=1)
    num_cohorts, num_devs = inc.shape
    actual_sum = dt_sum = 0.0
    actual_cells, dt_cells = [], []
    with torch.no_grad():
        for i in range(num_cohorts):
            last_dev = VAL_END - i
            if last_dev < 0 or last_dev >= num_devs - 1:
                continue
            obs_raw = inc[i, : last_dev + 1].reshape(-1, 1)
            obs_scaled_2col = scaler.transform_feature_array(
                np.column_stack([inc[i, : last_dev + 1], balos[i, : last_dev + 1]])
            )
            current_seq = obs_scaled_2col[:, :1].copy()
            forecasts_scaled = []
            while len(current_seq) < num_devs:
                seq_len = len(current_seq)
                X = np.zeros((1, num_devs, 1), dtype=np.float32)
                mask = np.zeros((1, num_devs), dtype=np.float32)
                X[0, :seq_len, :] = current_seq
                mask[0, :seq_len] = 1.0
                pred_scaled = model(torch.tensor(X), torch.tensor(mask), torch.tensor([peril_id], dtype=torch.long)).numpy()[0]
                forecasts_scaled.append(pred_scaled)
                current_seq = np.vstack([current_seq, pred_scaled[:1]])  # feed only the IncPaid prediction back
            forecasts_scaled = np.array(forecasts_scaled)
            forecasts_unscaled = np.maximum(scaler.inverse_transform_targets(forecasts_scaled), 0.0)

            for step_idx in range(len(forecasts_unscaled)):
                target_dev = last_dev + 1 + step_idx
                target_cal = i + target_dev
                if VAL_END < target_cal <= HOLDOUT_END and not np.isnan(inc[i, target_dev]):
                    act_val = inc[i, target_dev]
                    dt_val = forecasts_unscaled[step_idx, 0]
                    actual_sum += act_val
                    dt_sum += dt_val
                    if act_val > ACTIVE_CELL_THRESHOLD_RM:
                        actual_cells.append(act_val)
                        dt_cells.append(dt_val)
    actual_arr, dt_arr = np.array(actual_cells), np.array(dt_cells)
    metrics = compute_metrics(dt_arr, actual_arr)
    error_pct = float((dt_sum - actual_sum) / actual_sum * 100.0) if actual_sum else 0.0
    return {
        "reserve_rm_k": float(dt_sum) / 1000.0,
        "actual_rm_k": float(actual_sum) / 1000.0,
        "error_pct": error_pct,
        "metrics": metrics,
        "n_active": len(actual_cells),
        "param_count": model.count_parameters(),
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
        paid_path = str(data_dir / peril_name / "paid_qtr_triangle.csv")
        out_path = str(data_dir / peril_name / "outstanding_qtr_triangle.csv")
        inc, balos, _, _ = load_and_preprocess_triangles(paid_path, out_path)

        print(f"=== {peril_name}: paid-only GBM ===")
        gbm_result = _run_paid_only_gbm(inc, balos, peril_id)
        print(f"  error={gbm_result['error_pct']:+.2f}%  n_active={gbm_result['n_active']}  R2={gbm_result['metrics']['r2']:.4f}")

        print(f"=== {peril_name}: paid-only LSTM (seed {args.seed}) ===")
        ckpt = f"outputs/checkpoints/paid_only_lstm_{peril_name}.pt"
        Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
        lstm_result = _run_paid_only_lstm(inc, balos, peril_id, args.seed, args.epochs, ckpt)
        print(f"  error={lstm_result['error_pct']:+.2f}%  n_active={lstm_result['n_active']}  "
              f"R2={lstm_result['metrics']['r2']:.4f}  params={lstm_result['param_count']}")

        out[peril_name] = {"gbm_paid_only": gbm_result, "lstm_paid_only": lstm_result}

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/paid_only_results.json", "w") as f:
        json.dump(_to_native(out), f, indent=2)
    print("\n[SUCCESS] Wrote outputs/paid_only_results.json")


if __name__ == "__main__":
    main()
