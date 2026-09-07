"""
Auto-regressive forecasting and evaluation-metric utilities, shared by the
classical, GBM and LSTM models in the pipeline.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.data_pipeline import MinMaxSequenceScaler
from src.ml.lstm import DeepTriangleLSTM


def compute_duan_smearing_psi(
    model: DeepTriangleLSTM,
    scaler: MinMaxSequenceScaler,
    train_tensors: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Duan (1983) smearing factors (psi_inc, psi_balos), estimated from this
    model's own one-step-ahead training residuals in log1p space (Section
    4.1.3, Eq 4.9): psi = mean(exp(e_i)), e_i = true_log1p - pred_log1p.
    Uses the same teacher-forced training examples train_deeptriangle_model
    computes loss over (train_tensors["X"]/["Y"]/["mask"]/["peril_ids"]), not
    a held-out set, matching Duan's own definition of the smearing residual.
    """
    model.eval()
    with torch.no_grad():
        preds_scaled = model(
            train_tensors["X"].to(device), train_tensors["mask"].to(device), train_tensors["peril_ids"].to(device)
        ).cpu().numpy()
    targets_scaled = train_tensors["Y"].numpy()

    pred_log = scaler.unscale_to_log1p(preds_scaled)
    true_log = scaler.unscale_to_log1p(targets_scaled)
    residual = true_log - pred_log

    psi_inc = float(np.mean(np.exp(residual[..., 0])))
    psi_balos = float(np.mean(np.exp(residual[..., 1])))
    return psi_inc, psi_balos


def predict_auto_regressive(
    model: DeepTriangleLSTM,
    scaler: MinMaxSequenceScaler,
    initial_seq_scaled: np.ndarray,
    peril_id: int = 0,
    max_total_len: int = 65,
    device: str = "cpu",
    smearing_psi: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Recursively feed the model's own predictions at quarter j back into the
    input sequence for quarter j+1, until the sequence reaches max_total_len.
    Returns an array of shape (num_forecast_steps, 2) in RM (unscaled).

    If initial_seq_scaled has more than 2 columns (e.g. a claim-count channel
    appended alongside IncPaid/BALOS), the model still only predicts the first
    two - it has no head for the extra channel(s). Future rollout steps carry
    the extra channel(s) forward at their last-observed scaled value (a
    persistence assumption): this repo does not forecast its own exogenous
    inputs, and using true future values would leak holdout information.
    """
    model.eval()
    current_seq = initial_seq_scaled.copy()
    input_dim = current_seq.shape[1]
    n_extra = input_dim - 2
    last_extra = current_seq[-1, 2:].copy() if n_extra > 0 else None
    forecasts_scaled = []

    with torch.no_grad():
        while len(current_seq) < max_total_len:
            seq_len = len(current_seq)
            X = np.zeros((1, max_total_len, input_dim), dtype=np.float32)
            mask = np.zeros((1, max_total_len), dtype=np.float32)
            X[0, :seq_len, :] = current_seq
            mask[0, :seq_len] = 1.0

            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            mask_t = torch.tensor(mask, dtype=torch.float32).to(device)
            peril_t = torch.tensor([peril_id], dtype=torch.long).to(device)

            pred_scaled = model(X_t, mask_t, peril_t).cpu().numpy()[0]
            forecasts_scaled.append(pred_scaled)
            next_row = np.concatenate([pred_scaled, last_extra]) if n_extra > 0 else pred_scaled
            current_seq = np.vstack([current_seq, next_row])

    if len(forecasts_scaled) == 0:
        return np.empty((0, 2))

    forecasts_scaled = np.array(forecasts_scaled)
    forecasts_unscaled = scaler.inverse_transform_targets(forecasts_scaled, smearing_psi=smearing_psi)
    forecasts_unscaled = np.maximum(forecasts_unscaled, 0.0)
    return forecasts_unscaled


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, MAPE and R^2 over paired active-cell arrays, NaN-safe."""
    valid_mask = ~np.isnan(predictions) & ~np.isnan(actuals)
    preds_clean = predictions[valid_mask]
    acts_clean = actuals[valid_mask]

    if len(preds_clean) == 0:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0}

    errors = preds_clean - acts_clean
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    denom = np.maximum(np.abs(acts_clean), 1.0)
    mape = float(np.mean(np.abs(errors) / denom) * 100.0)

    ss_res = np.sum(errors**2)
    ss_tot = np.sum((acts_clean - np.mean(acts_clean)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
