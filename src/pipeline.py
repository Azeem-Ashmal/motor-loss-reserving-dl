"""
Single-peril reserving evaluation pipeline.
Ties together data loading, the classical benchmarks (Mack, ICL, Munich, BF),
the GBM control, and the DeepTriangle LSTM into one reproducible evaluation on
the holdout window (calendar quarters TRAIN_CUTOFF+1..HOLDOUT_END).

Every number this function returns is computed here, not looked up from a
constants table - this is what makes outputs/results.json traceable to code
per the reproducibility commitment in the dissertation's Chapter 8.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from src.classical.bf import BornhuetterFerguson
from src.classical.icl import IncurredChainLadder
from src.classical.mack import MackPaidChainLadder
from src.classical.munich import MunichChainLadder
from src.data_pipeline import MinMaxSequenceScaler, load_and_preprocess_triangles, pad_and_tensorize
from src.evaluate import compute_metrics, predict_auto_regressive
from src.ml.gbm import GBMReservingControl
from src.ml.lstm import DeepTriangleLSTM
from src.seeding import set_deterministic_seed
from src.train import train_deeptriangle_model

# Materiality threshold for the "active cell" holdout subset (RMSE/MAE/R^2
# are computed only on cells whose realised incremental payment exceeds
# this), matching the RM 1,000 threshold stated in the dissertation
# (Section 3.3). A prior version of this file hardcoded 1.0 (RM 1) here,
# which barely filtered anything and did not match the documented rule.
ACTIVE_CELL_THRESHOLD_RM = 1000.0


def _load_dataset_config() -> Dict[str, int]:
    """
    Single source of truth for split cutoffs: config/default.yaml. Falls back
    to the study's stated design (train k<=47, val 48-55, holdout 56-64) if the
    file is unavailable, e.g. when this module is imported as a standalone
    library outside the repository layout.
    """
    import yaml
    from pathlib import Path

    defaults = {"train_cutoff": 47, "val_cutoff": 55, "holdout_end": 64,
                "bf_mature_cohort_start": 24, "bf_mature_cohort_end": 36}
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    if not config_path.exists():
        return defaults
    with open(config_path) as f:
        cfg = yaml.safe_load(f).get("dataset", {})
    return {**defaults, **cfg}


_CFG = _load_dataset_config()
TRAIN_CUTOFF = _CFG["train_cutoff"]
VAL_END = _CFG["val_cutoff"]
HOLDOUT_END = _CFG["holdout_end"]
BF_MATURE_COHORT_RANGE = range(_CFG["bf_mature_cohort_start"], _CFG["bf_mature_cohort_end"])


def _build_lstm_splits(inc, balos, peril_id, extra_arrays=None):
    """
    extra_arrays: optional list of 2D matrices, same shape as inc, appended as
    further input-only feature columns (e.g. incremental claim counts). They
    are never targets - only IncPaid/BALOS are predicted.
    """
    extra_arrays = extra_arrays or []
    num_cohorts, num_devs = inc.shape
    train_seqs, train_tgts, train_meta = [], [], []
    val_seqs, val_tgts, val_meta = [], [], []
    for i in range(num_cohorts):
        for t in range(num_devs - 1):
            k = i + t + 1
            if k >= num_cohorts:
                continue
            if np.isnan(inc[i, t]) or np.isnan(balos[i, t]) or np.isnan(inc[i, t + 1]) or np.isnan(balos[i, t + 1]):
                continue
            if any(np.isnan(ex[i, t]) for ex in extra_arrays):
                continue
            cols = [inc[i, : t + 1], balos[i, : t + 1]] + [ex[i, : t + 1] for ex in extra_arrays]
            seq = np.column_stack(cols)
            tgt = np.array([inc[i, t + 1], balos[i, t + 1]])
            meta = {"cohort_idx": i, "dev_idx": t, "target_dev_idx": t + 1, "target_cal_idx": k, "peril_id": peril_id}
            if k <= TRAIN_CUTOFF:
                train_seqs.append(seq); train_tgts.append(tgt); train_meta.append(meta)
            elif TRAIN_CUTOFF < k <= VAL_END:
                val_seqs.append(seq); val_tgts.append(tgt); val_meta.append(meta)
    return train_seqs, train_tgts, train_meta, val_seqs, val_tgts, val_meta


def _load_incremental_count(count_csv_path: str) -> np.ndarray:
    """Loads a cumulative claim-count triangle (same shape/convention as the
    paid/outstanding triangles) and derives incrementals the same way."""
    import pandas as pd

    df = pd.read_csv(count_csv_path)
    dev_cols = [c for c in df.columns if c.startswith("Dev_")]
    cum = df[dev_cols].values.astype(np.float64)
    inc = np.zeros_like(cum)
    inc[:, 0] = cum[:, 0]
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]
    return inc


def _load_exposure(exposure_path: str, num_cohorts: int) -> np.ndarray:
    """Reads a two-column Cohort,Exposure CSV aligned to the triangle's rows."""
    import pandas as pd

    df = pd.read_csv(exposure_path)
    if len(df) != num_cohorts:
        raise ValueError(f"Exposure file has {len(df)} rows, expected {num_cohorts} to match the triangle")
    return df["Exposure"].values.astype(float)


def run_peril_evaluation(
    paid_path: str,
    out_path: str,
    peril_id: int = 1,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 25,
    use_pooled_embedding: bool = False,
    checkpoint_path: Optional[str] = None,
    train_lstm: bool = True,
    exposure_path: Optional[str] = None,
    bf_mature_cohorts: Optional[range] = None,
    rnn_type: str = "lstm",
    count_path: Optional[str] = None,
) -> Dict[str, Any]:
    set_deterministic_seed(seed)

    inc, balos, cohorts, dev_cols = load_and_preprocess_triangles(paid_path, out_path)
    cum_paid = np.cumsum(inc, axis=1)
    num_cohorts, num_devs = inc.shape

    # --- Classical benchmarks, calibrated on k <= VAL_END only ---
    mack = MackPaidChainLadder(cum_paid, cutoff_cal_idx=VAL_END)
    f_P, C_full = mack.fit_predict()
    mack_se = mack.mack_se_full_runoff()

    icl = IncurredChainLadder(cum_paid, balos, cutoff_cal_idx=VAL_END)
    f_I, I_full, icl_paid_allocation = icl.fit_predict(f_P)

    mcl = MunichChainLadder(cum_paid, balos, cutoff_cal_idx=VAL_END)
    mcl_res = mcl.fit_predict()

    mack_inc_full = np.zeros_like(C_full)
    mack_inc_full[:, 0] = C_full[:, 0]
    mack_inc_full[:, 1:] = C_full[:, 1:] - C_full[:, :-1]

    # --- GBM control, trained on k <= TRAIN_CUTOFF only ---
    gbm = GBMReservingControl().fit(inc, balos, cutoff_cal_idx=TRAIN_CUTOFF)

    # --- Bornhuetter-Ferguson, only if an exposure series is supplied ---
    bf_model = None
    exposure = None
    if exposure_path is not None:
        exposure = _load_exposure(exposure_path, num_cohorts)
        mature_range = bf_mature_cohorts if bf_mature_cohorts is not None else BF_MATURE_COHORT_RANGE
        prior_bc = BornhuetterFerguson.compute_prior_burning_cost(cum_paid, exposure, mature_range)
        bf_model = BornhuetterFerguson(prior_bc)

    # --- LSTM ---
    count_inc = _load_incremental_count(count_path) if count_path is not None else None
    extra_arrays = [count_inc] if count_inc is not None else []
    input_dim = 2 + len(extra_arrays)
    train_seqs, train_tgts, train_meta, val_seqs, val_tgts, val_meta = _build_lstm_splits(
        inc, balos, peril_id, extra_arrays=extra_arrays,
    )
    scaler = MinMaxSequenceScaler(use_log_transform=True, n_extra_features=len(extra_arrays)).fit(train_seqs, train_tgts)
    dt_model = DeepTriangleLSTM(
        input_dim=input_dim, hidden_dim=64, dropout=0.2, output_activation="softplus",
        use_peril_embedding=use_pooled_embedding, num_perils=2, embed_dim=8,
        rnn_type=rnn_type,
    )
    param_count = dt_model.count_parameters()

    if train_lstm:
        train_tensors = pad_and_tensorize(train_seqs, train_tgts, train_meta, scaler)
        val_tensors = pad_and_tensorize(val_seqs, val_tgts, val_meta, scaler)
        ckpt = checkpoint_path or "outputs/_tmp_lstm.pt"
        train_history = train_deeptriangle_model(
            dt_model, train_tensors, val_tensors, epochs=epochs,
            learning_rate=1e-3, patience=patience, checkpoint_path=ckpt,
        )
        dt_model.load_state_dict(torch.load(ckpt))
    dt_model.eval()
    if not train_lstm:
        train_history = None

    # --- Roll every cohort forward to the holdout window, cell by cell ---
    actual_sum = mack_sum = icl_sum = mcl_sum = gbm_sum = dt_sum = 0.0
    actual_cells, mack_cells, icl_cells, mcl_cells, gbm_cells, dt_cells = [], [], [], [], [], []
    active_cell_dev, active_cell_cal = [], []
    all_actual, all_mack, all_icl, all_mcl, all_gbm, all_dt, all_cal, all_dev = [], [], [], [], [], [], [], []
    dev_error_mack: Dict[int, float] = {}
    ay_error_mack: Dict[int, float] = {}
    last_dev_by_cohort = [None] * num_cohorts

    for i in range(num_cohorts):
        last_dev = VAL_END - i
        if last_dev < 0 or last_dev >= num_devs - 1:
            continue
        last_dev_by_cohort[i] = last_dev
        obs_cols = [inc[i, : last_dev + 1], balos[i, : last_dev + 1]] + [ex[i, : last_dev + 1] for ex in extra_arrays]
        obs_seq_raw = np.column_stack(obs_cols)
        obs_seq_scaled = scaler.transform_feature_array(obs_seq_raw)
        dt_forecast = predict_auto_regressive(dt_model, scaler, obs_seq_scaled, peril_id=peril_id, max_total_len=num_devs)

        curr_inc, curr_bal, curr_cum = inc[i, last_dev], balos[i, last_dev], cum_paid[i, last_dev]
        gbm_forecast = gbm.rollout(last_dev, curr_inc, curr_bal, curr_cum, num_devs - 1 - last_dev)

        for step_idx in range(len(dt_forecast)):
            target_dev = last_dev + 1 + step_idx
            target_cal = i + target_dev
            if VAL_END < target_cal <= HOLDOUT_END and not np.isnan(inc[i, target_dev]):
                act_val = inc[i, target_dev]
                mack_val = mack_inc_full[i, target_dev]
                icl_val = icl_paid_allocation[i, target_dev]
                mcl_val = mcl_res["cl_inc_paid"][i, target_dev]
                gbm_val = gbm_forecast[step_idx]
                dt_val = dt_forecast[step_idx, 0]

                actual_sum += act_val; mack_sum += mack_val; icl_sum += icl_val
                mcl_sum += mcl_val; gbm_sum += gbm_val; dt_sum += dt_val
                dev_error_mack[target_dev] = dev_error_mack.get(target_dev, 0.0) + float(mack_val - act_val)
                ay_error_mack[i] = ay_error_mack.get(i, 0.0) + float(mack_val - act_val)
                all_actual.append(float(act_val)); all_mack.append(float(mack_val)); all_icl.append(float(icl_val))
                all_mcl.append(float(mcl_val)); all_gbm.append(float(gbm_val)); all_dt.append(float(dt_val))
                all_cal.append(target_cal); all_dev.append(target_dev)

                if act_val > ACTIVE_CELL_THRESHOLD_RM:
                    actual_cells.append(act_val); mack_cells.append(mack_val)
                    icl_cells.append(icl_val); mcl_cells.append(mcl_val)
                    gbm_cells.append(gbm_val); dt_cells.append(dt_val)
                    active_cell_dev.append(target_dev); active_cell_cal.append(target_cal)

    bf_sum = None
    if bf_model is not None:
        bf_sum = bf_model.allocate_holdout_reserve(
            exposure, f_P, cutoff_cal_idx=VAL_END, holdout_end_idx=HOLDOUT_END,
            last_dev_by_cohort=last_dev_by_cohort,
        )

    actual_arr = np.array(actual_cells)
    metrics = {
        "mack": compute_metrics(np.array(mack_cells), actual_arr),
        "icl": compute_metrics(np.array(icl_cells), actual_arr),
        "munich": compute_metrics(np.array(mcl_cells), actual_arr),
        "gbm": compute_metrics(np.array(gbm_cells), actual_arr),
        "lstm": compute_metrics(np.array(dt_cells), actual_arr),
    }

    def pct(x):
        return float((x - actual_sum) / actual_sum * 100.0) if actual_sum else 0.0

    reserves_rm_k = {
        "actual": float(actual_sum) / 1000.0,
        "mack": float(mack_sum) / 1000.0,
        "icl": float(icl_sum) / 1000.0,
        "munich": float(mcl_sum) / 1000.0,
        "gbm": float(gbm_sum) / 1000.0,
        "lstm": float(dt_sum) / 1000.0,
    }
    if bf_sum is not None:
        reserves_rm_k["bf"] = float(bf_sum) / 1000.0
    error_pct = {k: pct(v * 1000.0) for k, v in reserves_rm_k.items() if k != "actual"}

    result = {
        "n_active": len(actual_cells),
        "reserves_rm_k": reserves_rm_k,
        "error_pct": error_pct,
        "metrics": metrics,
        "mack_link_ratios": f_P.tolist(),
        "icl_link_ratios": f_I.tolist(),
        "mack_se_full_runoff": mack_se,
        "dev_error_mack_rm": dev_error_mack,
        "ay_error_mack_rm": ay_error_mack,
        "lstm_param_count": param_count,
        "seed": seed,
        "cells": {
            "actual": actual_cells, "mack": mack_cells, "icl": icl_cells,
            "munich": mcl_cells, "gbm": gbm_cells, "lstm": [float(x) for x in dt_cells],
            "dev_idx": active_cell_dev, "cal_idx": active_cell_cal,
        },
        "cells_all": {
            "actual": all_actual, "mack": all_mack, "icl": all_icl,
            "munich": all_mcl, "gbm": all_gbm, "lstm": all_dt, "cal_idx": all_cal, "dev_idx": all_dev,
        },
        "lstm_train_history": train_history,
    }
    if bf_model is not None:
        result["bf_prior_burning_cost"] = bf_model.prior_bc
    return result


def run_pooled_lstm_evaluation(
    theft_paid_path: str,
    theft_out_path: str,
    ws_paid_path: str,
    ws_out_path: str,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 25,
    checkpoint_path: Optional[str] = None,
    train_lstm: bool = True,
) -> Dict[str, Any]:
    """
    Trains a single DeepTriangle LSTM on both perils simultaneously, with an
    8-dimensional peril embedding distinguishing Theft (peril_id=1) from
    Windscreen (peril_id=0) at every timestep, then evaluates holdout
    performance separately per peril using that one shared model.

    This is the "pooled multi-peril" architecture named in the original
    execution plan (Archived/Thesis_Deliverables_Old/topic4_execution_plan.md,
    Structural Risk 1: "Implement an Embedding Layer to pool multiple lines of
    business simultaneously... whilst utilising categorical embeddings to
    preserve peril-specific structural nuances") as the mitigation for sparse
    single-peril training data (1,128 training cells for Theft alone). It was
    scaffolded in DeepTriangleLSTM (use_peril_embedding) from early on but
    never actually run end-to-end until this function existed.

    Returns per-peril reserve/error/metric figures in the same shape as
    run_peril_evaluation's LSTM entries, plus the combined parameter count.
    """
    set_deterministic_seed(seed)

    theft_inc, theft_balos, _, _ = load_and_preprocess_triangles(theft_paid_path, theft_out_path)
    ws_inc, ws_balos, _, _ = load_and_preprocess_triangles(ws_paid_path, ws_out_path)

    # peril_id convention matches run_all.py: theft=1, windscreen=0
    t_train_s, t_train_t, t_train_m, t_val_s, t_val_t, t_val_m = _build_lstm_splits(theft_inc, theft_balos, peril_id=1)
    w_train_s, w_train_t, w_train_m, w_val_s, w_val_t, w_val_m = _build_lstm_splits(ws_inc, ws_balos, peril_id=0)

    train_seqs = t_train_s + w_train_s
    train_tgts = t_train_t + w_train_t
    train_meta = t_train_m + w_train_m
    val_seqs = t_val_s + w_val_s
    val_tgts = t_val_t + w_val_t
    val_meta = t_val_m + w_val_m

    # One scaler fit across both perils' training data, consistent with the
    # single-peril pipeline's own scaler being fit only on that peril's
    # training cells - here "training data" means the pooled set.
    scaler = MinMaxSequenceScaler(use_log_transform=True).fit(train_seqs, train_tgts)

    dt_model = DeepTriangleLSTM(
        input_dim=2, hidden_dim=64, dropout=0.2, output_activation="softplus",
        use_peril_embedding=True, num_perils=2, embed_dim=8,
    )
    param_count = dt_model.count_parameters()

    if train_lstm:
        train_tensors = pad_and_tensorize(train_seqs, train_tgts, train_meta, scaler)
        val_tensors = pad_and_tensorize(val_seqs, val_tgts, val_meta, scaler)
        ckpt = checkpoint_path or "outputs/_tmp_pooled_lstm.pt"
        train_history = train_deeptriangle_model(
            dt_model, train_tensors, val_tensors, epochs=epochs,
            learning_rate=1e-3, patience=patience, checkpoint_path=ckpt,
        )
        dt_model.load_state_dict(torch.load(ckpt))
    else:
        train_history = None
    dt_model.eval()

    def _evaluate_peril(inc, balos, peril_id):
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
            dt_forecast = predict_auto_regressive(dt_model, scaler, obs_seq_scaled, peril_id=peril_id, max_total_len=num_devs)
            for step_idx in range(len(dt_forecast)):
                target_dev = last_dev + 1 + step_idx
                target_cal = i + target_dev
                if VAL_END < target_cal <= HOLDOUT_END and not np.isnan(inc[i, target_dev]):
                    act_val = inc[i, target_dev]
                    dt_val = dt_forecast[step_idx, 0]
                    actual_sum += act_val
                    dt_sum += dt_val
                    if act_val > ACTIVE_CELL_THRESHOLD_RM:
                        actual_cells.append(act_val)
                        dt_cells.append(dt_val)
        actual_arr = np.array(actual_cells)
        dt_arr = np.array(dt_cells)
        metrics = compute_metrics(dt_arr, actual_arr) if len(actual_arr) else None
        error_pct = float((dt_sum - actual_sum) / actual_sum * 100.0) if actual_sum else 0.0
        return {
            "reserve_rm_k": float(dt_sum) / 1000.0,
            "actual_rm_k": float(actual_sum) / 1000.0,
            "error_pct": error_pct,
            "metrics": metrics,
            "n_active": len(actual_cells),
        }

    return {
        "theft": _evaluate_peril(theft_inc, theft_balos, peril_id=1),
        "windscreen": _evaluate_peril(ws_inc, ws_balos, peril_id=0),
        "param_count": param_count,
        "seed": seed,
        "lstm_train_history": train_history,
    }
