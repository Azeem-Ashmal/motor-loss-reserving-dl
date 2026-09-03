"""
Occlusion-based interpretability for the DeepTriangle LSTM.

Addresses the explainability phase named in the original research plan
(Archived/Thesis_Deliverables_Old/topic4_comprehensive_prospectus.md, Concern 2
"Model Interpretability" and Phase 5 "Explainability & Audit"), which proposed
SHAP values. SHAP's DeepExplainer does not have first-class support for
variable-length, packed-sequence recurrent models like this one; occlusion
(ablating one input channel at a time and measuring the resulting change in
the model's own forecast and error) answers the same practical question - how
much does this model's prediction actually depend on each input signal - with
a method that works directly on this architecture without a bolted-on
compatibility layer, at the cost of coarser per-timestep attribution than SHAP
would give. That trade-off is a deliberate design choice, not a shortcut taken
because SHAP was too much work.

Method: for each active holdout cohort, replace one entire input channel
(IncPaid history, or BALOS history) with its own scaled-space midpoint (0.5)
across every observed timestep, re-run the exact same autoregressive rollout,
and compare the resulting holdout reserve and cell-level error against the
un-occluded forecast. A channel the model actually relies on should degrade
the forecast when removed; a channel it ignores should not.
"""

from typing import Dict, List

import numpy as np

from src.data_pipeline import MinMaxSequenceScaler
from src.evaluate import compute_metrics, predict_auto_regressive
from src.pipeline import ACTIVE_CELL_THRESHOLD_RM
from src.ml.lstm import DeepTriangleLSTM


def occlusion_analysis(
    model: DeepTriangleLSTM,
    scaler: MinMaxSequenceScaler,
    inc: np.ndarray,
    balos: np.ndarray,
    peril_id: int,
    val_end: int,
    holdout_end: int,
) -> Dict[str, Dict]:
    """
    Runs the normal forecast and two occluded forecasts (IncPaid occluded,
    BALOS occluded) for every holdout cohort, and returns aggregate reserve
    and cell-level metrics under each condition.
    """
    num_cohorts, num_devs = inc.shape

    results: Dict[str, Dict] = {
        "normal": {"actual_cells": [], "pred_cells": [], "reserve": 0.0, "actual_reserve": 0.0},
        "occlude_incpaid": {"actual_cells": [], "pred_cells": [], "reserve": 0.0, "actual_reserve": 0.0},
        "occlude_balos": {"actual_cells": [], "pred_cells": [], "reserve": 0.0, "actual_reserve": 0.0},
    }

    for i in range(num_cohorts):
        last_dev = val_end - i
        if last_dev < 0 or last_dev >= num_devs - 1:
            continue

        obs_seq_raw = np.column_stack((inc[i, : last_dev + 1], balos[i, : last_dev + 1]))
        obs_seq_scaled = scaler.transform_feature_array(obs_seq_raw)

        variants = {
            "normal": obs_seq_scaled,
            "occlude_incpaid": obs_seq_scaled.copy(),
            "occlude_balos": obs_seq_scaled.copy(),
        }
        variants["occlude_incpaid"][:, 0] = 0.5
        variants["occlude_balos"][:, 1] = 0.5

        for label, seq in variants.items():
            forecast = predict_auto_regressive(model, scaler, seq, peril_id=peril_id, max_total_len=num_devs)
            for step_idx in range(len(forecast)):
                target_dev = last_dev + 1 + step_idx
                target_cal = i + target_dev
                if val_end < target_cal <= holdout_end and not np.isnan(inc[i, target_dev]):
                    act_val = inc[i, target_dev]
                    pred_val = forecast[step_idx, 0]
                    results[label]["reserve"] += pred_val
                    results[label]["actual_reserve"] += act_val
                    if act_val > ACTIVE_CELL_THRESHOLD_RM:
                        results[label]["actual_cells"].append(act_val)
                        results[label]["pred_cells"].append(pred_val)

    summary = {}
    for label, d in results.items():
        actual_arr = np.array(d["actual_cells"])
        pred_arr = np.array(d["pred_cells"])
        metrics = compute_metrics(pred_arr, actual_arr)
        error_pct = (d["reserve"] - d["actual_reserve"]) / d["actual_reserve"] * 100.0 if d["actual_reserve"] else 0.0
        summary[label] = {
            "reserve_rm_k": d["reserve"] / 1000.0,
            "error_pct": error_pct,
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        }

    # Importance = how much worse RMSE gets when a channel is occluded,
    # relative to the normal forecast's own RMSE.
    base_rmse = summary["normal"]["rmse"]
    summary["incpaid_importance_pct"] = (
        (summary["occlude_incpaid"]["rmse"] - base_rmse) / base_rmse * 100.0 if base_rmse else 0.0
    )
    summary["balos_importance_pct"] = (
        (summary["occlude_balos"]["rmse"] - base_rmse) / base_rmse * 100.0 if base_rmse else 0.0
    )
    return summary
