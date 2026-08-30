"""
Data Engineering & Tensorization Module for Topic 4 (DeepTriangle Framework).

This module processes 2D actuarial claims triangles (Cumulative Paid and Outstanding BALOS)
into 3D sequential tensors suitable for Multi-Task Seq2Seq LSTM architectures.

Key Features:
- Incremental Paid Loss derivation
- Sliding window temporal sequence generation
- Strict out-of-sample calendar year splitting (<= 2023 Q4 Train, >= 2024 Q1 Test)
- Zero-leakage Min-Max scaling fitted strictly on training data
- Sequence padding (max_len=65) and complementary binary masking
- Multi-peril pooling support with categorical peril IDs
"""

import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch


class MinMaxSequenceScaler:
    """
    Min-Max Scaler with log1p transformation designed for actuarial sequence tensors and target vectors.
    Scales features to the [0, 1] interval. Parameters are fitted strictly on
    the training dataset to guarantee zero out-of-sample data leakage.
    """

    def __init__(
        self,
        use_log_transform: bool = True,
        clip_range: Tuple[float, float] = (0.0, 1.0),
        n_extra_features: int = 0,
    ):
        self.use_log_transform = use_log_transform
        self.clip_min, self.clip_max = clip_range
        self.inc_paid_min: float = 0.0
        self.inc_paid_max: float = 1.0
        self.balos_min: float = 0.0
        self.balos_max: float = 1.0
        # Extra exogenous input-only channels beyond IncPaid/BALOS (e.g. claim
        # counts). Never predicted as a target, only ever fed as history.
        self.n_extra_features = n_extra_features
        self.extra_min: List[float] = [0.0] * n_extra_features
        self.extra_max: List[float] = [1.0] * n_extra_features
        self.is_fitted: bool = False

    def fit(self, sequences: List[np.ndarray], targets: List[np.ndarray]) -> "MinMaxSequenceScaler":
        """
        Fit scaler using list of raw sequence arrays and target arrays.
        Each sequence array is of shape (seq_len, 2 + n_extra_features): feature 0
        is IncPaid, feature 1 is BALOS, any further columns are exogenous extras
        (fit here, but never predicted - targets stay 2-dimensional).
        """
        all_inc_paid = []
        all_balos = []
        all_extra: List[list] = [[] for _ in range(self.n_extra_features)]

        for seq in sequences:
            inc = np.log1p(np.maximum(seq[:, 0], 0.0)) if self.use_log_transform else seq[:, 0]
            bal = np.log1p(np.maximum(seq[:, 1], 0.0)) if self.use_log_transform else seq[:, 1]
            all_inc_paid.extend(inc)
            all_balos.extend(bal)
            for e in range(self.n_extra_features):
                col = seq[:, 2 + e]
                col = np.log1p(np.maximum(col, 0.0)) if self.use_log_transform else col
                all_extra[e].extend(col)

        for tgt in targets:
            inc = np.log1p(np.maximum(tgt[0], 0.0)) if self.use_log_transform else tgt[0]
            bal = np.log1p(np.maximum(tgt[1], 0.0)) if self.use_log_transform else tgt[1]
            all_inc_paid.append(inc)
            all_balos.append(bal)

        self.inc_paid_min = float(np.min(all_inc_paid))
        self.inc_paid_max = float(np.max(all_inc_paid))
        self.balos_min = float(np.min(all_balos))
        self.balos_max = float(np.max(all_balos))

        # Add small epsilon safety to prevent zero-division
        if self.inc_paid_max == self.inc_paid_min:
            self.inc_paid_max += 1e-6
        if self.balos_max == self.balos_min:
            self.balos_max += 1e-6

        for e in range(self.n_extra_features):
            mn, mx = float(np.min(all_extra[e])), float(np.max(all_extra[e]))
            if mx == mn:
                mx += 1e-6
            self.extra_min[e] = mn
            self.extra_max[e] = mx

        self.is_fitted = True
        return self

    def transform_feature_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Transform a feature array of shape (..., 2) or (..., 2 + n_extra_features)
        into scaled [0, 1] space. Feature 0: IncPaid, Feature 1: BALOS, any
        further columns: exogenous extras in fit order.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data.")

        inc = np.log1p(np.maximum(arr[..., 0], 0.0)) if self.use_log_transform else arr[..., 0]
        bal = np.log1p(np.maximum(arr[..., 1], 0.0)) if self.use_log_transform else arr[..., 1]

        scaled = np.zeros_like(arr, dtype=np.float32)
        scaled[..., 0] = (inc - self.inc_paid_min) / (self.inc_paid_max - self.inc_paid_min)
        scaled[..., 1] = (bal - self.balos_min) / (self.balos_max - self.balos_min)

        n_extra_present = arr.shape[-1] - 2
        for e in range(n_extra_present):
            col = arr[..., 2 + e]
            col = np.log1p(np.maximum(col, 0.0)) if self.use_log_transform else col
            scaled[..., 2 + e] = (col - self.extra_min[e]) / (self.extra_max[e] - self.extra_min[e])
        return np.clip(scaled, self.clip_min, self.clip_max)

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets of shape (..., 2) back to original RM values.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming.")

        inc_log = targets_scaled[..., 0] * (self.inc_paid_max - self.inc_paid_min) + self.inc_paid_min
        bal_log = targets_scaled[..., 1] * (self.balos_max - self.balos_min) + self.balos_min

        unscaled = np.zeros_like(targets_scaled, dtype=np.float32)
        if self.use_log_transform:
            unscaled[..., 0] = np.expm1(np.maximum(inc_log, 0.0))
            unscaled[..., 1] = np.expm1(np.maximum(bal_log, 0.0))
        else:
            unscaled[..., 0] = inc_log
            unscaled[..., 1] = bal_log
        return unscaled

    def save_meta(self, filepath: str) -> None:
        """Save scaler parameters to JSON metadata file."""
        meta = {
            "use_log_transform": self.use_log_transform,
            "inc_paid_min": self.inc_paid_min,
            "inc_paid_max": self.inc_paid_max,
            "balos_min": self.balos_min,
            "balos_max": self.balos_max,
            "clip_range": [self.clip_min, self.clip_max],
            "is_fitted": self.is_fitted,
            "n_extra_features": self.n_extra_features,
            "extra_min": self.extra_min,
            "extra_max": self.extra_max,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_meta(cls, filepath: str) -> "MinMaxSequenceScaler":
        """Load scaler parameters from JSON metadata file."""
        with open(filepath, "r") as f:
            meta = json.load(f)
        scaler = cls(
            use_log_transform=meta.get("use_log_transform", True),
            clip_range=tuple(meta["clip_range"]),
            n_extra_features=meta.get("n_extra_features", 0),
        )
        scaler.inc_paid_min = meta["inc_paid_min"]
        scaler.inc_paid_max = meta["inc_paid_max"]
        scaler.balos_min = meta["balos_min"]
        scaler.balos_max = meta["balos_max"]
        scaler.extra_min = meta.get("extra_min", [])
        scaler.extra_max = meta.get("extra_max", [])
        scaler.is_fitted = meta["is_fitted"]
        return scaler


def load_and_preprocess_triangles(
    paid_csv_path: str,
    outstanding_csv_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load paid and outstanding CSV triangles, derive incremental paid losses,
    and return clean matrices.

    Returns:
        inc_paid_matrix: shape (num_cohorts, num_dev_periods)
        balos_matrix: shape (num_cohorts, num_dev_periods)
        cohorts: list of cohort names (e.g., '2010 Q1')
        dev_cols: list of dev column names (e.g., 'Dev_0', ..., 'Dev_64')
    """
    paid_df = pd.read_csv(paid_csv_path)
    out_df = pd.read_csv(outstanding_csv_path)

    cohorts = paid_df["Cohort"].tolist()
    dev_cols = [c for c in paid_df.columns if c.startswith("Dev_")]

    cum_paid = paid_df[dev_cols].values.astype(np.float64)
    balos = out_df[dev_cols].values.astype(np.float64)

    # Derive incremental paid losses
    inc_paid = np.zeros_like(cum_paid)
    inc_paid[:, 0] = cum_paid[:, 0]
    inc_paid[:, 1:] = cum_paid[:, 1:] - cum_paid[:, :-1]

    return inc_paid, balos, cohorts, dev_cols


def build_samples_and_split(
    inc_paid_matrix: np.ndarray,
    balos_matrix: np.ndarray,
    cohorts: List[str],
    peril_id: int = 0,
    train_cutoff_calendar_idx: int = 55,  # 2023 Q4 (index 55 out of 64)
) -> Dict[str, Any]:
    """
    Extract sliding window historical sequences and target vectors for each cohort and dev step.
    Splits strictly based on calendar index of target period:
    - Train: target_cal_idx <= train_cutoff_calendar_idx (2010 Q1 - 2023 Q4)
    - Out-of-Sample Val: train_cutoff_calendar_idx < target_cal_idx <= 64 (2024 Q1 - 2026 Q1)
    """
    num_cohorts, num_devs = inc_paid_matrix.shape

    train_sequences, train_targets, train_meta = [], [], []
    val_sequences, val_targets, val_meta = [], [], []

    for i in range(num_cohorts):
        cohort_label = cohorts[i]
        for t in range(num_devs - 1):
            target_cal_idx = i + t + 1
            if target_cal_idx >= num_cohorts:
                # Target is beyond currently available calendar range (future unobserved)
                continue

            # Check if history at t and target at t+1 are valid (not NaN)
            hist_inc = inc_paid_matrix[i, : t + 1]
            hist_bal = balos_matrix[i, : t + 1]
            target_inc = inc_paid_matrix[i, t + 1]
            target_bal = balos_matrix[i, t + 1]

            if (
                np.isnan(hist_inc).any()
                or np.isnan(hist_bal).any()
                or np.isnan(target_inc)
                or np.isnan(target_bal)
            ):
                continue

            seq = np.column_stack((hist_inc, hist_bal))
            target = np.array([target_inc, target_bal], dtype=np.float64)

            meta_info = {
                "cohort_idx": i,
                "cohort_label": cohort_label,
                "dev_idx": t,
                "target_dev_idx": t + 1,
                "target_cal_idx": target_cal_idx,
                "peril_id": peril_id,
            }

            if target_cal_idx <= train_cutoff_calendar_idx:
                train_sequences.append(seq)
                train_targets.append(target)
                train_meta.append(meta_info)
            else:
                val_sequences.append(seq)
                val_targets.append(target)
                val_meta.append(meta_info)

    return {
        "train_sequences": train_sequences,
        "train_targets": train_targets,
        "train_meta": train_meta,
        "val_sequences": val_sequences,
        "val_targets": val_targets,
        "val_meta": val_meta,
    }


def pad_and_tensorize(
    sequences: List[np.ndarray],
    targets: List[np.ndarray],
    meta: List[Dict[str, Any]],
    scaler: MinMaxSequenceScaler,
    max_len: int = 65,
) -> Dict[str, torch.Tensor]:
    """
    Scale sequences and targets, left-pad sequences with 0.0 up to max_len,
    and build complementary binary mask tensors.

    Returns dictionary containing:
        - X: Tensor of shape (N, max_len, input_dim) - input_dim inferred from
          the sequences (2 for IncPaid+BALOS only, more if exogenous extras
          such as claim counts are appended as further columns)
        - Y: Tensor of shape (N, 2)
        - mask: FloatTensor of shape (N, max_len) (1.0 for valid, 0.0 for padded)
        - peril_ids: LongTensor of shape (N,)
    """
    n_samples = len(sequences)
    input_dim = sequences[0].shape[1] if n_samples > 0 else 2
    X = np.zeros((n_samples, max_len, input_dim), dtype=np.float32)
    Y = np.zeros((n_samples, 2), dtype=np.float32)
    mask = np.zeros((n_samples, max_len), dtype=np.float32)
    peril_ids = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        seq = sequences[i]
        tgt = targets[i]
        seq_len = len(seq)

        # Transform using fitted scaler
        seq_scaled = scaler.transform_feature_array(seq)
        tgt_scaled = scaler.transform_feature_array(tgt.reshape(1, 2)).squeeze(0)

        # Right-padding for PyTorch pack_padded_sequence (sequence at start, 0s at end)
        X[i, :seq_len, :] = seq_scaled
        mask[i, :seq_len] = 1.0
        Y[i, :] = tgt_scaled
        peril_ids[i] = meta[i]["peril_id"]

    return {
        "X": torch.tensor(X, dtype=torch.float32),
        "Y": torch.tensor(Y, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "peril_ids": torch.tensor(peril_ids, dtype=torch.long),
    }


def process_topic4_data(
    base_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Master data engineering function for Topic 4.
    Processes Windscreen and Theft loss triangles, fits Min-Max scalers,
    constructs train and out-of-sample validation tensors, and saves processed outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    ws_paid_path = os.path.join(base_dir, "Data/ws_triangles/paid_qtr_triangle.csv")
    ws_out_path = os.path.join(base_dir, "Data/ws_triangles/outstanding_qtr_triangle.csv")

    theft_paid_path = os.path.join(base_dir, "Data/theft_triangles_all_covers/paid_qtr_triangle.csv")
    theft_out_path = os.path.join(base_dir, "Data/theft_triangles_all_covers/outstanding_qtr_triangle.csv")

    # 1. Load raw triangles
    ws_inc, ws_balos, ws_cohorts, dev_cols = load_and_preprocess_triangles(ws_paid_path, ws_out_path)
    theft_inc, theft_balos, theft_cohorts, _ = load_and_preprocess_triangles(theft_paid_path, theft_out_path)

    # 2. Extract sequences and train/val split (peril_id=0 for WS, 1 for Theft)
    ws_split = build_samples_and_split(ws_inc, ws_balos, ws_cohorts, peril_id=0)
    theft_split = build_samples_and_split(theft_inc, theft_balos, theft_cohorts, peril_id=1)

    # 3. Fit scalers strictly on training data
    ws_scaler = MinMaxSequenceScaler().fit(ws_split["train_sequences"], ws_split["train_targets"])
    theft_scaler = MinMaxSequenceScaler().fit(theft_split["train_sequences"], theft_split["train_targets"])

    # Pooled scaler fitted on combined training data
    pooled_train_seqs = ws_split["train_sequences"] + theft_split["train_sequences"]
    pooled_train_tgts = ws_split["train_targets"] + theft_split["train_targets"]
    pooled_scaler = MinMaxSequenceScaler().fit(pooled_train_seqs, pooled_train_tgts)

    # 4. Tensorize Windscreen
    ws_train_tensors = pad_and_tensorize(
        ws_split["train_sequences"], ws_split["train_targets"], ws_split["train_meta"], ws_scaler
    )
    ws_val_tensors = pad_and_tensorize(
        ws_split["val_sequences"], ws_split["val_targets"], ws_split["val_meta"], ws_scaler
    )

    # 5. Tensorize Theft
    theft_train_tensors = pad_and_tensorize(
        theft_split["train_sequences"], theft_split["train_targets"], theft_split["train_meta"], theft_scaler
    )
    theft_val_tensors = pad_and_tensorize(
        theft_split["val_sequences"], theft_split["val_targets"], theft_split["val_meta"], theft_scaler
    )

    # 6. Tensorize Pooled Multi-Peril Dataset
    pooled_train_meta = ws_split["train_meta"] + theft_split["train_meta"]
    pooled_val_seqs = ws_split["val_sequences"] + theft_split["val_sequences"]
    pooled_val_tgts = ws_split["val_targets"] + theft_split["val_targets"]
    pooled_val_meta = ws_split["val_meta"] + theft_split["val_meta"]

    pooled_train_tensors = pad_and_tensorize(
        pooled_train_seqs, pooled_train_tgts, pooled_train_meta, pooled_scaler
    )
    pooled_val_tensors = pad_and_tensorize(
        pooled_val_seqs, pooled_val_tgts, pooled_val_meta, pooled_scaler
    )

    # Save scalers
    ws_scaler_path = os.path.join(output_dir, "ws_scaler_meta.json")
    theft_scaler_path = os.path.join(output_dir, "theft_scaler_meta.json")
    pooled_scaler_path = os.path.join(output_dir, "pooled_scaler_meta.json")

    ws_scaler.save_meta(ws_scaler_path)
    theft_scaler.save_meta(theft_scaler_path)
    pooled_scaler.save_meta(pooled_scaler_path)

    # Save tensors
    ws_tensor_path = os.path.join(output_dir, "ws_tensors.pt")
    theft_tensor_path = os.path.join(output_dir, "theft_tensors.pt")
    pooled_tensor_path = os.path.join(output_dir, "pooled_tensors.pt")

    torch.save(
        {
            "train": ws_train_tensors,
            "val": ws_val_tensors,
            "train_meta": ws_split["train_meta"],
            "val_meta": ws_split["val_meta"],
        },
        ws_tensor_path,
    )

    torch.save(
        {
            "train": theft_train_tensors,
            "val": theft_val_tensors,
            "train_meta": theft_split["train_meta"],
            "val_meta": theft_split["val_meta"],
        },
        theft_tensor_path,
    )

    torch.save(
        {
            "train": pooled_train_tensors,
            "val": pooled_val_tensors,
            "train_meta": pooled_train_meta,
            "val_meta": pooled_val_meta,
        },
        pooled_tensor_path,
    )

    print("Data Engineering & Tensorization complete.")
    print(f"  Windscreen Train samples: {len(ws_train_tensors['X'])}, Val: {len(ws_val_tensors['X'])}")
    print(f"  Theft Train samples: {len(theft_train_tensors['X'])}, Val: {len(theft_val_tensors['X'])}")
    print(f"  Pooled Train samples: {len(pooled_train_tensors['X'])}, Val: {len(pooled_val_tensors['X'])}")

    return {
        "ws_tensor_path": ws_tensor_path,
        "theft_tensor_path": theft_tensor_path,
        "pooled_tensor_path": pooled_tensor_path,
    }
