"""
Occlusion-based channel-importance analysis, single-peril and pooled.
Trains the seed-42 model(s) fresh (same building blocks as run_all.py /
run_pooled_single_seed.py) and runs src.interpretability.occlusion_analysis
against them, writing outputs/occlusion_results.json and
outputs/occlusion_pooled_results.json.

Previously produced by an ad-hoc, unsaved call predating the active-cell
threshold fix (commit d45aabf) - this script makes both results reproducible
from code and on the corrected RM 1,000 threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import MinMaxSequenceScaler, load_and_preprocess_triangles, pad_and_tensorize
from src.interpretability import occlusion_analysis
from src.ml.lstm import DeepTriangleLSTM
from src.pipeline import VAL_END, HOLDOUT_END, _build_lstm_splits
from src.seeding import set_deterministic_seed
from src.train import train_deeptriangle_model


def _to_native(obj):
    """Recursively cast numpy scalars to native Python types for json.dump."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _train_single_peril(paid_path, out_path, peril_id, seed, epochs):
    set_deterministic_seed(seed)
    inc, balos, _, _ = load_and_preprocess_triangles(paid_path, out_path)
    train_s, train_t, train_m, val_s, val_t, val_m = _build_lstm_splits(inc, balos, peril_id=peril_id)
    scaler = MinMaxSequenceScaler(use_log_transform=True).fit(train_s, train_t)
    model = DeepTriangleLSTM(input_dim=2, hidden_dim=64, dropout=0.2, output_activation="softplus")
    train_tensors = pad_and_tensorize(train_s, train_t, train_m, scaler)
    val_tensors = pad_and_tensorize(val_s, val_t, val_m, scaler)
    ckpt = f"outputs/_tmp_occlusion_single_{peril_id}.pt"
    train_deeptriangle_model(model, train_tensors, val_tensors, epochs=epochs,
                              learning_rate=1e-3, patience=25, checkpoint_path=ckpt)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    return model, scaler, inc, balos


def _train_pooled(theft_paid, theft_out, ws_paid, ws_out, seed, epochs):
    set_deterministic_seed(seed)
    theft_inc, theft_balos, _, _ = load_and_preprocess_triangles(theft_paid, theft_out)
    ws_inc, ws_balos, _, _ = load_and_preprocess_triangles(ws_paid, ws_out)
    t_train_s, t_train_t, t_train_m, t_val_s, t_val_t, t_val_m = _build_lstm_splits(theft_inc, theft_balos, peril_id=1)
    w_train_s, w_train_t, w_train_m, w_val_s, w_val_t, w_val_m = _build_lstm_splits(ws_inc, ws_balos, peril_id=0)
    train_seqs, train_tgts, train_meta = t_train_s + w_train_s, t_train_t + w_train_t, t_train_m + w_train_m
    val_seqs, val_tgts, val_meta = t_val_s + w_val_s, t_val_t + w_val_t, t_val_m + w_val_m
    scaler = MinMaxSequenceScaler(use_log_transform=True).fit(train_seqs, train_tgts)
    model = DeepTriangleLSTM(input_dim=2, hidden_dim=64, dropout=0.2, output_activation="softplus",
                              use_peril_embedding=True, num_perils=2, embed_dim=8)
    train_tensors = pad_and_tensorize(train_seqs, train_tgts, train_meta, scaler)
    val_tensors = pad_and_tensorize(val_seqs, val_tgts, val_meta, scaler)
    ckpt = "outputs/_tmp_occlusion_pooled.pt"
    train_deeptriangle_model(model, train_tensors, val_tensors, epochs=epochs,
                              learning_rate=1e-3, patience=25, checkpoint_path=ckpt)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    return model, scaler, theft_inc, theft_balos, ws_inc, ws_balos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    theft_paid = str(data_dir / "theft" / "paid_qtr_triangle.csv")
    theft_out = str(data_dir / "theft" / "outstanding_qtr_triangle.csv")
    ws_paid = str(data_dir / "windscreen" / "paid_qtr_triangle.csv")
    ws_out = str(data_dir / "windscreen" / "outstanding_qtr_triangle.csv")

    Path("outputs").mkdir(exist_ok=True)

    print("=== Single-peril occlusion (Theft) ===")
    t_model, t_scaler, t_inc, t_balos = _train_single_peril(theft_paid, theft_out, 1, args.seed, args.epochs)
    theft_occ = occlusion_analysis(t_model, t_scaler, t_inc, t_balos, peril_id=1, val_end=VAL_END, holdout_end=HOLDOUT_END)

    print("=== Single-peril occlusion (Windscreen) ===")
    w_model, w_scaler, w_inc, w_balos = _train_single_peril(ws_paid, ws_out, 0, args.seed, args.epochs)
    ws_occ = occlusion_analysis(w_model, w_scaler, w_inc, w_balos, peril_id=0, val_end=VAL_END, holdout_end=HOLDOUT_END)

    single_out = {"theft": theft_occ, "windscreen": ws_occ}
    with open("outputs/occlusion_results.json", "w") as f:
        json.dump(_to_native(single_out), f, indent=2)
    print(f"Theft normal:      error={theft_occ['normal']['error_pct']:+.2f}%  R2={theft_occ['normal']['r2']:.4f}")
    print(f"Windscreen normal: error={ws_occ['normal']['error_pct']:+.2f}%  R2={ws_occ['normal']['r2']:.4f}")
    print("Wrote outputs/occlusion_results.json")

    print("\n=== Pooled occlusion ===")
    p_model, p_scaler, p_t_inc, p_t_balos, p_w_inc, p_w_balos = _train_pooled(
        theft_paid, theft_out, ws_paid, ws_out, args.seed, args.epochs
    )
    theft_pooled_occ = occlusion_analysis(p_model, p_scaler, p_t_inc, p_t_balos, peril_id=1, val_end=VAL_END, holdout_end=HOLDOUT_END)
    ws_pooled_occ = occlusion_analysis(p_model, p_scaler, p_w_inc, p_w_balos, peril_id=0, val_end=VAL_END, holdout_end=HOLDOUT_END)

    pooled_out = {"theft": theft_pooled_occ, "windscreen": ws_pooled_occ}
    with open("outputs/occlusion_pooled_results.json", "w") as f:
        json.dump(_to_native(pooled_out), f, indent=2)
    print(f"Theft (pooled) normal:      error={theft_pooled_occ['normal']['error_pct']:+.2f}%  R2={theft_pooled_occ['normal']['r2']:.4f}")
    print(f"Windscreen (pooled) normal: error={ws_pooled_occ['normal']['error_pct']:+.2f}%  R2={ws_pooled_occ['normal']['r2']:.4f}")
    print("Wrote outputs/occlusion_pooled_results.json")


if __name__ == "__main__":
    main()
