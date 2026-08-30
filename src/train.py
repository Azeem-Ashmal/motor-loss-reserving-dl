"""
Training and Optimization Module for Topic 4 (DeepTriangle Framework).

Implements PyTorch training loops, Adam optimizer, early stopping callback,
and model checkpointing to enforce robust generalization and prevent overfitting.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ml.lstm import DeepTriangleLSTM, MultiTaskLoss


class EarlyStopping:
    """
    Early Stopping callback that monitors validation loss and saves the optimal model parameters.
    Halts training when out-of-sample validation loss fails to improve after specified patience.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_deeptriangle_model(
    model: DeepTriangleLSTM,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    checkpoint_path: str = "models/best_model.pt",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Execute DeepTriangle training loop with Adam optimizer and Early Stopping.

    Returns:
        history: Dict containing loss trajectories across epochs.
    """
    model = model.to(device)

    # DataLoader setup
    train_dataset = TensorDataset(
        train_tensors["X"],
        train_tensors["Y"],
        train_tensors["mask"],
        train_tensors["peril_ids"],
    )
    val_dataset = TensorDataset(
        val_tensors["X"],
        val_tensors["Y"],
        val_tensors["mask"],
        val_tensors["peril_ids"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = MultiTaskLoss(w_inc_paid=0.5, w_balos=0.5)
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_inc_paid_loss": [],
        "train_balos_loss": [],
        "val_inc_paid_loss": [],
        "val_balos_loss": [],
    }

    print(f"Beginning model training on device '{device}' for up to {epochs} epochs...")
    print(f"  Train set: {len(train_dataset)} samples | Val set: {len(val_dataset)} samples")

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        running_inc_loss = 0.0
        running_bal_loss = 0.0

        for X_b, Y_b, mask_b, peril_b in train_loader:
            X_b, Y_b, mask_b, peril_b = (
                X_b.to(device),
                Y_b.to(device),
                mask_b.to(device),
                peril_b.to(device),
            )

            optimizer.zero_grad()
            preds = model(X_b, mask_b, peril_b)
            total_loss, inc_loss, bal_loss = criterion(preds, Y_b)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item() * len(X_b)
            running_inc_loss += inc_loss.item() * len(X_b)
            running_bal_loss += bal_loss.item() * len(X_b)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_inc = running_inc_loss / len(train_dataset)
        epoch_train_bal = running_bal_loss / len(train_dataset)

        # 2. Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_inc = 0.0
        val_running_bal = 0.0

        with torch.no_grad():
            for X_b, Y_b, mask_b, peril_b in val_loader:
                X_b, Y_b, mask_b, peril_b = (
                    X_b.to(device),
                    Y_b.to(device),
                    mask_b.to(device),
                    peril_b.to(device),
                )
                preds = model(X_b, mask_b, peril_b)
                total_loss, inc_loss, bal_loss = criterion(preds, Y_b)

                val_running_loss += total_loss.item() * len(X_b)
                val_running_inc += inc_loss.item() * len(X_b)
                val_running_bal += bal_loss.item() * len(X_b)

        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_inc = val_running_inc / len(val_dataset)
        epoch_val_bal = val_running_bal / len(val_dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_inc_paid_loss"].append(epoch_train_inc)
        history["train_balos_loss"].append(epoch_train_bal)
        history["val_inc_paid_loss"].append(epoch_val_inc)
        history["val_balos_loss"].append(epoch_val_bal)

        improved = early_stopping(epoch_val_loss, model)

        if epoch % 10 == 0 or epoch == 1 or improved:
            imp_marker = " [*Saved Best Model]" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}{imp_marker}"
            )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch}. Best Val Loss: {early_stopping.best_loss:.6f}")
            break

    # Load best checkpoint into model before returning
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    return history
