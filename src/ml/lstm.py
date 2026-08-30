"""
DeepTriangle-Inspired Multi-Task Recurrent Neural Network Architecture.
PyTorch LSTM (or GRU) with two independent dense heads predicting next-quarter
incremental paid losses and case reserves (BALOS), trained autoregressively over
the 65-quarter development horizon.

Verified parameter counts (input_dim excludes any peril embedding, hidden_dim=64):
- Paid-only   (d_in=1): 21,378
- Single-peril joint (d_in=2, IncPaid + BALOS): 21,634
- Pooled multi-peril (d_in=2, +8-dim peril embedding): 23,698
These are computed by count_parameters(), never hand-typed; see tests/test_reconciliation.py.

Kuo (2019)'s original DeepTriangle used a GRU, not an LSTM; this repo defaults
to LSTM but exposes `rnn_type="gru"` so that choice can be tested directly
against this dataset rather than assumed (see run_multiseed.py --rnn-type and
the dissertation's architecture-comparison chapter).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class DeepTriangleLSTM(nn.Module):
    """
    Multi-task RNN: one LSTM or GRU encoder over the observed history of a
    cohort, two-layer dense heads (64 -> 32 -> 1) predicting the next
    incremental paid loss and next case reserve. Non-negativity enforced via
    Softplus/ReLU. Class name kept as DeepTriangleLSTM for backward
    compatibility with existing checkpoints/tests even when rnn_type="gru".
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_peril_embedding: bool = False,
        num_perils: int = 2,
        embed_dim: int = 8,
        output_activation: str = "softplus",
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_peril_embedding = use_peril_embedding
        self.embed_dim = embed_dim if use_peril_embedding else 0
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        if self.use_peril_embedding:
            self.peril_embed = nn.Embedding(num_embeddings=num_perils, embedding_dim=embed_dim)
            lstm_input_dim = input_dim + embed_dim
        else:
            self.peril_embed = None
            lstm_input_dim = input_dim

        self.input_dropout = nn.Dropout(p=dropout)
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.lstm = rnn_cls(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dropout = nn.Dropout(p=dropout)

        if output_activation.lower() == "softplus":
            self.non_neg_act = nn.Softplus()
        elif output_activation.lower() == "relu":
            self.non_neg_act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        self.head_inc_paid = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(32, 1),
            self.non_neg_act,
        )
        self.head_balos = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(32, 1),
            self.non_neg_act,
        )

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        peril_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        X: (B, T, input_dim), feature 0 = IncPaid, feature 1 = BALOS.
        mask: (B, T), 1.0 for valid (right-padded) timesteps, 0.0 for padding.
        Returns: (B, 2) next-step predictions, column 0 = IncPaid, column 1 = BALOS.
        """
        batch_size, seq_len, _ = X.shape
        lengths = torch.clamp(mask.sum(dim=1).long().cpu(), min=1)

        if self.use_peril_embedding:
            if peril_ids is None:
                raise ValueError("peril_ids must be provided when use_peril_embedding=True")
            peril_emb = self.peril_embed(peril_ids).unsqueeze(1).repeat(1, seq_len, 1)
            X_in = torch.cat([X, peril_emb], dim=-1)
        else:
            X_in = X

        X_in = self.input_dropout(X_in)
        packed_in = pack_padded_sequence(X_in, lengths, batch_first=True, enforce_sorted=False)
        if self.rnn_type == "gru":
            _, h_n = self.lstm(packed_in)
        else:
            _, (h_n, _) = self.lstm(packed_in)
        last_hidden = self.hidden_dropout(h_n[-1])

        pred_inc_paid = self.head_inc_paid(last_hidden)
        pred_balos = self.head_balos(last_hidden)
        return torch.cat([pred_inc_paid, pred_balos], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskLoss(nn.Module):
    """Weighted MSE across the IncPaid and BALOS heads."""

    def __init__(self, w_inc_paid: float = 0.5, w_balos: float = 0.5):
        super().__init__()
        self.w_inc_paid = w_inc_paid
        self.w_balos = w_balos
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        loss_inc_paid = self.mse(predictions[:, 0], targets[:, 0])
        loss_balos = self.mse(predictions[:, 1], targets[:, 1])
        total_loss = self.w_inc_paid * loss_inc_paid + self.w_balos * loss_balos
        return total_loss, loss_inc_paid, loss_balos
