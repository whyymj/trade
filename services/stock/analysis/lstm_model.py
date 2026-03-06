#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 模型定义
简化的双头 LSTM 模型：分类（涨跌）+ 回归（涨跌幅）
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """LSTM 模型 - 双头输出（分类 + 回归）"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        n_magnitude_outputs: int = 5,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_magnitude_outputs = n_magnitude_outputs

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )
        self.fc_shared = nn.Linear(hidden_size, hidden_size)
        self.fc_direction = nn.Linear(hidden_size, num_classes)
        self.fc_magnitude = nn.Linear(hidden_size, n_magnitude_outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (direction_logits, magnitude_pred)
        """
        lstm_out, _ = self.lstm(x)
        last_h = lstm_out[:, -1]
        h = torch.relu(self.fc_shared(last_h))
        direction_logits = self.fc_direction(h)
        magnitude = self.fc_magnitude(h)
        if magnitude.shape[-1] == 1:
            magnitude = magnitude.squeeze(-1)
        return direction_logits, magnitude


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight_cls: float = 1.0,
    weight_reg: float = 1.0,
    reg_criterion: Optional[nn.Module] = None,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    criterion_cls = nn.CrossEntropyLoss()

    for X_b, y_dir_b, y_mag_b in loader:
        X_b = X_b.to(device)
        y_dir_b = y_dir_b.to(device)
        y_mag_b = y_mag_b.to(device)

        optimizer.zero_grad()
        logits, mag_pred = model(X_b)
        loss_cls = criterion_cls(logits, y_dir_b)

        if reg_criterion is None:
            loss_reg = nn.functional.mse_loss(mag_pred, y_mag_b)
        else:
            out = reg_criterion(mag_pred, y_mag_b)
            loss_reg = out[0] if isinstance(out, tuple) else out

        loss = weight_cls * loss_cls + weight_reg * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader) if len(loader) > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """评估模型"""
    model.eval()
    all_dir_true = []
    all_dir_pred = []
    all_mag_true = []
    all_mag_pred = []

    with torch.no_grad():
        for X_b, y_dir_b, y_mag_b in loader:
            X_b = X_b.to(device)
            logits, mag_pred = model(X_b)

            all_dir_true.extend(y_dir_b.numpy().tolist())
            all_dir_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_mag_true.extend(y_mag_b.numpy().tolist())
            all_mag_pred.extend(mag_pred.cpu().numpy().tolist())

    all_dir_true = np.array(all_dir_true)
    all_dir_pred = np.array(all_dir_pred)
    all_mag_true = np.array(all_mag_true)
    all_mag_pred = np.array(all_mag_pred)

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        recall_score,
    )

    acc = accuracy_score(all_dir_true, all_dir_pred)
    rec = recall_score(all_dir_true, all_dir_pred, average="binary", zero_division=0)
    f1 = f1_score(all_dir_true, all_dir_pred, average="binary", zero_division=0)
    mse = mean_squared_error(all_mag_true, all_mag_pred)

    return {
        "accuracy": float(acc),
        "recall": float(rec),
        "f1": float(f1),
        "mse": float(mse),
    }
