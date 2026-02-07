# -*- coding: utf-8 -*-
"""
改进的 LSTM 训练策略：AdamW、余弦退火、早停（波动匹配度）、时间序列数据增强。
"""
from __future__ import annotations

import copy
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from analysis.lstm_losses import get_regression_criterion


def 时间序列数据增强(
    X_b: torch.Tensor,
    y_dir_b: torch.Tensor,
    y_mag_b: torch.Tensor,
    noise_std: float = 0.01,
    scale_low: float = 0.98,
    scale_high: float = 1.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对当前 batch 做轻量增强：添加高斯噪声、逐样本缩放扰动。
    X_b: (batch, seq_len, n_features)，仅对 X 增强，y 不变。
    """
    X_aug = X_b + torch.randn_like(X_b, device=X_b.device, dtype=X_b.dtype) * noise_std
    scale = torch.rand(X_aug.size(0), 1, X_aug.size(2), device=X_aug.device, dtype=X_aug.dtype) * (scale_high - scale_low) + scale_low
    X_aug = X_aug * scale
    return X_aug, y_dir_b, y_mag_b


def 波动匹配度(预测幅度: torch.Tensor, 实际幅度: torch.Tensor) -> float:
    """1 - |std(预测)-std(实际)|/std(实际)，越接近 1 越好。"""
    pred_std = torch.std(预测幅度.detach(), unbiased=False).item()
    true_std = torch.std(实际幅度, unbiased=False).item()
    if true_std < 1e-12:
        return 1.0
    return float(1.0 - min(1.0, abs(pred_std - true_std) / true_std))


# 默认改进策略配置（与用户描述对齐，侧重回归幅度与波动匹配）
DEFAULT_IMPROVED_CONFIG = {
    "初始学习率": 0.001,
    "权重衰减": 0.01,
    "梯度裁剪": 1.0,
    "早停耐心": 25,
    "最小学习率": 1e-5,
    "T_0": 50,
    "T_mult": 2,
    "使用数据增强": True,
    "noise_std": 0.01,
    "scale_low": 0.98,
    "scale_high": 1.02,
    "reg_loss_type": "full",
    "weight_cls": 0.9,
    "weight_reg": 4.0,  # 回退微调：4.5 时同股反而更平滑，保留此前拟合较好的一档
    "max_epochs": 100,
}


def 改进的训练策略(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    device: Any,
    配置: Optional[dict[str, Any]] = None,
    stop_event: Optional[Any] = None,
) -> tuple[dict[str, Any], list[float], dict[str, float], int]:
    """
    专门针对波动预测的训练策略：AdamW、余弦退火、早停（波动匹配度）、可选数据增强。

    返回: (best_state_dict, training_loss_history, metrics_at_best, best_epoch)
    """
    if not _TORCH_AVAILABLE or torch is None or nn is None or DataLoader is None:
        raise ImportError("LSTM 训练需要 PyTorch，请安装: pip install torch scikit-learn")
    cfg = dict(DEFAULT_IMPROVED_CONFIG)
    if 配置:
        cfg.update(配置)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["初始学习率"],
        weight_decay=cfg["权重衰减"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["T_0"],
        T_mult=cfg["T_mult"],
        eta_min=cfg["最小学习率"],
    )
    reg_criterion = get_regression_criterion(cfg.get("reg_loss_type", "full"))
    if reg_criterion is not None:
        reg_criterion = reg_criterion.to(device)
    criterion_cls = nn.CrossEntropyLoss()
    weight_cls = cfg.get("weight_cls", 1.0)
    weight_reg = cfg.get("weight_reg", 1.0)
    use_aug = cfg.get("使用数据增强", True)
    noise_std = cfg.get("noise_std", 0.01)
    scale_low = cfg.get("scale_low", 0.98)
    scale_high = cfg.get("scale_high", 1.02)
    grad_clip = cfg.get("梯度裁剪", 1.0)
    patience = cfg.get("早停耐心", 20)
    max_epochs = cfg.get("max_epochs", 100)

    best_match = -1.0
    best_state: Optional[dict[str, Any]] = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_count = 0
    training_loss_history: list[float] = []

    for epoch in range(max_epochs):
        if stop_event is not None and stop_event.is_set():
            break
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_dir_b, y_mag_b in train_loader:
            X_b = X_b.to(device)
            y_dir_b = y_dir_b.to(device)
            y_mag_b = y_mag_b.to(device)
            if use_aug:
                X_b, y_dir_b, y_mag_b = 时间序列数据增强(
                    X_b, y_dir_b, y_mag_b, noise_std=noise_std, scale_low=scale_low, scale_high=scale_high
                )
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        training_loss_history.append(epoch_loss / max(n_batches, 1))
        scheduler.step()

        model.eval()
        all_dir_true: list[int] = []
        all_dir_pred: list[int] = []
        all_mag_pred: list[float] = []
        all_mag_true: list[float] = []
        with torch.no_grad():
            for X_b, y_dir_b, y_mag_b in val_loader:
                X_b = X_b.to(device)
                logits, mag_pred = model(X_b)
                all_dir_true.extend(y_dir_b.numpy().tolist())
                all_dir_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                all_mag_pred.extend(mag_pred.cpu().numpy().tolist())
                all_mag_true.extend(y_mag_b.numpy().tolist())
        pred_t = torch.tensor(all_mag_pred, dtype=torch.float32, device=device)
        true_t = torch.tensor(all_mag_true, dtype=torch.float32, device=device)
        match = 波动匹配度(pred_t, true_t)

        if match > best_match:
            best_match = match
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            pred_std = torch.std(pred_t).item()
            true_std = torch.std(true_t).item()
            n_val = len(all_mag_true)
            mse_val = float((pred_t - true_t).pow(2).mean().item()) if n_val else 0.0
            acc_val = sum(1 for a, b in zip(all_dir_pred, all_dir_true) if a == b) / max(n_val, 1)
            best_metrics = {
                "volatility_match": match,
                "val_pred_std": pred_std,
                "val_true_std": true_std,
                "mse": mse_val,
                "accuracy": acc_val,
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state or model.state_dict(), training_loss_history, best_metrics, best_epoch
