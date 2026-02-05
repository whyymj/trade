# -*- coding: utf-8 -*-
"""
LSTM 训练后验证：样本外测试与模型对比，仅在新模型显著优于旧模型时部署。

验证策略：
- 时间序列交叉验证：训练阶段已用 TimeSeriesSplit
- 样本外测试：保留最近 3 个月数据作为测试集
- 模型对比：新模型 vs 旧模型在测试集上的 F1 / MSE
- 决策规则：新模型 F1 提升不少于 min_f1_improvement 或 MSE 降低不少于 min_mse_ratio 才部署
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch
    from sklearn.metrics import f1_score, mean_squared_error
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    torch = None
    f1_score = None
    mean_squared_error = None

# 决策阈值：新模型 F1 至少比旧模型高 0.05，或 MSE 至多为旧模型的 90%
DEFAULT_MIN_F1_IMPROVEMENT = 0.05
DEFAULT_MIN_MSE_RATIO = 0.90  # new_mse <= old_mse * 0.90 即认为更好


def evaluate_model_on_holdout(
    model: Any,
    X_holdout: np.ndarray,
    y_dir: np.ndarray,
    y_mag: np.ndarray,
    device: Optional[Any] = None,
) -> dict[str, float]:
    """在样本外数据上评估模型，返回 accuracy, f1, mse, direction_accuracy。"""
    if not _AVAILABLE or model is None:
        return {}
    if device is None:
        device = torch.device("cpu")
    model.eval()
    X_t = torch.from_numpy(X_holdout).float().to(device)
    with torch.no_grad():
        logits, mag_pred = model(X_t)
    dir_pred = logits.argmax(dim=1).cpu().numpy()
    mag_pred_np = mag_pred.cpu().numpy()
    acc = float(np.mean((dir_pred == y_dir).astype(float)))
    f1 = float(f1_score(y_dir, dir_pred, average="binary", zero_division=0))
    mse = float(mean_squared_error(y_mag, mag_pred_np))
    mae = float(np.mean(np.abs(np.array(y_mag) - mag_pred_np)))
    rmse = float(np.sqrt(mse))
    return {
        "accuracy": acc,
        "f1": f1,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": acc,
    }


def should_deploy_new_model(
    new_metrics: dict[str, float],
    old_metrics: dict[str, float],
    *,
    min_f1_improvement: float = DEFAULT_MIN_F1_IMPROVEMENT,
    min_mse_ratio: float = DEFAULT_MIN_MSE_RATIO,
) -> tuple[bool, str]:
    """
    决策规则：仅当新模型显著优于旧模型时返回 (True, reason)。
    任一条件满足即部署：F1 提升 >= min_f1_improvement，或 MSE 降至旧模型的 min_mse_ratio 以下。
    """
    new_f1 = new_metrics.get("f1", 0)
    old_f1 = old_metrics.get("f1", 0)
    new_mse = new_metrics.get("mse", float("inf"))
    old_mse = old_metrics.get("mse", 1e-8)
    f1_ok = (new_f1 - old_f1) >= min_f1_improvement
    mse_ok = old_mse > 0 and (new_mse <= old_mse * min_mse_ratio)
    if f1_ok and mse_ok:
        return True, "F1 提升且 MSE 降低，部署新模型"
    if f1_ok:
        return True, "F1 提升达标，部署新模型"
    if mse_ok:
        return True, "MSE 降低达标，部署新模型"
    return False, f"新模型未显著优于旧模型 (新 F1={new_f1:.4f} vs 旧 {old_f1:.4f}, 新 MSE={new_mse:.6f} vs 旧 {old_mse:.6f})"
