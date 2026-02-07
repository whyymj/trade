# -*- coding: utf-8 -*-
"""
LSTM 预测平淡问题诊断：波动率、模型容量、特征相关性、损失曲线。
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np


def 计算特征相关性(
    特征: np.ndarray,
    实际值: np.ndarray,
    使用最后时间步: bool = True,
) -> np.ndarray:
    """
    计算每个特征与目标（实际涨跌幅）的 Pearson 相关系数。

    - 特征: (n_samples, seq_len, n_features) 或 (n_samples, n_features)
    - 实际值: (n_samples,)，与样本一一对应
    - 使用最后时间步: 当特征为 3 维时，True 用最后一帧 [:, -1, :]，False 用整序列均值
    - 返回: (n_features,) 每个特征与 实际值 的相关系数，NaN 会被置为 0
    """
    实际值 = np.asarray(实际值).ravel()
    n = len(实际值)
    if 特征.ndim == 3:
        if 使用最后时间步:
            F = 特征[:, -1, :]  # (n_samples, n_features)
        else:
            F = 特征.mean(axis=1)  # (n_samples, n_features)
    else:
        F = np.asarray(特征)
    if F.shape[0] != n:
        return np.zeros(F.shape[1], dtype=np.float64)
    n_features = F.shape[1]
    out = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        col = F[:, j].astype(np.float64)
        if np.std(col) < 1e-12 or np.std(实际值) < 1e-12:
            out[j] = 0.0
            continue
        r = np.corrcoef(col, 实际值)[0, 1]
        out[j] = r if not np.isnan(r) else 0.0
    return out


def _模型参数数量(模型: Any) -> int:
    """统计 PyTorch 模型可训练参数总数。"""
    try:
        import torch
        if hasattr(模型, "parameters"):
            return sum(p.numel() for p in 模型.parameters())
        return 0
    except Exception:
        return 0


def 诊断LSTM预测平淡问题(
    预测值: Union[np.ndarray, list],
    实际值: Union[np.ndarray, list],
    模型: Any,
    训练数据: Union[dict, Any],
) -> dict[str, str]:
    """
    快速诊断预测曲线过于平坦的可能原因。

    参数
    -----
    预测值 : 模型输出的涨跌幅预测，一维
    实际值 : 真实涨跌幅，一维，与 预测值 等长
    模型   : PyTorch nn.Module（如 LSTMDualHead），用于统计参数量；
             若模型有属性 训练损失（列表），则用于过度平滑检查
    训练数据 : 支持两种形式：
              - dict：需含 'X' 或 '特征'（(n, seq_len, n_features)），可选 '训练损失'（列表）
              - 任意对象：需有 .特征 属性，可选 .训练损失

    返回
    -----
    诊断结果 : 仅包含“有问题”的项，键为问题类型，值为描述文案
    """
    预测值 = np.asarray(预测值, dtype=np.float64).ravel()
    实际值 = np.asarray(实际值, dtype=np.float64).ravel()
    if len(预测值) != len(实际值) or len(预测值) == 0:
        return {"输入错误": "预测值与实际值长度不一致或为空"}

    诊断结果: dict[str, str] = {}

    # 1. 预测 vs 实际 波动率
    预测波动率 = float(np.std(预测值))
    实际波动率 = float(np.std(实际值))
    波动率比 = 预测波动率 / (实际波动率 + 1e-12)
    if 实际波动率 < 1e-12:
        诊断结果["波动率问题"] = "实际值几乎无波动，无法据此判断预测是否过平"
    elif 波动率比 < 0.3:
        诊断结果["波动率问题"] = (
            f"预测波动率过低（仅为实际值的 {波动率比 * 100:.1f}%），"
            "预测曲线易呈近似直线"
        )

    # 2. 模型容量
    参数数量 = _模型参数数量(模型)
    if 0 < 参数数量 < 10000:
        诊断结果["模型容量"] = f"模型可能过小（约 {参数数量} 个参数），表达能力有限"

    # 3. 输入特征与目标的相关性
    特征矩阵 = None
    平均相关性_val: Optional[float] = None
    if isinstance(训练数据, dict):
        特征矩阵 = 训练数据.get("X") or 训练数据.get("特征")
    elif hasattr(训练数据, "特征"):
        特征矩阵 = getattr(训练数据, "特征")
    if 特征矩阵 is not None:
        特征矩阵 = np.asarray(特征矩阵)
        n_align = min(特征矩阵.shape[0], len(实际值))
        特征相关性 = 计算特征相关性(
            特征矩阵[:n_align], 实际值[:n_align]
        )
        平均相关性_val = float(np.mean(np.abs(特征相关性)))
        if 平均相关性_val < 0.2:
            诊断结果["特征问题"] = (
                f"特征与目标平均相关性较低（约 {平均相关性_val:.3f}），"
                "可能难以支撑幅度预测"
            )

    # 4. 训练损失是否过低（过度平滑风险）
    训练损失列表: Optional[list] = None
    末损失: Optional[float] = None
    if isinstance(训练数据, dict):
        训练损失列表 = 训练数据.get("训练损失")
    elif hasattr(训练数据, "训练损失"):
        训练损失列表 = getattr(训练数据, "训练损失", None)
    if 训练损失列表 is None and hasattr(模型, "训练损失"):
        训练损失列表 = getattr(模型, "训练损失", None)
    if 训练损失列表 and len(训练损失列表) > 0:
        末损失 = float(训练损失列表[-1])
        if 末损失 < 0.01:
            诊断结果["过度拟合"] = (
                f"训练损失过低（{末损失:.4f}），预测可能过度平滑、趋近常数"
            )

    # 5. 始终附加摘要，便于前端始终展示诊断结果
    平均相关性_str = f"，特征相关性≈{平均相关性_val:.3f}" if 平均相关性_val is not None else ""
    诊断结果["诊断摘要"] = (
        f"预测波动率={预测波动率:.4f}，实际={实际波动率:.4f}，"
        f"波动率比={波动率比:.1%}，参数量={参数数量}"
        + 平均相关性_str
        + (f"，末轮损失={末损失:.4f}" if 末损失 is not None else "")
    )
    return 诊断结果
