# -*- coding: utf-8 -*-
"""
LSTM 回归头损失函数：缓解预测过度平滑。

- MSE：传统，易导致预测趋近均值
- Huber：对异常值更鲁棒
- 波动增强损失：MSE + 预测波动不低于实际波动一半的惩罚
- 改进的损失函数：MSE + 波动匹配 + 方向一致性 + 分位数（尾部）
"""
from __future__ import annotations

from typing import Any, Optional, Union

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _safe_std(x: "torch.Tensor", dim: Optional[int] = None) -> "torch.Tensor":
    s = torch.std(x, dim=dim, unbiased=False)
    return s.clamp(min=1e-8)


if _TORCH_AVAILABLE and nn is not None:

    class 波动增强损失(nn.Module):
        """MSE + 惩罚预测波动过小，鼓励预测曲线有一定波动。"""

        def __init__(self, 波动惩罚系数: float = 0.1, 最小波动比例: float = 0.5):
            super().__init__()
            self.波动惩罚系数 = 波动惩罚系数
            self.最小波动比例 = 最小波动比例

        def forward(
            self,
            预测值: torch.Tensor,
            实际值: torch.Tensor,
        ) -> torch.Tensor:
            mse = nn.functional.mse_loss(预测值, 实际值)
            预测波动 = _safe_std(预测值)
            目标波动 = _safe_std(实际值)
            波动惩罚 = torch.relu(目标波动 * self.最小波动比例 - 预测波动)
            return mse + self.波动惩罚系数 * 波动惩罚

    class 改进的损失函数(nn.Module):
        """
        MSE 或 Huber 为主 + 波动匹配（防塌缩为常数）+ 方向一致性 + 分位数。
        增加「每步波动匹配」与「序列内方向」：让多步预测每条曲线有起伏、跟随真实 5 日走势，缓解预测像直线。
        use_huber=True 时主项用 Huber，减轻 MSE 对「预测均值」的偏好，有利于学到幅度。
        """

        def __init__(
            self,
            波动权重: float = 0.55,
            方向权重: float = 0.14,
            分位数权重: float = 0.08,
            每步波动权重: float = 0.28,
            序列内方向权重: float = 0.14,
            use_huber: bool = False,
            huber_delta: float = 0.1,
        ):
            super().__init__()
            self.波动权重 = 波动权重
            self.方向权重 = 方向权重
            self.分位数权重 = 分位数权重
            self.每步波动权重 = 每步波动权重
            self.序列内方向权重 = 序列内方向权重
            self.use_huber = use_huber
            self.huber_delta = huber_delta

        def forward(
            self,
            预测值: torch.Tensor,
            实际值: torch.Tensor,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
            if self.use_huber:
                base_loss = nn.functional.huber_loss(预测值, 实际值, delta=self.huber_delta)
            else:
                base_loss = nn.functional.mse_loss(预测值, 实际值)
            mse_loss = nn.functional.mse_loss(预测值, 实际值)  # 仅用于 aux 记录

            # 整体波动匹配（防整条曲线塌成常数）
            预测波动 = _safe_std(预测值)
            实际波动 = _safe_std(实际值)
            波动损失 = (torch.abs(预测波动 - 实际波动) / 实际波动).clamp(max=10.0)

            # 每步波动匹配：多步输出 (B,5) 时，每列预测的 std 与真实每列 std 匹配，避免 5 日预测都挤在一起
            每步波动损失 = torch.tensor(0.0, device=预测值.device, dtype=预测值.dtype)
            if 预测值.dim() >= 2 and 预测值.shape[-1] > 1:
                pred_std_per_step = _safe_std(预测值, dim=0)  # (n_steps,)
                true_std_per_step = _safe_std(实际值, dim=0)
                每步波动损失 = (torch.abs(pred_std_per_step - true_std_per_step) / true_std_per_step).clamp(max=10.0).mean()

            # 序列方向：batch 内相邻样本变化方向一致
            预测方向 = torch.sign((预测值[1:] - 预测值[:-1]).detach())
            实际方向 = torch.sign(实际值[1:] - 实际值[:-1])
            方向一致 = ((预测方向 == 实际方向).float()).mean()
            方向损失 = 1.0 - 方向一致

            # 序列内方向：每个样本的 5 步预测，相邻步变化方向与真实一致，让预测曲线跟随真实拐弯
            序列内方向损失 = torch.tensor(0.0, device=预测值.device, dtype=预测值.dtype)
            if 预测值.dim() >= 2 and 预测值.shape[-1] > 1:
                pred_diff = 预测值[:, 1:] - 预测值[:, :-1]  # (B, n_steps-1)
                true_diff = 实际值[:, 1:] - 实际值[:, :-1]
                seq_dir_match = (torch.sign(pred_diff.detach()) == torch.sign(true_diff)).float().mean()
                序列内方向损失 = 1.0 - seq_dir_match

            def _分位数损失(err: torch.Tensor, q: float) -> torch.Tensor:
                return torch.max((q - 1) * err, q * err).mean()

            误差 = 实际值 - 预测值
            q50 = _分位数损失(误差, 0.5)
            q75 = _分位数损失(误差, 0.75)
            q25 = _分位数损失(误差, 0.25)
            分位数总损失 = (q50 + q75 + q25) / 3.0

            总损失 = (
                base_loss
                + self.波动权重 * 波动损失
                + self.每步波动权重 * 每步波动损失
                + self.方向权重 * 方向损失
                + self.序列内方向权重 * 序列内方向损失
                + self.分位数权重 * 分位数总损失
            )
            aux = {
                "mse": mse_loss.item(),
                "波动损失": 波动损失.item(),
                "每步波动损失": 每步波动损失.item() if 预测值.dim() >= 2 else 0.0,
                "方向损失": 方向损失.item(),
                "序列内方向损失": 序列内方向损失.item() if 预测值.dim() >= 2 else 0.0,
                "分位数损失": 分位数总损失.item(),
            }
            return 总损失, aux

else:
    波动增强损失 = None  # type: ignore[assignment, misc]
    改进的损失函数 = None  # type: ignore[assignment, misc]


def get_regression_criterion(
    kind: str = "mse",
    **kwargs: Any,
) -> Optional[Any]:
    """
    获取回归头损失函数。用于 train_epoch 的 reg_criterion。
    kind: "mse" | "huber" | "volatility" | "full" | "full_huber"
    - mse: 默认 MSE
    - huber: nn.HuberLoss(delta=kwargs.get("delta", 1.0))
    - volatility: 波动增强损失
    - full: 改进的损失函数（MSE + 波动+方向+分位数）
    - full_huber: 同上但主项用 Huber，减轻对均值的偏好，利于幅度学习
    未安装 torch 时返回 None。
    """
    if not _TORCH_AVAILABLE or nn is None or 波动增强损失 is None or 改进的损失函数 is None:
        return None
    if kind == "mse":
        return None
    if kind == "huber":
        delta = kwargs.get("delta", 1.0)
        return nn.HuberLoss(delta=delta)
    if kind == "volatility":
        return 波动增强损失(
            波动惩罚系数=kwargs.get("波动惩罚系数", 0.2),
            最小波动比例=kwargs.get("最小波动比例", 0.7),
        )
    if kind == "full":
        return 改进的损失函数(
            波动权重=kwargs.get("波动权重", 0.55),
            方向权重=kwargs.get("方向权重", 0.14),
            分位数权重=kwargs.get("分位数权重", 0.08),
            每步波动权重=kwargs.get("每步波动权重", 0.28),
            序列内方向权重=kwargs.get("序列内方向权重", 0.14),
            use_huber=False,
        )
    if kind == "full_huber":
        return 改进的损失函数(
            波动权重=kwargs.get("波动权重", 0.55),
            方向权重=kwargs.get("方向权重", 0.14),
            分位数权重=kwargs.get("分位数权重", 0.08),
            每步波动权重=kwargs.get("每步波动权重", 0.28),
            序列内方向权重=kwargs.get("序列内方向权重", 0.14),
            use_huber=True,
            huber_delta=kwargs.get("huber_delta", 0.1),
        )
    return None
