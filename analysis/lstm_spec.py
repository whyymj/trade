# -*- coding: utf-8 -*-
"""
LSTM 训练与架构规格：集中导出参数与结构，用于生成文档与 AI 分析。

- get_training_spec()：返回完整规格字典（架构 + 训练参数 + 特征 + 损失等）
- generate_training_spec_document()：生成 docs/LSTM_TRAINING_SPEC.md 与 .json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.lstm_constants import DEFAULT_MODEL_DIR, FORECAST_DAYS
from analysis.lstm_volatility_features import VOLATILITY_FEATURE_NAMES

# 从 lstm_model 延迟取，避免循环依赖与 torch 强依赖
def _get_model_constants() -> dict[str, Any]:
    try:
        from analysis.lstm_model import (
            SEQ_LEN,
            BASE_FEATURE_NAMES,
            DEFAULT_FEATURE_NAMES,
            CPU_FRIENDLY_PARAM_GRID,
            CPU_FRIENDLY_CV_SPLITS,
        )
        return {
            "SEQ_LEN": SEQ_LEN,
            "BASE_FEATURE_NAMES": BASE_FEATURE_NAMES,
            "DEFAULT_FEATURE_NAMES": DEFAULT_FEATURE_NAMES,
            "CPU_FRIENDLY_PARAM_GRID": CPU_FRIENDLY_PARAM_GRID,
            "CPU_FRIENDLY_CV_SPLITS": CPU_FRIENDLY_CV_SPLITS,
        }
    except Exception:
        return {
            "SEQ_LEN": 60,
            "BASE_FEATURE_NAMES": [
                "close_norm", "volume_norm", "rsi", "macd_hist", "bb_position",
                "volatility", "obv_norm", "mfi", "aroon_up",
            ],
            "DEFAULT_FEATURE_NAMES": [],
            "CPU_FRIENDLY_PARAM_GRID": {"lr": [5e-4], "hidden_size": [32], "epochs": [25], "batch_size": 32},
            "CPU_FRIENDLY_CV_SPLITS": 3,
        }


def _get_improved_config() -> dict[str, Any]:
    try:
        from analysis.lstm_training import DEFAULT_IMPROVED_CONFIG
        return dict(DEFAULT_IMPROVED_CONFIG)
    except Exception:
        return {
            "初始学习率": 0.001,
            "权重衰减": 0.01,
            "梯度裁剪": 1.0,
            "早停耐心": 20,
            "最小学习率": 1e-5,
            "T_0": 50,
            "T_mult": 2,
            "使用数据增强": True,
            "noise_std": 0.01,
            "scale_low": 0.98,
            "scale_high": 1.02,
            "reg_loss_type": "full",
            "weight_cls": 0.9,
            "weight_reg": 4.0,
            "max_epochs": 100,
        }


def _get_default_cv_param_grid() -> dict[str, Any]:
    return {
        "lr": [1e-3, 5e-4],
        "hidden_size": [32, 64],
        "epochs": [30, 50],
        "batch_size": 32,
    }


def _get_validation_defaults() -> dict[str, Any]:
    try:
        from analysis.lstm_validation import DEFAULT_MIN_F1_IMPROVEMENT, DEFAULT_MIN_MSE_RATIO
        return {
            "min_f1_improvement": DEFAULT_MIN_F1_IMPROVEMENT,
            "min_mse_ratio": DEFAULT_MIN_MSE_RATIO,
            "holdout_ratio_description": "最近约 3 个月或至少 10 条",
        }
    except Exception:
        return {
            "min_f1_improvement": 0.05,
            "min_mse_ratio": 0.90,
            "holdout_ratio_description": "最近约 3 个月或至少 10 条",
        }


def get_training_spec() -> dict[str, Any]:
    """
    返回当前 LSTM 模块的完整训练与架构规格，便于文档生成与 AI 分析。
    所有数值与列表与运行时一致。
    """
    mc = _get_model_constants()
    improved = _get_improved_config()
    validation = _get_validation_defaults()

    n_base = len(mc["BASE_FEATURE_NAMES"])
    n_vol = len(VOLATILITY_FEATURE_NAMES)
    n_total = n_base + n_vol

    spec = {
        "version": "1.0",
        "purpose": "LSTM 训练与架构规格，供文档与 AI 分析使用",
        "data": {
            "input": {
                "sequence_length": mc["SEQ_LEN"],
                "description": "过去 N 个交易日日线",
                "min_trading_days_required": mc["SEQ_LEN"] + FORECAST_DAYS,
            },
            "output": {
                "forecast_days": FORECAST_DAYS,
                "direction": "二分类 0/1（跌/涨），基于未来 5 日累计涨跌幅",
                "magnitude": "回归，未来 5 日逐日收益率，形状 (batch, 5)",
            },
            "features": {
                "count_total": n_total,
                "count_base": n_base,
                "count_volatility": n_vol,
                "base_feature_names": mc["BASE_FEATURE_NAMES"],
                "volatility_feature_names": list(VOLATILITY_FEATURE_NAMES),
                "feature_names_full": mc["BASE_FEATURE_NAMES"] + list(VOLATILITY_FEATURE_NAMES),
                "base_feature_sources": {
                    "close_norm": "收盘价在 SEQ_LEN 内 min-max 归一化",
                    "volume_norm": "成交量在 SEQ_LEN 内 min-max 归一化",
                    "rsi": "RSI(14)，归一化到 [0,1]",
                    "macd_hist": "MACD(12,26,9) 柱，裁剪后按最大值缩放",
                    "bb_position": "(close-mid)/(upper-lower)，布林带位置",
                    "volatility": "20 日滚动波动率(年化)，裁剪到 [0,2]/2",
                    "obv_norm": "OBV 在 SEQ_LEN 内 min-max 归一化",
                    "mfi": "MFI(14)，归一化到 [0,1]",
                    "aroon_up": "Aroon(20) up，归一化到 [0,1]",
                },
            },
        },
        "architecture": {
            "model_types": ["LSTMDualHead", "LSTMDualHeadEnhanced"],
            "LSTMDualHead": {
                "description": "基础：单层单向 LSTM + Dropout + 双头",
                "input_size": n_total,
                "hidden_size_default": 64,
                "num_layers": 1,
                "dropout": 0.3,
                "batch_first": True,
                "num_classes": 2,
                "n_magnitude_outputs": FORECAST_DAYS,
                "structure": [
                    "LSTM(input_size, hidden_size, num_layers=1, dropout=0 if num_layers<=1 else dropout)",
                    "取 last_h = lstm_out[:, -1]",
                    "Dropout -> Linear(hidden, hidden) ReLU -> Dropout",
                    "fc_direction: Linear(hidden, 2)",
                    "fc_magnitude: Linear(hidden, 5)",
                ],
            },
            "LSTMDualHeadEnhanced": {
                "description": "增强：双向 LSTM + 序列注意力 + 跳跃连接 + 更深全连接",
                "input_size": n_total,
                "hidden_size": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "bidirectional": True,
                "hidden_out_per_direction": 128,
                "attention": "SeqAttention(hidden*2)：对 seq_len 维 softmax 得时间步权重",
                "skip_connection": "Linear(input_size, hidden*2) + ReLU，输入最后一帧",
                "combined": "concat(attention_context, skip) -> (batch, hidden*4)",
                "fc_shared": "Linear(256, 256) -> LayerNorm -> ReLU -> Dropout -> Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU",
                "fc_direction": "Linear(64, 2)",
                "fc_magnitude": "Linear(64, 5)",
                "init_weights": "LSTM: xavier/orthogonal/constant; fc_shared: xavier; direction/magnitude: normal(0,0.01)",
            },
            "default_use_enhanced": True,
        },
        "training": {
            "default_optimizer": "AdamW (改进策略) 或 Adam (非改进)",
            "improved_strategy": {
                "name": "改进的训练策略",
                "config": improved,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=最小学习率)",
                "early_stopping_metric": "波动匹配度 = 1 - |std(预测)-std(实际)|/std(实际)，取验证集上最佳",
                "data_augmentation": "时间序列数据增强：高斯噪声(noise_std) + 逐样本缩放(scale_low~scale_high)",
                "gradient_clip": "梯度裁剪 max_norm=梯度裁剪",
            },
            "cross_validation": {
                "method": "TimeSeriesSplit",
                "default_n_splits": 5,
                "cpu_friendly_n_splits": mc["CPU_FRIENDLY_CV_SPLITS"],
                "default_param_grid": _get_default_cv_param_grid(),
                "cpu_friendly_param_grid": mc["CPU_FRIENDLY_PARAM_GRID"],
                "scoring": "score = avg_f1 - 0.1 * log1p(avg_mse)",
            },
            "train_and_save_defaults": {
                "lr": 5e-4,
                "hidden_size": 64,
                "epochs": 50,
                "batch_size": 32,
                "use_enhanced_model": True,
                "reg_loss_type": "full",
                "use_improved_training": True,
            },
            "regression_loss_options": {
                "mse": "nn.MSE（易导致预测过平滑）",
                "huber": "nn.HuberLoss(delta=1.0)",
                "volatility": "波动增强损失：MSE + 惩罚预测波动过小，最小波动比例 0.7，惩罚系数 0.2",
                "full": "改进的损失函数：MSE + 整体波动匹配(0.35) + 每步波动匹配(0.15) + 方向一致(0.1) + 序列内方向(0.1) + 分位数(0.05)",
            },
            "classification_loss": "CrossEntropyLoss",
            "loss_weights_default": {"weight_cls": 0.9, "weight_reg": 4.0},
        },
        "validation": {
            "post_training_holdout": validation["holdout_ratio_description"],
            "deploy_rule": "仅当新模型显著优于旧模型时部署",
            "min_f1_improvement": validation["min_f1_improvement"],
            "min_mse_ratio": validation["min_mse_ratio"],
            "evaluate_metrics": ["accuracy", "f1", "mse", "mae", "rmse", "direction_accuracy"],
        },
        "paths": {
            "default_model_dir": str(DEFAULT_MODEL_DIR),
        },
    }
    return spec


def generate_training_spec_document(
    docs_dir: Path | str | None = None,
    write_json: bool = True,
) -> tuple[str, str | None]:
    """
    生成 LSTM 训练规格文档（Markdown + 可选 JSON）。
    返回 (md_path, json_path_or_none)。
    """
    docs_dir = Path(docs_dir) if docs_dir else Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    spec = get_training_spec()

    md_path = docs_dir / "LSTM_TRAINING_SPEC.md"
    md_lines = [
        "# LSTM 训练与架构规格",
        "",
        "本文档由 `analysis.lstm_spec.generate_training_spec_document()` 生成，便于 AI 与人工分析训练参数与模型结构。",
        "",
        "---",
        "",
        "## 1. 数据规格",
        "",
        "| 项 | 值 |",
        "|----|----|",
        f"| 输入序列长度 (SEQ_LEN) | {spec['data']['input']['sequence_length']} 个交易日 |",
        f"| 最少所需交易日 | {spec['data']['input']['min_trading_days_required']} |",
        f"| 预测未来天数 (FORECAST_DAYS) | {spec['data']['output']['forecast_days']} |",
        "| 方向目标 | 二分类 0/1（跌/涨），基于未来 5 日累计涨跌幅 |",
        "| 幅度目标 | 回归，未来 5 日逐日收益率，形状 (batch, 5) |",
        "",
        "## 2. 特征列表",
        "",
        f"总特征数：**{spec['data']['features']['count_total']}**（基础 {spec['data']['features']['count_base']} + 波动增强 {spec['data']['features']['count_volatility']}）",
        "",
        "### 2.1 基础特征 (base)",
        "",
        "```",
        json.dumps(spec["data"]["features"]["base_feature_names"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 2.2 波动增强特征 (volatility)",
        "",
        "```",
        json.dumps(spec["data"]["features"]["volatility_feature_names"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 2.3 基础特征说明",
        "",
        "| 特征名 | 说明 |",
        "|--------|------|",
    ]
    for k, v in spec["data"]["features"]["base_feature_sources"].items():
        md_lines.append(f"| {k} | {v} |")
    md_lines.extend([
        "",
        "## 3. 模型架构",
        "",
        "### 3.1 LSTMDualHead（基础）",
        "",
        "- 单层单向 LSTM → 取最后时间步 → Dropout → 全连接 → 双头（分类 2 类 + 回归 5 维）",
        "- 默认 hidden_size=64, num_layers=1, dropout=0.3",
        "",
        "### 3.2 LSTMDualHeadEnhanced（增强，默认）",
        "",
        "- 双向 LSTM(hidden=128, layers=3) → 序列注意力 → 与最后一帧跳跃连接 concat → 多层全连接 → 双头",
        "- 输入维度 = 特征总数；输出方向 2 类、幅度 5 维",
        "",
        "## 4. 训练参数",
        "",
        "### 4.1 改进训练策略默认配置",
        "",
        "```json",
        json.dumps(spec["training"]["improved_strategy"]["config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### 4.2 交叉验证",
        "",
        "- 方法：TimeSeriesSplit",
        f"- 默认折数：{spec['training']['cross_validation']['default_n_splits']}，快速训练：{spec['training']['cross_validation']['cpu_friendly_n_splits']}",
        "- 默认超参网格：",
        "```json",
        json.dumps(spec["training"]["cross_validation"]["default_param_grid"], indent=2),
        "```",
        "- 快速训练超参网格：",
        "```json",
        json.dumps(spec["training"]["cross_validation"]["cpu_friendly_param_grid"], indent=2),
        "```",
        "",
        "### 4.3 回归损失选项",
        "",
        "| kind | 说明 |",
        "|------|------|",
    ])
    for k, v in spec["training"]["regression_loss_options"].items():
        md_lines.append(f"| {k} | {v} |")
    md_lines.extend([
        "",
        "### 4.4 训练后验证与部署",
        "",
        f"- 留出集：{spec['validation']['post_training_holdout']}",
        f"- 部署条件：新模型 F1 提升 ≥ {spec['validation']['min_f1_improvement']} 或 MSE ≤ 旧模型 × {spec['validation']['min_mse_ratio']}",
        "",
        "## 5. 完整规格 JSON（机器可读）",
        "",
        "以下为 `get_training_spec()` 的完整输出，便于 AI 或脚本解析。",
        "",
        "```json",
        json.dumps(spec, ensure_ascii=False, indent=2),
        "```",
        "",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path = None
    if write_json:
        json_path = docs_dir / "LSTM_TRAINING_SPEC.json"
        json_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(md_path), str(json_path) if json_path else None
