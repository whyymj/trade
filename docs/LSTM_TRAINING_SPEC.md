# LSTM 训练与架构规格

本文档由 `analysis.lstm_spec.generate_training_spec_document()` 生成，便于 AI 与人工分析训练参数与模型结构。

---

## 1. 数据规格

| 项 | 值 |
|----|----|
| 输入序列长度 (SEQ_LEN) | 60 个交易日 |
| 最少所需交易日 | 65 |
| 预测未来天数 (FORECAST_DAYS) | 5 |
| 方向目标 | 二分类 0/1（跌/涨），基于未来 5 日累计涨跌幅 |
| 幅度目标 | 回归，未来 5 日逐日收益率，形状 (batch, 5) |

## 2. 特征列表

总特征数：**21**（基础 9 + 波动增强 12）

### 2.1 基础特征 (base)

```
[
  "close_norm",
  "volume_norm",
  "rsi",
  "macd_hist",
  "bb_position",
  "volatility",
  "obv_norm",
  "mfi",
  "aroon_up"
]
```

### 2.2 波动增强特征 (volatility)

```
[
  "日收益率",
  "收益率绝对值",
  "收益率符号",
  "日内振幅",
  "开盘缺口",
  "历史波动率_5日",
  "历史波动率_20日",
  "波动率比",
  "大涨标记",
  "大跌标记",
  "成交量变化率",
  "量价背离"
]
```

### 2.3 基础特征说明

| 特征名 | 说明 |
|--------|------|
| close_norm | 收盘价在 SEQ_LEN 内 min-max 归一化 |
| volume_norm | 成交量在 SEQ_LEN 内 min-max 归一化 |
| rsi | RSI(14)，归一化到 [0,1] |
| macd_hist | MACD(12,26,9) 柱，裁剪后按最大值缩放 |
| bb_position | (close-mid)/(upper-lower)，布林带位置 |
| volatility | 20 日滚动波动率(年化)，裁剪到 [0,2]/2 |
| obv_norm | OBV 在 SEQ_LEN 内 min-max 归一化 |
| mfi | MFI(14)，归一化到 [0,1] |
| aroon_up | Aroon(20) up，归一化到 [0,1] |

## 3. 模型架构

### 3.1 LSTMDualHead（基础）

- 单层单向 LSTM → 取最后时间步 → Dropout → 全连接 → 双头（分类 2 类 + 回归 5 维）
- 默认 hidden_size=64, num_layers=1, dropout=0.3

### 3.2 LSTMDualHeadEnhanced（增强，默认）

- 双向 LSTM(hidden=128, layers=3) → 序列注意力 → 与最后一帧跳跃连接 concat → 多层全连接 → 双头
- 输入维度 = 特征总数；输出方向 2 类、幅度 5 维

## 4. 训练参数

### 4.1 改进训练策略默认配置

```json
{
  "初始学习率": 0.001,
  "权重衰减": 0.01,
  "梯度裁剪": 1.0,
  "早停耐心": 20,
  "最小学习率": 1e-05,
  "T_0": 50,
  "T_mult": 2,
  "使用数据增强": true,
  "noise_std": 0.01,
  "scale_low": 0.98,
  "scale_high": 1.02,
  "reg_loss_type": "full",
  "weight_cls": 1.0,
  "weight_reg": 1.0,
  "max_epochs": 60
}
```

### 4.2 交叉验证

- 方法：TimeSeriesSplit
- 默认折数：5，快速训练：3
- 默认超参网格：
```json
{
  "lr": [
    0.001,
    0.0005
  ],
  "hidden_size": [
    32,
    64
  ],
  "epochs": [
    30,
    50
  ],
  "batch_size": 32
}
```
- 快速训练超参网格：
```json
{
  "lr": [
    0.0005
  ],
  "hidden_size": [
    32
  ],
  "epochs": [
    25
  ],
  "batch_size": 32
}
```

### 4.3 回归损失选项

| kind | 说明 |
|------|------|
| mse | nn.MSE（易导致预测过平滑） |
| huber | nn.HuberLoss(delta=1.0) |
| volatility | 波动增强损失：MSE + 惩罚预测波动过小，最小波动比例 0.7，惩罚系数 0.2 |
| full | 改进的损失函数：MSE + 整体波动匹配(0.35) + 每步波动匹配(0.15) + 方向一致(0.1) + 序列内方向(0.1) + 分位数(0.05) |

### 4.4 训练后验证与部署

- 留出集：最近约 3 个月或至少 10 条
- 部署条件：新模型 F1 提升 ≥ 0.05 或 MSE ≤ 旧模型 × 0.9

## 5. 完整规格 JSON（机器可读）

以下为 `get_training_spec()` 的完整输出，便于 AI 或脚本解析。

```json
{
  "version": "1.0",
  "purpose": "LSTM 训练与架构规格，供文档与 AI 分析使用",
  "data": {
    "input": {
      "sequence_length": 60,
      "description": "过去 N 个交易日日线",
      "min_trading_days_required": 65
    },
    "output": {
      "forecast_days": 5,
      "direction": "二分类 0/1（跌/涨），基于未来 5 日累计涨跌幅",
      "magnitude": "回归，未来 5 日逐日收益率，形状 (batch, 5)"
    },
    "features": {
      "count_total": 21,
      "count_base": 9,
      "count_volatility": 12,
      "base_feature_names": [
        "close_norm",
        "volume_norm",
        "rsi",
        "macd_hist",
        "bb_position",
        "volatility",
        "obv_norm",
        "mfi",
        "aroon_up"
      ],
      "volatility_feature_names": [
        "日收益率",
        "收益率绝对值",
        "收益率符号",
        "日内振幅",
        "开盘缺口",
        "历史波动率_5日",
        "历史波动率_20日",
        "波动率比",
        "大涨标记",
        "大跌标记",
        "成交量变化率",
        "量价背离"
      ],
      "feature_names_full": [
        "close_norm",
        "volume_norm",
        "rsi",
        "macd_hist",
        "bb_position",
        "volatility",
        "obv_norm",
        "mfi",
        "aroon_up",
        "日收益率",
        "收益率绝对值",
        "收益率符号",
        "日内振幅",
        "开盘缺口",
        "历史波动率_5日",
        "历史波动率_20日",
        "波动率比",
        "大涨标记",
        "大跌标记",
        "成交量变化率",
        "量价背离"
      ],
      "base_feature_sources": {
        "close_norm": "收盘价在 SEQ_LEN 内 min-max 归一化",
        "volume_norm": "成交量在 SEQ_LEN 内 min-max 归一化",
        "rsi": "RSI(14)，归一化到 [0,1]",
        "macd_hist": "MACD(12,26,9) 柱，裁剪后按最大值缩放",
        "bb_position": "(close-mid)/(upper-lower)，布林带位置",
        "volatility": "20 日滚动波动率(年化)，裁剪到 [0,2]/2",
        "obv_norm": "OBV 在 SEQ_LEN 内 min-max 归一化",
        "mfi": "MFI(14)，归一化到 [0,1]",
        "aroon_up": "Aroon(20) up，归一化到 [0,1]"
      }
    }
  },
  "architecture": {
    "model_types": [
      "LSTMDualHead",
      "LSTMDualHeadEnhanced"
    ],
    "LSTMDualHead": {
      "description": "基础：单层单向 LSTM + Dropout + 双头",
      "input_size": 21,
      "hidden_size_default": 64,
      "num_layers": 1,
      "dropout": 0.3,
      "batch_first": true,
      "num_classes": 2,
      "n_magnitude_outputs": 5,
      "structure": [
        "LSTM(input_size, hidden_size, num_layers=1, dropout=0 if num_layers<=1 else dropout)",
        "取 last_h = lstm_out[:, -1]",
        "Dropout -> Linear(hidden, hidden) ReLU -> Dropout",
        "fc_direction: Linear(hidden, 2)",
        "fc_magnitude: Linear(hidden, 5)"
      ]
    },
    "LSTMDualHeadEnhanced": {
      "description": "增强：双向 LSTM + 序列注意力 + 跳跃连接 + 更深全连接",
      "input_size": 21,
      "hidden_size": 128,
      "num_layers": 3,
      "dropout": 0.2,
      "bidirectional": true,
      "hidden_out_per_direction": 128,
      "attention": "SeqAttention(hidden*2)：对 seq_len 维 softmax 得时间步权重",
      "skip_connection": "Linear(input_size, hidden*2) + ReLU，输入最后一帧",
      "combined": "concat(attention_context, skip) -> (batch, hidden*4)",
      "fc_shared": "Linear(256, 256) -> LayerNorm -> ReLU -> Dropout -> Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 64) -> ReLU",
      "fc_direction": "Linear(64, 2)",
      "fc_magnitude": "Linear(64, 5)",
      "init_weights": "LSTM: xavier/orthogonal/constant; fc_shared: xavier; direction/magnitude: normal(0,0.01)"
    },
    "default_use_enhanced": true
  },
  "training": {
    "default_optimizer": "AdamW (改进策略) 或 Adam (非改进)",
    "improved_strategy": {
      "name": "改进的训练策略",
      "config": {
        "初始学习率": 0.001,
        "权重衰减": 0.01,
        "梯度裁剪": 1.0,
        "早停耐心": 20,
        "最小学习率": 1e-05,
        "T_0": 50,
        "T_mult": 2,
        "使用数据增强": true,
        "noise_std": 0.01,
        "scale_low": 0.98,
        "scale_high": 1.02,
        "reg_loss_type": "full",
        "weight_cls": 1.0,
        "weight_reg": 1.0,
        "max_epochs": 60
      },
      "optimizer": "AdamW",
      "scheduler": "CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=最小学习率)",
      "early_stopping_metric": "波动匹配度 = 1 - |std(预测)-std(实际)|/std(实际)，取验证集上最佳",
      "data_augmentation": "时间序列数据增强：高斯噪声(noise_std) + 逐样本缩放(scale_low~scale_high)",
      "gradient_clip": "梯度裁剪 max_norm=梯度裁剪"
    },
    "cross_validation": {
      "method": "TimeSeriesSplit",
      "default_n_splits": 5,
      "cpu_friendly_n_splits": 3,
      "default_param_grid": {
        "lr": [
          0.001,
          0.0005
        ],
        "hidden_size": [
          32,
          64
        ],
        "epochs": [
          30,
          50
        ],
        "batch_size": 32
      },
      "cpu_friendly_param_grid": {
        "lr": [
          0.0005
        ],
        "hidden_size": [
          32
        ],
        "epochs": [
          25
        ],
        "batch_size": 32
      },
      "scoring": "score = avg_f1 - 0.1 * log1p(avg_mse)"
    },
    "train_and_save_defaults": {
      "lr": 0.0005,
      "hidden_size": 64,
      "epochs": 50,
      "batch_size": 32,
      "use_enhanced_model": true,
      "reg_loss_type": "full",
      "use_improved_training": true
    },
    "regression_loss_options": {
      "mse": "nn.MSE（易导致预测过平滑）",
      "huber": "nn.HuberLoss(delta=1.0)",
      "volatility": "波动增强损失：MSE + 惩罚预测波动过小，最小波动比例 0.7，惩罚系数 0.2",
      "full": "改进的损失函数：MSE + 整体波动匹配(0.35) + 每步波动匹配(0.15) + 方向一致(0.1) + 序列内方向(0.1) + 分位数(0.05)"
    },
    "classification_loss": "CrossEntropyLoss",
    "loss_weights_default": {
      "weight_cls": 1.0,
      "weight_reg": 1.0
    }
  },
  "validation": {
    "post_training_holdout": "最近约 3 个月或至少 10 条",
    "deploy_rule": "仅当新模型显著优于旧模型时部署",
    "min_f1_improvement": 0.05,
    "min_mse_ratio": 0.9,
    "evaluate_metrics": [
      "accuracy",
      "f1",
      "mse",
      "mae",
      "rmse",
      "direction_accuracy"
    ]
  },
  "paths": {
    "default_model_dir": "/Users/wuhao/Desktop/python/trade/analysis_temp/lstm"
  }
}
```
