# -*- coding: utf-8 -*-
"""
股票数据分析模块包。

模块列表:
- time_domain: 时域与统计特征分析
- frequency_domain: 频域（周期性）分析
- arima_model: 预测模型构建（ARIMA）
- shape_similarity: 几何形态与相似度匹配
- complexity: 非线性与复杂度分析
- full_report: 综合分析报告生成器
- lstm_model: LSTM 深度学习预测（方向+幅度）、交叉验证、SHAP 可解释性
"""
import sys

# 仅做清理时不必加载 matplotlib 等重量级依赖，直接清理并退出
if any(a in ("--cleanup", "-c") for a in sys.argv[1:]):
    from analysis.cleanup_temp import cleanup_analysis_temp_dirs
    n = cleanup_analysis_temp_dirs()
    print(f"已清理 {n} 个分析临时目录")
    sys.exit(0)

from analysis.time_domain import (
    analyze_time_domain,
    calc_basic_stats,
    calc_max_drawdown,
    calc_moving_averages,
    decompose_stl,
)

from analysis.frequency_domain import (
    analyze_frequency_domain,
    calc_fft_spectrum,
    calc_power_spectrum,
    calc_returns_from_prices,
    calc_wavelet_transform,
    find_dominant_periods,
)

from analysis.arima_model import (
    adf_test,
    auto_diff,
    build_arima_model,
    calc_model_metrics,
    fit_arima_model,
    forecast_arima,
    search_best_arima_params,
)

from analysis.shape_similarity import (
    calc_dtw_similarity,
    compare_curves_shape,
    find_similar_patterns,
    normalize_series,
)

from analysis.complexity import (
    analyze_complexity,
    calc_approximate_entropy,
    calc_hurst_exponent,
    calc_sample_entropy,
    create_phase_space,
)
from analysis.technical import (
    analyze_technical,
    calc_bollinger_bands,
    calc_macd,
    calc_rsi,
    calc_rolling_volatility,
    calc_var_historical,
)

try:
    from analysis.lstm_model import (
        build_features_from_df,
        cross_validate_and_tune,
        load_model,
        run_lstm_pipeline,
        train_and_save,
    )
    _LSTM_AVAILABLE = True
except Exception:
    _LSTM_AVAILABLE = False
    build_features_from_df = None
    cross_validate_and_tune = None
    load_model = None
    run_lstm_pipeline = None
    train_and_save = None

__all__ = [
    # 时域分析
    "analyze_time_domain",
    "calc_basic_stats",
    "calc_max_drawdown",
    "calc_moving_averages",
    "decompose_stl",
    # 频域分析
    "analyze_frequency_domain",
    "calc_fft_spectrum",
    "calc_power_spectrum",
    "calc_returns_from_prices",
    "calc_wavelet_transform",
    "find_dominant_periods",
    # ARIMA预测模型
    "adf_test",
    "auto_diff",
    "build_arima_model",
    "calc_model_metrics",
    "fit_arima_model",
    "forecast_arima",
    "search_best_arima_params",
    # 形态相似度匹配
    "calc_dtw_similarity",
    "compare_curves_shape",
    "find_similar_patterns",
    "normalize_series",
    # 非线性与复杂度分析
    "analyze_complexity",
    "calc_approximate_entropy",
    "calc_hurst_exponent",
    "calc_sample_entropy",
    "create_phase_space",
    # 技术指标与风险
    "analyze_technical",
    "calc_rsi",
    "calc_macd",
    "calc_bollinger_bands",
    "calc_rolling_volatility",
    "calc_var_historical",
    # LSTM 预测（可选，依赖 torch）
    "build_features_from_df",
    "cross_validate_and_tune",
    "load_model",
    "run_lstm_pipeline",
    "train_and_save",
]
