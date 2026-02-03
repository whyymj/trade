#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块4：几何形态与相似度匹配

使用动态时间规整(DTW)算法比较价格曲线的形状相似度，包括：
- DTW距离计算
- 最优匹配路径提取
- 曲线对齐与可视化

使用示例:
    import pandas as pd
    from analysis.shape_similarity import compare_curves_shape

    curve1 = pd.Series([1.0, 2.0, 3.0, 2.5, 2.0])
    curve2 = pd.Series([1.5, 2.5, 3.5, 3.0, 2.5, 2.0])
    result = compare_curves_shape(curve1, curve2)
    print(f"DTW距离: {result['dtw_distance']}")
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def normalize_series(series: pd.Series, method: str = "zscore") -> pd.Series:
    """
    归一化序列，消除量纲和尺度差异。

    Args:
        series: 原始序列
        method: 归一化方法
            - 'zscore': Z-score标准化 (x - mean) / std
            - 'minmax': Min-Max归一化到[0,1]
            - 'first': 以第一个值为基准的相对变化

    Returns:
        归一化后的序列
    """
    s = series.dropna().astype(float)

    if method == "zscore":
        mean_val = s.mean()
        std_val = s.std()
        if std_val == 0:
            return s - mean_val
        return (s - mean_val) / std_val

    elif method == "minmax":
        min_val = s.min()
        max_val = s.max()
        if max_val == min_val:
            return pd.Series(0.5, index=s.index)
        return (s - min_val) / (max_val - min_val)

    elif method == "first":
        first_val = s.iloc[0]
        if first_val == 0:
            return s
        return s / first_val - 1  # 相对变化率

    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    计算两个序列的DTW距离和累积成本矩阵。

    纯Python实现，无需外部DTW库。

    Args:
        seq1: 第一个序列
        seq2: 第二个序列
        window: Sakoe-Chiba带宽约束（None表示无约束）

    Returns:
        tuple: (DTW距离, 累积成本矩阵)
    """
    n, m = len(seq1), len(seq2)

    # 初始化成本矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # 带宽约束
    if window is None:
        window = max(n, m)

    # 填充DTW矩阵
    for i in range(1, n + 1):
        # Sakoe-Chiba带约束
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # 插入
                dtw_matrix[i, j - 1],      # 删除
                dtw_matrix[i - 1, j - 1],  # 匹配
            )

    return dtw_matrix[n, m], dtw_matrix[1:, 1:]


def dtw_path(dtw_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    从DTW累积成本矩阵回溯最优匹配路径。

    Args:
        dtw_matrix: DTW累积成本矩阵

    Returns:
        最优路径列表 [(i1, j1), (i2, j2), ...]
    """
    n, m = dtw_matrix.shape
    i, j = n - 1, m - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # 找最小成本的前驱
            candidates = [
                (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
                (dtw_matrix[i - 1, j], i - 1, j),
                (dtw_matrix[i, j - 1], i, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

        path.append((i, j))

    path.reverse()
    return path


def calc_dtw_similarity(
    curve1: pd.Series,
    curve2: pd.Series,
    normalize: str = "zscore",
    window: int | None = None,
) -> dict[str, Any]:
    """
    计算两条曲线的DTW距离和匹配路径。

    Args:
        curve1: 第一条价格曲线
        curve2: 第二条价格曲线
        normalize: 归一化方法 ('zscore', 'minmax', 'first', None)
        window: DTW带宽约束

    Returns:
        dict: 包含 dtw_distance, normalized_distance, path, 
              normalized_curve1, normalized_curve2
    """
    # 转换为numpy数组
    s1 = curve1.dropna().values.astype(float)
    s2 = curve2.dropna().values.astype(float)

    # 归一化
    if normalize:
        s1_norm = normalize_series(pd.Series(s1), method=normalize).values
        s2_norm = normalize_series(pd.Series(s2), method=normalize).values
    else:
        s1_norm = s1
        s2_norm = s2

    # 计算DTW
    distance, dtw_matrix = dtw_distance(s1_norm, s2_norm, window=window)

    # 提取最优路径
    path = dtw_path(dtw_matrix)

    # 归一化距离（除以路径长度，便于比较不同长度的序列）
    normalized_distance = distance / len(path)

    return {
        "dtw_distance": distance,
        "normalized_distance": normalized_distance,
        "path": path,
        "dtw_matrix": dtw_matrix,
        "curve1_normalized": s1_norm,
        "curve2_normalized": s2_norm,
        "curve1_original": s1,
        "curve2_original": s2,
    }


def plot_dtw_alignment(
    result: dict[str, Any],
    title: str = "DTW曲线对齐",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制DTW对齐结果图。

    包含：归一化曲线对比、匹配连线、DTW累积成本矩阵热力图。

    Args:
        result: calc_dtw_similarity 返回的结果字典
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    s1 = result["curve1_normalized"]
    s2 = result["curve2_normalized"]
    path = result["path"]
    dtw_matrix = result["dtw_matrix"]
    distance = result["dtw_distance"]
    norm_dist = result["normalized_distance"]

    fig = plt.figure(figsize=(14, 10))

    # 子图1：归一化曲线叠加显示
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(s1, label=f"曲线1 (长度={len(s1)})", color="tab:blue", linewidth=1.5)
    ax1.plot(s2, label=f"曲线2 (长度={len(s2)})", color="tab:orange", linewidth=1.5)
    ax1.set_title("归一化曲线叠加对比")
    ax1.set_xlabel("时间点")
    ax1.set_ylabel("归一化值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：DTW匹配对齐图（带连线）
    ax2 = fig.add_subplot(2, 2, 2)

    # 将曲线2向上平移以便于显示匹配线
    offset = max(s1.max(), s2.max()) - min(s1.min(), s2.min()) + 0.5
    s2_shifted = s2 + offset

    ax2.plot(range(len(s1)), s1, "b-", label="曲线1", linewidth=1.5)
    ax2.plot(range(len(s2)), s2_shifted, "r-", label="曲线2 (平移)", linewidth=1.5)

    # 绘制匹配连线（采样以避免过于密集）
    step = max(1, len(path) // 50)
    for idx, (i, j) in enumerate(path):
        if idx % step == 0:
            ax2.plot([i, j], [s1[i], s2_shifted[j]], "g-", alpha=0.3, linewidth=0.5)

    ax2.set_title("DTW匹配对齐（绿线为匹配关系）")
    ax2.set_xlabel("时间点")
    ax2.set_ylabel("归一化值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：DTW累积成本矩阵热力图
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(dtw_matrix.T, origin="lower", cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax3, label="累积成本")

    # 绘制最优路径
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax3.plot(path_x, path_y, "b-", linewidth=2, label="最优路径")

    ax3.set_title("DTW累积成本矩阵与最优路径")
    ax3.set_xlabel("曲线1时间点")
    ax3.set_ylabel("曲线2时间点")
    ax3.legend()

    # 子图4：对齐后的曲线对比
    ax4 = fig.add_subplot(2, 2, 4)

    # 根据DTW路径对齐曲线
    aligned_s1 = [s1[i] for i, j in path]
    aligned_s2 = [s2[j] for i, j in path]

    ax4.plot(aligned_s1, label="曲线1 (对齐后)", color="tab:blue", linewidth=1.5)
    ax4.plot(aligned_s2, label="曲线2 (对齐后)", color="tab:orange", linewidth=1.5)
    ax4.fill_between(
        range(len(aligned_s1)),
        aligned_s1,
        aligned_s2,
        alpha=0.2,
        color="gray",
        label="差异区域",
    )
    ax4.set_title("DTW对齐后的曲线对比")
    ax4.set_xlabel("对齐时间点")
    ax4.set_ylabel("归一化值")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 总标题
    fig.suptitle(
        f"{title}\nDTW距离: {distance:.4f}, 归一化距离: {norm_dist:.4f}",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_original_curves(
    curve1: pd.Series,
    curve2: pd.Series,
    label1: str = "曲线1",
    label2: str = "曲线2",
    title: str = "原始曲线对比",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制原始曲线对比图（双Y轴）。

    Args:
        curve1: 第一条曲线
        curve2: 第二条曲线
        label1: 曲线1标签
        label2: 曲线2标签
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 曲线1（左Y轴）
    color1 = "tab:blue"
    ax1.set_xlabel("时间点")
    ax1.set_ylabel(label1, color=color1)
    line1 = ax1.plot(curve1.values, color=color1, linewidth=1.5, label=label1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # 曲线2（右Y轴）
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel(label2, color=color2)
    line2 = ax2.plot(curve2.values, color=color2, linewidth=1.5, label=label2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def interpret_similarity(normalized_distance: float) -> str:
    """
    根据归一化DTW距离解读相似度等级。

    Args:
        normalized_distance: 归一化后的DTW距离

    Returns:
        相似度等级描述
    """
    if normalized_distance < 0.1:
        return "极高相似度 - 曲线形态几乎一致"
    elif normalized_distance < 0.3:
        return "高相似度 - 曲线整体走势相似"
    elif normalized_distance < 0.5:
        return "中等相似度 - 存在部分相似特征"
    elif normalized_distance < 0.8:
        return "低相似度 - 曲线形态差异较大"
    else:
        return "极低相似度 - 曲线形态明显不同"


def compare_curves_shape(
    curve1: pd.Series,
    curve2: pd.Series,
    normalize: str = "zscore",
    window: int | None = None,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    label1: str = "曲线1",
    label2: str = "曲线2",
) -> dict[str, Any]:
    """
    比较两条价格曲线的形状相似度（主函数）。

    使用动态时间规整(DTW)算法计算两条曲线间的最优匹配路径和DTW距离，
    并进行可视化展示匹配关系。

    Args:
        curve1: 第一条价格序列（Pandas Series）
        curve2: 第二条价格序列（Pandas Series），长度可与curve1不同
        normalize: 归一化方法
            - 'zscore': Z-score标准化（推荐，消除量纲差异）
            - 'minmax': Min-Max归一化到[0,1]
            - 'first': 以首值为基准的相对变化
            - None: 不归一化
        window: DTW带宽约束（限制匹配范围，加速计算）
        save_dir: 图表保存目录
        show_plots: 是否显示图表
        label1: 曲线1的标签名称
        label2: 曲线2的标签名称

    Returns:
        dict: 包含以下字段：
            - dtw_distance: 原始DTW距离
            - normalized_distance: 归一化DTW距离（除以路径长度）
            - similarity_level: 相似度等级描述
            - path: 最优匹配路径
            - figures: 生成的图表列表

    Example:
        >>> import pandas as pd
        >>> from analysis.shape_similarity import compare_curves_shape
        >>> curve1 = df1["收盘"]
        >>> curve2 = df2["收盘"]
        >>> result = compare_curves_shape(curve1, curve2)
        >>> print(f"DTW距离: {result['dtw_distance']:.4f}")
    """
    # 数据验证
    if not isinstance(curve1, pd.Series) or not isinstance(curve2, pd.Series):
        raise TypeError("输入必须为 Pandas Series")

    s1 = curve1.dropna()
    s2 = curve2.dropna()

    if len(s1) < 5 or len(s2) < 5:
        raise ValueError("每条曲线至少需要5个数据点")

    # 准备保存目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    print("=" * 60)
    print("【几何形态与相似度匹配】")
    print("=" * 60)
    print(f"{label1} 长度: {len(s1)} 个数据点")
    print(f"{label2} 长度: {len(s2)} 个数据点")
    print(f"归一化方法: {normalize or '无'}")

    # 计算DTW
    print("\n正在计算DTW距离...")
    dtw_result = calc_dtw_similarity(s1, s2, normalize=normalize, window=window)

    distance = dtw_result["dtw_distance"]
    norm_dist = dtw_result["normalized_distance"]
    path = dtw_result["path"]

    similarity_level = interpret_similarity(norm_dist)

    print("\n【DTW相似度结果】")
    print("-" * 40)
    print(f"  DTW距离: {distance:.4f}")
    print(f"  归一化距离: {norm_dist:.4f}")
    print(f"  匹配路径长度: {len(path)}")
    print(f"  相似度评级: {similarity_level}")
    print("-" * 40)

    # 绘图
    print("\n生成可视化图表...")

    # 原始曲线对比图
    fig_orig = plot_original_curves(
        s1, s2,
        label1=label1,
        label2=label2,
        title=f"原始曲线对比 - {label1} vs {label2}",
        save_path=save_dir / "original_curves.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_orig)

    # DTW对齐结果图
    fig_dtw = plot_dtw_alignment(
        dtw_result,
        title=f"DTW曲线对齐分析 - {label1} vs {label2}",
        save_path=save_dir / "dtw_alignment.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_dtw)

    print("=" * 60)

    # 汇总结果
    result = {
        "dtw_distance": distance,
        "normalized_distance": norm_dist,
        "similarity_level": similarity_level,
        "path": path,
        "path_length": len(path),
        "curve1_length": len(s1),
        "curve2_length": len(s2),
        "dtw_matrix": dtw_result["dtw_matrix"],
        "curve1_normalized": dtw_result["curve1_normalized"],
        "curve2_normalized": dtw_result["curve2_normalized"],
        "figures": figures,
    }

    if save_dir:
        print(f"\n图表已保存至: {save_dir.resolve()}")

    return result


def find_similar_patterns(
    target_curve: pd.Series,
    search_curve: pd.Series,
    pattern_length: int,
    top_n: int = 5,
    step: int = 1,
    normalize: str = "zscore",
) -> list[dict[str, Any]]:
    """
    在长序列中搜索与目标模式最相似的子序列。

    Args:
        target_curve: 目标模式曲线
        search_curve: 待搜索的长序列
        pattern_length: 搜索窗口长度（应与target_curve长度相近）
        top_n: 返回最相似的前N个结果
        step: 滑动步长
        normalize: 归一化方法

    Returns:
        list[dict]: 每个元素包含 start_idx, end_idx, dtw_distance, subsequence
    """
    target = target_curve.dropna().values.astype(float)
    search = search_curve.dropna().values.astype(float)

    if len(search) < pattern_length:
        raise ValueError("搜索序列长度必须大于模式长度")

    results = []
    n_windows = (len(search) - pattern_length) // step + 1

    print(f"搜索窗口数量: {n_windows}")

    for i in range(0, len(search) - pattern_length + 1, step):
        subsequence = search[i:i + pattern_length]

        # 计算DTW距离
        dtw_result = calc_dtw_similarity(
            pd.Series(target),
            pd.Series(subsequence),
            normalize=normalize,
        )

        results.append({
            "start_idx": i,
            "end_idx": i + pattern_length,
            "dtw_distance": dtw_result["dtw_distance"],
            "normalized_distance": dtw_result["normalized_distance"],
            "subsequence": subsequence,
        })

    # 按距离排序，返回最相似的top_n个
    results.sort(key=lambda x: x["dtw_distance"])
    return results[:top_n]


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python -m analysis.shape_similarity <csv1> <csv2> [output_dir]")
        print("示例: python -m analysis.shape_similarity data/stock1.csv data/stock2.csv output/")
        sys.exit(1)

    csv1_path = sys.argv[1]
    csv2_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"读取曲线1: {csv1_path}")
    print(f"读取曲线2: {csv2_path}")

    # 读取数据
    df1 = pd.read_csv(csv1_path, encoding="utf-8-sig")
    df2 = pd.read_csv(csv2_path, encoding="utf-8-sig")

    # 找收盘价列
    price_col = None
    for col in ["收盘", "close", "Close", "收盘价"]:
        if col in df1.columns:
            price_col = col
            break

    if price_col is None:
        print("错误: 未找到收盘价列")
        sys.exit(1)

    curve1 = df1[price_col].astype(float)
    curve2 = df2[price_col].astype(float)

    # 提取文件名作为标签
    label1 = Path(csv1_path).stem
    label2 = Path(csv2_path).stem

    result = compare_curves_shape(
        curve1, curve2,
        save_dir=output_dir,
        show_plots=True,
        label1=label1,
        label2=label2,
    )

    print("\n分析完成。")
