#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析临时目录清理（仅用标准库，不依赖 matplotlib 等，便于单独执行）。
"""
import shutil
import tempfile
from pathlib import Path

ANALYSIS_TEMP_PREFIX = "trade_analysis_"

# 旧版或未加前缀时可能产生的临时目录名模式（仅当其下存在 time_domain 子目录时视为分析产出）
LEGACY_TMP_PREFIX = "tmp"


def _is_likely_analysis_temp(path: Path) -> bool:
    """判断是否为分析产生的临时目录：含有 time_domain 子目录。"""
    return (path / "time_domain").is_dir()


def cleanup_analysis_temp_dirs() -> int:
    """
    清理系统中遗留的分析临时目录：
    1) 以 trade_analysis_ 开头的目录（当前 run_analysis_from_dataframe 创建）；
    2) 以 tmp 开头且含有 time_domain 子目录的目录（旧版或无前缀的临时目录）。
    返回删除的目录数量。
    """
    removed = 0
    tmp_root = Path(tempfile.gettempdir())
    if not tmp_root.exists():
        return 0
    for path in tmp_root.iterdir():
        if not path.is_dir():
            continue
        to_remove = False
        if path.name.startswith(ANALYSIS_TEMP_PREFIX):
            to_remove = True
        elif path.name.startswith(LEGACY_TMP_PREFIX) and _is_likely_analysis_temp(path):
            to_remove = True
        if to_remove:
            try:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            except Exception:
                pass
    return removed


if __name__ == "__main__":
    n = cleanup_analysis_temp_dirs()
    print(f"已清理 {n} 个分析临时目录")
