#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理分析临时目录（仅标准库，不加载 analysis 包，无需 matplotlib）。
在项目根目录执行: python cleanup_analysis_temp.py
"""
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

# 直接加载 analysis/cleanup_temp.py，不经过 analysis 包，避免触发 matplotlib 等依赖
_root = Path(__file__).resolve().parent
_cleanup_path = _root / "analysis" / "cleanup_temp.py"
_loader = importlib.machinery.SourceFileLoader("cleanup_temp", str(_cleanup_path))
_spec = importlib.util.spec_from_loader("cleanup_temp", _loader, origin=str(_cleanup_path))
if _spec is None or _spec.loader is None:
    raise RuntimeError("无法加载 analysis/cleanup_temp.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["cleanup_temp"] = _mod
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    n = _mod.cleanup_analysis_temp_dirs()
    print(f"已清理 {n} 个分析临时目录")
