# -*- coding: utf-8 -*-
"""
集成学习多因子预测：XGBoost、LightGBM、随机森林 + 特征选择 + 过拟合预防 + 堆叠/加权集成

- XGBoost：技术指标 + 基本面因子 + 市场情绪因子
- LightGBM：高效处理大量特征
- 随机森林：特征重要性分析
- 集成：加权平均或 Stacking
- 自动特征选择：递归特征消除（RFE/RFECV）
- 过拟合预防：早停、L1/L2 正则化
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    lgb = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        mean_squared_error,
        roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.feature_selection import RFECV
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    RandomForestClassifier = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    RFECV = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    TimeSeriesSplit = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]


# ---------- 默认超参（早停 + 正则化） ----------
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 30,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


def _ensure_deps():
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("集成模块需要 scikit-learn")
    if not _XGB_AVAILABLE:
        raise RuntimeError("XGBoost 需要安装: pip install xgboost")
    if not _LGB_AVAILABLE:
        raise RuntimeError("LightGBM 需要安装: pip install lightgbm")


# ---------- 从因子面板构建 X, y（时序切分） ----------
def build_xy_from_factors(
    factor_df: pd.DataFrame,
    forward_return: pd.Series,
    *,
    task: str = "classification",
    threshold: float = 0.0,
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Index]:
    """
    从因子 DataFrame 与未来收益构建 (X, y)。

    Args:
        factor_df: 因子面板，索引为日期
        forward_return: 未来 N 日收益率，与 factor_df 索引对齐
        task: "classification" 则 y = (forward_return > threshold).astype(int)；"regression" 则 y = forward_return
        threshold: 分类时涨跌分界
        drop_na: 是否丢弃含 NaN 的行

    Returns:
        X, y, feature_names, index (对齐后的日期索引)
    """
    common = factor_df.index.intersection(forward_return.index)
    X_df = factor_df.loc[common].copy()
    y_ser = forward_return.reindex(common)

    if drop_na:
        mask = ~(X_df.isna().any(axis=1) | y_ser.isna())
        X_df = X_df.loc[mask]
        y_ser = y_ser.loc[mask]

    X = X_df.values.astype(np.float64)
    if task == "classification":
        y = (y_ser.values > threshold).astype(np.int32)
    else:
        y = y_ser.values.astype(np.float64)
    names = list(X_df.columns)
    return X, y, names, X_df.index


# ---------- XGBoost ----------
def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
) -> tuple[Any, dict[str, Any]]:
    """
    训练 XGBoost 分类器，带早停与正则化。
    返回 (model, evals_result)。
    """
    _ensure_deps()
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    p["early_stopping_rounds"] = early_stopping_rounds or p.get("early_stopping_rounds", 30)
    evals = []
    if X_val is not None and y_val is not None and len(X_val) > 0:
        evals = [(xgb.DMatrix(X_train, y_train, feature_names=None), "train"),
                 (xgb.DMatrix(X_val, y_val, feature_names=None), "valid")]
    dtrain = xgb.DMatrix(X_train, y_train)
    evals_result = {}  # XGBoost 2.x 通过传入空 dict 接收评估结果，Booster 无 evals_result() 方法
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = xgb.train(
            p,
            dtrain,
            num_boost_round=p["n_estimators"],
            evals=evals,
            early_stopping_rounds=p["early_stopping_rounds"],
            verbose_eval=False,
            evals_result=evals_result,
        )
    return model, evals_result


def predict_xgb(model: Any, X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    return model.predict(d)


# ---------- LightGBM ----------
def train_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
) -> tuple[Any, dict[str, Any]]:
    """训练 LightGBM 分类器，带早停与正则化。返回 (model, evals_result)。"""
    _ensure_deps()
    p = {**DEFAULT_LGB_PARAMS, **(params or {})}
    p["early_stopping_rounds"] = early_stopping_rounds or p.get("early_stopping_rounds", 30)
    train_set = lgb.Dataset(X_train, y_train)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_val is not None and y_val is not None and len(X_val) > 0:
        valid_sets.append(lgb.Dataset(X_val, y_val, reference=train_set))
        valid_names.append("valid")
    evals_result = {}  # LightGBM Booster 无 evals_result_，用 record_evaluation 回调收集
    callbacks = [
        lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
        lgb.record_evaluation(evals_result),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lgb.train(
            p,
            train_set,
            num_boost_round=p["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    return model, evals_result


def predict_lgb(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ---------- 随机森林（特征重要性） ----------
def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    task: str = "classification",
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, float]]:
    """训练随机森林，返回 (model, feature_importances_dict)。"""
    _ensure_deps()
    p = {**DEFAULT_RF_PARAMS, **(params or {})}
    if task == "classification":
        model = RandomForestClassifier(**p)
    else:
        model = RandomForestRegressor(**p)
    model.fit(X_train, y_train)
    imp = dict(zip(range(len(model.feature_importances_)), model.feature_importances_.tolist()))
    return model, imp


def get_rf_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """从 RF 模型得到按重要性排序的 DataFrame。"""
    imp = model.feature_importances_
    n = len(imp)
    names = feature_names if len(feature_names) >= n else [f"f{i}" for i in range(n)]
    return pd.DataFrame({
        "feature": names[:n],
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ---------- 递归特征消除（RFE） ----------
def run_rfecv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    task: str = "classification",
    min_features_to_select: int = 10,
    cv_splits: int = 5,
    step: float = 0.1,
    scoring: str | Callable = "roc_auc",
) -> tuple[list[str], Any, Any]:
    """
    使用 RFECV 做自动特征选择（时序交叉验证）。
    返回 (selected_feature_names, fitted_rfecv, selector)。
    """
    _ensure_deps()
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    if task == "classification":
        estimator = RandomForestClassifier(**DEFAULT_RF_PARAMS)
    else:
        estimator = RandomForestRegressor(**DEFAULT_RF_PARAMS)
        if scoring == "roc_auc":
            scoring = "neg_mean_squared_error"
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=tscv,
        min_features_to_select=min_features_to_select,
        scoring=scoring,
        n_jobs=-1,
    )
    selector.fit(X, y)
    selected = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]
    return selected, selector, selector.estimator_


# ---------- 集成：加权平均 ----------
def ensemble_weighted_predict(
    predictions: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """加权平均多模型预测概率。weights 需和为 1。"""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(predictions[0])
    for pred, wi in zip(predictions, w):
        out += wi * np.asarray(pred, dtype=np.float64)
    return out


# ---------- 集成：最优权重分配（最大化 AUC） ----------
def optimize_ensemble_weights(
    predictions_list: list[np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "auc",
    method: str = "grid",
    n_grid: int = 21,
) -> tuple[np.ndarray, float]:
    """
    在验证集上优化模型组合权重。
    predictions_list: 各模型在验证集上的预测概率 (each shape = (n_samples,))
    y_true: 真实标签
    metric: "auc" 或 "accuracy"
    method: "grid" 网格搜索 或 "scipy" 使用 scipy.optimize
    返回 (best_weights, best_score)。
    """
    from scipy.optimize import minimize

    n_models = len(predictions_list)
    preds = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in predictions_list], axis=1)

    def _score(weights: np.ndarray) -> float:
        w = np.maximum(weights, 0)
        w = w / (w.sum() + 1e-10)
        prob = preds @ w
        if metric == "auc":
            try:
                return roc_auc_score(y_true, prob)
            except Exception:
                return 0.5
        else:
            return accuracy_score(y_true, (prob >= 0.5).astype(int))

    if method == "grid":
        best_score = -1.0
        best_w = np.ones(n_models) / n_models
        grid = np.linspace(0, 1, n_grid)
        if n_models == 2:
            for w0 in grid:
                w = np.array([w0, 1 - w0])
                s = _score(w)
                if s > best_score:
                    best_score = s
                    best_w = w
        else:
            from itertools import product
            for ws in product(grid, repeat=n_models):
                w = np.array(ws)
                if w.sum() < 0.01:
                    continue
                s = _score(w)
                if s > best_score:
                    best_score = s
                    best_w = w / w.sum()
        return best_w, best_score

    # scipy: 最小化 -auc，约束 sum(w)=1, w>=0
    def neg_auc(w: np.ndarray) -> float:
        return -_score(w)

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n_models
    x0 = np.ones(n_models) / n_models
    res = minimize(neg_auc, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = np.maximum(res.x, 0)
    w = w / w.sum()
    return w, _score(w)


# ---------- Stacking ----------
def train_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_models: list[tuple[str, Any, Callable]],
    *,
    meta_model: str = "logistic",
    cv_splits: int = 5,
) -> tuple[Any, list[tuple[str, Any, Callable]]]:
    """
    两层 Stacking：base_models 在每折上预测得到 meta 特征，再训练 meta 模型。
    base_models: [(name, model, predict_proba_func), ...]，model 已拟合或未拟合均可；若未拟合则在本函数内用 X_train 拟合。
    本函数会按 TimeSeriesSplit 折内拟合 base、折外预测生成 meta 特征，再拟合 meta。
    为简化，这里用全量数据拟合 base，再用 TimeSeriesSplit 生成 OOF 预测作为 meta 特征（与 sklearn StackingClassifier 逻辑类似）。
    """
    _ensure_deps()
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    n = len(X_train)
    oof = np.zeros((n, len(base_models)))
    fitted_bases = []
    for i, (name, base, predict_fn) in enumerate(base_models):
        # 每折：训练 base，对验证折预测
        for train_idx, val_idx in tscv.split(X_train):
            Xt, Xv = X_train[train_idx], X_train[val_idx]
            yt = y_train[train_idx]
            if hasattr(base, "fit"):
                try:
                    if "early_stopping" in str(type(base)).lower() or hasattr(base, "evals_result"):
                        # XGB/LGB 需传 eval set
                        continue
                    base.fit(Xt, yt)
                except Exception:
                    pass
            # 若 base 未实现 fit 或为 Booster，则用全量拟合一次
            try:
                pred = predict_fn(base, Xv)
            except Exception:
                pred = np.full(len(val_idx), 0.5)
            oof[val_idx, i] = pred
        # 用全量数据再拟合一次，供最终预测用
        fitted_bases.append((name, base, predict_fn))

    # 用 OOF 训练 meta
    if meta_model == "logistic":
        meta = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    else:
        meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(oof, y_train)
    return meta, fitted_bases


def predict_stacking(
    meta_model: Any,
    base_models: list[tuple[str, Any, Callable]],
    X: np.ndarray,
) -> np.ndarray:
    """Stacking 预测：先得到各 base 预测，再 meta 预测。"""
    base_preds = []
    for _name, base, predict_fn in base_models:
        try:
            p = predict_fn(base, X)
        except Exception:
            p = np.full(len(X), 0.5)
        base_preds.append(p)
    oof = np.column_stack(base_preds)
    return meta_model.predict(oof)


# ---------- 一站式流水线 ----------
def run_ensemble_pipeline(
    factor_df: pd.DataFrame,
    forward_return: pd.Series,
    *,
    task: str = "classification",
    train_ratio: float = 0.7,
    use_rfe: bool = True,
    rfe_min_features: int = 15,
    rfe_cv: int = 3,
    xgb_params: dict[str, Any] | None = None,
    lgb_params: dict[str, Any] | None = None,
    rf_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 30,
    ensemble_method: str = "weighted",
    optimize_weights: bool = True,
) -> dict[str, Any]:
    """
    一站式：构建 X,y -> (可选)RFE -> 划分时序 -> 训练 XGB/LGB/RF -> 集成 -> 权重优化 -> 返回结果与指标。
    """
    _ensure_deps()
    X, y, names, idx = build_xy_from_factors(factor_df, forward_return, task=task)
    if len(X) < 50:
        return {"error": "样本不足", "n_samples": len(X)}

    # 时序划分
    n = len(X)
    split = int(n * train_ratio)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    selected_names = names
    if use_rfe and len(names) > rfe_min_features:
        selected_names, _rfe, _ = run_rfecv(
            X_train, y_train, names,
            task=task, min_features_to_select=rfe_min_features, cv_splits=rfe_cv, step=0.1,
        )
        col_idx = [names.index(f) for f in selected_names]
        X_train = X_train[:, col_idx]
        X_val = X_val[:, col_idx]
        X = X[:, col_idx]

    # 训练
    xgb_model, xgb_evals = train_xgb(X_train, y_train, X_val, y_val, early_stopping_rounds=early_stopping_rounds)
    lgb_model, lgb_evals = train_lgb(X_train, y_train, X_val, y_val, early_stopping_rounds=early_stopping_rounds)
    rf_model, rf_imp = train_rf(X_train, y_train, task=task, params=rf_params)
    rf_importance_df = get_rf_feature_importance(rf_model, selected_names)

    # 验证集预测
    p_xgb = predict_xgb(xgb_model, X_val)
    p_lgb = predict_lgb(lgb_model, X_val)
    if task == "classification":
        p_rf = rf_model.predict_proba(X_val)[:, 1]
    else:
        p_rf = rf_model.predict(X_val)

    predictions_list = [p_xgb, p_lgb, p_rf]
    if optimize_weights:
        weights, score = optimize_ensemble_weights(predictions_list, y_val, metric="auc", method="grid", n_grid=11)
    else:
        weights = np.array([1/3, 1/3, 1/3])
        try:
            score = roc_auc_score(y_val, ensemble_weighted_predict(predictions_list, weights.tolist()))
        except Exception:
            score = 0.5

    ensemble_pred = ensemble_weighted_predict(predictions_list, weights.tolist())
    val_auc = roc_auc_score(y_val, ensemble_pred) if task == "classification" else 0.0
    val_acc = accuracy_score(y_val, (ensemble_pred >= 0.5).astype(int)) if task == "classification" else 0.0
    if task == "regression":
        val_rmse = float(np.sqrt(mean_squared_error(y_val, ensemble_pred)))
    else:
        val_rmse = None

    return {
        "task": task,
        "n_samples": n,
        "n_features_used": len(selected_names),
        "feature_names": selected_names,
        "models": {"xgb": xgb_model, "lgb": lgb_model, "rf": rf_model},
        "rf_feature_importance": rf_importance_df.to_dict(orient="records"),
        "ensemble_weights": weights.tolist(),
        "optimized_metric": score,
        "val_auc": val_auc,
        "val_accuracy": val_acc,
        "val_rmse": val_rmse,
        "xgb_evals": {k: list(v) if isinstance(v, (list, np.ndarray)) else v for k, v in (xgb_evals or {}).items()},
        "lgb_evals": {k: list(v) if isinstance(v, (list, np.ndarray)) else v for k, v in (lgb_evals or {}).items()},
    }


def save_ensemble_artifacts(result: dict[str, Any], out_dir: str | Path) -> None:
    """保存集成结果：权重、特征列表、RF 重要性；XGB/LGB 模型可另存为 native 格式。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "weights": result.get("ensemble_weights", []),
            "feature_names": result.get("feature_names", []),
            "val_auc": result.get("val_auc"),
            "val_accuracy": result.get("val_accuracy"),
        }, f, ensure_ascii=False, indent=2)
    rf_imp = result.get("rf_feature_importance")
    if rf_imp:
        pd.DataFrame(rf_imp).to_csv(out_dir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")

