from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from .coeff_diagnostics import physics_features

# ============================================================
# 基础指标
# ============================================================
def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _relative_l2(y_true, y_pred):
    eps = np.finfo(np.float64).eps
    return float(np.linalg.norm(y_true - y_pred) / max(np.linalg.norm(y_true), eps))

def _nrmse_global_std(y_true, y_pred, global_std):
    eps = np.finfo(np.float64).eps
    return _rmse(y_true, y_pred) / max(global_std, eps)

# ============================================================
# Ridge Pipeline
# ============================================================
def _ridge_cv_pipeline(alphas):
    """标准化输入特征，再进行RidgeCV。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])

# ============================================================
# 边界伪外推测试
# ============================================================
def _boundary_holdout(X, y, mask_train, mask_test, alphas, global_std):
    """
    用训练网格最外一层模拟外推：
    例如：
        train: H < 3.0
        test : H = 3.0
    """
    if np.sum(mask_train) < 5 or np.sum(mask_test) == 0:
        return {
            "r2": np.nan,
            "nrmse_std": np.nan,
            "relative_l2": np.nan,
        }
    model = _ridge_cv_pipeline(alphas)
    model.fit(X[mask_train], y[mask_train])
    pred = model.predict(X[mask_test])
    if np.sum(mask_test) > 1:
        test_r2 = float(r2_score(y[mask_test], pred))
    else:
        test_r2 = np.nan
    return {
        "r2": test_r2,
        "nrmse_std": _nrmse_global_std(y[mask_test], pred, global_std),
        "relative_l2": _relative_l2(y[mask_test], pred),
    }

# ============================================================
# 主诊断函数
# ============================================================
def ridge_trend_diagnostics(VH: np.ndarray, beta: np.ndarray, alphas=None):
    """
    对每一个POD模态：
    1. 用物理增强特征训练Ridge
    2. 计算训练R2
    3. LOO交叉验证
    4. 分析Ridge残差比例
    5. 最高航速边界伪外推
    6. 最大潜深边界伪外推
    返回：
        dataframe
        residuals
    """
    VH = np.asarray(VH, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if alphas is None:
        alphas = np.logspace(-8, 4, 40)
    X, feature_names = physics_features(VH)
    V = VH[:, 0]
    H = VH[:, 1]
    max_V = np.max(V)
    max_H = np.max(H)
    n_samples = beta.shape[0]
    n_modes = beta.shape[1]
    residuals = np.zeros_like(beta, dtype=np.float64)
    rows = []

    for mode_id in range(n_modes):
        print(f"Trend diagnostic {mode_id + 1}/{n_modes}")
        y = beta[:, mode_id]
        global_std = float(np.std(y, ddof=1))

        # 1. 全数据拟合
        model = _ridge_cv_pipeline(alphas)
        model.fit(X, y)
        train_pred = model.predict(X)
        residual = y - train_pred
        residuals[:, mode_id] = residual
        ridge_model = model.named_steps["ridge"]
        best_alpha = float(ridge_model.alpha_)
        train_r2 = float(r2_score(y, train_pred))

        # 2. 真正的LOO诊断
        #    每个外层fold内部重新RidgeCV
        loo = LeaveOneOut()
        loo_model = _ridge_cv_pipeline(alphas)
        loo_pred = cross_val_predict(loo_model, X, y, cv=loo, n_jobs=None)
        loo_r2 = float(r2_score(y, loo_pred))
        loo_rmse = _rmse(y, loo_pred)
        eps = np.finfo(np.float64).eps
        loo_nrmse = loo_rmse / max(global_std, eps)
        residual_std_ratio = np.std(residual, ddof=1) / max(global_std, eps)

        # 3. 最大速度边界伪外推
        speed_test = np.isclose(V, max_V)
        speed_train = ~speed_test
        speed_result = _boundary_holdout(
            X=X, y=y, mask_train=speed_train, mask_test=speed_test,
            alphas=alphas, global_std=global_std
        )

        # 4. 最大潜深边界伪外推
        depth_test = np.isclose(H, max_H)
        depth_train = ~depth_test
        depth_result = _boundary_holdout(
            X=X, y=y, mask_train=depth_train, mask_test=depth_test,
            alphas=alphas, global_std=global_std
        )

        rows.append({
            "mode": mode_id + 1,
            "ridge_alpha": best_alpha,
            "ridge_train_R2": train_r2,
            "ridge_LOO_R2": loo_r2,
            "ridge_LOO_RMSE": loo_rmse,
            "ridge_LOO_NRMSE_std": float(loo_nrmse),
            "residual_std_ratio": float(residual_std_ratio),
            "maxV_holdout_R2": speed_result["r2"],
            "maxV_holdout_NRMSE_std": speed_result["nrmse_std"],
            "maxV_holdout_relative_L2": speed_result["relative_l2"],
            "maxH_holdout_R2": depth_result["r2"],
            "maxH_holdout_NRMSE_std": depth_result["nrmse_std"],
            "maxH_holdout_relative_L2": depth_result["relative_l2"],
        })
    return pd.DataFrame(rows), residuals
