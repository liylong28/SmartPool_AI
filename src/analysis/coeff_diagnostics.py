from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, spearmanr

# ============================================================
# 输入检查
# ============================================================
def validate_inputs(VH: np.ndarray, beta: np.ndarray):
    """
    VH: (n_samples, 2)
    beta: (n_samples, n_modes)
    """
    VH = np.asarray(VH, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    if VH.ndim != 2 or VH.shape[1] != 2:
        raise ValueError(f"VH should have shape (N, 2), got {VH.shape}.")
    if beta.ndim != 2:
        raise ValueError(f"beta should be 2D, got {beta.shape}.")
    if VH.shape[0] != beta.shape[0]:
        raise ValueError("Sample number mismatch between VH and beta.")
    if not np.all(np.isfinite(VH)):
        raise ValueError("VH contains NaN or Inf.")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains NaN or Inf.")

    unique_pairs = np.unique(VH, axis=0)
    if unique_pairs.shape[0] != VH.shape[0]:
        raise ValueError("Duplicated (V, H) conditions were found.")
    return VH, beta

# ============================================================
# 构造 V-H 参数网格
# ============================================================
def build_beta_grid(VH: np.ndarray, beta: np.ndarray, require_full_grid: bool = True):
    """
    返回：
        speeds: (n_V,)
        depths: (n_H,)
        beta_grid: (n_modes, n_H, n_V)
    beta_grid[k, ih, iv] 表示第k个POD系数在 H=depths[ih], V=speeds[iv] 下的值。
    """
    VH, beta = validate_inputs(VH, beta)
    V = VH[:, 0]
    H = VH[:, 1]
    
    speeds = np.sort(np.unique(V))
    depths = np.sort(np.unique(H))
    n_modes = beta.shape[1]
    beta_grid = np.full((n_modes, len(depths), len(speeds)), np.nan, dtype=np.float64)

    for sample_id in range(VH.shape[0]):
        v = V[sample_id]
        h = H[sample_id]
        iv = int(np.argmin(np.abs(speeds - v)))
        ih = int(np.argmin(np.abs(depths - h)))

        if not np.isclose(speeds[iv], v):
            raise ValueError(f"Cannot locate V={v} in speed grid.")
        if not np.isclose(depths[ih], h):
            raise ValueError(f"Cannot locate H={h} in depth grid.")
        beta_grid[:, ih, iv] = beta[sample_id]

    if require_full_grid:
        expected = len(speeds) * len(depths)
        if expected != VH.shape[0]:
            raise ValueError(f"Expected a full V-H Cartesian grid with {expected} samples, but got {VH.shape[0]}.")
        if np.isnan(beta_grid).any():
            raise ValueError("Some V-H grid cells are missing.")
    return speeds, depths, beta_grid

# ============================================================
# 每个模态基本统计量
# ============================================================
def coefficient_statistics(beta: np.ndarray, energy_ratio: np.ndarray | None = None):
    """
    对每个beta模态计算：
        mean, std, min, max, range, abs_max, skewness, kurtosis
    """
    beta = np.asarray(beta, dtype=np.float64)
    n_modes = beta.shape[1]
    rows = []

    for i in range(n_modes):
        y = beta[:, i]
        row = {
            "mode": i + 1,
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=1)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "range": float(np.ptp(y)),
            "abs_max": float(np.max(np.abs(y))),
            "skewness": float(skew(y, bias=False)),
            "kurtosis": float(kurtosis(y, fisher=True, bias=False))
        }
        if energy_ratio is not None and i < len(energy_ratio):
            row["pod_energy_ratio"] = float(energy_ratio[i])
        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# 单调性评价
# ============================================================
def _monotonicity_score(y: np.ndarray):
    """
    返回 [0.5, 1] 左右的量。
    越接近1：越接近整体单调。
    越接近0.5：越容易反复改变趋势。
    """
    y = np.asarray(y, dtype=np.float64)
    diff = np.diff(y)
    if diff.size == 0:
        return np.nan

    scale = max(np.max(np.abs(y)), 1.0)
    tol = 1e-12 * scale
    increasing = np.mean(diff >= -tol)
    decreasing = np.mean(diff <= tol)
    return float(max(increasing, decreasing))

# ============================================================
# 曲线粗糙度
# ============================================================
def _roughness_score(x: np.ndarray, y: np.ndarray):
    """
    基于二阶导数量级定义无量纲粗糙度。
    越小：趋势越平滑。
    越大：局部非线性 / 振荡越明显。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return np.nan

    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    x_span = np.ptp(x)
    y_span = np.ptp(y)
    eps = np.finfo(np.float64).eps
    score = np.mean(np.abs(d2y)) * x_span ** 2 / max(y_span, eps)
    return float(score)

# ============================================================
# V/H 可加性诊断
# ============================================================
def _additive_r2(field: np.ndarray):
    """
    检查：beta(V,H) 能否被 mean + effect_V(V) + effect_H(H) 简单解释。
    高：主要为平滑主趋势。
    低：V-H耦合或复杂局部结构较强。
    """
    field = np.asarray(field, dtype=np.float64)
    grand_mean = np.mean(field)
    effect_h = np.mean(field, axis=1) - grand_mean
    effect_v = np.mean(field, axis=0) - grand_mean
    pred = grand_mean + effect_h[:, None] + effect_v[None, :]

    sse = np.sum((field - pred) ** 2)
    sst = np.sum((field - grand_mean) ** 2)
    eps = np.finfo(np.float64).eps
    if sst <= eps:
        return 1.0
    return float(1.0 - sse / sst)

# ============================================================
# 每个POD系数的二维曲面诊断
# ============================================================
def surface_diagnostics(speeds: np.ndarray, depths: np.ndarray, beta_grid: np.ndarray):
    """
    计算：
        monotonicity_V, monotonicity_H
        roughness_V, roughness_H
        additive_R2
    """
    n_modes = beta_grid.shape[0]
    rows = []

    for mode_id in range(n_modes):
        field = beta_grid[mode_id]
        # 固定H，分析随V变化
        mono_v, rough_v = [], []
        for ih in range(len(depths)):
            y = field[ih, :]
            mono_v.append(_monotonicity_score(y))
            rough_v.append(_roughness_score(speeds, y))

        # 固定V，分析随H变化
        mono_h, rough_h = [], []
        for iv in range(len(speeds)):
            y = field[:, iv]
            mono_h.append(_monotonicity_score(y))
            rough_h.append(_roughness_score(depths, y))

        rows.append({
            "mode": mode_id + 1,
            "monotonicity_V": float(np.nanmean(mono_v)),
            "monotonicity_H": float(np.nanmean(mono_h)),
            "roughness_V": float(np.nanmean(rough_v)),
            "roughness_H": float(np.nanmean(rough_h)),
            "additive_R2": _additive_r2(field)
        })
    return pd.DataFrame(rows)

# ============================================================
# 物理增强特征
# ============================================================
def physics_features(VH: np.ndarray):
    """
    与论文Trend-GP中的物理增强特征一致：
        [V, V^2, V^4, 1/V^2, H, exp(-H)]
    """
    VH = np.asarray(VH, dtype=np.float64)
    V = VH[:, 0]
    H = VH[:, 1]
    X = np.column_stack([V, V**2, V**4, 1.0/(V**2), H, np.exp(-H)])
    names = ["V", "V2", "V4", "inv_V2", "H", "exp_neg_H"]
    return X, names

# ============================================================
# 特征与POD系数相关性
# ============================================================
def feature_correlations(VH: np.ndarray, beta: np.ndarray):
    """
    返回：
        Pearson correlation
        Spearman correlation
    """
    X, feature_names = physics_features(VH)
    n_modes = beta.shape[1]
    pearson_rows, spearman_rows = [], []

    for mode_id in range(n_modes):
        y = beta[:, mode_id]
        p_row = {"mode": mode_id + 1}
        s_row = {"mode": mode_id + 1}
        for j, name in enumerate(feature_names):
            x = X[:, j]
            p_row[name] = float(np.corrcoef(x, y)[0, 1])
            s_row[name] = float(spearmanr(x, y).statistic)
        pearson_rows.append(p_row)
        spearman_rows.append(s_row)
    return pd.DataFrame(pearson_rows), pd.DataFrame(spearman_rows)
