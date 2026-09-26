from __future__ import annotations
import numpy as np

def _validate_pair(true: np.ndarray, pred: np.ndarray):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} vs {pred.shape}")
    if not np.all(np.isfinite(true)):
        raise ValueError("true contains NaN or Inf.")
    if not np.all(np.isfinite(pred)):
        raise ValueError("pred contains NaN or Inf.")
    return true, pred


def global_relative_l2(true: np.ndarray, pred: np.ndarray) -> float:
    """
    全局相对L2误差，单位：%
    E = ||true-pred||_2 / ||true||_2 * 100
    """
    true, pred = _validate_pair(true, pred)
    numerator = np.linalg.norm(true - pred)
    denominator = np.linalg.norm(true)
    eps = np.finfo(np.float64).eps
    return float(numerator / max(denominator, eps) * 100.0)


def sample_relative_l2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    每个工况分别计算相对L2误差。
    true/pred:
        shape=(n_samples, n_features)
    return:
        shape=(n_samples,)
        单位 %
    """
    true, pred = _validate_pair(true, pred)
    if true.ndim != 2:
        raise ValueError("sample_relative_l2 requires 2D arrays.")
    numerator = np.linalg.norm(true - pred, axis=1)
    denominator = np.linalg.norm(true, axis=1)
    eps = np.finfo(np.float64).eps
    return numerator / np.maximum(denominator, eps) * 100.0


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = _validate_pair(true, pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = _validate_pair(true, pred)
    return float(np.mean(np.abs(true - pred)))


def max_abs_error(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = _validate_pair(true, pred)
    return float(np.max(np.abs(true - pred)))


def summarize_reconstruction(true: np.ndarray, pred: np.ndarray) -> dict:
    sample_error = sample_relative_l2(true, pred)
    return {
        "global_relative_l2_percent": global_relative_l2(true, pred),
        "mean_sample_relative_l2_percent": float(np.mean(sample_error)),
        "median_sample_relative_l2_percent": float(np.median(sample_error)),
        "max_sample_relative_l2_percent": float(np.max(sample_error)),
        "min_sample_relative_l2_percent": float(np.min(sample_error)),
        "rmse": rmse(true, pred),
        "mae": mae(true, pred),
        "max_abs_error": max_abs_error(true, pred),
    }
