from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np

class POD:
    """
    Proper Orthogonal Decomposition (POD)
    输入矩阵:
        X.shape = (n_samples, n_features)
    本项目:
        n_samples  = 132
        n_features = 54000
    拟合后:
        modes_.shape  = (n_modes, 54000)
        coeff.shape   = (132, n_modes)
    重构:
        X_hat = coeff @ modes_
    如果 center=True:
        X_hat = coeff @ modes_ + mean_
    """
    def __init__(
        self,
        energy_threshold: float = 0.99,
        n_modes: Optional[int] = None,
        center: bool = False,
        solver: str = "svd",
    ):
        """
        Parameters
        ----------
        energy_threshold : float
            累积能量阈值，例如 0.99。
        n_modes : int or None
            如果指定，则强制保留固定模态数。
            如果为 None，则根据 energy_threshold 自动选择。
        center : bool
            是否在 POD 前减去训练集平均波场。
        solver : str
            "svd":
                直接进行经济型SVD。
            "snapshot":
                使用 snapshot method，
                特别适合 n_samples << n_features 的情况。
        """
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1].")
        if solver not in {"svd", "snapshot"}:
            raise ValueError("solver must be 'svd' or 'snapshot'.")

        self.energy_threshold = float(energy_threshold)
        self.n_modes = n_modes
        self.center = bool(center)
        self.solver = solver

        self.mean_: Optional[np.ndarray] = None
        self.modes_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.energy_ratio_: Optional[np.ndarray] = None
        self.cumulative_energy_: Optional[np.ndarray] = None
        self.n_modes_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.is_fitted_: bool = False

    # =========================================================
    # 基础检查
    # =========================================================
    @staticmethod
    def _validate_matrix(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D matrix, got shape {X.shape}.")
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf.")
        return X

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("POD model has not been fitted yet.")

    # =========================================================
    # fit
    # =========================================================
    def fit(self, X: np.ndarray) -> "POD":
        X = self._validate_matrix(X)
        self.n_samples_, self.n_features_ = X.shape

        # Step 1: 中心化
        if self.center:
            self.mean_ = np.mean(X, axis=0)
            X_work = X - self.mean_
        else:
            self.mean_ = np.zeros(self.n_features_, dtype=np.float64)
            X_work = X

        # Step 2: POD / SVD
        if self.solver == "svd":
            modes_all, singular_values = self._fit_svd(X_work)
        else:
            modes_all, singular_values = self._fit_snapshot(X_work)

        # Step 3: 模态能量
        energy = singular_values ** 2
        total_energy = np.sum(energy)
        if total_energy <= 0:
            raise ValueError("Total POD energy is zero.")
        energy_ratio = energy / total_energy
        cumulative_energy = np.cumsum(energy_ratio)

        # Step 4: 自动确定截断阶数
        if self.n_modes is None:
            k = int(np.searchsorted(cumulative_energy, self.energy_threshold, side="left") + 1)
        else:
            k = int(self.n_modes)
            if k <= 0:
                raise ValueError("n_modes must be positive.")
            if k > modes_all.shape[0]:
                raise ValueError(f"Requested n_modes={k}, but only {modes_all.shape[0]} modes are available.")

        # Step 5: 保存截断后的空间模态
        self.n_modes_ = k
        self.modes_ = modes_all[:k].copy()
        self.singular_values_ = singular_values.copy()
        self.energy_ratio_ = energy_ratio.copy()
        self.cumulative_energy_ = cumulative_energy.copy()
        self.is_fitted_ = True
        return self

    # =========================================================
    # 直接SVD
    # =========================================================
    @staticmethod
    def _fit_svd(X: np.ndarray):
        """
        X = U S V^T
        V^T 的每一行即空间 POD 模态。
        """
        _, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt, singular_values

    # =========================================================
    # Snapshot POD
    # =========================================================
    @staticmethod
    def _fit_snapshot(X: np.ndarray):
        """
        Method of Snapshots
        当:
            n_samples << n_features
        例如:
            132 << 54000
        可以先求:
            C = X X^T
        C shape:
            (132,132)
        再恢复空间模态。
        """
        C = X @ X.T
        eigenvalues, U = np.linalg.eigh(C)
        # 从大到小排列
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        U = U[:, order]

        # 数值误差可能导致极小负特征值
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        singular_values = np.sqrt(eigenvalues)
        if singular_values.size == 0:
            raise ValueError("No singular values found.")

        tol = np.finfo(np.float64).eps * max(X.shape) * singular_values[0]
        valid = singular_values > tol
        singular_values = singular_values[valid]
        U = U[:, valid]

        # V^T = S^{-1} U^T X
        Vt = (U.T @ X) / singular_values[:, None]
        # 再归一化，避免浮点累积误差
        norms = np.linalg.norm(Vt, axis=1, keepdims=True)
        Vt = Vt / np.maximum(norms, np.finfo(np.float64).eps)
        return Vt, singular_values

    # =========================================================
    # transform
    # =========================================================
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        高维波场 -> POD系数
        X:
            (n_samples, 54000)
        return:
            (n_samples, n_modes)
        """
        self._check_fitted()
        X = self._validate_matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_}, got {X.shape[1]}.")
        X_work = X - self.mean_
        coeff = X_work @ self.modes_.T
        return coeff

    # =========================================================
    # inverse_transform
    # =========================================================
    def inverse_transform(self, coeff: np.ndarray) -> np.ndarray:
        """POD系数 -> 完整波场"""
        self._check_fitted()
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.ndim == 1:
            coeff = coeff[None, :]
        if coeff.ndim != 2:
            raise ValueError("coeff must be 1D or 2D.")
        if coeff.shape[1] != self.n_modes_:
            raise ValueError(f"Expected {self.n_modes_} coefficients, got {coeff.shape[1]}.")
        X_rec = coeff @ self.modes_
        X_rec = X_rec + self.mean_
        return X_rec

    # =========================================================
    # fit_transform
    # =========================================================
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    # =========================================================
    # 单个模态贡献
    # =========================================================
    def mode_contribution(self, coeff: np.ndarray, mode_index: int) -> np.ndarray:
        """
        返回某一个模态对波场的贡献。
        mode_index 从0开始。
        """
        self._check_fitted()
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.ndim == 1:
            coeff = coeff[None, :]
        if not 0 <= mode_index < self.n_modes_:
            raise IndexError("Invalid mode_index.")
        beta_i = coeff[:, mode_index:mode_index + 1]
        phi_i = self.modes_[mode_index:mode_index + 1]
        return beta_i @ phi_i

    # =========================================================
    # 保存
    # =========================================================
    def save(self, filepath: Union[str, Path]):
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            mean=self.mean_,
            modes=self.modes_,
            singular_values=self.singular_values_,
            energy_ratio=self.energy_ratio_,
            cumulative_energy=self.cumulative_energy_,
            n_modes=np.array(self.n_modes_, dtype=np.int64),
            n_samples=np.array(self.n_samples_, dtype=np.int64),
            n_features=np.array(self.n_features_, dtype=np.int64),
            energy_threshold=np.array(self.energy_threshold, dtype=np.float64),
            center=np.array(self.center, dtype=np.bool_),
        )

    # =========================================================
    # 加载
    # =========================================================
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "POD":
        data = np.load(filepath, allow_pickle=False)
        obj = cls(
            energy_threshold=float(data["energy_threshold"]),
            n_modes=int(data["n_modes"]),
            center=bool(data["center"]),
        )
        obj.mean_ = data["mean"]
        obj.modes_ = data["modes"]
        obj.singular_values_ = data["singular_values"]
        obj.energy_ratio_ = data["energy_ratio"]
        obj.cumulative_energy_ = data["cumulative_energy"]
        obj.n_modes_ = int(data["n_modes"])
        obj.n_samples_ = int(data["n_samples"])
        obj.n_features_ = int(data["n_features"])
        obj.is_fitted_ = True
        return obj
