from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM


class BaseAnomalyModel(ABC):
    """
    窗口级接口：fit(H) + predict(P)
    - 允许一维或多维输入：
      * 一维: shape (n,) 或 (n, 1)
      * 多维: shape (n, d)
    - predict 返回 shape (m,) 的 0/1 标记
    """

    @abstractmethod
    def fit(self, history_values: np.ndarray) -> 'BaseAnomalyModel':
        pass

    @abstractmethod
    def predict(self, future_values: np.ndarray) -> np.ndarray:
        pass


class ThreeSigmaModel(BaseAnomalyModel):
    """
    标准化 + 3-sigma 阈值（支持多维）：
    - 在 H 上拟合 StandardScaler，将 H 与 P 映射到标准化空间
    - 对于每一维计算 mean±k*std，P 若任一维越界则判为异常
    """

    def __init__(self, k: float = 3.0) -> None:
        self.k = k
        self.scaler: Optional[StandardScaler] = None
        self.lower: Optional[np.ndarray] = None
        self.upper: Optional[np.ndarray] = None

    def fit(self, history_values: np.ndarray) -> 'ThreeSigmaModel':
        H = np.asarray(history_values, dtype=float)
        if H.ndim == 1:
            H = H.reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(H)
        mu = Hz.mean(axis=0)
        sd = Hz.std(axis=0, ddof=1)
        self.lower = mu - self.k * sd
        self.upper = mu + self.k * sd
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.lower is not None and self.upper is not None, 'fit() must be called before predict()'
        P = np.asarray(future_values, dtype=float)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        Pz = self.scaler.transform(P)
        below = (Pz < self.lower).any(axis=1)
        above = (Pz > self.upper).any(axis=1)
        return (below | above).astype(int)


class KDEModel(BaseAnomalyModel):
    def __init__(self, bandwidth: float = 0.2, kernel: str = 'gaussian', q: float = 0.005) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.q = q
        self.scaler: Optional[StandardScaler] = None
        self.kde: Optional[KernelDensity] = None
        self.thresh: float = -np.inf

    def fit(self, history_values: np.ndarray) -> 'KDEModel':
        H = np.asarray(history_values, dtype=float)
        if H.ndim == 1:
            H = H.reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(H)
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(Hz)
        H_scores = self.kde.score_samples(Hz)
        self.thresh = float(np.quantile(H_scores, self.q))
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.kde is not None, 'fit() must be called before predict()'
        P = np.asarray(future_values, dtype=float)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        Pz = self.scaler.transform(P)
        P_scores = self.kde.score_samples(Pz)
        return (P_scores < self.thresh).astype(int)


class OCSVMModel(BaseAnomalyModel):
    def __init__(self, nu: float = 0.01, gamma: str = 'scale', kernel: str = 'rbf') -> None:
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.scaler: Optional[StandardScaler] = None
        self.ocsvm: Optional[OneClassSVM] = None

    def fit(self, history_values: np.ndarray) -> 'OCSVMModel':
        H = np.asarray(history_values, dtype=float)
        if H.ndim == 1:
            H = H.reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(H)
        self.ocsvm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.ocsvm.fit(Hz)
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.ocsvm is not None, 'fit() must be called before predict()'
        P = np.asarray(future_values, dtype=float)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        Pz = self.scaler.transform(P)
        pred = self.ocsvm.predict(Pz)
        return (pred == -1).astype(int)


class EnsembleModel(BaseAnomalyModel):
    def __init__(self, models: List[BaseAnomalyModel], threshold: Optional[int] = None) -> None:
        assert len(models) > 0, 'EnsembleModel requires at least one sub-model'
        self.models = models
        self.threshold = threshold if threshold is not None else (len(models) // 2 + 1)

    def fit(self, history_values: np.ndarray) -> 'EnsembleModel':
        for m in self.models:
            m.fit(history_values)
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        votes: Optional[np.ndarray] = None
        for m in self.models:
            flags = m.predict(future_values).astype(int)
            votes = flags if votes is None else (votes + flags)
        assert votes is not None
        return (votes >= self.threshold).astype(int)


def run_window(history_values: np.ndarray, future_values: np.ndarray, model: BaseAnomalyModel) -> np.ndarray:
    return model.fit(history_values).predict(future_values)
