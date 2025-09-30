from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM


class BaseAnomalyModel(ABC):
    """
    抽象基类：窗口级使用
    - 先对历史窗口 H 拟合（fit），再对预测窗口 P 进行点级预测（predict）
    - 统一约定：输入为一维数值数组（shape = (n,) 或 (n,1)）
    - predict 返回 0/1 的 ndarray，1 表示异常
    """

    @abstractmethod
    def fit(self, history_values: np.ndarray) -> 'BaseAnomalyModel':
        pass

    @abstractmethod
    def predict(self, future_values: np.ndarray) -> np.ndarray:
        pass


class ThreeSigmaModel(BaseAnomalyModel):
    """
    标准化 + 3-sigma 阈值
    - 在 H 上拟合 StandardScaler，将 H 与 P 映射到标准化空间
    - 在标准化空间以 mean±k*std 为阈值
    """

    def __init__(self, k: float = 3.0) -> None:
        self.k = k
        self.scaler: Optional[StandardScaler] = None
        self.lower: float = -np.inf
        self.upper: float = np.inf

    def fit(self, history_values: np.ndarray) -> 'ThreeSigmaModel':
        v = np.asarray(history_values, dtype=float).reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(v).reshape(-1)
        mu = float(np.mean(Hz))
        sd = float(np.std(Hz, ddof=1)) if Hz.size > 1 else 0.0
        self.lower = mu - self.k * sd
        self.upper = mu + self.k * sd
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None, 'ThreeSigmaModel must be fit() before predict()'
        P = np.asarray(future_values, dtype=float).reshape(-1, 1)
        Pz = self.scaler.transform(P).reshape(-1)
        flags = (Pz < self.lower) | (Pz > self.upper)
        return flags.astype(int)


class KDEModel(BaseAnomalyModel):
    """
    标准化 + KDE 密度阈值
    - 在 H 上拟合 StandardScaler 与 KernelDensity
    - 用 H 的 log-density 分数分位数 q 做阈值，P 分数低于阈值判为异常
    """

    def __init__(self, bandwidth: float = 0.2, kernel: str = 'gaussian', q: float = 0.005) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.q = q
        self.scaler: Optional[StandardScaler] = None
        self.kde: Optional[KernelDensity] = None
        self.thresh: float = -np.inf

    def fit(self, history_values: np.ndarray) -> 'KDEModel':
        v = np.asarray(history_values, dtype=float).reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(v)
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(Hz)
        H_scores = self.kde.score_samples(Hz)
        self.thresh = float(np.quantile(H_scores, self.q))
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.kde is not None, 'KDEModel must be fit() before predict()'
        P = np.asarray(future_values, dtype=float).reshape(-1, 1)
        Pz = self.scaler.transform(P)
        P_scores = self.kde.score_samples(Pz)
        flags = (P_scores < self.thresh)
        return flags.astype(int)


class OCSVMModel(BaseAnomalyModel):
    """
    标准化 + One-Class SVM
    - 在 H 上拟合 StandardScaler 与 OCSVM
    - 对 P 逐点预测：predict 返回 {-1, 1}，其中 -1 视为异常
    """

    def __init__(self, nu: float = 0.01, gamma: str = 'scale', kernel: str = 'rbf') -> None:
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.scaler: Optional[StandardScaler] = None
        self.ocsvm: Optional[OneClassSVM] = None

    def fit(self, history_values: np.ndarray) -> 'OCSVMModel':
        v = np.asarray(history_values, dtype=float).reshape(-1, 1)
        self.scaler = StandardScaler()
        Hz = self.scaler.fit_transform(v)
        self.ocsvm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.ocsvm.fit(Hz)
        return self

    def predict(self, future_values: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.ocsvm is not None, 'OCSVMModel must be fit() before predict()'
        P = np.asarray(future_values, dtype=float).reshape(-1, 1)
        Pz = self.scaler.transform(P)
        pred = self.ocsvm.predict(Pz)  # 1正常，-1异常
        return (pred == -1).astype(int)


class EnsembleModel(BaseAnomalyModel):
    """
    简单多数决集成
    - 接收多个 BaseAnomalyModel 子类实例
    - fit: 依次在同一 H 上拟合所有子模型
    - predict: 将各子模型的 0/1 结果投票，>= threshold 判异常（默认多数决）
    """

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
    """便捷函数：单窗口拟合+预测，返回 P 的0/1标记。"""
    return model.fit(history_values).predict(future_values)
