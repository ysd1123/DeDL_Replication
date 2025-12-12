import numpy as np
import pandas as pd
from typing import Tuple
from utils import all_treatments, observed_treatments_m_plus_2

def generate_phi_functions(m: int, d_x: int, rng: np.random.RandomState):
    """
    生成 φ_j(x), j=0,...,m, 以及 φ_{m+1} 的参数：
    - A 为 (m+1) x d_x 的矩阵，元素 ~ U(-0.5, 0.5)
    - φ_j(x) = (A[j+1] x)^3, j=0,...,m
    - φ_{m+1} ~ U(10,20)
    """
    A = rng.uniform(-0.5, 0.5, size=(m+1, d_x))  # j=0,...,m
    phi_m1 = rng.uniform(10.0, 20.0)             # φ_{m+1}
    return A, phi_m1


def phi_eval(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    对每个样本 x 计算 φ_j(x), j=0,...,m。
    A: (m+1, d_x)
    x: (n, d_x)
    返回 φ: (n, m+1)
    """
    z = x @ A.T  # (n, m+1)
    return z ** 3


def generate_data_once(
    n: int, m: int, d_x: int, A: np.ndarray, phi_m1: float,
    observed_only: bool, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一次样本 (X, T, Y)：
    - X ~ U(0,1)^{d_x}
    - 若 observed_only=True，则 T 在 m+2 个组合上均匀取（部分观测 setting）
      否则，在所有 2^m 组合上均匀取
    - Y = φ_{m+1}(x) / (1 + exp(-(φ0(x)+Σ φ_k(x) t_k))) + ε, ε ~ U(-0.05,0.05)
    """
    X = rng.uniform(0.0, 1.0, size=(n, d_x)).astype(np.float32)
    phi = phi_eval(A, X)   # (n, m+1), 对应 φ0,...,φ_m
    phi0 = phi[:, 0]
    phi_vec = phi[:, 1:m+1]

    if observed_only:
        ts = observed_treatments_m_plus_2(m)
    else:
        ts = all_treatments(m)
    idx = rng.choice(ts.shape[0], size=n, replace=True)
    T = ts[idx]

    eta = phi0 + (phi_vec * T).sum(axis=1)
    sigma = 1.0 / (1.0 + np.exp(-eta))
    base = phi_m1
    g = base * sigma

    eps = rng.uniform(-0.05, 0.05, size=n)
    Y = g + eps

    return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32)


def estimate_true_ate_mc(
    m: int, d_x: int, A: np.ndarray, phi_m1: float,
    n_mc: int, rng: np.random.RandomState
) -> pd.DataFrame:
    """
    用 Monte Carlo 估计所有 2^m treatment 组合相对于 t0 的真 ATE。
    """
    ts_all = all_treatments(m)
    t0 = np.zeros(m, dtype=np.float32)
    X = rng.uniform(0.0, 1.0, size=(n_mc, d_x)).astype(np.float32)
    phi = phi_eval(A, X)
    phi0 = phi[:, 0]
    phi_vec = phi[:, 1:m+1]

    def outcome_for_t(t: np.ndarray) -> np.ndarray:
        eta = phi0 + (phi_vec * t).sum(axis=1)
        sigma = 1.0 / (1.0 + np.exp(-eta))
        g = phi_m1 * sigma
        return g  # 不加噪声，直接用条件期望

    y0 = outcome_for_t(t0)
    ate_list = []
    for idx, t in enumerate(ts_all):
        yt = outcome_for_t(t)
        ate = float((yt - y0).mean())
        ate_list.append({"combo_index": idx, "ATE_true": ate, "t_vec": t.tolist()})

    df_true = pd.DataFrame(ate_list)
    return df_true