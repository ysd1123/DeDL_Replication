import itertools
import numpy as np
import torch

def all_treatments(m: int) -> np.ndarray:
    """生成所有 2^m 个 treatment 组合，按字典序排序。"""
    return np.array(list(itertools.product([0, 1], repeat=m)), dtype=np.float32)

def observed_treatments_m_plus_2(m: int) -> np.ndarray:
    """
    生成论文中用于部分观测情形的 m+2 个组合：
    - 全 0
    - 单一 1 的 m 个组合
    - 一个包含两位为 1 的 overlap 组合，例如 (1,1,0,...,0)
    """
    t_list = []
    zero = np.zeros(m, dtype=np.float32)
    t_list.append(zero.copy())
    for k in range(m):
        t = zero.copy()
        t[k] = 1.0
        t_list.append(t)
    # overlap 组合：简单选择 (1,1,0,...,0)
    if m >= 2:
        t_overlap = zero.copy()
        t_overlap[0] = 1.0
        t_overlap[1] = 1.0
        t_list.append(t_overlap)
    return np.stack(t_list, axis=0)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def generalized_sigmoid_form_II(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    G(u(x), t) = θ_{m+1}(x) * σ(θ0(x) + Σ θk(x) t_k)
    u: (..., m+2)
    t: (..., m)
    """
    m = t.shape[-1]
    theta0 = u[..., 0]
    theta_vec = u[..., 1:m+1]
    theta_m1 = u[..., m+1]
    eta = theta0 + (theta_vec * t).sum(dim=-1)
    sigma = sigmoid(eta)
    return theta_m1 * sigma


def G_u_analytic(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    解析计算 G 对 u 的梯度 Gu(u,t) ∈ R^{m+2}。
    u: (..., m+2)
    t: (..., m)
    返回: (..., m+2)
    """
    m = t.shape[-1]
    theta0 = u[..., 0]
    theta_vec = u[..., 1:m+1]
    theta_m1 = u[..., m+1]
    eta = theta0 + (theta_vec * t).sum(dim=-1)
    sigma = sigmoid(eta)
    s_prime = sigma * (1.0 - sigma)

    # dG/dθ0, dG/dθ1..m, dG/dθ_{m+1}
    d_theta0 = theta_m1 * s_prime                 # (...)
    d_thetas = theta_m1.unsqueeze(-1) * s_prime.unsqueeze(-1) * t  # (..., m)
    d_theta_m1 = sigma                           # (...)

    return torch.cat(
        [d_theta0.unsqueeze(-1), d_thetas, d_theta_m1.unsqueeze(-1)],
        dim=-1
    )


def loss_mse(y: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    g = generalized_sigmoid_form_II(u, t)
    return (y - g) ** 2


def loss_grad_u(y: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    ℓ_u(y,t,u) = 2 (G(u,t)-y) Gu(u,t)
    返回形状: (..., m+2)
    """
    g = generalized_sigmoid_form_II(u, t)
    Gu = G_u_analytic(u, t)
    return 2.0 * (g - y).unsqueeze(-1) * Gu
