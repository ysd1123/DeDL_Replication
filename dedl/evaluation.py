from __future__ import annotations
import itertools
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import StructuredNet
from .training import train_plain_dnn


def _enumerate_treatments(m: int) -> np.ndarray:
    combos = list(itertools.product([0, 1], repeat=m))
    return np.array([[1, *c] for c in combos], dtype=float)


def _la_baseline(T: np.ndarray, Y: np.ndarray, t_star: np.ndarray) -> float:
    mask = (T == t_star).all(axis=1)
    if mask.any():
        return float(Y[mask].mean())
    return float(Y.mean())


def _lr_baseline(X: np.ndarray, T: np.ndarray, Y: np.ndarray, t_star: np.ndarray) -> float:
    design = np.concatenate([X, T], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
    return float(np.dot(np.concatenate([X.mean(axis=0), t_star]), coef))


def _compute_gradient(link: str, c: float, u: float, t: np.ndarray) -> np.ndarray:
    if link == "sigmoid":
        grad_beta = c * np.exp(-u) / (1 + np.exp(-u)) ** 2 * t
        grad_d = 1 / (1 + np.exp(-u))
    elif link == "linear":
        grad_beta = t
        grad_d = 1.0
    elif link == "softplus":
        grad_beta = c * 1 / (1 + np.exp(-u)) * t
        grad_d = 1.0
    else:
        raise ValueError(f"Unsupported link: {link}")
    return np.concatenate([grad_beta, [grad_d]])


def _dedl_predict(model: StructuredNet, X: np.ndarray, T: np.ndarray, Y: np.ndarray, t_star: np.ndarray, config: Dict) -> float:
    ridge = float(config.get("debias", {}).get("ridge", 1e-3))
    model.eval()
    with torch.no_grad():
        beta_list = []
        pred_list = []
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=config.get("training", {}).get("batch_size", 256), shuffle=False)
        for batch_x, batch_t in loader:
            pred, beta = model(batch_x, batch_t, return_beta=True)
            beta_list.append(beta.numpy())
            pred_list.append(pred.numpy())
    beta_all = np.concatenate(beta_list, axis=0)
    pred_all = np.concatenate(pred_list, axis=0)

    m = T.shape[1] - 1
    t_candidates = _enumerate_treatments(m)
    corrected_pred = []
    for i in range(len(X)):
        beta = beta_all[i]
        u_obs = beta.dot(T[i])
        g_obs = _compute_gradient(model.link_function, float(model.c_param.item()), u_obs, T[i])
        lambda_mat = np.zeros((len(g_obs), len(g_obs)))
        for t_prime in t_candidates:
            u_prime = beta.dot(t_prime)
            g_prime = _compute_gradient(model.link_function, float(model.c_param.item()), u_prime, t_prime)
            lambda_mat += np.outer(g_prime, g_prime)
        lambda_mat += ridge * np.eye(lambda_mat.shape[0])
        lambda_inv = np.linalg.inv(lambda_mat)
        residual = Y[i] - pred_all[i]
        u_star = beta.dot(t_star)
        g_star = _compute_gradient(model.link_function, float(model.c_param.item()), u_star, t_star)
        corrected_pred.append(float(pred_all[i] + g_star @ lambda_inv @ g_obs * residual))
    return float(np.mean(corrected_pred))


def evaluate_methods(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    trained_model: StructuredNet,
    config: Dict,
    t_stars: List[np.ndarray],
) -> List[Dict[str, float]]:
    """Evaluate baseline and DeDL estimators for each target treatment."""
    results = []
    m = T.shape[1] - 1
    t_candidates = _enumerate_treatments(m)

    pdl_layers = config.get("model", {}).get("pdl_layers", [64, 64])
    pdl_model = train_plain_dnn(X, T, Y, pdl_layers, config)

    baseline_t0 = t_candidates[0]
    la_base = _la_baseline(T, Y, baseline_t0)
    lr_base = _lr_baseline(X, T, Y, baseline_t0)
    pdl_base = float(pdl_model(torch.tensor(np.concatenate([X, np.repeat(baseline_t0[None, :], len(X), axis=0)], axis=1), dtype=torch.float32)).detach().mean().item())
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        t_base_batch = torch.tensor(np.repeat(baseline_t0[None, :], len(X), axis=0), dtype=torch.float32)
        sdl_base = trained_model(x_tensor, t_base_batch)[0].mean().item()
    dedl_base = _dedl_predict(trained_model, X, T, Y, baseline_t0, config)

    for t_star in t_stars:
        la_pred = _la_baseline(T, Y, t_star)
        lr_pred = _lr_baseline(X, T, Y, t_star)

        pdl_pred = float(pdl_model(torch.tensor(np.concatenate([X, np.repeat(t_star[None, :], len(X), axis=0)], axis=1), dtype=torch.float32)).detach().mean().item())
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            t_star_batch = torch.tensor(np.repeat(t_star[None, :], len(X), axis=0), dtype=torch.float32)
            sdl_pred = trained_model(x_tensor, t_star_batch)[0].mean().item()

        dedl_pred = _dedl_predict(trained_model, X, T, Y, t_star, config)

        results.append({
            "treatment": t_star.tolist(),
            "la": la_pred,
            "lr": lr_pred,
            "pdl": pdl_pred,
            "sdl": sdl_pred,
            "dedl": dedl_pred,
            "la_diff": la_pred - la_base,
            "lr_diff": lr_pred - lr_base,
            "pdl_diff": pdl_pred - pdl_base,
            "sdl_diff": sdl_pred - sdl_base,
            "dedl_diff": dedl_pred - dedl_base,
        })
    return results
