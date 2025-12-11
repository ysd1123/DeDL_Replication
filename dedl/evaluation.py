from __future__ import annotations
import itertools
from typing import Dict, List, Sequence, Tuple, Union, Optional

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
        grad_d = 1.0
    elif link == "linear":
        grad_beta = t
        grad_d = 1.0
    elif link == "softplus":
        grad_beta = c * 1 / (1 + np.exp(-u)) * t
        grad_d = 1.0
    else:
        raise ValueError(f"Unsupported link: {link}")
    return np.concatenate([grad_beta, [grad_d]])


def _get_model_stats(models: Union[StructuredNet, Sequence[StructuredNet]]) -> Tuple[str, float]:
    if isinstance(models, (list, tuple)):
        link = models[0].link_function
        c_val = float(np.mean([float(m.c_param.item()) for m in models]))
    else:
        link = models.link_function
        c_val = float(models.c_param.item())
    return link, c_val


def _collect_predictions(
    models: Union[StructuredNet, Sequence[StructuredNet]], 
    X: np.ndarray, 
    T: np.ndarray, 
    config: Dict,
    fold_indices: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect predictions from model(s).
    
    When fold_indices is provided (cross-fitting case), each model predicts only on its held-out fold.
    Otherwise, all models predict on all data and predictions are averaged.
    """
    batch_size = config.get("training", {}).get("batch_size", 256)
    
    # Handle cross-fitting case
    if isinstance(models, (list, tuple)) and fold_indices is not None:
        # Each model predicts only on its held-out fold
        pred_all = np.zeros(len(X))
        beta_all = np.zeros((len(X), T.shape[1]))
        
        with torch.no_grad():
            for model, fold_idx in zip(models, fold_indices):
                X_fold = X[fold_idx]
                T_fold = T[fold_idx]
                
                dataset = TensorDataset(
                    torch.tensor(X_fold, dtype=torch.float32),
                    torch.tensor(T_fold, dtype=torch.float32)
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                fold_preds = []
                fold_betas = []
                for batch_x, batch_t in loader:
                    p, b = model(batch_x, batch_t, return_beta=True)
                    fold_preds.append(p.numpy())
                    fold_betas.append(b.numpy())
                
                pred_all[fold_idx] = np.concatenate(fold_preds, axis=0)
                beta_all[fold_idx] = np.concatenate(fold_betas, axis=0)
        
        return pred_all, beta_all
    
    # Original behavior for non-cross-fitting or when fold_indices not provided
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_list = []
    beta_list = []
    with torch.no_grad():
        for batch_x, batch_t in loader:
            if isinstance(models, (list, tuple)):
                preds = []
                betas = []
                for m in models:
                    p, b = m(batch_x, batch_t, return_beta=True)
                    preds.append(p.numpy())
                    betas.append(b.numpy())
                pred_list.append(np.mean(np.stack(preds, axis=0), axis=0))
                beta_list.append(np.mean(np.stack(betas, axis=0), axis=0))
            else:
                pred, beta = models(batch_x, batch_t, return_beta=True)
                pred_list.append(pred.numpy())
                beta_list.append(beta.numpy())
    return np.concatenate(pred_list, axis=0), np.concatenate(beta_list, axis=0)


def _predict_sdl(
    models: Union[StructuredNet, Sequence[StructuredNet]], 
    X: np.ndarray, 
    t_vec: np.ndarray,
    fold_indices: Optional[List[np.ndarray]] = None
) -> float:
    """
    Predict using SDL (Structured Deep Learning) model.
    
    When fold_indices is provided (cross-fitting case), each model predicts only on its held-out fold.
    Otherwise, all models predict on all data and predictions are averaged.
    """
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        t_batch = torch.tensor(np.repeat(t_vec[None, :], len(X), axis=0), dtype=torch.float32)
        
        # Handle cross-fitting case
        if isinstance(models, (list, tuple)) and fold_indices is not None:
            pred_all = np.zeros(len(X))
            for model, fold_idx in zip(models, fold_indices):
                x_fold = x_tensor[fold_idx]
                t_fold = t_batch[fold_idx]
                pred, _ = model(x_fold, t_fold)
                pred_all[fold_idx] = pred.detach().cpu().numpy().squeeze()
            return float(np.mean(pred_all))
        
        # Original behavior
        if isinstance(models, (list, tuple)):
            preds = []
            for m in models:
                pred, _ = m(x_tensor, t_batch)
                preds.append(pred.detach().cpu().numpy())
            return float(np.mean(np.concatenate(preds, axis=0)))
        pred, _ = models(x_tensor, t_batch)
        return float(pred.mean().item())


def _dedl_predict(
    models: Union[StructuredNet, Sequence[StructuredNet]], 
    X: np.ndarray, 
    T: np.ndarray, 
    Y: np.ndarray, 
    t_star: np.ndarray, 
    config: Dict,
    fold_indices: Optional[List[np.ndarray]] = None
) -> float:
    ridge = float(config.get("debias", {}).get("ridge", 1e-3))
    lambda_weighting = config.get("debias", {}).get("lambda_weighting", "uniform")
    pred_all, beta_all = _collect_predictions(models, X, T, config, fold_indices)

    link, c_param = _get_model_stats(models)

    m = T.shape[1] - 1
    t_candidates = _enumerate_treatments(m)

    unique_T, counts_T = np.unique(T, axis=0, return_counts=True)
    total = len(T)
    prob_map = {tuple(row): count / total for row, count in zip(unique_T, counts_T)}

    corrected_pred = []
    for i in range(len(X)):
        beta = beta_all[i]
        u_obs = beta.dot(T[i])
        g_obs = _compute_gradient(link, c_param, u_obs, T[i])
        lambda_mat = np.zeros((len(g_obs), len(g_obs)))
        for t_prime in t_candidates:
            u_prime = beta.dot(t_prime)
            g_prime = _compute_gradient(link, c_param, u_prime, t_prime)
            if lambda_weighting == "empirical":
                w = prob_map.get(tuple(t_prime.tolist()), 0.0)
            else:
                w = 1.0
            lambda_mat += w * np.outer(g_prime, g_prime)
        lambda_mat += ridge * np.eye(lambda_mat.shape[0])
        lambda_inv = np.linalg.inv(lambda_mat)
        residual = Y[i] - pred_all[i]
        u_star = beta.dot(t_star)
        g_star = _compute_gradient(link, c_param, u_star, t_star)
        corrected_pred.append(float(pred_all[i] + g_star @ lambda_inv @ g_obs * residual))
    return float(np.mean(corrected_pred))


def evaluate_methods(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    trained_model: Union[StructuredNet, Sequence[StructuredNet]],
    config: Dict,
    t_stars: List[np.ndarray],
    fold_indices: Optional[List[np.ndarray]] = None,
) -> List[Dict[str, float]]:
    """
    Evaluate baseline and DeDL estimators for each target treatment.
    
    When fold_indices is provided (cross-fitting case), each model predicts only on its held-out fold.
    This ensures proper cross-fitting methodology where predictions are made only on data not seen during training.
    """
    results = []
    m = T.shape[1] - 1
    t_candidates = _enumerate_treatments(m)

    pdl_layers = config.get("model", {}).get("pdl_layers", [64, 64])
    pdl_model = train_plain_dnn(X, T, Y, pdl_layers, config)

    baseline_t0 = t_candidates[0]
    la_base = _la_baseline(T, Y, baseline_t0)
    lr_base = _lr_baseline(X, T, Y, baseline_t0)
    pdl_base = float(pdl_model(torch.tensor(np.concatenate([X, np.repeat(baseline_t0[None, :], len(X), axis=0)], axis=1), dtype=torch.float32)).detach().mean().item())
    sdl_base = _predict_sdl(trained_model, X, baseline_t0, fold_indices)
    dedl_base = _dedl_predict(trained_model, X, T, Y, baseline_t0, config, fold_indices)

    for t_star in t_stars:
        la_pred = _la_baseline(T, Y, t_star)
        lr_pred = _lr_baseline(X, T, Y, t_star)

        pdl_pred = float(pdl_model(torch.tensor(np.concatenate([X, np.repeat(t_star[None, :], len(X), axis=0)], axis=1), dtype=torch.float32)).detach().mean().item())
        sdl_pred = _predict_sdl(trained_model, X, t_star, fold_indices)

        dedl_pred = _dedl_predict(trained_model, X, T, Y, t_star, config, fold_indices)

        treatment_key = "".join(str(int(b)) for b in t_star[1:])
        results.append({
            "treatment": t_star.tolist(),
            "treatment_key": treatment_key,
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
