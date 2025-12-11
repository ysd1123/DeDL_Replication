from __future__ import annotations
import itertools
from typing import Dict, List, Sequence, Tuple, Union

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


def _get_cv_seed(config: Dict) -> int:
    """Get the cross-validation seed from config."""
    return config.get("training", {}).get("cv_seed", config.get("data", {}).get("seed", 42))


def _assign_to_folds(n_samples: int, n_folds: int, cv_seed: int) -> np.ndarray:
    """Assign data indices to folds using the same logic as cross_fit."""
    rng = np.random.RandomState(cv_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    # Create an array mapping each index to its fold number
    fold_assignment = np.zeros(n_samples, dtype=int)
    for fold_idx, fold_indices in enumerate(folds):
        fold_assignment[fold_indices] = fold_idx
    return fold_assignment


def _collect_predictions(models: Union[StructuredNet, Sequence[StructuredNet]], X: np.ndarray, T: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = config.get("training", {}).get("batch_size", 256)
    
    # Set models to evaluation mode before processing
    if isinstance(models, (list, tuple)):
        for m in models:
            m.eval()
    else:
        models.eval()
    
    # Handle cross-fitting case
    if isinstance(models, (list, tuple)) and len(models) > 1:
        # Get fold assignments for the data
        n_folds = len(models)
        cv_seed = _get_cv_seed(config)
        fold_assignment = _assign_to_folds(len(X), n_folds, cv_seed)
        
        # Predict each fold with its corresponding model
        pred_all = np.zeros(len(X))
        beta_all = np.zeros((len(X), T.shape[1]))
        
        for fold_idx in range(n_folds):
            # Get indices for this fold
            fold_mask = fold_assignment == fold_idx
            if not fold_mask.any():
                continue
            
            fold_X = X[fold_mask]
            fold_T = T[fold_mask]
            
            # Create dataloader for this fold
            dataset = TensorDataset(torch.tensor(fold_X, dtype=torch.float32), torch.tensor(fold_T, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            pred_list = []
            beta_list = []
            with torch.no_grad():
                for batch_x, batch_t in loader:
                    pred, beta = models[fold_idx](batch_x, batch_t, return_beta=True)
                    pred_list.append(pred.numpy())
                    beta_list.append(beta.numpy())
            
            pred_all[fold_mask] = np.concatenate(pred_list, axis=0)
            beta_all[fold_mask] = np.concatenate(beta_list, axis=0)
        
        return pred_all, beta_all
    else:
        # Single model case
        if isinstance(models, (list, tuple)):
            model = models[0]
        else:
            model = models
        
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        pred_list = []
        beta_list = []
        with torch.no_grad():
            for batch_x, batch_t in loader:
                pred, beta = model(batch_x, batch_t, return_beta=True)
                pred_list.append(pred.numpy())
                beta_list.append(beta.numpy())
        return np.concatenate(pred_list, axis=0), np.concatenate(beta_list, axis=0)


def _predict_sdl(models: Union[StructuredNet, Sequence[StructuredNet]], X: np.ndarray, t_vec: np.ndarray, config: Dict) -> float:
    # Set models to evaluation mode before processing
    if isinstance(models, (list, tuple)):
        for m in models:
            m.eval()
    else:
        models.eval()
    
    with torch.no_grad():
        t_batch = torch.tensor(np.repeat(t_vec[None, :], len(X), axis=0), dtype=torch.float32)
        
        # Handle cross-fitting case
        if isinstance(models, (list, tuple)) and len(models) > 1:
            # Get fold assignments for the data
            n_folds = len(models)
            cv_seed = _get_cv_seed(config)
            fold_assignment = _assign_to_folds(len(X), n_folds, cv_seed)
            
            # Predict each fold with its corresponding model
            pred_all = np.zeros(len(X))
            
            for fold_idx in range(n_folds):
                # Get indices for this fold
                fold_mask = fold_assignment == fold_idx
                if not fold_mask.any():
                    continue
                
                x_tensor = torch.tensor(X[fold_mask], dtype=torch.float32)
                t_fold = t_batch[fold_mask]
                
                pred, _ = models[fold_idx](x_tensor, t_fold)
                pred_all[fold_mask] = pred.detach().cpu().numpy()
            
            return float(pred_all.mean())
        else:
            # Single model case
            if isinstance(models, (list, tuple)):
                model = models[0]
            else:
                model = models
            
            x_tensor = torch.tensor(X, dtype=torch.float32)
            pred, _ = model(x_tensor, t_batch)
            return float(pred.mean().item())


def _dedl_predict(models: Union[StructuredNet, Sequence[StructuredNet]], X: np.ndarray, T: np.ndarray, Y: np.ndarray, t_star: np.ndarray, config: Dict) -> float:
    ridge = float(config.get("debias", {}).get("ridge", 1e-3))
    lambda_weighting = config.get("debias", {}).get("lambda_weighting", "uniform")
    pred_all, beta_all = _collect_predictions(models, X, T, config)

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
                w = prob_map.get(tuple(t_prime), 0.0)
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
    sdl_base = _predict_sdl(trained_model, X, baseline_t0, config)
    dedl_base = _dedl_predict(trained_model, X, T, Y, baseline_t0, config)

    for t_star in t_stars:
        la_pred = _la_baseline(T, Y, t_star)
        lr_pred = _lr_baseline(X, T, Y, t_star)

        pdl_pred = float(pdl_model(torch.tensor(np.concatenate([X, np.repeat(t_star[None, :], len(X), axis=0)], axis=1), dtype=torch.float32)).detach().mean().item())
        sdl_pred = _predict_sdl(trained_model, X, t_star, config)

        dedl_pred = _dedl_predict(trained_model, X, T, Y, t_star, config)

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
