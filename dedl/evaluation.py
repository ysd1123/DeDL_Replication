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


def _collect_predictions(
    models: Union[StructuredNet, Sequence[StructuredNet]],
    X: np.ndarray,
    T: np.ndarray,
    config: Dict,
    respect_folds: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = config.get("training", {}).get("batch_size", 256)
    
    # Set models to evaluation mode before processing
    if isinstance(models, (list, tuple)):
        for m in models:
            m.eval()
    else:
        models.eval()
    
    # Handle cross-fitting case
    if respect_folds and isinstance(models, (list, tuple)) and len(models) > 1:
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
        # Single model case or averaged prediction for unseen data
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


def _structured_predictions_for_t(
    models: Union[StructuredNet, Sequence[StructuredNet]],
    X: np.ndarray,
    t_vec: np.ndarray,
    config: Dict,
) -> np.ndarray:
    T_target = np.repeat(t_vec[None, :], len(X), axis=0)
    preds, _ = _collect_predictions(models, X, T_target, config, respect_folds=False)
    return preds


def _predict_sdl(
    models: Union[StructuredNet, Sequence[StructuredNet]],
    X: np.ndarray,
    t_vec: np.ndarray,
    config: Dict,
    return_array: bool = False,
):
    preds = _structured_predictions_for_t(models, X, t_vec, config)
    if return_array:
        return preds
    return float(preds.mean())


def _prepare_la_stats(T: np.ndarray, Y: np.ndarray) -> Dict[Tuple[int, ...], float]:
    stats = {}
    unique_rows = np.unique(T, axis=0)
    for row in unique_rows:
        mask = np.all(T == row, axis=1)
        if mask.any():
            stats[tuple(row.astype(int))] = float(Y[mask].mean())
    return stats


def _la_predict(stats: Dict[Tuple[int, ...], float], t_vec: np.ndarray) -> float:
    key = tuple(t_vec.astype(int))
    if key in stats:
        return stats[key]
    baseline_key = tuple([1] + [0] * (len(t_vec) - 1))
    baseline = stats.get(baseline_key)
    if baseline is None:
        return float("nan")
    additive = 0.0
    for idx, val in enumerate(t_vec[1:], start=1):
        if val == 1:
            single = [1] + [0] * (len(t_vec) - 1)
            single[idx] = 1
            additive += stats.get(tuple(single), baseline) - baseline
    return baseline + additive


def _fit_lr(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    design = np.concatenate([X, T], axis=1)
    coef, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
    return coef


def _lr_predict(coef: np.ndarray, X: np.ndarray, t_star: np.ndarray) -> float:
    target_design = np.concatenate([X, np.repeat(t_star[None, :], len(X), axis=0)], axis=1)
    preds = target_design @ coef
    return float(preds.mean())


def _pdl_predict(model: torch.nn.Module, X: np.ndarray, t_star: np.ndarray) -> float:
    inp = np.concatenate([X, np.repeat(t_star[None, :], len(X), axis=0)], axis=1)
    tensor_inp = torch.tensor(inp, dtype=torch.float32)
    with torch.no_grad():
        preds = model(tensor_inp).detach().cpu().numpy()
    return float(preds.mean())


def _dedl_predict(
    models: Union[StructuredNet, Sequence[StructuredNet]],
    target_X: np.ndarray,
    ref_X: np.ndarray,
    ref_T: np.ndarray,
    ref_Y: np.ndarray,
    t_star: np.ndarray,
    config: Dict,
) -> float:
    ridge = float(config.get("debias", {}).get("ridge", 1e-3))
    lambda_weighting = config.get("debias", {}).get("lambda_weighting", "uniform")
    target_preds = _predict_sdl(models, target_X, t_star, config, return_array=True)
    pred_ref, beta_ref = _collect_predictions(models, ref_X, ref_T, config, respect_folds=True)

    link, c_param = _get_model_stats(models)

    m = ref_T.shape[1] - 1
    t_candidates = _enumerate_treatments(m)

    unique_T, counts_T = np.unique(ref_T, axis=0, return_counts=True)
    total = len(ref_T)
    prob_map = {tuple(row): count / total for row, count in zip(unique_T, counts_T)}

    corrections = []
    for i in range(len(ref_X)):
        beta = beta_ref[i]
        u_obs = beta.dot(ref_T[i])
        g_obs = _compute_gradient(link, c_param, u_obs, ref_T[i])
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
        residual = ref_Y[i] - pred_ref[i]
        u_star = beta.dot(t_star)
        g_star = _compute_gradient(link, c_param, u_star, t_star)
        corrections.append(float(g_star @ lambda_inv @ g_obs * residual))
    correction = float(np.mean(corrections)) if corrections else 0.0
    return float(target_preds.mean() + correction)


def evaluate_methods(
    train_split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    trained_model: Union[StructuredNet, Sequence[StructuredNet]],
    config: Dict,
    t_stars: List[np.ndarray],
) -> List[Dict[str, float]]:
    """Evaluate baseline and DeDL estimators for each target treatment."""
    results = []
    x_train, t_train, y_train = train_split
    x_test, t_test, y_test = test_split
    m = t_train.shape[1] - 1
    t_candidates = _enumerate_treatments(m)

    pdl_layers = config.get("model", {}).get("pdl_layers", [64, 64])
    pdl_model = train_plain_dnn(x_train, t_train, y_train, pdl_layers, config)

    baseline_t0 = t_candidates[0]
    la_stats = _prepare_la_stats(t_train, y_train)
    lr_coef = _fit_lr(x_train, t_train, y_train)

    la_base = _la_predict(la_stats, baseline_t0)
    lr_base = _lr_predict(lr_coef, x_test, baseline_t0)
    pdl_base = _pdl_predict(pdl_model, x_test, baseline_t0)
    sdl_base = _predict_sdl(trained_model, x_test, baseline_t0, config)
    dedl_base = _dedl_predict(trained_model, x_test, x_train, t_train, y_train, baseline_t0, config)

    for t_star in t_stars:
        la_pred = _la_predict(la_stats, t_star)
        lr_pred = _lr_predict(lr_coef, x_test, t_star)
        pdl_pred = _pdl_predict(pdl_model, x_test, t_star)
        sdl_pred = _predict_sdl(trained_model, x_test, t_star, config)
        dedl_pred = _dedl_predict(trained_model, x_test, x_train, t_train, y_train, t_star, config)

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
