from __future__ import annotations
import itertools
import pathlib
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch

ArrayLike = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_treatments(t_combo_obs: List[List[int]], t_dist_obs: List[float], n: int) -> np.ndarray:
    combos = np.array(t_combo_obs)
    probs = np.array(t_dist_obs)
    probs = probs / probs.sum()
    indices = np.random.choice(len(combos), size=n, p=probs)
    return combos[indices]


def _generate_coefficients(dist: str, size: int, scale: float) -> np.ndarray:
    if dist == "normal":
        return np.random.normal(scale=scale, size=size)
    if dist == "uniform":
        return np.random.uniform(-scale, scale, size=size)
    if dist == "laplace":
        return np.random.laplace(scale=scale, size=size)
    raise ValueError(f"Unsupported coefficient distribution: {dist}")


def _apply_outcome_function(
    x: np.ndarray,
    t: np.ndarray,
    coef: np.ndarray,
    c_true: float,
    d_true: float,
    noise_level: float,
    fn: str,
    noise_type: str = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
    u = x.dot(coef[: x.shape[1]]) + t.dot(coef[x.shape[1] :])
    if fn == "sigmoid":
        y_true = c_true / (1 + np.exp(-u)) + d_true
    elif fn == "linear":
        y_true = u + d_true
    elif fn == "polynomial":
        y_true = u + 0.1 * u ** 2 + d_true
    else:
        raise ValueError(f"Unsupported outcome function: {fn}")

    if noise_type == "normal":
        noise = np.random.normal(scale=noise_level, size=len(x))
    elif noise_type == "laplace":
        noise = np.random.laplace(scale=noise_level, size=len(x))
    else:
        noise = np.random.uniform(-noise_level, noise_level, size=len(x))
    return y_true + noise, y_true


def _generate_synthetic(config: Dict) -> Tuple[ArrayLike, ArrayLike, np.ndarray]:
    data_cfg = config["data"]
    _set_seed(data_cfg.get("seed"))
    m = int(data_cfg["m"])
    d_c = int(data_cfg["d_c"])
    train_size = int(data_cfg["train_size"])
    test_size = int(data_cfg.get("test_size", train_size))
    n = train_size + test_size

    t_combo_obs = data_cfg.get("t_combo_obs") or [list(seq) for seq in itertools.product([0, 1], repeat=m)]
    t_dist_obs = data_cfg.get("t_dist_obs") or [1 / (2 ** m)] * (2 ** m)

    noise_level = float(data_cfg.get("noise_level", 0.1))
    noise_type = data_cfg.get("noise_type", "normal")
    outcome_fn = data_cfg.get("outcome_fn", "sigmoid")

    coef_scale = float(data_cfg.get("coef_scale", 1.0))
    coef_dist = data_cfg.get("coef_dist", "normal")
    coef = _generate_coefficients(coef_dist, d_c + m + 1, coef_scale)

    c_true_range = data_cfg.get("c_true_range", [1.0, 2.0])
    c_true = float(np.random.uniform(c_true_range[0], c_true_range[1]))
    d_true = float(data_cfg.get("d_true", 0.0))

    x = np.random.normal(size=(n, d_c))
    if data_cfg.get("cov_shift", False):
        weights = np.linspace(1, 2, d_c)
        x = x * weights

    t = _sample_treatments(t_combo_obs, t_dist_obs, n)
    t = np.concatenate([np.ones((n, 1)), t], axis=1)

    y, y_true = _apply_outcome_function(x, t, coef, c_true, d_true, noise_level, outcome_fn, noise_type)

    train_idx = np.arange(train_size)
    test_idx = np.arange(train_size, n)
    sim_info = {
        "x": x,
        "t": t,
        "y_true": y_true,
        "coef": coef,
        "c_true": c_true,
        "d_true": d_true,
        "noise_level": noise_level,
    }
    return (x[train_idx], t[train_idx], y[train_idx]), (x[test_idx], t[test_idx], y[test_idx]), sim_info


def _process_real(config: Dict) -> Tuple[ArrayLike, ArrayLike, np.ndarray]:
    data_cfg = config["data"]
    _set_seed(data_cfg.get("seed"))
    path = pathlib.Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    factor_cols = data_cfg.get("factor_cols", [])
    feature_cols = data_cfg.get("feature_cols", [])
    outcome_col = data_cfg.get("outcome_col")
    dropna = data_cfg.get("dropna", True)

    if dropna:
        df = df.dropna(subset=factor_cols + feature_cols + [outcome_col])
    else:
        # Check for missing values in factor_cols
        if df[factor_cols].isnull().any().any():
            raise ValueError("Missing values found in factor_cols. Binary treatments cannot be imputed with mean. Please clean your data or use dropna=True.")
        df = df.fillna(df.mean(numeric_only=True))

    seed = data_cfg.get("seed", 42)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)

    t = df[factor_cols].to_numpy(dtype=float)
    t = np.concatenate([np.ones((len(t), 1)), t], axis=1)
    x = df[feature_cols].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)

    train_size = int(data_cfg.get("train_size", int(0.7 * len(df))))
    # Clamp train_size to at most len(df)
    train_size = min(train_size, len(df))
    test_size = int(data_cfg.get("test_size", len(df) - train_size))
    # Clamp test_size so that train_size + test_size <= len(df)
    test_size = min(test_size, len(df) - train_size)
    if train_size + test_size < len(df):
        # Optionally warn if not all data is used
        import warnings
        warnings.warn(
            f"train_size + test_size ({train_size} + {test_size}) < total rows ({len(df)}). Some data will be unused."
        )
    train_idx = np.arange(train_size)
    test_idx = np.arange(train_size, train_size + test_size)

    return (x[train_idx], t[train_idx], y[train_idx]), (x[test_idx], t[test_idx], y[test_idx]), y


def load_data(config: Dict) -> Tuple[ArrayLike, ArrayLike, np.ndarray]:
    """Load synthetic or real data according to config."""
    data_type = config["data"]["type"]
    if data_type == "synthetic":
        return _generate_synthetic(config)
    if data_type == "real":
        train, test, y = _process_real(config)
        sim_info = {"x": np.concatenate([train[0], test[0]]), "t": np.concatenate([train[1], test[1]]), "y_obs": np.concatenate([train[2], test[2]])}
        return train, test, sim_info
    raise ValueError(f"Unsupported data type: {data_type}")
