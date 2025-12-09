from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import StructuredNet, PlainNet


def _build_dataloader(x: np.ndarray, t: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _maybe_l1(model: torch.nn.Module, l1_weight: float) -> torch.Tensor:
    if l1_weight <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return sum(p.abs().sum() for p in model.parameters()) * l1_weight


def train_model(model: StructuredNet, dataloader: DataLoader, config: Dict) -> List[float]:
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    epochs = int(train_cfg.get("epochs", 1000))
    mse_threshold = train_cfg.get("mse_threshold", None)
    if mse_threshold is not None:
        mse_threshold = float(mse_threshold)
    patience = int(train_cfg.get("patience", 10))
    l1_weight = float(train_cfg.get("l1_weight", 0.0))

    criterion_name = train_cfg.get("loss_fn", "mse").lower()
    if criterion_name == "mae":
        criterion: nn.Module = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses: List[float] = []
    best_mse = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()
            pred, _ = model(batch_x, batch_t)
            loss = criterion(pred, batch_y) + _maybe_l1(model, l1_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)

        if epoch % train_cfg.get("log_every", 100) == 0:
            pass

        model.eval()
        with torch.no_grad():
            all_pred, _ = model(dataloader.dataset.tensors[0], dataloader.dataset.tensors[1])
            mse = nn.MSELoss()(all_pred, dataloader.dataset.tensors[2]).item()
        if mse < best_mse:
            best_mse = mse
            no_improve = 0
        else:
            no_improve += 1
        if mse_threshold is not None and mse < mse_threshold:
            break
        if no_improve >= patience:
            break
    return losses


def train_plain_dnn(x: np.ndarray, t: np.ndarray, y: np.ndarray, layers: List[int], config: Dict) -> PlainNet:
    batch_size = int(config.get("training", {}).get("batch_size", 256))
    lr = float(config.get("training", {}).get("lr", 1e-3))
    weight_decay = float(config.get("training", {}).get("weight_decay", 0.0))
    epochs = int(config.get("training", {}).get("epochs", 500))

    dataset = TensorDataset(torch.tensor(np.concatenate([x, t], axis=1), dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PlainNet(dataset.tensors[0].shape[1], layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model


def cross_fit(model_factory, x: np.ndarray, t: np.ndarray, y: np.ndarray, config: Dict) -> List[StructuredNet]:
    k = int(config.get("training", {}).get("cv_folds", 1))
    if k <= 1:
        model = model_factory()
        loader = _build_dataloader(x, t, y, int(config.get("training", {}).get("batch_size", 256)))
        train_model(model, loader, config)
        return [model]

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    models: List[StructuredNet] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        loader = _build_dataloader(x[train_idx], t[train_idx], y[train_idx], int(config.get("training", {}).get("batch_size", 256)))
        model = model_factory()
        train_model(model, loader, config)
        models.append(model)
    return models

