from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from dnn_arch import DeDLNet, PDLNet
from utils import loss_mse

def train_dedl_model(X_train, T_train, Y_train, config: Dict) -> DeDLNet:
    """
    训练用于 DeDL/SDL 的结构化 DNN。
    """
    device = config["device"]
    m = config["m"]
    d_x = config["d_x"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    model = DeDLNet(d_x=d_x, m=m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(T_train),
        torch.from_numpy(Y_train)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, tb, yb in dl:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            u = model(xb)
            loss = loss_mse(yb, tb, u).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(ds)
        print(f"[DeDL/SDL] Epoch {ep+1}/{epochs}, train MSE={avg_loss:.4f}")

    return model


def train_pdl_model(X_train, T_train, Y_train, config: Dict) -> PDLNet:
    """
    训练 PDL 模型（完全自由 DNN）。
    """
    device = config["device"]
    m = config["m"]
    d_x = config["d_x"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    model = PDLNet(d_x=d_x, m=m).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(T_train),
        torch.from_numpy(Y_train)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, tb, yb in dl:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)

            y_hat = model(xb, tb)
            loss = ((yb - y_hat) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(ds)
        print(f"[PDL] Epoch {ep+1}/{epochs}, train MSE={avg_loss:.4f}")

    return model