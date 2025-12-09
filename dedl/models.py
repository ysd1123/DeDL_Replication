from __future__ import annotations
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredNet(nn.Module):
    """Structured deep network that predicts beta(x) and applies a link function."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        self.m = int(config["data"]["m"])
        input_dim = int(config["data"]["d_c"]) + self.m + 1
        layers: Iterable[int] = model_cfg.get("layers", [64, 64])

        modules = []
        prev_dim = input_dim
        for dim in layers:
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(nn.ReLU())
            prev_dim = dim
        modules.append(nn.Linear(prev_dim, self.m + 1))
        self.network = nn.Sequential(*modules)

        self.link_function = model_cfg.get("link_function", "sigmoid")
        c_init = float(model_cfg.get("c_init", 1.0))
        d_init = float(model_cfg.get("d_init", 0.0))
        self.learn_scale = bool(model_cfg.get("learn_scale", True))
        self.learn_shift = bool(model_cfg.get("learn_shift", True))
        if self.learn_scale:
            self.c_param = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
        else:
            self.register_buffer("c_param", torch.tensor([c_init], dtype=torch.float32))
        if self.learn_shift:
            self.d_param = nn.Parameter(torch.tensor([d_init], dtype=torch.float32))
        else:
            self.register_buffer("d_param", torch.tensor([d_init], dtype=torch.float32))

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_beta: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        xt = torch.cat([x, t], dim=1)
        beta = self.network(xt)
        u = (beta * t).sum(dim=1)
        if self.link_function == "sigmoid":
            y_hat = self.c_param / (1 + torch.exp(-u)) + self.d_param
        elif self.link_function == "linear":
            y_hat = u + self.d_param
        elif self.link_function == "softplus":
            y_hat = self.c_param * F.softplus(u) + self.d_param
        else:
            raise ValueError(f"Unsupported link function: {self.link_function}")
        if return_beta:
            return y_hat, beta
        return y_hat, beta.detach()


class PlainNet(nn.Module):
    """Simple fully-connected network used for PDL baselines."""

    def __init__(self, input_dim: int, layers: Iterable[int]):
        super().__init__()
        modules = []
        prev_dim = input_dim
        for dim in layers:
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(nn.ReLU())
            prev_dim = dim
        modules.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
