import torch
import torch.nn as nn

# DNN 结构定义
# DeDL/SDL 方法共用 DNN 结构

class DeDLNet(nn.Module):
    """
    输出 u(x) = (θ_0, θ_1,...,θ_m, θ_{m+1}) 的 DNN。
    结构：简单两层全连接 + ReLU + 输出层
    """
    def __init__(self, d_x: int, m: int, hidden_width: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_x, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc_out = nn.Linear(hidden_width, m + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        u = self.fc_out(h)
        return u

class PDLNet(nn.Module):
    """
    PDL：完全自由的 DNN，用 (x,t) 一起作为输入，直接输出 y_pred。
    """
    def __init__(self, d_x: int, m: int, hidden_width: int = 10):
        super().__init__()
        inp_dim = d_x + m
        self.fc1 = nn.Linear(inp_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(torch.cat([x, t], dim=-1)))
        h = torch.relu(self.fc2(h))
        y = self.fc3(h).squeeze(-1)
        return y