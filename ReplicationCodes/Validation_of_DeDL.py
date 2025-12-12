#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeDL synthetic experiments: Validation of DeDL vs LA/LR/PDL/SDL

References:
- Ye et al. (2025), "Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments"
    - Online Appendix D.1/D.2 (Synthetic Experiments Design)
    - Appendix C (Benchmark: LA/LR/PDL/SDL)

实验流程：
    1. 生成合成数据（训练集 + 推断集 + 计算真值用的大样本）
    2. 训练 DeDL/SDL/PDL 模型，以及 LR 基准
    3. 计算所有 2^m 个 treatment 组合的 ATE 估计值
    4. 输出中间数据与结果到 CSV，同时保存模型权重到 models/ 目录
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from dgp import generate_phi_functions, generate_data_once,estimate_true_ate_mc
from infer import estimate_ate_all_methods

# 全局配置
CONFIG = {
    "m": 4,                 # 实验个数 m（e.g. 4,6,8,10）
    "d_x": 10,              # 协变量维度 d_X
    "n_train": 2000,        # 训练阶段样本数（论文用 500m，可按需调大）
    "n_infer": 2000,        # 推断阶段样本数
    "n_true_mc": 20000,     # 计算真值 ATE 的 Monte Carlo 样本量
    "batch_size": 512,      # 训练批次大小
    "lr": 1e-3,             # 学习率
    "epochs": 1000,           # 训练轮数
    "lambda_reg_L": 5e-4,   # L(x) 的数值稳定正则化
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./output_dedl_synth",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "models"), exist_ok=True)

def set_seed(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 严格确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 主流程
def main():
    set_seed(1896)
    cfg = CONFIG
    rng = np.random.RandomState(12345)
    m = cfg["m"]
    d_x = cfg["d_x"]

    # 1) 生成 φ 参数
    A, phi_m1 = generate_phi_functions(m, d_x, rng)

    # 2) 生成训练和推断数据（部分观测机制：m+2 组合）
    X_train, T_train, Y_train = generate_data_once(
        n=cfg["n_train"], m=m, d_x=d_x,
        A=A, phi_m1=phi_m1,
        observed_only=True,
        rng=rng
    )
    X_infer, T_infer, Y_infer = generate_data_once(
        n=cfg["n_infer"], m=m, d_x=d_x,
        A=A, phi_m1=phi_m1,
        observed_only=True,
        rng=rng
    )

    # 保存中间数据
    df_train = pd.DataFrame(
        np.concatenate([X_train, T_train, Y_train.reshape(-1,1)], axis=1)
    )
    df_infer = pd.DataFrame(
        np.concatenate([X_infer, T_infer, Y_infer.reshape(-1,1)], axis=1)
    )
    df_train.to_csv(os.path.join(cfg["output_dir"], "data_train.csv"), index=False)
    df_infer.to_csv(os.path.join(cfg["output_dir"], "data_infer.csv"), index=False)

    # 3) 估计真值 ATE（MC, 全 2^m 组合）
    df_true = estimate_true_ate_mc(
        m=m, d_x=d_x, A=A, phi_m1=phi_m1,
        n_mc=cfg["n_true_mc"], rng=rng
    )
    df_true.to_csv(os.path.join(cfg["output_dir"], "true_ate.csv"), index=False)

    # 4) 估计所有方法的 ATE 并比较
    estimate_ate_all_methods(
        X_train, T_train, Y_train,
        X_infer, T_infer, Y_infer,
        df_true_ate=df_true,
        config=cfg
    )


if __name__ == "__main__":
    main()
