#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    python postprocess_dedl_synth.py --output_dir ./output_dedl_synth
"""

import os
import ast
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 95% 置信区间系数、数值容差
Z_975 = 1.96
TOL = 1e-8

# 参与比较的所有方法
ESTIMATORS: List[str] = ["LA", "LR", "PDL", "SDL", "DeDL"]

def parse_t_vec(s) -> Tuple[int, ...]:
    """把 true_ate.csv 里的字符串 '[0.0, 1.0, 0.0, ...]' 转成元组 (0,1,0,...)。"""
    if isinstance(s, str):
        t = ast.literal_eval(s)
    elif isinstance(s, (list, tuple, np.ndarray)):
        t = list(s)
    else:
        raise ValueError(f"无法解析 t_vec: {s!r}")
    # 强制转成 0/1 的整型，更方便做集合比较和显示
    return tuple(int(round(float(v))) for v in t)


def observed_treatments_m_plus_2(m: int):
    """
    区分观测/未观测的 treatment 组合：
      - 全 0
      - 每次一个 1（m 个）
      - 一个 overlap 组合 (1,1,0,...,0)
    返回值：set[tuple]，每个元素是长度为 m 的 0/1 元组。
    """
    zero = [0] * m
    combos = [tuple(zero)]
    for k in range(m):
        t = zero.copy()
        t[k] = 1
        combos.append(tuple(t))
    if m >= 2:
        t = zero.copy()
        t[0] = 1
        t[1] = 1
        combos.append(tuple(t))
    return set(combos)


def load_all(output_dir: str):
    """
    读取 output_dir 下的所有中间结果并合并成一个 DataFrame。

    返回：
        df: 每一行对应一个 treatment 组合，包含真值和各方法估计
        m:  treatment 维度
        idx0: baseline 组合（全 0）的行索引
    """
    true_path = os.path.join(output_dir, "true_ate.csv")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"找不到 {true_path}")

    df_true = pd.read_csv(true_path)
    if "t_vec" not in df_true.columns:
        raise ValueError("true_ate.csv 里需要包含列 't_vec' 才能解析 treatment 组合")

    df_true["t_tuple"] = df_true["t_vec"].apply(parse_t_vec)
    m = len(df_true["t_tuple"].iloc[0])

    df = df_true.copy()

    # 合并各方法的估计结果
    for est in ESTIMATORS:
        path = os.path.join(output_dir, f"est_ates_{est}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 {path}")
        tmp = pd.read_csv(path)
        needed_cols = {"combo_index", f"ATE_hat_{est}", f"SE_{est}"}
        missing = needed_cols.difference(tmp.columns)
        if missing:
            raise ValueError(f"{path} 缺少列: {missing}")
        tmp = tmp[list(needed_cols)]
        df = df.merge(tmp, on="combo_index", how="left")

    # 标记 observed / unobserved
    obs_set = observed_treatments_m_plus_2(m)
    df["is_observed"] = df["t_tuple"].apply(lambda t: tuple(t) in obs_set)
    df["is_unobserved"] = ~df["is_observed"]

    # baseline 组合（全 0）
    def is_baseline(t):
        return all(v == 0 for v in t)

    mask0 = df["t_tuple"].apply(is_baseline)
    if not mask0.any():
        raise RuntimeError("没有找到 baseline 组合 (0,...,0)，请检查 true_ate.csv。")
    idx0 = df.index[mask0][0]

    return df, m, idx0

# 误差指标：CDR / MAPE / MSE / MAE
def compute_cdr(mu_true, mu_hat, se_hat, mask=None) -> float:
    """
    计算 CDR（Correct Direction Ratio）：
    - 真值显著 => 估计也要显著且符号正确
    - 真值不显著 => 估计的 CI 必须包含 0
    """
    mu_true = np.asarray(mu_true)
    mu_hat = np.asarray(mu_hat)
    se_hat = np.asarray(se_hat)

    if mask is None:
        mask = np.ones_like(mu_true, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    mu_true = mu_true[mask]
    mu_hat = mu_hat[mask]
    se_hat = se_hat[mask]

    # 真值是否“显著”：这里没有 ground-truth 方差，只用 |μ|>0 近似
    sig_true = np.abs(mu_true) > TOL

    # 估计是否显著：0 是否落在 95% CI 里
    ci_low = mu_hat - Z_975 * se_hat
    ci_high = mu_hat + Z_975 * se_hat
    sig_hat = ~((ci_low <= 0.0) & (ci_high >= 0.0))

    sign_true = np.sign(mu_true)
    sign_hat = np.sign(mu_hat)

    correct = np.zeros_like(mu_true, dtype=bool)

    # 真值显著：要同时“显著 + 符号对”
    mask_sig = sig_true
    correct[mask_sig] = sig_hat[mask_sig] & (sign_true[mask_sig] == sign_hat[mask_sig])

    # 真值不显著：只要求估计也“不显著”
    mask_nonsig = ~sig_true
    correct[mask_nonsig] = ~sig_hat[mask_nonsig]

    return float(correct.mean())


def compute_mape_mse_mae(mu_true, mu_hat, mask=None):
    """
    计算 MAPE / MSE / MAE。
    - MAPE：只在 |μ_true| > TOL 的组合上算，结果用百分比表示。
    """
    mu_true = np.asarray(mu_true)
    mu_hat = np.asarray(mu_hat)

    if mask is None:
        mask = np.ones_like(mu_true, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    mu_true = mu_true[mask]
    mu_hat = mu_hat[mask]

    err = mu_hat - mu_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    # MAPE 只看真值非零的组合
    nz = np.abs(mu_true) > TOL
    if nz.any():
        ape = abs_err[nz] / np.abs(mu_true[nz])
        mape = float(ape.mean()) * 100.0
    else:
        mape = np.nan

    mse = float(sq_err.mean())
    mae = float(abs_err.mean())
    return mape, mse, mae

# Table 2：Ground-truth ATE + best treatment
def make_table2(df: pd.DataFrame, m: int, idx0: int, output_dir: str):
    """
    构造合成版 Table 2：
    - 每一行：t, 真值 ATE（可理解为 “relative effect size %”），observed or not, 是否为 true best。
    """
    factor = 100.0  # 把 ATE 看作“百分比”，仅用于展示

    # 真最佳 treatment（按真值 ATE 最大）
    best_idx = df["ATE_true"].idxmax()
    best_t = df.loc[best_idx, "t_tuple"]

    rows = []
    for _, row in df.sort_values("combo_index").iterrows():
        t = row["t_tuple"]
        ate_pct = row["ATE_true"] * factor
        rows.append({
            "t": "(" + ", ".join(str(int(v)) for v in t) + ")",
            "ATE_true_pct": ate_pct,
            "Observed": "Yes" if row["is_observed"] else "No",
            "Is_true_best": bool(tuple(t) == tuple(best_t)),
        })

    df_tab2 = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "Table2_ground_truth_synth.csv")
    df_tab2.to_csv(out_path, index=False)
    print(f"[Table 2] 已保存到 {out_path}")

# Table 3：ATE 的比较（unobserved / all）
def make_table3(df: pd.DataFrame, output_dir: str):
    """
    合成版 Table 3：
    - 上半部分：unobserved treatment 组合上的 CDRu / MAPEu / MSEu / MAEu
    - 下半部分：所有 treatment 组合上的 CDR / MAPE / MSE / MAE
    """
    mu_true = df["ATE_true"].values
    mask_unobs = df["is_unobserved"].values
    n_all = len(mu_true)
    n_unobs = int(mask_unobs.sum())

    rows_u = []
    rows_all = []

    for est in ESTIMATORS:
        mu_hat = df[f"ATE_hat_{est}"].values
        se_hat = df[f"SE_{est}"].values

        # --- Unobserved 组合 ---
        cdr_u = compute_cdr(mu_true, mu_hat, se_hat, mask_unobs)
        mape_u, mse_u, mae_u = compute_mape_mse_mae(mu_true, mu_hat, mask_unobs)
        num_correct_u = int(round(cdr_u * n_unobs))

        rows_u.append({
            "Estimator": est,
            "CDRu_frac": f"{num_correct_u}/{n_unobs}",
            "CDRu": cdr_u,
            "MAPEu_%": mape_u,
            "MSEu": mse_u,
            "MAEu": mae_u,
        })

        # --- 所有组合 ---
        cdr_all = compute_cdr(mu_true, mu_hat, se_hat, None)
        mape_all, mse_all, mae_all = compute_mape_mse_mae(mu_true, mu_hat, None)
        num_correct_all = int(round(cdr_all * n_all))

        rows_all.append({
            "Estimator": est,
            "CDR_frac": f"{num_correct_all}/{n_all}",
            "CDR": cdr_all,
            "MAPE_%": mape_all,
            "MSE": mse_all,
            "MAE": mae_all,
        })

    df_u = pd.DataFrame(rows_u)
    df_all = pd.DataFrame(rows_all)

    out_u = os.path.join(output_dir, "Table3_unobserved_ATE_synth.csv")
    out_all = os.path.join(output_dir, "Table3_all_ATE_synth.csv")
    df_u.to_csv(out_u, index=False)
    df_all.to_csv(out_all, index=False)
    print(f"[Table 3] Unobserved 部分已保存到 {out_u}")
    print(f"[Table 3] All 部分已保存到 {out_all}")

# Table 4：Best treatment identification
def make_table4_best(df: pd.DataFrame, output_dir: str):
    """
    合成版 Table 4：基于 τ(t) = μ(t*) - μ(t) 的指标。
    这里固定 t* 为真值 ATE 最大的组合，对每个估计方法计算：
      - τ_true(t)
      - τ_hat(t) = μ_hat(t*) - μ_hat(t)
      - se_tau ≈ sqrt(se(t*)^2 + se(t)^2)（忽略协方差）
    再复用 CDR / MAPE / MSE / MAE 的定义。
    """
    mu_true = df["ATE_true"].values
    idx_star = int(mu_true.argmax())
    mu_star_true = mu_true[idx_star]
    tau_true = mu_star_true - mu_true  # 长度 = 所有 treatment 组合数
    n = len(tau_true)

    rows = []

    for est in ESTIMATORS:
        mu_hat = df[f"ATE_hat_{est}"].values
        se_hat = df[f"SE_{est}"].values

        mu_hat_star = mu_hat[idx_star]
        se_hat_star = se_hat[idx_star]

        tau_hat = mu_hat_star - mu_hat
        # 粗略近似：方差 add up
        se_tau = np.sqrt(se_hat_star ** 2 + se_hat ** 2)

        cdr = compute_cdr(tau_true, tau_hat, se_tau, None)
        mape, mse, mae = compute_mape_mse_mae(tau_true, tau_hat, None)
        num_correct = int(round(cdr * n))

        rows.append({
            "Estimator": est,
            "CDR_frac": f"{num_correct}/{n}",
            "CDR": cdr,
            "MAPE_%": mape,
            "MSE": mse,
            "MAE": mae,
        })

    df4 = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "Table4_best_treatment_synth.csv")
    df4.to_csv(out_path, index=False)
    print(f"[Table 4] 已保存到 {out_path}")

# Figure 5：SDL vs DeDL + 真值
def plot_figure5_like(df: pd.DataFrame, output_dir: str):
    """
    画一张类似 Figure 5 的图：
      - X 轴：t 组合（去掉 baseline）
      - Y 轴：Relative Effect Size (%)，基于真值 ATE / SDL / DeDL
      - 三条系列：True Effect, DeDL, SDL，每条都有 95% CI（真值的 CI 用 DeDL SE 近似）
    """
    # 不画 baseline (0,...,0)
    def is_baseline(t):
        return all(v == 0 for v in t)

    df_plot = df[~df["t_tuple"].apply(is_baseline)].sort_values("combo_index")

    labels = df_plot["t_tuple"].apply(
        lambda t: "(" + ", ".join(str(int(v)) for v in t) + ")"
    ).tolist()
    x = np.arange(len(labels))

    factor = 100.0  # 当作“百分比”
    mu_true = df_plot["ATE_true"].values * factor
    mu_dedl = df_plot["ATE_hat_DeDL"].values * factor
    mu_sdl = df_plot["ATE_hat_SDL"].values * factor

    se_dedl = df_plot["SE_DeDL"].values * factor
    se_sdl = df_plot["SE_SDL"].values * factor

    se_true = df_plot["SE_true"].values * factor

    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(10, 4))

    offset = 0.12

    # True Effect（红色三角）
    ax.errorbar(
        x - offset,
        mu_true,
        yerr=Z_975 * se_true,
        fmt="v",
        markersize=6,
        capsize=4,
        color="red",
        label="True Effect",
        linewidth=1.0,
    )

    # DeDL（蓝色圆点）
    ax.errorbar(
        x,
        mu_dedl,
        yerr=Z_975 * se_dedl,
        fmt="o",
        markersize=5,
        capsize=4,
        color="blue",
        label="DeDL",
        linewidth=1.0,
    )

    # SDL（绿色菱形）
    ax.errorbar(
        x + offset,
        mu_sdl,
        yerr=Z_975 * se_sdl,
        fmt="D",
        markersize=5,
        capsize=4,
        color="green",
        label="SDL",
        linewidth=1.0,
    )

    # y=0 的虚线
    ax.axhline(0.0, linestyle="--", color="salmon", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Treatment Combination")
    ax.set_ylabel("Relative Effect Size (%)")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, "Figure5_synth_SDL_vs_DeDL.png")
    fig.savefig(fig_path, dpi=300)
    print(f"[Figure 5-like] 图已保存到 {fig_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dedl_synth",
        help="前一阶段 Validation_of_DeDL.py 输出的目录",
    )
    args = parser.parse_args()

    df, m, idx0 = load_all(args.output_dir)
    print(f"检测到 m = {m}, 总共有 {len(df)} 个 treatment 组合。")

    make_table2(df, m, idx0, args.output_dir)
    make_table3(df, args.output_dir)
    make_table4_best(df, args.output_dir)
    plot_figure5_like(df, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
