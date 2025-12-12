import os
import math
import numpy as np
import pandas as pd
from typing import Dict

import torch
from sklearn.linear_model import LinearRegression

from utils import G_u_analytic, generalized_sigmoid_form_II, loss_grad_u
from dnn_arch import DeDLNet
from dnn_train import train_dedl_model, train_pdl_model
from dgp import all_treatments, observed_treatments_m_plus_2

# DeDL / SDL / PDL / LR / LA 估计 ATE

def estimate_L_matrix_for_each_x(model: DeDLNet, X: np.ndarray, ts_support: np.ndarray, config: Dict) -> np.ndarray:
    """
    对每个样本 x_i 估计 L(x_i) = 2 E[Gu(u(x),T) Gu(u(x),T)^T]。
    使用已知的 treatment assignment 支持集 ts_support（均匀分布）。
    返回形状：(n, m+2, m+2)
    """
    device = config["device"]
    lambda_reg = config["lambda_reg_L"]
    n = X.shape[0]
    m = config["m"]
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        u = model(X_t)  # (n, m+2)
        u = u.cpu()

    ts = torch.from_numpy(ts_support)  # (K, m)
    K = ts.shape[0]

    L_list = []
    for i in range(n):
        u_i = u[i].unsqueeze(0)    # (1, m+2)
        # 对所有支持集中的 t 计算 Gu，再平均
        Gu_all = G_u_analytic(u_i.repeat(K, 1), ts)  # (K, m+2)
        # 均匀分布：ν(t) = 1/K
        G_outer = torch.einsum("ki,kj->ij", Gu_all, Gu_all) / K
        L_i = 2.0 * G_outer
        # 数值稳定：加小的对角正则
        L_i = L_i + lambda_reg * torch.eye(m + 2)
        L_list.append(L_i.numpy())
    return np.stack(L_list, axis=0)

def dedl_influence_psi(
    y: np.ndarray, x: np.ndarray, t_obs: np.ndarray,
    t_target: np.ndarray, t0: np.ndarray,
    model: DeDLNet, L_mats: np.ndarray, config: Dict
) -> np.ndarray:
    """
    对每个样本 i 计算 DeDL 的 influence function ψ_i(t_target)。
    #NOTE: 统一在 config["device"] 上做所有 torch 运算，只在和 L_mats 相乘时把向量搬回 CPU 并转成 numpy。
    """
    device = config["device"]
    m = config["m"]
    model.eval()

    X_t = torch.from_numpy(x).to(device) # (n, d_x)
    T_obs_t = torch.from_numpy(t_obs).to(device) # (n, m)
    y_t = torch.from_numpy(y).to(device) # (n,)
    t_target_t = torch.from_numpy(t_target.astype(np.float32)).to(device).unsqueeze(0) # (1, m)
    t0_t = torch.from_numpy(t0.astype(np.float32)).to(device).unsqueeze(0) # (1, m)

    # 一次性算出所有 u(x_i)
    with torch.no_grad():
        u_all = model(X_t)                                  # (n, m+2)

    psi_list = []

    for i in range(x.shape[0]):
        u_i = u_all[i:i+1]          # (1, m+2)，在 device 上
        t_obs_i = T_obs_t[i:i+1]    # (1, m)
        y_i = y_t[i:i+1]            # (1,)

        # H(x,u;t,t0) = G(u,t) - G(u,t0)
        G_t = generalized_sigmoid_form_II(u_i, t_target_t)  # (1,)
        G_t0 = generalized_sigmoid_form_II(u_i, t0_t)       # (1,)
        H_val = float((G_t - G_t0).item())

        # H_u(x,u;t,t0) = G_u(u,t) - G_u(u,t0)
        Gu_t = G_u_analytic(u_i, t_target_t)                # (1, m+2)
        Gu_t0 = G_u_analytic(u_i, t0_t)                     # (1, m+2)
        H_u_vec = (Gu_t - Gu_t0).squeeze(0).cpu()           # (m+2,) on CPU

        # ℓ_u(y, t_obs, u)
        ell_u_vec = loss_grad_u(y_i, t_obs_i, u_i)          # (1, m+2) on device
        ell_u_vec = ell_u_vec.squeeze(0).cpu()              # (m+2,) on CPU

        # L(x) 是 numpy 数组 (m+2, m+2)
        L_i = L_mats[i]
        L_inv = np.linalg.inv(L_i)

        # 现在三个量都在 CPU 上，用 numpy 做二次型
        delta = float(H_u_vec.numpy().T @ L_inv @ ell_u_vec.numpy())

        psi_list.append(H_val - delta)

    return np.array(psi_list, dtype=np.float32)

def estimate_ate_all_methods(
    X_train, T_train, Y_train,
    X_infer, T_infer, Y_infer,
    df_true_ate: pd.DataFrame,
    config: Dict
) -> None:
    """
    在给定训练/推断数据和真值 ATE 的情况下，估计并保存：
    - LA, LR, PDL, SDL, DeDL 各自的 ATE 估计
    - 以及 summary metrics（CDR, MAPE, MSE, MAE）
    """

    m = config["m"]
    device = config["device"]
    all_ts = all_treatments(m)
    t0 = np.zeros(m, dtype=np.float32)

    # ---------- 1) LA ----------
    # 每个单实验的 ATE：对训练数据简单做差异 in means
    la_mu_k = []
    la_var_k = []
    for k in range(m):
        mask_k1 = (T_train[:, k] == 1.0)
        mask_k0 = (T_train[:, k] == 0.0)
        y1 = Y_train[mask_k1]
        y0 = Y_train[mask_k0]
        mu_k = float(y1.mean() - y0.mean())
        var_k = float(y1.var() / max(len(y1), 1) + y0.var() / max(len(y0), 1))
        la_mu_k.append(mu_k)
        la_var_k.append(var_k)
    la_mu_k = np.array(la_mu_k)
    la_var_k = np.array(la_var_k)

    la_results = []
    for idx, t in enumerate(all_ts):
        mu_hat = float((la_mu_k * t).sum())
        var_hat = float((la_var_k * (t ** 2)).sum())
        se_hat = math.sqrt(max(var_hat, 1e-8))
        la_results.append(
            {"combo_index": idx, "ATE_hat_LA": mu_hat, "SE_LA": se_hat}
        )
    df_la = pd.DataFrame(la_results)

    # ---------- 2) LR ----------
    lr_model = LinearRegression()
    X_lr = np.concatenate([X_train, T_train], axis=1)
    lr_model.fit(X_lr, Y_train)
    lr_results = []
    for idx, t in enumerate(all_ts):
        t_rep = np.repeat(t.reshape(1, -1), X_infer.shape[0], axis=0)
        t0_rep = np.repeat(t0.reshape(1, -1), X_infer.shape[0], axis=0)
        X_lr_t = np.concatenate([X_infer, t_rep], axis=1)
        X_lr_t0 = np.concatenate([X_infer, t0_rep], axis=1)
        y_hat_t = lr_model.predict(X_lr_t)
        y_hat_t0 = lr_model.predict(X_lr_t0)
        delta = y_hat_t - y_hat_t0
        mu_hat = float(delta.mean())
        se_hat = float(delta.std(ddof=1) / math.sqrt(len(delta)))
        lr_results.append(
            {"combo_index": idx, "ATE_hat_LR": mu_hat, "SE_LR": se_hat}
        )
    df_lr = pd.DataFrame(lr_results)

    # ---------- 3) SDL / DeDL ----------
    dedl_model = train_dedl_model(X_train, T_train, Y_train, config)
    torch.save(
        dedl_model.state_dict(),
        os.path.join(config["output_dir"], "models", "dedl_sdl_model.pt")
    )

    # 估计 L(x)（使用部分观测机制：m+2 组合）
    ts_support = observed_treatments_m_plus_2(m)
    L_mats = estimate_L_matrix_for_each_x(dedl_model, X_infer, ts_support, config)

    # SDL: plug-in estimator
    dedl_model.eval()
    with torch.no_grad():
        u_inf = dedl_model(torch.from_numpy(X_infer).to(device)).cpu().numpy()

    sdl_rows = []
    dedl_rows = []
    for idx, t in enumerate(all_ts):
        t_vec = t.astype(np.float32)
        t0_vec = t0.astype(np.float32)

        # SDL plug-in
        t_torch = torch.from_numpy(t_vec).unsqueeze(0).to(device)
        t0_torch = torch.from_numpy(t0_vec).unsqueeze(0).to(device)
        u_torch = torch.from_numpy(u_inf).to(device)
        with torch.no_grad():
            G_t = generalized_sigmoid_form_II(u_torch, t_torch)
            G_t0 = generalized_sigmoid_form_II(u_torch, t0_torch)
            H_vals = (G_t - G_t0).cpu().numpy()
        mu_hat_sdl = float(H_vals.mean())
        se_hat_sdl = float(H_vals.std(ddof=1) / math.sqrt(len(H_vals)))
        sdl_rows.append(
            {"combo_index": idx, "ATE_hat_SDL": mu_hat_sdl, "SE_SDL": se_hat_sdl}
        )

        # DeDL influence function
        psi_vals = dedl_influence_psi(
            y=Y_infer, x=X_infer, t_obs=T_infer,
            t_target=t_vec, t0=t0_vec,
            model=dedl_model, L_mats=L_mats,
            config=config,
        )
        mu_hat_dedl = float(psi_vals.mean())
        var_hat = float(psi_vals.var(ddof=1))
        se_hat_dedl = math.sqrt(var_hat / len(psi_vals))
        dedl_rows.append(
            {"combo_index": idx, "ATE_hat_DeDL": mu_hat_dedl, "SE_DeDL": se_hat_dedl}
        )

    df_sdl = pd.DataFrame(sdl_rows)
    df_dedl = pd.DataFrame(dedl_rows)

    # ---------- 4) PDL ----------
    pdl_model = train_pdl_model(X_train, T_train, Y_train, config)
    torch.save(
        pdl_model.state_dict(),
        os.path.join(config["output_dir"], "models", "pdl_model.pt")
    )

    pdl_rows = []
    pdl_model.eval()
    with torch.no_grad():
        X_inf_t = torch.from_numpy(X_infer).to(device)
        for idx, t in enumerate(all_ts):
            t_vec = t.astype(np.float32)
            t0_vec = t0.astype(np.float32)
            t_rep = torch.from_numpy(
                np.repeat(t_vec.reshape(1, -1), X_infer.shape[0], axis=0)
            ).to(device)
            t0_rep = torch.from_numpy(
                np.repeat(t0_vec.reshape(1, -1), X_infer.shape[0], axis=0)
            ).to(device)

            y_hat_t = pdl_model(X_inf_t, t_rep).cpu().numpy()
            y_hat_t0 = pdl_model(X_inf_t, t0_rep).cpu().numpy()
            delta = y_hat_t - y_hat_t0
            mu_hat = float(delta.mean())
            se_hat = float(delta.std(ddof=1) / math.sqrt(len(delta)))
            pdl_rows.append(
                {"combo_index": idx, "ATE_hat_PDL": mu_hat, "SE_PDL": se_hat}
            )
    df_pdl = pd.DataFrame(pdl_rows)

    # ---------- 保存各方法 ATE 估计 ----------
    df_la.to_csv(os.path.join(config["output_dir"], "est_ates_LA.csv"), index=False)
    df_lr.to_csv(os.path.join(config["output_dir"], "est_ates_LR.csv"), index=False)
    df_sdl.to_csv(os.path.join(config["output_dir"], "est_ates_SDL.csv"), index=False)
    df_dedl.to_csv(os.path.join(config["output_dir"], "est_ates_DeDL.csv"), index=False)
    df_pdl.to_csv(os.path.join(config["output_dir"], "est_ates_PDL.csv"), index=False)

    # ---------- 汇总到一个表，计算误差指标 ----------
    df = df_true_ate.copy()  # 含 combo_index, ATE_true
    df = df.merge(df_la, on="combo_index")
    df = df.merge(df_lr, on="combo_index")
    df = df.merge(df_sdl, on="combo_index")
    df = df.merge(df_dedl, on="combo_index")
    df = df.merge(df_pdl, on="combo_index")

    # 误差列
    for method in ["LA", "LR", "SDL", "DeDL", "PDL"]:
        df[f"err_{method}"] = df[f"ATE_hat_{method}"] - df["ATE_true"]
        df[f"abs_err_{method}"] = df[f"err_{method}"].abs()
        df[f"sq_err_{method}"] = df[f"err_{method}"] ** 2

    # 定义显著性：|ATE_true| > 0（或你也可以用 MC 的 t-test）
    sig_mask = df["ATE_true"].abs() > 1e-4

    summary_rows = []
    for method in ["LA", "LR", "SDL", "DeDL", "PDL"]:
        # CDR：符号和显著性一致的比例（这里简单用符号一致）
        sign_true = np.sign(df["ATE_true"])
        sign_hat = np.sign(df[f"ATE_hat_{method}"])
        cdr = float((sign_true == sign_hat).mean())

        # MAPE：仅在真值显著的组合上
        ape = (df.loc[sig_mask, f"abs_err_{method}"] /
               df.loc[sig_mask, "ATE_true"].abs())
        mape = float(ape.mean())

        mse = float(df[f"sq_err_{method}"].mean())
        mae = float(df[f"abs_err_{method}"].mean())

        summary_rows.append({
            "method": method,
            "CDR": cdr,
            "MAPE": mape,
            "MSE": mse,
            "MAE": mae,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(config["output_dir"], "summary_metrics.csv"),
        index=False
    )

    print("Summary metrics:")
    print(df_summary)