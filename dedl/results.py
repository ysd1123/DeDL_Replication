from __future__ import annotations
import itertools
import json
import math
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


def _timestamp_dir(base: str = "results") -> pathlib.Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = pathlib.Path(base) / now
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _compute_ground_truth(config: Dict, sim_info: Dict) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute ground truth metrics for all possible treatment combinations based on the provided configuration and simulation information.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing data generation parameters such as:
        - "m": number of treatments (int)
        - "d_c": number of covariates (int)
        - "outcome_fn": outcome function type ("sigmoid", "linear", or "polynomial")
        - "t_combo_obs": list of observed treatment combinations (optional)
    sim_info : Dict
        Simulation information dictionary containing:
        - "x": covariate matrix (array-like, shape [n_samples, d_c])
        - "coef": coefficients for covariates and treatments (array-like)
        - "c_true": scaling constant for outcome (float, optional)
        - "d_true": offset for outcome (float, optional)
        - "noise_level": standard deviation of noise (float, optional)

    Returns
    -------
    gt_df : pd.DataFrame
        DataFrame with one row per treatment combination, including columns:
        - "treatment_key": string key for the treatment combination
        - "treatment": treatment vector
        - "observable": whether the treatment is observed in the data
        - "mu": mean outcome for the treatment
        - "mu_baseline": mean outcome for the baseline (all-control) treatment
        - "ate": average treatment effect relative to baseline
        - "relative_effect_pct": relative effect as a percentage
        - "p_value": p-value for the difference from baseline
        - "is_best": whether this treatment has the highest ATE
    best_info : Dict[str, object]
        Dictionary with information about the best treatment combination:
        - "treatment_key": string key for the best treatment
        - "treatment": treatment vector for the best treatment
        - "ate": ATE for the best treatment
    """
    m = int(config.get("data", {}).get("m", 0))
    d_c = int(config.get("data", {}).get("d_c", 0))
    outcome_fn = config.get("data", {}).get("outcome_fn", "sigmoid")
    t_combo_obs = config.get("data", {}).get("t_combo_obs") or [list(seq) for seq in itertools.product([0, 1], repeat=m)]

    x = np.asarray(sim_info.get("x"))
    coef = np.asarray(sim_info.get("coef"))
    c_true = float(sim_info.get("c_true", 1.0))
    d_true = float(sim_info.get("d_true", 0.0))
    noise_level = float(sim_info.get("noise_level", 0.0))

    coef_x = coef[:d_c]
    coef_t = coef[d_c:]

    combos = list(itertools.product([0, 1], repeat=m))

    treatment_assignments = sim_info.get("treatment_assignments")
    if treatment_assignments is None:
        # Fall back to the raw treatment matrix if assignments were not explicitly stored.
        t_full = np.asarray(sim_info.get("t"))
        if t_full.ndim == 2:
            if t_full.shape[1] == m:
                treatment_assignments = t_full
            elif t_full.shape[1] >= m + 1:
                # Drop intercept or any leading columns so only the m binary arms remain.
                treatment_assignments = t_full[:, -m:]
    if treatment_assignments is None:
        raise ValueError("Simulation info is missing treatment assignments required for ground truth computation.")
    treatment_assignments = np.asarray(treatment_assignments)
    if treatment_assignments.ndim != 2 or treatment_assignments.shape[1] != m:
        raise ValueError(
            f"Treatment assignments must have shape (n_samples, {m}) but received {treatment_assignments.shape}."
        )

    def _deterministic_outcome(t_vec: np.ndarray) -> np.ndarray:
        u = x @ coef_x + t_vec @ coef_t
        if outcome_fn == "sigmoid":
            return c_true / (1 + np.exp(-u)) + d_true
        if outcome_fn == "linear":
            return u + d_true
        if outcome_fn == "polynomial":
            return u + 0.1 * (u ** 2) + d_true
        raise ValueError(f"Unsupported outcome function: {outcome_fn}")

    rows = []
    baseline_combo = tuple([0] * m)
    mu_baseline = None
    for combo in combos:
        t_vec = np.array([1, *combo], dtype=float)
        y_true = _deterministic_outcome(t_vec)
        mu_c = float(np.mean(y_true))
        if combo == baseline_combo:
            mu_baseline = mu_c
        rows.append((combo, t_vec.tolist(), mu_c))

    if mu_baseline is None:
        raise ValueError("Baseline mean could not be computed")

    gt_rows = []
    for combo, t_vec, mu_c in rows:
        ate = mu_c - mu_baseline
        rel_pct = 100 * ate / mu_baseline if mu_baseline != 0 else np.nan
        # Compute group sizes for baseline (all-control) and the current treatment combo
        baseline_mask = np.all(treatment_assignments == baseline_combo, axis=1)
        current_mask = np.all(treatment_assignments == combo, axis=1)
        n1 = np.sum(baseline_mask)
        n2 = np.sum(current_mask)
        if n1 > 0 and n2 > 0:
            var_diff = (noise_level ** 2) / n1 + (noise_level ** 2) / n2
            t_stat = ate / math.sqrt(var_diff)
            p_val = 2 * (1 - _normal_cdf(abs(t_stat)))
        else:
            p_val = np.nan
        observable = list(combo) in t_combo_obs
        treatment_key = "".join(str(int(b)) for b in combo)
        gt_rows.append({
            "treatment_key": treatment_key,
            "treatment": t_vec,
            "observable": observable,
            "mu": mu_c,
            "mu_baseline": mu_baseline,
            "ate": ate,
            "relative_effect_pct": rel_pct,
            "p_value": p_val,
        })

    gt_df = pd.DataFrame(gt_rows)
    best_idx = gt_df["ate"].idxmax()
    best_row = gt_df.loc[best_idx]
    gt_df["is_best"] = gt_df.index == best_idx
    best_info = {
        "treatment_key": best_row["treatment_key"],
        "treatment": best_row["treatment"],
        "ate": best_row["ate"],
    }
    return gt_df, best_info


def _compute_estimator_metrics(combined_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    metrics_cfg = config.get("metrics", {})
    cdr_eps = float(metrics_cfg.get("cdr_eps", 0.0))
    mape_eps = float(metrics_cfg.get("mape_eps", 1e-6))
    estimators = ["la", "lr", "pdl", "sdl", "dedl"]
    rows = []
    for est in estimators:
        col = f"{est}_diff"
        if col not in combined_df:
            continue
        true_vals = combined_df["ate"]
        est_vals = combined_df[col]
        mask_valid = true_vals.notna() & est_vals.notna()
        sub_true = true_vals[mask_valid]
        sub_est = est_vals[mask_valid]
        mask_cdr = sub_true.abs() > cdr_eps
        if mask_cdr.any():
            cdr = float((np.sign(sub_true[mask_cdr]) == np.sign(sub_est[mask_cdr])).mean())
        else:
            cdr = np.nan
        mask_mape = sub_true.abs() > mape_eps
        if mask_mape.any():
            mape = float((np.abs(sub_est[mask_mape] - sub_true[mask_mape]) / sub_true[mask_mape].abs()).mean())
        else:
            mape = np.nan
        mae = float(np.abs(sub_est - sub_true).mean()) if len(sub_true) > 0 else np.nan
        unobs_mask = (combined_df.get("observable", False) == False) & mask_valid
        if unobs_mask.any():
            mseu = float(((combined_df.loc[unobs_mask, col] - combined_df.loc[unobs_mask, "ate"]) ** 2).mean())
        else:
            mseu = np.nan
        rows.append({"estimator": est, "CDR": cdr, "MAPE": mape, "MAE": mae, "MSEu": mseu})
    return pd.DataFrame(rows)


def _compute_best_treatment_metrics(combined_df: pd.DataFrame, best_info: Dict[str, object], m: int) -> pd.DataFrame:
    true_best_key = best_info.get("treatment_key")
    true_best_ate = best_info.get("ate")
    estimators = ["la", "lr", "pdl", "sdl", "dedl"]
    replications = combined_df["replication"].unique()
    baseline_key = "0" * m
    rows = []
    for est in estimators:
        col = f"{est}_diff"
        correct = []
        regrets = []
        for rep in replications:
            rep_df = combined_df[(combined_df["replication"] == rep) & (combined_df["treatment_key"] != baseline_key)]
            if rep_df.empty or col not in rep_df:
                continue
            idx_max = rep_df[col].idxmax()
            pred_key = rep_df.loc[idx_max, "treatment_key"]
            correct.append(1 if pred_key == true_best_key else 0)
            pred_true_ate = rep_df.loc[idx_max, "ate"]
            regrets.append(float(true_best_ate - pred_true_ate))
        if correct:
            bti = float(np.mean(correct))
            avg_regret = float(np.mean(regrets))
        else:
            bti = np.nan
            avg_regret = np.nan
        rows.append({"estimator": est, "BTI": bti, "AvgRegret": avg_regret})
    return pd.DataFrame(rows)


def _plot_sdl_vs_dedl(df: pd.DataFrame, out_path: pathlib.Path):
    plt.figure()
    plt.scatter(df["true_ate"], df["sdl_diff"], label="SDL", alpha=0.7)
    plt.scatter(df["true_ate"], df["dedl_diff"], label="DeDL", alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("True ATE")
    plt.ylabel("Estimated Incremental Effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_results(
    results: List[Dict],
    config: Dict,
    model: torch.nn.Module | None = None,
    sim_info: Dict | None = None,
) -> pathlib.Path:
    out_dir = _timestamp_dir()
    with (out_dir / "config.yml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)

    if model is not None:
        torch.save(model.state_dict(), out_dir / "model.pt")

    plot_df = df.copy()
    plot_df[plot_df.columns[0]] = plot_df[plot_df.columns[0]].astype(str)
    ax = plot_df.set_index(plot_df.columns[0])[['la', 'lr', 'pdl', 'sdl', 'dedl']].plot(kind='bar')
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png")
    plt.close()

    if sim_info is not None and config.get("data", {}).get("type") == "synthetic":
        gt_df, best_info = _compute_ground_truth(config, sim_info)
        gt_df.to_csv(out_dir / "ground_truth_ate.csv", index=False)
        with (out_dir / "ground_truth_best_treatment.json").open("w", encoding="utf-8") as fp:
            json.dump(best_info, fp, indent=2)

        combined_df = df.merge(
            gt_df[["treatment_key", "observable", "mu", "mu_baseline", "ate", "relative_effect_pct", "is_best"]],
            on="treatment_key",
            how="left",
        )
        combined_df.to_csv(out_dir / "combined_metrics.csv", index=False)

        sdl_dedl_df = combined_df[["treatment_key", "replication", "observable", "ate", "sdl_diff", "dedl_diff"]].copy()
        sdl_dedl_df.rename(columns={"ate": "true_ate"}, inplace=True)
        sdl_dedl_df["sdl_abs_error"] = np.abs(sdl_dedl_df["sdl_diff"] - sdl_dedl_df["true_ate"])
        sdl_dedl_df["dedl_abs_error"] = np.abs(sdl_dedl_df["dedl_diff"] - sdl_dedl_df["true_ate"])
        sdl_dedl_df.to_csv(out_dir / "sdl_vs_dedl.csv", index=False)
        _plot_sdl_vs_dedl(sdl_dedl_df, out_dir / "sdl_vs_dedl.png")

        metrics_df = _compute_estimator_metrics(combined_df, config)
        metrics_df.to_csv(out_dir / "ate_estimator_metrics.csv", index=False)

        best_metrics_df = _compute_best_treatment_metrics(combined_df, best_info, int(config.get("data", {}).get("m", 0)))
        best_metrics_df.to_csv(out_dir / "best_treatment_metrics.csv", index=False)

    return out_dir


def report(results_dir: pathlib.Path) -> pathlib.Path:
    metrics_path = results_dir / "metrics.csv"
    df = pd.read_csv(metrics_path)
    md_path = results_dir / "report.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# DeDL Experiment Report\n\n")
        fp.write(df.describe().to_markdown())
    return md_path
