from __future__ import annotations
import json
import pathlib
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml


def _timestamp_dir(base: str = "results") -> pathlib.Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = pathlib.Path(base) / now
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(results: List[Dict], config: Dict, model: torch.nn.Module | None = None) -> pathlib.Path:
    out_dir = _timestamp_dir()
    # save config copy
    with (out_dir / "config.yml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)

    if model is not None:
        torch.save(model.state_dict(), out_dir / "model.pt")

    ax = df.set_index(df.columns[0])[['la', 'lr', 'pdl', 'sdl', 'dedl']].plot(kind='bar')
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png")
    plt.close()

    return out_dir


def report(results_dir: pathlib.Path) -> pathlib.Path:
    metrics_path = results_dir / "metrics.csv"
    df = pd.read_csv(metrics_path)
    md_path = results_dir / "report.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# DeDL Experiment Report\n\n")
        fp.write(df.describe().to_markdown())
    return md_path
