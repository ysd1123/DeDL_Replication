from __future__ import annotations
import argparse
import itertools
from typing import List

import numpy as np

from dedl import (
    StructuredNet,
    build_dataloader,
    cross_fit,
    evaluate_methods,
    load_data,
    parse_config,
    save_results,
    train_model,
)


def set_global_seed(config):
    """
    Set the global random seed for reproducibility.

    The function sets the seed for Python's `random` module, NumPy, and PyTorch.
    The seed value is determined using the following precedence order:
        1. config["global_seed"] (if present)
        2. config["training"]["seed"] (if present)
        3. config["data"]["seed"] (if present)
    The first available value in this order is used as the seed.
    """
    import random

    import numpy as np
    import torch

    seed = (
        config.get("global_seed")
        or config.get("training", {}).get("seed")
        or config.get("data", {}).get("seed")
    )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Run DeDL experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = parse_config(args.config)
    set_global_seed(config)

    results_all: List[dict] = []
    n_rep = int(config.get("n_replications", 1))
    for rep in range(n_rep):
        (x_train, t_train, y_train), (x_test, t_test, y_test), sim_info = load_data(config)

        batch_size = int(config.get("training", {}).get("batch_size", 256))
        train_loader = build_dataloader(x_train, t_train, y_train, batch_size)

        cv_folds = int(config.get("training", {}).get("cv_folds", 1))
        if cv_folds > 1:
            def model_factory():
                return StructuredNet(config)

            trained_model = cross_fit(model_factory, x_train, t_train, y_train, config)
            model_for_save = trained_model[0]
        else:
            trained_model = StructuredNet(config)
            train_model(trained_model, train_loader, config)
            model_for_save = trained_model

        m = t_train.shape[1] - 1
        t_stars = [np.array([1, *combo], dtype=float) for combo in itertools.product([0, 1], repeat=m)]
        rep_results = evaluate_methods(x_test, t_test, y_test, trained_model, config, t_stars)
        for item in rep_results:
            item["replication"] = rep
        results_all.extend(rep_results)

    out_dir = save_results(results_all, config, model_for_save, sim_info)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
