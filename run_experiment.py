from __future__ import annotations
import argparse
import itertools
from typing import List

import numpy as np

from dedl import (
    StructuredNet,
    build_dataloader,
    evaluate_methods,
    load_data,
    parse_config,
    save_results,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Run DeDL experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = parse_config(args.config)

    results_all: List[dict] = []
    n_rep = int(config.get("n_replications", 1))
    for rep in range(n_rep):
        (x_train, t_train, y_train), (x_test, t_test, y_test), _ = load_data(config)

        batch_size = int(config.get("training", {}).get("batch_size", 256))
        train_loader = build_dataloader(x_train, t_train, y_train, batch_size)

        model = StructuredNet(config)
        train_model(model, train_loader, config)

        m = t_train.shape[1] - 1
        t_stars = [np.array([1, *combo], dtype=float) for combo in itertools.product([0, 1], repeat=m)]
        rep_results = evaluate_methods(x_test, t_test, y_test, model, config, t_stars)
        for item in rep_results:
            item["replication"] = rep
        results_all.extend(rep_results)

    out_dir = save_results(results_all, config, model)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
