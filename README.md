# DeDL Framework Toolkit

> This repository contains a reusable experiment framework derived from [the original synthetic experiment notebook](synthetic_experiments.ipynb) for the paper: [Zikun Ye, Zhiqi Zhang, Dennis J. Zhang, Heng Zhang, Renyu Zhang (2025) Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence. Management Science 0(0).](https://doi.org/10.1287/mnsc.2024.04625)
>
> The notebook is preserved for reference, and the reusable Python modules provide a command-line workflow for running new DeDL experiments on synthetic or real data.

## Introduction to DeDL Framework

The core logic of the toolkit follows the design approach proposed in Section 3 of the paper, which combines a **structured deep neural network** (DNN) with **Double Machine Learning** (DML): first, a **structured DNN** approximates the **nuisance functions** in the **data-generating process** (DGP), and then **influence functions** are used to **correct biases** in the predictions, enabling causal inference for unobserved combinations.

## Installation

1. Clone this repository:

    ```bash
    clone https://github.com/ysd1123/DeDL_Replication.git
    cd DeDL_Replication
    ```

2. (Optional) Create and activate a virtual environment.

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    uv pip install -r requirements.txt
    ```

    or if you don't use a virtual environment:

    ```bash
    pip install -r requirements.txt
    ```

## Quick start

Run the provided synthetic example:

```bash
python run_experiment.py --config configs/example.yml
```

The script will load data, train the structured network, evaluate baseline and debiased methods, and save metrics/plots under `results/<timestamp>/`.

## Configuration

Experiments are controlled by a `YAML` file. Key sections:

- **data**
  - `type`: `synthetic` or `real`.
  - Synthetic options: 
  	- `m` (number of factors), 
  	- `d_c` (covariate dimension), 
  	- `train_size`/`test_size` (number of samples in the training/test set), 
  	- `t_combo_obs` (observed treatment combos), 
  	- `t_dist_obs` (sampling probs), 
  	- `noise_level`, `noise_type` (observation noise magnitude and distribution type), 
  	- `outcome_fn` (`sigmoid`, `linear`, `polynomial`), 
  	- `coef_dist`/`coef_scale` (to generate the distribution and scale of causal effect coefficients to control for covariates and the magnitude of treatment effects), 
  	- `c_true_range`, `d_true` (to randomly generate true scaling factors $c_\text{true}$ and shift terms $d_\text{true}$ when `outcome_fn` is `sigmoid`), 
  	- `cov_shift`, 
  	- `seed`.
  - Real-data options: `path` (CSV), `factor_cols` (0/1 treatment columns), `feature_cols` (covariates), `outcome_col`, `dropna` (or fill mean), `train_size`/`test_size`, `seed`.
- **model**
  - `layers`: hidden sizes for the structured network producing \(\beta(x)\).
  - `link_function`: `sigmoid`, `linear`, or `softplus` (extendable), `learn_scale`/`learn_shift` to train link parameters, `pdl_layers` for the pure DNN baseline.
- **training**
  - `batch_size`, `lr`, `weight_decay`, `epochs`, `patience`, `mse_threshold`, `loss_fn` (`mse` or custom torch loss).
- **debias**
  - `ridge`: diagonal adjustment when inverting the information matrix for DeDL.
- `n_replications`: repeat the whole pipeline multiple times.

See `configs/example.yml` for a full sample configuration.

## Workflow

1. **Load data**: `dedl.data.load_data` draws synthetic data or loads a CSV according to the config.
2. **Build model**: `dedl.models.StructuredNet` constructs the structured network with the specified link function; a PDL baseline is built from the same config.
3. **Train**: `dedl.training.train_model` fits the structured model with early stopping and optional L1/L2 regularization.
4. **Evaluate**: `dedl.evaluation.evaluate_methods` reports LA/LR/PDL/SDL predictions and the DeDL debiased estimates for each treatment combination.
5. **Save**: `dedl.results.save_results` writes metrics, plots, config copy, and (optionally) model weights into a timestamped folder.

## Customization tips

- Modify `configs/example.yml` or create new YAML files to change data generation, network depth, link functions, or debiasing settings.
- Extend link functions or outcome generators by editing `dedl/models.py` and `dedl/data.py` respectively.
- To use your own dataset, set `data.type: real` and point `data.path` to your CSV with binary treatment columns listed in `data.factor_cols`.

## Repository contents

- `synthetic_experiments.ipynb`: Original notebook implementation for reference.
- `dedl/`: Modular Python package with data loading, modeling, training, evaluation, and result utilities.
	- `data.py`: Data generation and preprocessing: `_generate_synthetic` generates experimental data according to the configuration in synthetic data mode, while `_process_real` loads external CSV data
	- `models.py`: Defines the structured deep model `StructuredNet` and the baseline network `PlainNet`
	- `training.py`: Data loading, training, and cross-validation
	- `evaluation.py`: Evaluate baselines vs. DeDL inference
- `configs/`: Example YAML configurations.
- `run_experiment.py`: CLI entry point to launch experiments from a config file.
