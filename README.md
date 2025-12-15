# DeDL Framework Toolkit

> This repository contains a reusable experiment framework derived from [the original synthetic experiment notebook](https://github.com/zikunye2/deep_learning_based_causal_inference_for_combinatorial_experiments) for the paper: [Zikun Ye, Zhiqi Zhang, Dennis J. Zhang, Heng Zhang, Renyu Zhang (2025) Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence. Management Science 0(0).](https://doi.org/10.1287/mnsc.2024.04625)
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
cd ReplicationCodes
python Validation_of_DeDL.py
```

## Results and Visualizations

see `visualization/` for scripts to generate figures from the given result data.
