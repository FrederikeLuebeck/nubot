# Neural Unbalanced Optimal Transport for Modeling Single-Cell Dynamics

Frederike Lübeck*, Charlotte Bunne*, Gabriele Gut, Jacobo Sarabia Castillo, Lucas Pelkmans, and David Alvarez Melis

## Overview
<p align='center'><img src='fig_overview.png' alt='Overview.' width='100%'> </p>

**a.** A semi-coupling pair ( $\gamma_1$, $\gamma_2$ ) consists of two couplings that together solve the unbalanced OT problem. Intuitively, $\gamma_1$ describes where mass goes as it leaves from $\mu$, and $\gamma_2$ where it comes from as it arrives in $\nu$. **b.** NubOT parameterizes the semi-couplings ( $\gamma_1$, $\gamma_2$ ) as the composition of reweighting functions $\eta$ and $\zeta$ and the dual potentials $f$ and $g$ between the then balanced problem.

**Abstract.** Tracking the development of cells over time is a major challenge in biology, as measuring cells usually requires their destruction.
Optimal transport (OT) can help solve this challenge by learning an optimal coupling of samples taken at different points in time, thus enabling the reconstruction of pairwise correspondences between cells of different measurements. However, the classical formulation of OT assumes conservation of mass, which is violated in unbalanced scenarios in which the population size changes, e.g., when cells die or proliferate. In this work, we present NubOT, a neural unbalanced optimal transport model, that learns a parameterized optimal transport map between unbalanced distributions. To model variation of mass, we rely on the formalism of semi-couplings and propose a novel parameterization and algorithmic scheme that is based on a cycle-consistent learning procedure. We apply our model to the challenging task of predicting the responses of heterogeneous cells to cancer drugs on the level of single cells. By accurately modeling cell proliferation and death, our method yields notable improvements over previous neural optimal transport methods.

## Installation

To setup the environment, run the following commands:

```
conda create -y --name nubot python=3.8.5
conda activate nubot

conda install -y -c conda-forge jaxlib
conda install -y -c conda-forge jax

pip install -r requirements.txt

pip install -e .
```

The pre-processed data files can be downloaded [here](https://doi.org/10.3929/ethz-b-000631091).

## Running Model

NubOT and the baselines can now be trained by running the script ```python scripts/run_model.py``` by specifying the following command line parameters:

- `--config_model`: Model configuration file, i.e., one of the `.yaml` files in `config/models/`
- `--config_data`: Data configuration file, i.e., one of the `.yaml` files in `config/data/`
- `--target` *(optional)*: Target drug when using the cell data, e.g., `cisplatin`
- `--restart`*(optional)*: If this flag is set, the model restarts from scratch. Otherwise, the model is loaded from a previous checkpoint, if existent.
- `--experiment_name`,  `--experiment_dir` and `--outdir` can be provided to save the outputs in specific folders.

For example, the command for training NubOT on the cell data at 8h for the drug cisplatin is:

```
python scripts/run_model.py --config_model ./config/models/cell_nubot/submission_base.yaml --config_data ./config/data/4i-8h-raw.yaml --target cisplatin --restart
```
