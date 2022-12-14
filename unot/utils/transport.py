import anndata
import pandas as pd
import numpy as np
import torch
import ot

import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import pointcloud
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt


def transport(config, model, dataset, return_as="anndata", dosage=None, **kwargs):
    name = config.model.get("name", "cellot")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    if name == "cellot":
        outputs = transport_cellot(model, inputs)

    elif name == "nubot":
        div_by_h_bw = True
        outputs, weights = transport_nubot(model, inputs, div_by_h_bw)

    elif name == "gan":
        outputs, weights = transport_gan(model, inputs)

    elif name == "cae":
        outputs = transport_cae(model, inputs, target=config.data.target)

    elif name == "gaussian":
        outputs = transport_gaussian(model, inputs)

    else:
        raise ValueError

    if dosage is not None:
        outputs = (1 - dosage) * inputs + dosage * outputs

    if return_as == "anndata":
        outputs = anndata.AnnData(
            outputs.detach().numpy(),
            obs=dataset.adata.obs.copy(),
            var=dataset.adata.var.copy(),
        )
        if name == "nubot" or name == "gan":
            outputs.obsm["weights"] = pd.DataFrame(
                weights.detach().numpy(),
                index=dataset.adata.obs.index.copy(),
                columns=["weights"],
            )

    return outputs


def transport_cellot(model, inputs):
    f, g = model
    g.eval()
    outputs = g.transport(inputs.requires_grad_(True))
    return outputs


def transport_gan(model, inputs):
    D, G = model
    G.eval()
    outputs, weights = G(inputs)
    return outputs, weights


def transport_scgen(model, inputs, source, target, decode=True):
    model.eval()
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(inputs)
    if not decode:
        return codes + shift

    outputs = model.decode(codes + shift)
    return outputs


def transport_cae(model, inputs, target):
    model.eval()
    target_code = model.conditions.index(target)
    outputs = model.outputs(inputs, decode_as=target_code).recon
    return outputs


def transport_nubot(model, inputs, div_by_h_bw=False):
    f, g, h_fw, h_bw = model
    g.eval()
    h_fw.eval()
    h_bw.eval()
    outputs = g.transport(inputs.requires_grad_(True))
    w_x = h_fw(inputs)
    if div_by_h_bw:
        print("divide w_y_hat by w_x_hat")
        w_y = h_bw(outputs)
        w_x = w_x / w_y
    return outputs, w_x


def transport_ot(x, y, epsilon=0.1):
    # compute Sinkhorn map on entire set
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    geom_xy = pointcloud.PointCloud(x.to_numpy(), y.to_numpy(), epsilon=epsilon)

    # solve ot problem
    out_xy = sinkhorn.sinkhorn(geom_xy, a, b, max_iterations=10000, min_iterations=10)
    gamma = out_xy.matrix

    # transport all source points
    v = (gamma.T / np.sum(gamma, axis=1)).T
    transported = np.matmul(v, y)

    def plot_kde(x, y, transported, i):
        plt.figure()
        sns.kdeplot(x.to_numpy()[:, i], label="source")
        sns.kdeplot(y.to_numpy()[:, i], label="target")
        sns.kdeplot(transported.to_numpy()[:, i], label="transported")

    plot_kde(x, y, transported, 0)
    plot_kde(x, y, transported, 5)
    plot_kde(x, y, transported, 20)
    plt.close()

    return transported


def transport_ubot(x, y, epsilon=0.001, reg_m=0.01):
    a = torch.ones(len(x)) / len(x)
    b = torch.ones(len(y)) / len(y)
    M = ot.dist(torch.tensor(x.to_numpy()), torch.tensor(y.to_numpy()))
    M /= M.max()

    gamma = ot.unbalanced.sinkhorn_unbalanced(
        a,
        b,
        M,
        epsilon,
        reg_m,
        method="sinkhorn",
        # numItermax=1000,
    )
    v = (gamma.T / gamma.sum(axis=1)).T
    transported = np.matmul(v, y.to_numpy())

    def plot_kde(x, y, transported, i):
        plt.figure()
        sns.kdeplot(x.to_numpy()[:, i], label="source")
        sns.kdeplot(y.to_numpy()[:, i], label="target")
        sns.kdeplot(transported.numpy()[:, i], label="transported")

    plot_kde(x, y, transported, 0)
    plot_kde(x, y, transported, 5)
    plot_kde(x, y, transported, 20)
    plt.close()

    return pd.DataFrame(transported.numpy(), columns=y.columns)


def transport_gaussian(model, inputs):
    return model(inputs)
