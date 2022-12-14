import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import wandb
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import umap


def visualize(x, y, T, data, eps=None, epoch=None, **kwargs):
    x = x.detach().numpy()
    y = y.detach().numpy()
    T = T.detach().numpy()
    if eps is not None:
        eps = eps.detach().numpy()

    # plot 1D-marginals
    if data == "gaussian" or data == "toy":
        for dim in range(x.shape[1]):
            if dim > 3:  # log only first 3 dims to wandb
                break
            name, plot = plot_kde(x, y, T, eps, dim=dim)
            if epoch is not None:
                plot.title("EPOCH " + str(epoch))
            wandb.log({name: wandb.Image(plot)})
            plt.close()

    if data == "toy":
        name, plot = plot_scatter(x, y, T, eps=eps)
        if epoch is not None:
            plot.title("EPOCH " + str(epoch))
        wandb.log({name: wandb.Image(plot)})
        plt.close()

        if eps is not None:
            plt.figure()
            plt.scatter(x[:, 0], x[:, 1], c=eps[:, 0])
            plt.clim(0, 2)
            plt.colorbar()
            wandb.log({"eps_x": wandb.Image(plt)})
            plt.close()

    if data == "cell":
        # pick n axes and plot their 1D marginals
        axes = kwargs.pop("axes", None)
        if axes is None:
            np.random.seed(0)
            n = 5
            axes = np.array([0, 5, 10, 15, 20, 40])

        for dim in axes:
            name, plot = plot_kde(x, y, T, eps, dim=dim)
            if epoch is not None:
                plot.title("EPOCH " + str(epoch))
            wandb.log({name: wandb.Image(plot)})
            plt.close()

    else:
        return

    plt.close("all")


def plot_kde(x, y, T, eps=None, dim=0):
    plt.figure()
    kwargs = {"fill": True, "common_norm": True}
    sns.kdeplot(x[:, dim], label="source", **kwargs)
    sns.kdeplot(y[:, dim], label="target", **kwargs)
    sns.kdeplot(T[:, dim], label="transported", **kwargs)

    if eps is not None:
        sns.kdeplot(
            x=x[:, dim],
            weights=eps[:, 0],
            label="scaled source (renormalized)",
            color="red",
            **kwargs
        )

    plt.legend()
    name = "dim_" + str(dim)
    return name, plt


def plot_scatter(x, y, T, eps=None):
    """
    Scatterplot (2D) of target, source and transported data.
    """
    scaling = eps

    kwargs = {}
    s = 10
    plt.figure()
    sns.scatterplot(x=x[:, 0], y=x[:, 1], s=s, label="source", **kwargs)
    sns.scatterplot(x=y[:, 0], y=y[:, 1], s=s, label="target", **kwargs)
    sns.scatterplot(x=T[:, 0], y=T[:, 1], s=s, label="T", **kwargs)
    if eps is not None:
        sns.scatterplot(x=T[:, 0], y=T[:, 1], s=s * scaling, label="scaled T", **kwargs)

    # arrow from source (x) to transported (T)
    for i in range(len(x)):
        dx = T[i, 0] - x[i, 0]
        dy = T[i, 1] - x[i, 1]
        plt.arrow(x[i, 0], x[i, 1], dx, dy, alpha=0.05, color="grey")

    plt.legend()

    return "scatter", plt


def plot_kde_cells(x, y, T, dim=0):
    """kde for cell features"""

    kwargs = {"fill": True, "palette": "Paired", "common_norm": False}
    plt.figure()
    df = pd.DataFrame(
        {"source": x[:, dim], "target": y[:, dim], "transported": T[:, dim]}
    )
    plot = sns.kdeplot(data=df, **kwargs)
    name = "dim_" + str(dim)
    return name, plot
