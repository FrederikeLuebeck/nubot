from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
from unot.losses.mmd import mmd_distance, mmd_distance_weigthed, resampled_mmd
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
import anndata
import torch
import numpy as np
from unot.utils.helpers import load_item_from_save
from unot.utils.loaders import load_data, load_model, load
from unot.utils import load_config
from unot.utils.transport import transport_ubot, transport_ot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import ot

import jax.numpy as jnp
from ott.core import sinkhorn
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from matplotlib.ticker import MaxNLocator


from unot.utils.transport import transport


def compute_knn_enrichment(pushfwd, treated, return_joint=False, ncells=None):
    if ncells is None:
        ncells = min(len(pushfwd), len(treated))
    assert ncells <= len(pushfwd)
    assert ncells <= len(treated)

    pushfwd = pushfwd.sample(n=ncells, random_state=0)
    treated = treated.sample(n=ncells, random_state=0)

    joint = pd.concat((pushfwd, treated), axis=0)

    labels = pd.concat(
        (
            pd.Series("pushfwd", index=pushfwd.index),
            pd.Series("treated", index=treated.index),
        )
    ).astype("category")

    n_neighbors = min([251, ncells])
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(joint)
    dists, knn = model.kneighbors(pushfwd)
    # assert np.all(knn[:, 0] == np.arange(len(knn)))
    knn = pd.DataFrame(knn[:, 1:], index=pushfwd.index)

    enrichment = knn.applymap(lambda x: labels.iloc[x] == "pushfwd")
    knn = knn.applymap(lambda x: labels.index[x])

    if return_joint:
        return knn, enrichment, joint

    return knn, enrichment


def compute_drug_signature_differences(control, treated, pushfwd):
    base = control.mean(0)

    true = treated.mean(0) - base
    pred = pushfwd.mean(0) - base

    diff = true - pred
    return diff


def compute_mmd_df(
    target, transport, gammas=None, subsample=False, ncells=None, nreps=5
):

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    gammas = list(gammas)

    def compute(ncells=None):
        for g in tqdm(gammas * nreps, desc="mmd", leave=False):
            mmd = mmd_distance(
                target if ncells is None else target.sample(ncells, random_state=0),
                transport
                if ncells is None
                else transport.sample(ncells, random_state=0),
                g,
            )

            yield g, mmd

    if subsample:
        if ncells is None:
            ncells = min(len(target), len(transport))
        else:
            ncells = min(len(target), len(transport), ncells)
    elif ncells is not None:
        assert ncells <= min(len(target), len(transport))

    mmd = pd.DataFrame(compute(ncells), columns=["gamma", "mmd"])
    return mmd


def compute_weighted_mmd_df(
    target, transport, w_x, gammas=None, subsample=False, ncells=None, nreps=5
):

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    gammas = list(gammas)

    def compute(ncells=None):
        for g in tqdm(gammas * nreps, desc="mmd", leave=False):
            y = target if ncells is None else target.sample(ncells, random_state=0)
            imp = (
                transport
                if ncells is None
                else transport.sample(ncells, random_state=0)
            )
            w = w_x.loc[imp.index, :]

            mmd = mmd_distance_weigthed(
                imp,
                y,
                w,
                g,
            )

            yield g, mmd

    if subsample:
        if ncells is None:
            ncells = min(len(target), len(transport))
        else:
            ncells = min(len(target), len(transport), ncells)
    elif ncells is not None:
        assert ncells <= min(len(target), len(transport))

    mmd = pd.DataFrame(compute(ncells), columns=["gamma", "mmd"])
    return mmd


def compute_weighted_mmd_df_resample(
    target, transport, w_x, gammas=None, subsample=False, ncells=None, nreps=5
):

    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    gammas = list(gammas)

    def compute(ncells=None):
        for g in tqdm(gammas * nreps, desc="mmd", leave=False):
            y = target if ncells is None else target.sample(ncells, random_state=0)
            imp = (
                transport
                if ncells is None
                else transport.sample(ncells, random_state=0)
            )
            w = w_x.loc[imp.index, :]

            mmd = resampled_mmd(
                imp,
                y,
                w,
                g,
            )

            yield g, mmd

    if subsample:
        if ncells is None:
            ncells = min(len(target), len(transport))
        else:
            ncells = min(len(target), len(transport), ncells)
    elif ncells is not None:
        assert ncells <= min(len(target), len(transport))

    mmd = pd.DataFrame(compute(ncells), columns=["gamma", "mmd"])
    return mmd


def load_all_inputs(config, setting, where, expdir="", save_csv=False):
    dataset, model_kwargs = load_data(
        config,
        split_on=["split", "transport"],
        return_as="dataset",
        include_model_kwargs=True,
    )

    if setting == "iid":
        to_pushfwd = dataset.test.source

    elif setting == "ood":
        to_pushfwd = dataset.ood.source

    else:
        raise ValueError(f"unknown setting, {setting} must be [iid, ood]")

    if setting == "iid":
        # control = pd.concat(
        #     (dataset.test.source.adata.to_df(), dataset.train.source.adata.to_df())
        # )
        control = dataset.test.source.adata.to_df()

        # treated = pd.concat(
        #     (dataset.test.target.adata.to_df(), dataset.train.target.adata.to_df())
        # )
        treated = dataset.test.target.adata.to_df()

    elif setting == "ood":
        control = dataset.ood.source.adata.to_df()

        if "target" in dataset.ood.keys():
            treated = dataset.ood.target.adata.to_df()
        else:
            treated = pd.concat(
                (dataset.test.target.adata.to_df(), dataset.train.target.adata.to_df())
            )

    else:
        raise ValueError(f"unknown setting, {setting} must be [iid, ood]")

    if config.model.name in ["nubot", "cellot"] and save_csv:
        filepath = Path(expdir / "data")
        filepath.mkdir(parents=True, exist_ok=True)
        dataset.test.source.adata.to_df().to_csv(expdir / "data" / "test_control.csv")
        dataset.test.target.adata.to_df().to_csv(expdir / "data" / "test_treated.csv")
        dataset.train.source.adata.to_df().to_csv(expdir / "data" / "train_control.csv")
        dataset.train.target.adata.to_df().to_csv(expdir / "data" / "train_treated.csv")

    obs = load_data(config, split_on=[], return_as="anndata").obs
    return control, treated, to_pushfwd, obs, model_kwargs


def load_conditions(expdir, where, setting, save_csv=False):
    config = load_config(expdir / "config.yaml")

    control, treated, to_pushfwd, obs, model_kwargs = load_all_inputs(
        config,
        setting,
        where,
        expdir,
        save_csv,
    )

    if config.model.name == "identity":
        imputed = to_pushfwd.adata  # .to_df()

    elif config.model.name == "random":
        treated, imputed = np.array_split(
            treated.sample(n=2 * len(to_pushfwd), random_state=0, replace=True), 2
        )
        imputed.index = to_pushfwd.adata.obs.index
    elif config.model.name == "ubot":
        # UBOT is computed on test&train set
        # take test samples
        transported = transport_ubot(control, treated)
        test_mask = control.index.isin(to_pushfwd.adata.obs.index)
        imputed = transported[test_mask]
        imputed.index = to_pushfwd.adata.obs.index
        imputed = anndata.AnnData(imputed, obs=obs.loc[to_pushfwd.adata.obs.index])

    elif config.model.name == "ot":
        # entropic reg sinkhorn
        transported = transport_ot(control, treated)
        test_mask = control.index.isin(to_pushfwd.adata.obs.index)
        imputed = transported[test_mask]
        imputed.index = to_pushfwd.adata.obs.index
        imputed = anndata.AnnData(imputed, obs=obs.loc[to_pushfwd.adata.obs.index])

    else:
        assert config.model.name in {"cellot", "nubot", "gan", "gaussian"}
        model, *_ = load_model(
            config, restore=expdir / "cache" / "last.pt", **model_kwargs
        )
        d = expdir / "cache" / "last.pt"
        step = load_item_from_save(d, "step", 0)
        # assert step == (config.training.n_iter - 1)
        print(step)

        imputed = transport(config, model, to_pushfwd)  # .to_df()

        if config.model.name in ["nubot", "cellot"] and save_csv:
            test_source = pd.read_csv(expdir / "data" / "test_control.csv", index_col=0)
            test_predicted = transport(config, model, to_pushfwd)
            if config.model.name == "nubot":
                test_predicted_w = test_predicted.obsm["weights"]
                test_predicted_w.to_csv(expdir / "data" / "test_predicted_weights.csv")
            test_predicted = test_predicted.to_df()
            test_predicted.to_csv(expdir / "data" / "test_predicted.csv")

    return control, treated, imputed


def visualize_weights_UMAP(
    umap,
    imputed,
    weights,
    treated,
    outdir,
    feature="ClCasp3",
    name="UMAP_",
    timestep="",
):

    sns.set_context(context="talk", font_scale=1.2)

    def scatter(axs, x, y, c, cmap, title="", clim=None, **kwargs):
        plt.sca(axs)
        # axs.axis('off')
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        # axs.spines['bottom'].set_visible(False)
        # axs.spines['left'].set_visible(False)
        plt.scatter(x, y, c=c, s=1, **kwargs)
        if clim is not None:
            plt.clim(0, clim)
        plt.colorbar(ticks=MaxNLocator(integer=True))
        plt.set_cmap(cmap)
        plt.title(title)
        plt.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )

    imp_umap = umap[umap["is_imputed"] == True]
    tre_umap = umap[umap["is_imputed"] == False]

    imp_umap = imp_umap.join(weights, how="left")

    imp_cell = (
        imputed.filter(regex="cell-" + feature, axis=1).loc[imp_umap.index, :].values
    )
    imp_nucl = (
        imputed.filter(regex="nuclei-" + feature, axis=1).loc[imp_umap.index, :].values
    )
    tre_cell = (
        treated.filter(regex="cell-" + feature, axis=1).loc[tre_umap.index, :].values
    )
    tre_nucl = (
        treated.filter(regex="nuclei-" + feature, axis=1).loc[tre_umap.index, :].values
    )

    fname_cell = imputed.filter(regex="cell-" + feature, axis=1).columns.to_list()[0]
    fname_nucl = imputed.filter(regex="nuclei-" + feature, axis=1).columns.to_list()[0]

    quantiles = pd.read_csv("./datasets/4i/feature_quantiles_09999.csv", index_col=0)

    clim_cell = quantiles.loc[fname_cell, "quantile"]
    clim_nucl = quantiles.loc[fname_nucl, "quantile"]

    if len(imp_cell) == 0:
        print("Feature does not exist.")
        return

    cmap = "BBR"

    if feature in ["ClCasp3", "Ki67", "MelA", "Sox9"]:
        cmap_r = cmap + "_r"
    else:
        cmap_r = cmap

    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4)
    )

    scatter(
        axes[0],
        tre_umap["UMAP1"],
        tre_umap["UMAP2"],
        c=tre_cell,
        cmap=cmap,
        title=f"Observed {timestep}\n" + feature + " (Cell)",
        clim=clim_cell,
    )
    scatter(
        axes[1],
        imp_umap["UMAP1"],
        imp_umap["UMAP2"],
        c=imp_cell,
        cmap=cmap,
        title=f"Predicted {timestep}\n" + feature + " (Cell)",
        clim=clim_cell,
    )
    scatter(
        axes[2],
        imp_umap["UMAP1"],
        imp_umap["UMAP2"],
        c=imp_umap["weights"],
        cmap=cmap_r,
        title=f"Predicted {timestep}\n Weights",
        clim=2.0,
    )

    pat = outdir / (name + "_" + feature + "_cell" + ".pdf")

    plt.savefig(pat, format="pdf", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4)
    )

    scatter(
        axes[0],
        tre_umap["UMAP1"],
        tre_umap["UMAP2"],
        c=tre_nucl,
        cmap=cmap,
        title=f"Observed {timestep}\n" + feature + " (Nuclei)",
        clim=clim_nucl,
    )
    scatter(
        axes[1],
        imp_umap["UMAP1"],
        imp_umap["UMAP2"],
        c=imp_nucl,
        cmap=cmap,
        title=f"Predicted {timestep}\n" + feature + " (Nuclei)",
        clim=clim_nucl,
    )
    scatter(
        axes[2],
        imp_umap["UMAP1"],
        imp_umap["UMAP2"],
        c=imp_umap["weights"],
        cmap=cmap_r,
        title=f"Predicted {timestep}\n Weights",
        clim=2.0,
    )

    pat = outdir / (name + "_" + feature + "_nuclei" + ".pdf")

    plt.savefig(pat, format="pdf", bbox_inches="tight")
    plt.close()


def visualize_weights_UMAP_control(
    umap,
    control,
    imputed,
    weights,
    outdir,
    feature="ClCasp3",
    name="UMAP_",
    timestep="",
):
    sns.set_context(context="talk", font_scale=1.3)

    def scatter(axs, x, y, c, cmap, title="", clim=None, **kwargs):
        plt.sca(axs)
        # axs.axis('off')
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        # axs.spines['bottom'].set_visible(False)
        # axs.spines['left'].set_visible(False)
        plt.scatter(x, y, c=c, s=1, **kwargs)
        if clim is not None:
            plt.clim(0, clim)
        plt.colorbar(ticks=MaxNLocator(integer=True))
        plt.set_cmap(cmap)
        plt.title(title)
        plt.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )

    umap = umap.join(weights, how="left")

    ctr_cell = control.filter(regex="cell-" + feature, axis=1).loc[umap.index, :].values
    ctr_nucl = (
        control.filter(regex="nuclei-" + feature, axis=1).loc[umap.index, :].values
    )
    pre_cell = imputed.filter(regex="cell-" + feature, axis=1).loc[umap.index, :].values
    pre_nucl = (
        imputed.filter(regex="nuclei-" + feature, axis=1).loc[umap.index, :].values
    )

    fname_cell = imputed.filter(regex="cell-" + feature, axis=1).columns.to_list()[0]
    fname_nucl = imputed.filter(regex="nuclei-" + feature, axis=1).columns.to_list()[0]

    quantiles = pd.read_csv("./datasets/4i/feature_quantiles_09999.csv", index_col=0)

    clim_cell = quantiles.loc[fname_cell, "quantile"]
    clim_nucl = quantiles.loc[fname_nucl, "quantile"]

    if len(ctr_cell) == 0:
        print("Feature does not exist.")
        return

    cmap = "BBR"

    if feature in ["ClCasp3", "Ki67", "MelA", "Sox9", "PCNA"]:
        cmap_r = cmap + "_r"
    else:
        cmap_r = cmap

    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4)
    )

    scatter(
        axes[0],
        umap["UMAP1"],
        umap["UMAP2"],
        c=ctr_cell,
        cmap=cmap,
        title=f"Control {timestep}\n " + feature + " (Cell)\n in Control",
        clim=clim_cell,
    )
    scatter(
        axes[1],
        umap["UMAP1"],
        umap["UMAP2"],
        c=pre_cell,
        cmap=cmap,
        title=f"Control {timestep}\n " + feature + " (Cell)\n in Predicted",
        clim=clim_cell,
    )
    scatter(
        axes[2],
        umap["UMAP1"],
        umap["UMAP2"],
        c=umap["weights"],
        cmap=cmap_r,
        title=f"Control {timestep}\n Weights",
        clim=2.0,
    )

    pat = outdir / (name + "_" + feature + "_cell" + ".pdf")

    plt.savefig(pat, format="pdf", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 4)
    )

    scatter(
        axes[0],
        umap["UMAP1"],
        umap["UMAP2"],
        c=ctr_nucl,
        cmap=cmap,
        title=f"Control {timestep}\n " + feature + " (Nuclei)\n in Control",
        clim=clim_nucl,
    )
    scatter(
        axes[1],
        umap["UMAP1"],
        umap["UMAP2"],
        c=pre_nucl,
        cmap=cmap,
        title=f"Control {timestep}\n" + feature + " (Nuclei)\n in Predicted",
        clim=clim_nucl,
    )
    scatter(
        axes[2],
        umap["UMAP1"],
        umap["UMAP2"],
        c=umap["weights"],
        cmap=cmap_r,
        title=f"Control {timestep}\n Weights",
        clim=2.0,
    )

    pat = outdir / (name + "_" + feature + "_nuclei" + ".pdf")

    plt.savefig(pat, format="pdf", bbox_inches="tight")
    plt.close()


# def compute_wasserstein_distance(source, target, reg=0.5, ncells=None):
#     """Computes transport cost between x and y via Sinkhorn algorithm."""
# x = source if ncells is None else source.sample(ncells, random_state=0)
# y = target if ncells is None else target.sample(ncells, random_state=0)

# a = np.ones(len(x)) / len(x)
# b = np.ones(len(y)) / len(y)

# M = ot.dist(x.to_numpy(), y.to_numpy())

# w2 = ot.bregman.sinkhorn2(a, b, M, reg)
# return w2


def compute_wasserstein_distance(
    source, target, epsilon, ncells=None, tau_b=1.0, use_POT=False
):
    """Computes transport between x and y via Sinkhorn algorithm."""
    x = source if ncells is None else source.sample(ncells, random_state=0)
    y = target if ncells is None else target.sample(ncells, random_state=0)

    if use_POT:
        x = torch.tensor(x.to_numpy())
        y = torch.tensor(y.to_numpy())

        a = torch.ones(len(x)) / len(x)
        b = torch.ones(len(y)) / len(y)

        print(a.sum())
        print(b.sum())

        M = ot.dist(x, y)  # , metric="euclidean")
        M /= M.max()
        ot_dist = ot.bregman.sinkhorn2(a, b, M, reg=epsilon)
        return ot_dist.numpy()
    else:

        a = jnp.ones(len(x)) / len(x)
        b = jnp.ones(len(y)) / len(y)

        # compute cost
        # geom_xy = pointcloud.PointCloud(x.to_numpy(), y.to_numpy(), epsilon=epsilon)

        # solve ot problem
        # out_xy = sinkhorn.sinkhorn(geom_xy, a, b, max_iterations=10000, min_iterations=10)

        out_xy = sinkhorn_divergence(
            pointcloud.PointCloud,
            x.to_numpy(),
            y.to_numpy(),
            a=a,
            b=b,
            epsilon=epsilon,
            sinkhorn_kwargs={"tau_b": tau_b},
        )

        # return regularized ot cost
        # return out_xy.reg_ot_cost
        return out_xy.divergence


def compute_wasserstein_distance_resampled(
    source, target, w_x, epsilon, ncells=None, tau_b=1.0, use_POT=False
):
    """Computes wasserstein distance on a resampled sample."""
    p = np.array(w_x["weights"].to_numpy()).astype("float64")
    p /= p.sum()

    source_rs = source.iloc[np.random.choice(len(source), size=len(source), p=p), :]
    return compute_wasserstein_distance(
        source_rs, target, epsilon=epsilon, ncells=ncells, tau_b=tau_b, use_POT=use_POT
    )


def compute_wasserstein_distance_weighted(
    source, target, w_x, epsilon, ncells=None, tau_b=1.0, use_POT=False
):
    """use weights in a,b"""
    x = source if ncells is None else source.sample(ncells, random_state=0)
    y = target if ncells is None else target.sample(ncells, random_state=0)

    w = w_x.loc[x.index, :]

    p = w.values.squeeze(-1)
    p /= p.mean()

    if use_POT:
        x = torch.tensor(x.to_numpy())
        y = torch.tensor(y.to_numpy())

        a = torch.ones(len(x)) / len(x) * p
        b = torch.ones(len(y)) / len(y)

        print(a.sum())
        print(b.sum())

        M = ot.dist(x, y)  # , metric="euclidean")
        M /= M.max()
        ot_dist = ot.bregman.sinkhorn2(a, b, M, reg=epsilon)
        return ot_dist.numpy()

    else:

        a = jnp.ones(len(x)) / len(x) * p
        b = jnp.ones(len(y)) / len(y)

        print(a.sum())
        print(b.sum())

        # compute cost
        # geom_xy = pointcloud.PointCloud(x.to_numpy(), y.to_numpy(), epsilon=epsilon)

        # solve ot problem
        # out_xy = sinkhorn.sinkhorn(geom_xy, a, b, max_iterations=10000, min_iterations=10)
        out_xy = sinkhorn_divergence(
            pointcloud.PointCloud,
            x.to_numpy(),
            y.to_numpy(),
            a=a,
            b=b,
            epsilon=epsilon,
            sinkhorn_kwargs={"tau_b": tau_b},
        )

        # return regularized ot cost
        # return out_xy.reg_ot_cost
        return out_xy.divergence


def compute_transport_cost(source, predicted, w=None, resample=True):
    if w is None:
        return ((source.to_numpy() - predicted.to_numpy()) ** 2).sum(axis=1)
    else:
        # normalize weights (sum=1)
        p = np.array(w["weights"].to_numpy()).astype("float64")
        p /= p.sum()

        if resample:

            idx = np.random.choice(len(source), size=len(source), p=p)
            source_rs = source.iloc[idx, :]
            predicted_rs = predicted.iloc[idx, :]

            # removed "p*" because it is resampled already.
            return ((source_rs.to_numpy() - predicted_rs.to_numpy()) ** 2).sum(axis=1)
        else:
            # make sum=len(p)
            p = p * len(p)
            return p * ((source.to_numpy() - predicted.to_numpy()) ** 2).sum(axis=1)


def visualize_joint_UMAPS(umap_control, umap_treated, weights):
    def scatter(axs, x, y, c, cmap, title="", clim=False, **kwargs):
        plt.sca(axs)
        plt.scatter(x, y, c=c, s=5, **kwargs)
        if clim:
            plt.clim(0, 2.0)
        plt.colorbar()
        plt.set_cmap(cmap)
        plt.title(title)

    imp_umap = umap_treated[umap_treated["is_imputed"] == True]
    tre_umap = umap_treated[umap_treated["is_imputed"] == False]

    cmap = "BBR"
    cmap_r = "BBR_r"

    fig, axes = plt.subplots(
        nrows=3, ncols=3, sharex=True, sharey=True, figsize=(20, 20)
    )

    ctrl_8 = umap_control[umap_control.h == 8]
    scatter(
        axes[0, 0],
        ctrl_8["UMAP1"],
        ctrl_8["UMAP2"],
        c=weights.loc[ctrl_8.index, "weights"],
        cmap=cmap_r,
        title="control, 8h",
        clim=True,
    )
    ctrl_24 = umap_control[umap_control.h == 24]
    scatter(
        axes[0, 1],
        ctrl_24["UMAP1"],
        ctrl_24["UMAP2"],
        c=ctrl_24["weights"],
        cmap=cmap_r,
        title="control, 24h",
        clim=True,
    )
    ctrl_48 = umap_control[umap_control.h == 48]
    scatter(
        axes[0, 2],
        ctrl_48["UMAP1"],
        ctrl_48["UMAP2"],
        c=ctrl_48["weights"],
        cmap=cmap_r,
        title="control, 48h",
        clim=True,
    )

    name = "joint_UMAP_timecourse"

    plt.savefig(name, format="pdf", bbox_inches="tight")
    plt.close()
