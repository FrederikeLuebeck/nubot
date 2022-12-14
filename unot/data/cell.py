#!/usr/bin/python3
# author: Stefan Stark

import anndata
import re
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import scanpy as sc

# from cellot.utils import load_config
from unot.data.utils import cast_dataset_to_loader
from unot.utils.helpers import nest_dict
from absl import logging
import re
from math import ceil


class AnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        value = self.adata.X[idx]

        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def read_list(arg):

    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg

    return list(lst)


def read_single_anndata(config, path=None):
    if path is None:
        path = config.data.path

    data = anndata.read(path)

    if "features" in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()

    # select subgroup of individuals
    if "individuals" in config.data:
        data = data[
            data.obs[config.data.individuals[0]].isin(config.data.individuals[1])
        ]

    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    transport_mapper = dict()
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)

    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    mask = data.obs["transport"].notna()
    if "subset" in config.data:
        for key, value in config.data.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # write train/test/valid into split column
    if "datasplit" in config:
        data.obs["split"] = split_cell_data(data, **config.datasplit)

    return data[mask].copy()


def read_merged_anndata(config, pathlist=None):
    def read_list(arg):
        if isinstance(arg, list):
            return arg

        path = Path(arg)
        assert path.exists()
        return path.read_text().split()

    def iterate(pathlist):
        for path in pathlist:
            data = read_single_anndata(config, path)
            if np.isnan(data.X).any():
                logging.warning(f"Omitting due to NA {path}")
                continue

            yield data.uns["sample"], data

    assert (config.data.type == "cell-merged") or (config.data.type == "tupro-cohort")

    if pathlist is None:
        pathlist = read_list(config.data.paths)

    merged = anndata.concat(dict(iterate(pathlist)), label="sample", index_unique="-")

    return merged


def select_cohort(table, key=None, value=None, regex=False):
    if key is None:
        assert value is None
        return pd.Series(True, index=table.index)

    if regex:
        pattr = re.compile(value)
        mask = table[key].str.match(pattr)
    else:
        mask = table[key] == value

    return mask.fillna(False)


def label_cohort_as_train_or_eval(
    samples, random_state=0, holdout=None, trainset=None, evalset=None
):

    if trainset is None:
        trainset = list()
    elif isinstance(trainset, str):
        trainset = [trainset]
    assert isinstance(trainset, list)

    if evalset is None:
        evalset = list()
    elif isinstance(evalset, str):
        evalset = [evalset]

    assert isinstance(evalset, list)
    assert not np.in1d(trainset, evalset).any()

    cohort = pd.Series("train", samples)

    if holdout == 0 or holdout is None:
        return cohort

    if holdout < 1:
        holdout = ceil(len(cohort) * holdout)

    nremove = holdout - len(evalset)
    assert nremove >= 0

    decide = [x for x in samples if (x not in evalset) and (x not in trainset)]

    # set seed just for random choice
    np.random.seed(None)
    st0 = np.random.get_state()
    np.random.seed(random_state)
    evalset.extend(np.random.choice(decide, nremove, replace=False))
    np.random.set_state(st0)

    cohort.loc[evalset] = "eval"

    return cohort


def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",  # ["loader", "anndata"]
    include_model_kwargs=False,
    **kwargs,
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition

    if data is None:
        if config.data.type == "cell":
            data = read_single_anndata(config, **kwargs)

        elif config.data.type == "cell-merged":
            data = read_merged_anndata(config, **kwargs)

    if config.data.get("select") is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    # if config.data.get("death") is not None:
    #     # let one cluster in the target population die / disappear
    #     target = data[data.obs["transport"] == "target"].copy()
    #     sc.pp.neighbors(target, n_neighbors=15)
    #     sc.tl.leiden(target)
    #     n_clusters = len(np.unique(target.obs["leiden"]))
    #     idx = np.random.choice(n_clusters)
    #     target.obs["dead"] = target.obs["leiden"] == str(idx)
    #     keep = pd.Series(True, index=data.obs_names)
    #     keep[target.obs[target.obs["dead"] == True].index] = False
    #     data.obs["keep"] = keep

    #     # data.raw = data.copy()
    #     # data = data[keep].copy()
    #     data.obs.loc[~keep, "transport"] = "dead"

    if "dimension_reduction" in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == "pca":
            dims = config.data.dimension_reduction.get(
                "dims", data.obsm["X_pca"].shape[1]
            )

            data = anndata.AnnData(
                data.obsm["X_pca"][:, :dims], obs=data.obs.copy(), uns=data.uns.copy()
            )
            data.uns["genes"] = genes

    # if "ae_emb" in config.data:
    #     # load path to autoencoder
    #     assert config.get("model.name", "cellot") == "cellot"
    #     path_ae = Path(config.data.ae_emb.path)
    #     model_kwargs = {"input_dim": data.n_vars}
    #     config_ae = load_config(path_ae / "config.yaml")
    #     ae_model, _ = load_autoencoder_model(
    #         config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
    #     )

    #     inputs = torch.Tensor(
    #         data.X if not sparse.issparse(data.X) else data.X.todense()
    #     )

    #     genes = data.var_names.to_list()
    #     data = anndata.AnnData(
    #         ae_model.eval().encode(inputs).detach().numpy(),
    #         obs=data.obs.copy(),
    #         uns=data.uns.copy(),
    #     )
    #     data.uns["genes"] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs["input_dim"] = data.n_vars

    if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

    if split_on is None:
        split_on = ["split", "transport"]
        # if config.model.name == "cellot":
        #     # datasets & dataloaders accessed as loader.train.source
        #     split_on = ["split", "transport"]

        # elif config.model.name == "scgen" or config.model.name == "cae":
        #     split_on = ["split"]

        # else:
        #     raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else ".".join(key)): data[index]
            for key, index in data.obs[split_on].groupby(split_on).groups.items()
        }

        dataset = nest_dict(
            {
                key: AnnDataDataset(val.copy(), **dataset_args)
                for key, val in splits.items()
            },
            as_dot_dict=True,
        )

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if "loader" in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault("drop_last", True)

        if config.training.get("adapt_batch_size", False):
            growth_rates_per_drug = pd.read_csv("./datasets/4i/growth_rates.csv")
            if len(re.findall("48h", config.data.path)) == 1:
                hours = "_48h"
            elif len(re.findall("24h", config.data.path)) == 1:
                hours = "_24h"
            else:
                hours = "_08h"
            col_name = "growth_factor" + hours
            if config.data["target"] == "all":
                raise NotImplementedError
            else:
                growth_rate = growth_rates_per_drug[
                    growth_rates_per_drug["Condition"].str.lower()
                    == config.data.target.lower()
                ][col_name].values[0]
                if growth_rate + 0.1 <= 1.0:
                    # include 10% more cells (rather oversampling than missing cells)
                    growth_rate = growth_rate + 0.1

                kwargs["batch_size_target_factor"] = growth_rate

        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "anndata":
            returns.append(data)

        elif key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def split_cell_data_train_test(
    data, groupby=None, random_state=0, holdout=None, **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(
    data,
    test_size=0.15,
    eval_size=0.15,
    groupby=None,
    random_state=0,
    holdout=None,
    **kwargs,
):

    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size
        )

        trainobs, testobs = train_test_split(
            training, random_state=random_state, test_size=test_size
        )

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):

    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """

    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype("category")
