import sys
import pandas as pd
import numpy as np
import anndata
from pathlib import Path
from absl import flags
from scripts.preprocess import parse_raw_dd_data
from unot.utils.helpers import parse_config_cli, dump_config
from unot.data.cell import split_cell_data
import yaml

FLAGS = flags.FLAGS

flags.DEFINE_string("datapath", "", "")
flags.DEFINE_string("outpath", "", "")
flags.DEFINE_string("name", "", "")
flags.DEFINE_string("config", "", "")
flags.DEFINE_boolean("dry", False, "")


def apply_filter(config, data):
    if "min_well_size" in config:
        keep = data.obs["well_name"].value_counts() >= config.min_well_size
        mask = data.obs["well_name"].isin(keep.index[keep])
        data = data[mask].copy()

    return data


def remove_background(view, q=0.75, key="drug", value="sec.abcl"):
    df = view.to_df()
    secondary = df.loc[view.obs[key] == value].quantile(q)

    return (df - secondary).clip(0).values


def nan_zero_inflated_cells(view, n, notin=None):
    df = view.to_df()
    obs = view.obs

    if notin is None:
        keep = pd.Series(False, index=view.obs_names)
    else:
        key = notin["key"]
        values = list(notin["values"])
        keep = obs[key].isin(values)

    # only remove cells within the mask
    nmissing = (df.loc[~keep] == 0).sum(axis=1)
    keep.update(nmissing <= n)
    df.loc[~keep.astype(bool)] = np.nan

    return df.values


def qscale(data, path=None, q=None, key=None, value=None):
    if path is not None:
        scales = pd.read_csv(path, index_col=0, squeeze=True)
    elif q is not None:
        df = data.to_df()
        if (key is not None) or (value is not None):
            assert (key is not None) and (value is not None)
            mask = data.obs[key] == value
            train_only = data.obs["split"] == "train"
            df = df.loc[mask & train_only]
        scales = df.quantile(q)
    view = data[:, scales.index]
    view.X = view.X / scales.values
    return


def preprocess(config, data):

    for item in config.preprocess:
        name = item.pop("name")
        features = item.pop("features", None)
        kwargs = dict(item)

        if features is not None:
            view = data[:, data.var_names.str.match(features)]
        else:
            view = data

        if name == "log1p":
            view.X = np.log(view.X + 1)

        elif name == "arcsinh":
            cf = item.get("cofactor", 1)
            view.X = np.arcsinh(view.X / cf) * cf

        elif name == "qscale":
            assert features is None
            qscale(data, **kwargs)

        elif name == "remove_background":
            view.X = remove_background(view, **kwargs)

        elif name == "nan_zero_inflated_cells":
            view.X = nan_zero_inflated_cells(view, **kwargs)

        elif name == "dropna":
            keep = view.to_df().notna().any(axis=1)
            data = data[keep].copy()

        else:
            raise ValueError

    return data


def main(config, outpath, datapath):
    if datapath.suffix == ".csv":
        data = parse_raw_dd_data(pd.read_csv(datapath))
    else:
        data = anndata.read(datapath)

    # split data into train and test
    # config = yaml.load(Path("configs/data/4i-8h-raw.yaml"), Loader=yaml.FullLoader)
    # config = load_config(Path("configs/data/4i-8h-raw.yaml"))
    split = split_cell_data(data, **config.datasplit)
    data.obs["split"] = split

    data = apply_filter(config, data)
    if config.get("features") is not None:
        features = Path(config.features).read_text().split()
        assert np.in1d(features, data.var_names).all()
        data = data[:, features].copy()

    if (data.obs["drug"] == "control").sum() < 500:
        raise ValueError

    data = preprocess(config, data)

    assert not data.obs_names.duplicated().any()
    assert not data.var_names.duplicated().any()

    data.write(outpath)
    dump_config(outpath.with_suffix(".yaml"), config)
    return


if __name__ == "__main__":
    _, *unparsed = flags.FLAGS(sys.argv, known_only=True)
    config = parse_config_cli(FLAGS.config, unparsed)

    datapath = Path(FLAGS.datapath)
    if len(FLAGS.outpath) == 0:
        assert len(FLAGS.name) > 0
        outpath = datapath.with_name(FLAGS.name + ".h5ad")
    else:
        outpath = FLAGS.outpath
    outpath = Path(outpath)

    if FLAGS.dry:
        print(datapath)
        print(outpath)
        print(config)
        sys.exit()

    main(config, outpath, datapath)
