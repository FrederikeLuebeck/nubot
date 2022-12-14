from pathlib import Path
import sys
import numpy as np
import pandas as pd
from absl import app, flags
from unot.utils.evaluate import (
    compute_mmd_df,
    compute_wasserstein_distance,
    compute_wasserstein_distance_resampled,
    compute_weighted_mmd_df_resample,
    compute_weighted_mmd_df,
    compute_transport_cost,
    compute_wasserstein_distance_weighted,
)
from unot.data.utils import cast_loader_to_iterator
from unot.utils.loaders import load

from unot.plotting.setup import setup_plt
from unot.utils.helpers import fix_random_seeds
from unot.utils import load_config
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from unot.data.toy import load_toy_data


FLAGS = flags.FLAGS
flags.DEFINE_string("outdir", "", "Path to outdir.")


def main(argv):
    _ = flags.FLAGS(argv, known_only=True)
    expdir = Path(FLAGS.outdir)

    outdir = expdir / f"evals"
    outdir.mkdir(exist_ok=True, parents=True)

    config = load_config(expdir / "config.yaml")
    assert config.data.type == "toy"

    # set batchsize to 400 for evaluation
    config.data.loader.batch_size = 400

    if config.model.name == "nubot_v1":
        (f, g, h_fw, h_bw), opts, loader = load(
            config, restore=expdir / "cache" / "last.pt"
        )
        data = cast_loader_to_iterator(loader, cycle_all=True)

        control = next(data.test.source)
        treated = next(data.test.target)

        imputed = g.transport(control.requires_grad_(True))
        w_x = h_fw(control)
        w_y = h_bw(imputed)

        w_final = w_x / w_y

        # to df
        control = pd.DataFrame(control.detach().numpy())
        treated = pd.DataFrame(treated.detach().numpy())
        imputed = pd.DataFrame(imputed.detach().numpy())
        weights = pd.DataFrame(w_final.detach().numpy(), columns=["weights"])

    if config.model.name == "gan":
        (D, G), opts, loader = load(config, restore=expdir / "cache" / "last.pt")
        data = cast_loader_to_iterator(loader, cycle_all=True)
        control = next(data.test.source)
        treated = next(data.test.target)
        G.eval()
        D.eval()
        T, eps = G(control)

        # to df
        control = pd.DataFrame(control.detach().numpy())
        treated = pd.DataFrame(treated.detach().numpy())
        imputed = pd.DataFrame(T.detach().numpy())
        weights = pd.DataFrame(eps.detach().numpy(), columns=["weights"])

    if config.model.name == "cellot":
        (f, g), opts, loader = load(config, restore=expdir / "cache" / "last.pt")
        data = cast_loader_to_iterator(loader, cycle_all=True)
        control = next(data.test.source)
        treated = next(data.test.target)
        imputed = g.transport(control.requires_grad_(True))

        # to df
        control = pd.DataFrame(control.detach().numpy())
        treated = pd.DataFrame(treated.detach().numpy())
        imputed = pd.DataFrame(imputed.detach().numpy())
        weights = None

    # compute metrics
    mmddf = compute_mmd_df(treated, imputed, subsample=True, ncells=5000)
    w2_imp_tre = compute_wasserstein_distance(imputed, treated)
    cost = compute_transport_cost(control.loc[imputed.index, :], imputed)
    total_cost = cost.sum()
    avg_cost = cost.mean()

    summary = pd.Series(
        {
            "mmd": mmddf["mmd"].mean(),
            "w2": w2_imp_tre,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
        }
    )
    summary.to_csv(outdir / "evals.csv", header=None)

    if weights is not None:
        mmddf = compute_weighted_mmd_df(
            treated, imputed, weights, subsample=True, ncells=5000
        )
        mmddf_rs = compute_weighted_mmd_df_resample(
            treated, imputed, weights, subsample=True, ncells=5000
        )
        w2_imp_tre_rw = compute_wasserstein_distance_resampled(
            imputed, treated, weights
        )
        w2_imp_tre_w = compute_wasserstein_distance_weighted(imputed, treated, weights)

        cost = compute_transport_cost(control.loc[imputed.index, :], imputed, weights)
        total_cost = cost.sum()
        avg_cost = cost.mean()

        mmddf.to_csv(outdir / "mmd_w.csv")
        mmddf_rs.to_csv(outdir / "mmd_w_rs.csv")

        summary = pd.Series(
            {
                "mmd_w": mmddf["mmd"].mean(),
                "mmd_w_rs": mmddf_rs["mmd"].mean(),
                "weights_mean": weights.values.mean(),
                "weights_std": weights.values.std(),
                "w2_w_rs": w2_imp_tre_rw,
                "w2_w": w2_imp_tre_w,
                "total_cost_w": total_cost,
                "avg_cost_w": avg_cost,
            }
        )
        summary.to_csv(outdir / "evals_weights.csv", header=None)

        data_dir = outdir
        data_dir.mkdir(exist_ok=True, parents=True)

    imputed.to_csv(data_dir / "imputed.csv")
    treated.to_csv(data_dir / "treated.csv")
    control.to_csv(data_dir / "control.csv")
    if weights is not None:
        weights.to_csv(data_dir / "weights.csv")

    return


if __name__ == "__main__":
    fix_random_seeds()
    setup_plt()
    main(sys.argv)
