from math import radians
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from absl import app, flags
from unot.utils.evaluate import (
    compute_drug_signature_differences,
    compute_knn_enrichment,
    compute_mmd_df,
    compute_wasserstein_distance,
    compute_wasserstein_distance_resampled,
    compute_weighted_mmd_df_resample,
    load_conditions,
    visualize_weights_UMAP,
    compute_weighted_mmd_df,
    visualize_joint_UMAPS,
    compute_transport_cost,
    visualize_weights_UMAP_control,
    # wasserstein_loss,
    compute_wasserstein_distance_weighted,
)

from unot.plotting.setup import setup_plt
from unot.utils.helpers import fix_random_seeds
from unot.utils import load_config
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from unot.data.cell import read_single_anndata


FLAGS = flags.FLAGS
flags.DEFINE_boolean("predictions", True, "Run predictions.")
flags.DEFINE_string("outdir", "", "Path to outdir.")
flags.DEFINE_list("outdirs_24_48", [], "Path to outdir.")
flags.DEFINE_integer("n_markers", None, "Number of marker genes.")
flags.DEFINE_string("subset", None, "Name of obs entry to use as subset.")
flags.DEFINE_string("subset_name", None, "Name of subset.")
flags.DEFINE_enum("setting", "iid", ["iid", "ood"], "Evaluate iid or ood setting.")
flags.DEFINE_enum(
    "where",
    "data_space",
    ["data_space", "latent_space"],
    "In which space to conduct analysis.",
)
flags.DEFINE_multi_string("via", "", "Directory containing compositional map.")
flags.DEFINE_string("subname", "", "")
flags.DEFINE_boolean("save_csv", False, "Save csv files of data.")
flags.DEFINE_boolean("plot_UMAPs", False, "Plot UMAPs of weights.")
flags.DEFINE_boolean("UMAP", False, "Compute umap and save.")


def main(argv):

    _ = flags.FLAGS(argv, known_only=True)
    expdir = Path(FLAGS.outdir)
    setting = FLAGS.setting
    where = FLAGS.where
    subset = FLAGS.subset
    subset_name = FLAGS.subset_name

    if subset is None:
        outdir = expdir / f"evals_{setting}_{where}"
    else:
        assert subset is not None
        outdir = expdir / f"evals_{setting}_{where}_{subset}_{subset_name}"

    if len(FLAGS.subname) > 0:
        outdir = outdir / FLAGS.subname

    outdir.mkdir(exist_ok=True, parents=True)

    config = load_config(expdir / "config.yaml")
    cache = outdir / "imputed.h5ad"

    control, treated, imputed = load_conditions(expdir, where, setting, FLAGS.save_csv)
    # imputed.write(cache)
    if config.model.name != "random":
        if "weights" in imputed.obsm.keys():
            weights = imputed.obsm["weights"]
        imputed = imputed.to_df()

    imputed.columns = imputed.columns.astype(str)
    treated.columns = treated.columns.astype(str)
    control.columns = control.columns.astype(str)

    assert imputed.columns.equals(treated.columns)

    # evaluation of transport map without weights

    ncells = len(treated)
    print("ncells=", ncells)

    l2ds = compute_drug_signature_differences(control, treated, imputed)
    knn, enrichment, joint = compute_knn_enrichment(imputed, treated, return_joint=True)
    mmddf = compute_mmd_df(treated, imputed, subsample=True, ncells=ncells)

    w2_dict = {}
    epsilons = [1, 2, 3, 4, 5]
    tau_bs = [0.95, 1.0]
    for eps in epsilons:
        for tau_b in tau_bs:
            w2_imp_tre = compute_wasserstein_distance(
                imputed,
                treated,
                epsilon=eps,
                ncells=ncells,
                tau_b=tau_b,
            )
            key = "w2_" + str(eps) + "_" + str(tau_b)
            w2_dict[key] = w2_imp_tre
    cost = compute_transport_cost(control.loc[imputed.index, :], imputed)
    total_cost = cost.sum()
    avg_cost = cost.mean()

    l2ds.to_csv(outdir / "drug_signature_diff.csv")
    enrichment.to_csv(outdir / "knn_enrichment.csv")
    knn.to_csv(outdir / "knn_neighbors.csv")
    mmddf.to_csv(outdir / "mmd.csv")
    pd.DataFrame(cost, index=imputed.index, columns=["cost"]).to_csv(
        outdir / "cost.csv"
    )

    summary = {
        "l2DS": np.sqrt((l2ds**2).sum()),
        "enrichment-k50": enrichment.iloc[:, :50].values.mean(),
        "enrichment-k100": enrichment.iloc[:, :100].values.mean(),
        "mmd": mmddf["mmd"].mean(),
        # "w2": w2_imp_tre,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
    }

    summary.update(w2_dict)
    summary = pd.Series(summary)
    summary.to_csv(outdir / "evals.csv", header=None)

    if FLAGS.UMAP:
        umap = pd.DataFrame(
            UMAP(random_state=0).fit_transform(joint),
            index=joint.index,
            columns=["UMAP1", "UMAP2"],
        )
        umap["is_imputed"] = umap.index.isin(imputed.index)
        umap.to_csv(outdir / "umap.csv")

    # check if weights exist. If yes, compute weight specific evaluation metrics.
    if "weights" in locals():
        # resampling, then mmd
        mmddf = compute_weighted_mmd_df(
            treated, imputed, weights, subsample=True, ncells=ncells
        )

        mmddf_rs = compute_weighted_mmd_df_resample(
            treated, imputed, weights, subsample=True, ncells=ncells
        )

        w2_dict = {}

        for eps in epsilons:
            for tau_b in tau_bs:
                w2_imp_tre_rw = compute_wasserstein_distance_resampled(
                    imputed, treated, weights, epsilon=eps, tau_b=tau_b, ncells=ncells
                )
                w2_imp_tre_w = compute_wasserstein_distance_weighted(
                    imputed, treated, weights, epsilon=eps, tau_b=tau_b, ncells=ncells
                )

                key = "w2_w_" + str(eps) + "_" + str(tau_b)
                w2_dict[key] = w2_imp_tre_w

                key = "w2_w_rs_" + str(eps) + "_" + str(tau_b)
                w2_dict[key] = w2_imp_tre_rw

        cost_rs = compute_transport_cost(
            control.loc[imputed.index, :], imputed, weights, resample=True
        )
        total_cost_rs = cost.sum()
        avg_cost_rs = cost.mean()

        cost = compute_transport_cost(
            control.loc[imputed.index, :], imputed, weights, resample=False
        )
        total_cost = cost.sum()
        avg_cost = cost.mean()

        mmddf.to_csv(outdir / "mmd_w.csv")
        mmddf_rs.to_csv(outdir / "mmd_w_rs.csv")

        summary = {
            "mmd_w": mmddf["mmd"].mean(),
            "mmd_w_rs": mmddf_rs["mmd"].mean(),
            "weights_mean": weights.values.mean(),
            "weights_std": weights.values.std(),
            # "w2_w_rs": w2_imp_tre_rw,
            # "w2_w": w2_imp_tre_w,
            "total_cost_w": total_cost,
            "avg_cost_w": avg_cost,
            "total_cost_w_rs": total_cost_rs,
            "avg_cost_w_rs": avg_cost_rs,
        }

        summary.update(w2_dict)
        summary = pd.Series(summary)
        summary.to_csv(outdir / "evals_weights.csv", header=None)

        if FLAGS.plot_UMAPs:
            full_joint = joint

            # embedded on full joint
            umap_full = pd.DataFrame(
                UMAP(random_state=0).fit_transform(full_joint),
                index=full_joint.index,
                columns=["UMAP1", "UMAP2"],
            )
            umap_full["is_imputed"] = umap_full.index.isin(imputed.index)

            # embedded on treated only (target)
            reducer = UMAP(random_state=0).fit(treated)
            umap_treated = pd.DataFrame(
                reducer.transform(full_joint),
                index=full_joint.index,
                columns=["UMAP1", "UMAP2"],
            )
            umap_treated["is_imputed"] = umap_treated.index.isin(imputed.index)

            timestep = config.data.path.split("/")[-1].replace("_subm.h5ad", "")

            for umap, n in zip([umap_full, umap_treated], ["full", "treated"]):

                name = config.data.target + "_UMAP_fitted_on_" + n

                for feature in ["ClCasp3", "Ki67", "Sox9", "MelA", "PCNA"]:
                    visualize_weights_UMAP(
                        umap,
                        imputed,
                        weights,
                        treated,
                        outdir,
                        feature=feature,
                        name=name,
                        timestep=timestep,
                    )

            test_control = control.loc[imputed.index, :]
            umap_ctr = pd.DataFrame(
                UMAP(random_state=0).fit_transform(test_control),
                index=test_control.index,
                columns=["UMAP1", "UMAP2"],
            )

            for feature in ["ClCasp3", "Ki67", "Sox9", "MelA", "PCNA"]:
                name = config.data.target + "_UMAP_fitted_on_control"
                visualize_weights_UMAP_control(
                    umap_ctr,
                    test_control,
                    imputed,
                    weights,
                    outdir,
                    feature=feature,
                    name=name,
                    timestep=timestep,
                )

    return


if __name__ == "__main__":

    fix_random_seeds()
    setup_plt()
    main(sys.argv)
