import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import trange
import numpy as np

from unot.utils.loaders import load_data
from unot.data.utils import cast_loader_to_iterator
from unot.models.gaussian_closed_form import get_closed_form_joint, load_gaussian_model
from unot.utils.helpers import load_item_from_save


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fit_gaussian_approx(config, outdir):

    cachedir = outdir / "cache"

    dataset, model_kwargs = load_data(
        config, include_model_kwargs=True, return_as=["dataset"]
    )

    input_dim = model_kwargs["input_dim"]
    # data = cast_loader_to_iterator(loader, cycle_all=True)

    source = dataset.train.source.adata.X
    target = dataset.train.target.adata.X

    print("len(source)=" + str(len(source)))

    gamma = config.training.get("gamma", None)
    if gamma == "None":
        gamma = None

    joint_mean, joint_cov = get_closed_form_joint(
        source, target, sigma=0.1, gamma=gamma
    )

    # save state dict of Linear transformation
    # torch.save(s
    #     state_dict(f, g, opts, step=step),
    #     cachedir / "model.pt",
    # )

    OT_map, *_ = load_gaussian_model(config, **model_kwargs)

    assert len(joint_mean) == input_dim * 2
    assert len(joint_cov) == input_dim * 2

    # conditional expectation of joint given x is
    # m2 + cov21 cov22^-1 (a-m1)
    A = joint_cov[0:input_dim, 0:input_dim]
    B = joint_cov[input_dim:, input_dim:]
    C1 = joint_cov[0:input_dim, input_dim:]
    C2 = joint_cov[input_dim:, 0:input_dim]

    M = C2 @ np.linalg.inv(A)

    OT_map.weight = torch.nn.Parameter(torch.tensor(M))

    OT_map.bias = torch.nn.Parameter(
        torch.tensor(joint_mean[input_dim:] - (M @ joint_mean[0:input_dim].T).T)
    )

    torch.save(OT_map.state_dict(), cachedir / "last.pt")
    torch.save(OT_map.state_dict(), cachedir / "model.pt")
    print("Saved model.")
