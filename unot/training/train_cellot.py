import torch
import wandb
from tqdm import trange
import numpy as np

from unot.utils.loaders import load
from unot.training.eval import compute_metrics
from unot.plotting.plots import visualize, plot_scatter
from unot.data.utils import cast_loader_to_iterator
from unot.models.cellot import (
    compute_loss_f,
    compute_loss_g,
    state_dict,
)
from unot.utils.helpers import load_item_from_save


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def train_cellot(config, outdir):

    cachedir = outdir / "cache"

    (f, g), opts, loader = load(config, restore=cachedir / "last.pt")
    data = cast_loader_to_iterator(loader, cycle_all=True)

    step = 0
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    n_iter = config.training.n_iter
    ticker = trange(step, n_iter, initial=step, total=n_iter)

    for step in ticker:
        global EPOCH
        EPOCH = step

        y = next(data.train.target)

        for _ in range(config.training.n_iter_inner):
            x = next(data.train.source)
            x.requires_grad_(True)
            opts.g.zero_grad()

            yhat = g.transport(x)

            loss_g = compute_loss_g(f, g, x, yhat).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                loss_g = loss_g + g.penalize_w()

            loss_g.backward()
            opts.g.step()

            wandb.log({"train_loss": loss_g})

        x = next(data.train.source)
        x.requires_grad_(True)

        opts.f.zero_grad()
        loss_f = compute_loss_f(f, g, x, y).mean()
        loss_f.backward()
        opts.f.step()
        f.clamp_w()

        wandb.log({"train_loss": loss_f})

        check_loss(loss_g, loss_f)

        log_freq = config.training.get("log_freq", 500)
        if step % log_freq == 0 or step == config.training.n_iter - 1:
            if config.data.type == "cell":
                viz = False
            else:
                viz = True
            evaluate(config, g, f, data, viz=viz)
            torch.save(
                state_dict(f, g, opts, step=step),
                cachedir / "model.pt",
            )

        if step % config.training.get("cache_freq", 500) == 0:
            torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")

    torch.save(state_dict(f, g, opts, step=step), cachedir / "last.pt")


def evaluate(config, g, f, data, viz=True):
    x = next(data.test.source)
    y = next(data.test.target)
    x.requires_grad_(True)
    T = g.transport(x)
    T = T.detach()

    log_dict = compute_metrics(x, y, T)
    log_dict["epoch"] = EPOCH

    with torch.no_grad():
        log_dict["test_loss_g"] = compute_loss_g(f, g, x, transport=T).mean()
        log_dict["test_loss_f"] = compute_loss_f(f, g, x, y, transport=T).mean()

    wandb.log(log_dict)

    if viz:
        visualize(x, y, T, config.data.type, epoch=EPOCH)

        # check backward direction (nabla f transports y to x)
        y.requires_grad_(True)
        xhat = f.transport(y)

        _, plot = plot_scatter(y.detach(), x.detach(), xhat.detach())
        wandb.log({"backwards_scatter": wandb.Image(plot)})
