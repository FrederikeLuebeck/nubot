import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import trange
import numpy as np
from sys import exit

from unot.utils.loaders import load
from unot.training.eval import compute_metrics
from unot.data.utils import cast_loader_to_iterator
from unot.plotting.plots import visualize, plot_scatter
from unot.models.nubot import (
    compute_loss_f,
    compute_loss_g,
    compute_loss_h,
    compute_weights_unbs,
    state_dict,
)
from unot.utils.helpers import load_item_from_save

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def load_lr_scheduler(optim, config):
    if "scheduler" not in config:
        return None

    return torch.optim.lr_scheduler.StepLR(optim, **config.scheduler)


def train_nubot(config, outdir):

    cachedir = outdir / "cache"

    # load networks, optimizers, data
    (f, g, h_fw, h_bw), opts, loader = load(config, restore=cachedir / "last.pt")
    data = cast_loader_to_iterator(loader, cycle_all=True)
    scheduler_f = load_lr_scheduler(opts.f, config)
    scheduler_g = load_lr_scheduler(opts.g, config)

    step = 0
    step = load_item_from_save(cachedir / "last.pt", "step", 0)

    if scheduler_f is not None and step > 0:
        scheduler_f.last_epoch = step
    if scheduler_g is not None and step > 0:
        scheduler_g.last_epoch = step

    n_iter = config.training.n_iter
    ticker = trange(step, n_iter, initial=step, total=n_iter)

    reg = config.sinkhorn.reg
    reg_m = config.sinkhorn.reg_m
    log_freq = config.training.get("log_freq", 300)

    for step in ticker:
        global EPOCH
        EPOCH = step

        for i in range(config.training.n_iter_inner):
            x = next(data.train.source)
            x.requires_grad_(True)

            y = next(data.train.target)
            y.requires_grad_(True)

            opts.h_fw.zero_grad()
            opts.h_bw.zero_grad()
            opts.g.zero_grad()

            yhat = g.transport(x)

            # unbalanced sinkhorn: map yhat on y
            # wtilde are weights of yhat (i.e. x)
            wtilde_yhat = compute_weights_unbs(yhat, y, reg, reg_m)
            loss_g = compute_loss_g(f, g, x, wtilde_yhat, transport=yhat)

            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                loss_g = loss_g + g.penalize_w()

            xhat = f.transport(y)

            # unbalanced sinkhorn: map xhat on x
            # wtilde are weights of xhat (i.e. y)
            wtilde_xhat = compute_weights_unbs(xhat, x, reg, reg_m)

            # logging & plotting (wandb)
            if step % log_freq == 0:
                wandb.log(
                    {
                        "wtilde_yhat_mean": wtilde_yhat.mean().item(),
                        "wtilde_xhat_mean": wtilde_xhat.mean().item(),
                        "wtilde_yhat_sum": wtilde_yhat.sum().item(),
                        "wtilde_xhat_sum": wtilde_xhat.sum().item(),
                    }
                )
                try:
                    w_final = h_fw(x) / h_bw(yhat)
                    if x.size(1) <= 2:
                        plot_weights(x, w_final, "w_final")
                    hist_weights(w_final.detach(), "w_final_hist")
                except:
                    print("Dividing h_fw(x) by h_bw(yhat) failed (or plotting).")
                    print(min(h_bw(yhat)))
                # if data is 2-dimensional, plot weights
                if x.size(1) <= 2:
                    plot_weights(x, wtilde_yhat, name="wtilde_yhat")
                    plot_weights(y, wtilde_xhat, name="wtilde_xhat")

                hist_weights(wtilde_yhat, "wtilde_yhat_hist")
                hist_weights(wtilde_xhat, "wtilde_xhat_hist")

            # predicted weights with eta and zeta
            what_yhat = h_fw(x)
            what_xhat = h_bw(y)

            # MSE loss for rescaling functions eta and zeta
            loss_h_fw = compute_loss_h(what_yhat, wtilde_yhat)
            loss_h_bw = compute_loss_h(what_xhat, wtilde_xhat)

            # update h (eta and zeta)
            loss_h_fw.backward()
            opts.h_fw.step()
            loss_h_bw.backward()
            opts.h_bw.step()

            # update g
            loss_g.backward(retain_graph=True)
            opts.g.step()

            check_loss(loss_g, loss_h_fw, loss_h_bw)

        # update f
        opts.f.zero_grad()
        loss_f = compute_loss_f(f, g, x, y, wtilde_yhat, wtilde_xhat)
        loss_f.backward()
        opts.f.step()
        f.clamp_w()

        # log losses
        if step % log_freq == 0:
            wandb.log(
                {
                    "g_loss": loss_g,
                    "h_fw_loss": loss_h_fw,
                    "h_bw_loss": loss_h_bw,
                    "f_loss": loss_f,
                }
            )

        check_loss(loss_f)

        if scheduler_f is not None:
            scheduler_f.step()
        if scheduler_g is not None:
            scheduler_g.step()

        # evaluate
        if step % log_freq == 0 or step == config.training.n_iter - 1:
            if config.data.type == "cell":
                viz = False
            else:
                viz = True
            evaluate(config, g, f, h_fw, h_bw, data, viz=viz)
            torch.save(
                state_dict(f, g, h_fw, h_bw, opts, step=step),
                cachedir / "model.pt",
            )

        if step % config.training.get("cache_freq", 100) == 0:
            torch.save(
                state_dict(f, g, h_fw, h_bw, opts, step=step), cachedir / "last.pt"
            )

    torch.save(state_dict(f, g, h_fw, h_bw, opts, step=step), cachedir / "last.pt")


def evaluate(config, g, f, h_fw, h_bw, data, viz=True):
    x = next(data.test.source)
    y = next(data.test.target)
    x.requires_grad_(True)
    T = g.transport(x)
    T = T.detach()
    what_yhat = h_fw(x).detach()
    what_xhat = h_bw(y).detach()

    # histogram of weights
    hist_weights(what_yhat, "what_yhat_hist")
    hist_weights(what_xhat, "what_xhat_hist")
    if x.size(1) <= 2:
        plot_weights(x, what_yhat, "what_yhat")

    # divide weigths of eta and zeta
    try:
        what_yhat = what_yhat / h_bw(T)
        hist_weights(what_yhat.detach(), "w_final_hist")
    except:
        print("Dividing h_fw(x) by h_bw(yhat) failed (or plotting).")
        print(min(h_bw(T)))

    log_dict = compute_metrics(x, y, T, eps=what_yhat)
    log_dict["epoch"] = EPOCH
    wandb.log(log_dict)

    if viz:
        visualize(x, y, T, config.data.type, eps=what_yhat, epoch=EPOCH)

        # check backward map
        y.requires_grad_(True)
        xhat = f.transport(y)
        _, plot = plot_scatter(y.detach(), x.detach(), xhat.detach())
        wandb.log({"backwards_scatter": wandb.Image(plot)})
        plt.close()

    # exit if w2 distance is too high, i.e., the model diverged
    if log_dict["w2_monge"] > 10000:
        print("w2_monge > 10000")
        exit()


def plot_weights(x, w, name="weights"):
    x = x.detach()
    w = w.detach()
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=w)
    plt.colorbar()
    plt.clim(0, 2)
    wandb.log({name: wandb.Image(plt)})
    plt.close()


def hist_weights(w, name="hist_weights"):
    if w.max() <= 2.0:
        bins = np.linspace(0, 2, num=20)
    else:
        bins = None
    plt.figure()
    plt.hist(w.squeeze(-1).numpy(), bins=bins, color="black")
    wandb.log({name: wandb.Image(plt)})
    plt.close()
