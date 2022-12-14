import wandb
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from unot.utils.loaders import load
from unot.utils.helpers import load_item_from_save
from unot.models.gan import compute_loss_D, compute_loss_G, state_dict

from unot.plotting.plots import visualize
from unot.data.utils import cast_loader_to_iterator
from unot.training.eval import compute_metrics
from unot.training.train_nubot import hist_weights


def check_loss(*args):
    for arg in args:
        if torch.isnan(arg):
            raise ValueError


def train_unbalanced_gan(config, outdir):

    cachedir = outdir / "cache"

    (D, G), opts, loader = load(config, restore=cachedir / "last.pt")
    data = cast_loader_to_iterator(loader, cycle_all=True)

    step = 0
    step = load_item_from_save(cachedir / "last.pt", "step", 0)
    n_iter = config.training.n_iter
    ticker = trange(step, n_iter, initial=step, total=n_iter)

    # lambdas (defaults as in code from paper)
    lamb0 = config.training.get("lamb0", 0.001)
    lamb1 = config.training.get("lamb1", 1.0)
    lamb2 = config.training.get("lamb2", 1.0)
    lambG = config.training.get("lambG", 10)
    lambG2 = config.training.get("lambG2", 0.01)

    critic_iter = 2

    gradient_penalty = config.training.get("gradient_penalty", False)

    for step in ticker:
        x = next(data.train.source)
        y = next(data.train.target)

        # update discriminator
        for i in range(config.training.n_iter_D):
            G.eval()
            D.train()
            opts.D.zero_grad()
            T, eps = G(x)
            loss_D = compute_loss_D(
                D, x, y, T, eps, config.training.psi, gradient_penalty, lambG
            )
            loss_D.backward()
            opts.D.step()

            check_loss(loss_D)

            wandb.log({"loss_D": loss_D})

        # update generator
        for i in range(config.training.n_iter_G):
            D.eval()
            G.train()
            opts.G.zero_grad()
            T, eps = G(x)
            loss_G = compute_loss_G(
                D,
                G,
                x,
                T,
                eps,
                config.training.c1,
                config.training.c2,
                config.training.psi,
                lamb0,
                lamb1,
                lamb2,
                gradient_penalty,
                lambG2,
            )
            loss_G.backward()
            opts.G.step()

            check_loss(loss_G)

            wandb.log({"loss_G": loss_G})

        if step % config.training.get("log_freq", 500) == 0:
            plot_weights(x, eps)
            hist_weights(eps.detach(), "weights_hist")

        log_freq = config.training.get("log_freq", 500)
        if step % log_freq == 0 or step == config.training.n_iter - 1:
            evaluate(config, G, D, data, viz=True)
            torch.save(
                state_dict(D, G, opts, step=step),
                cachedir / "model.pt",
            )

        if step % config.training.get("cache_freq", 500) == 0:
            torch.save(state_dict(D, G, opts, step=step), cachedir / "last.pt")


def evaluate(config, G, D, data, viz=True):
    x = next(data.test.source)
    y = next(data.test.target)
    T, eps = G(x)
    T = T.detach()
    eps = eps.detach()

    log_dict = compute_metrics(x, y, T, eps=eps)

    wandb.log(log_dict)
    if viz:
        visualize(x, y, T, config.data.type, eps)


def plot_weights(x, w, name="weights"):
    x = x.detach()
    w = w.detach()
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=w)
    plt.colorbar()
    wandb.log({name: wandb.Image(plt)})
    plt.close()
