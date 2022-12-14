from unot.training.train_cellot import train_cellot
from unot.training.train_unbalanced_gan import train_unbalanced_gan

from unot.training.train_nubot import train_nubot
from unot.training.fit_gaussian_approx import fit_gaussian_approx

FUNS = {
    "cellot": train_cellot,
    "gan": train_unbalanced_gan,
    "nubot": train_nubot,
    "gaussian": fit_gaussian_approx,
}
