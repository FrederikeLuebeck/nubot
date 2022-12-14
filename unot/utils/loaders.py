from unot.data.cell import load_cell_data
from unot.data.gaussians import load_gaussian_data
from unot.data.toy import load_toy_data

import unot.models


def load_data(config, **kwargs):
    if config.data.type == "gaussian":
        load_fxn = load_gaussian_data
    elif config.data.type == "toy":
        load_fxn = load_toy_data
    elif config.data.type == "cell":
        load_fxn = load_cell_data
    else:
        raise ValueError

    return load_fxn(config, **kwargs)


def load_model(config, restore=None, **kwargs):
    name = config.get("model.name", "")
    if name == "cellot" or name == "cellot_cycle" or name == "cellot_given_w":
        loadfxn = unot.models.load_cellot_model
    elif name == "nubot":
        loadfxn = unot.models.load_nubot_model
    elif name == "gan":
        loadfxn = unot.models.load_gan_model
    elif name == "gaussian":
        loadfxn = unot.models.load_gaussian_model
    else:
        raise ValueError

    return loadfxn(config, restore=restore, **kwargs)


def load(config, restore=True, include_model_kwargs=False, **kwargs):

    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)

    model, opt = load_model(config, restore=restore, **model_kwargs)

    if include_model_kwargs:
        return model, opt, loader, model_kwargs

    return model, opt, loader
