import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from unot.data.utils import cast_loader_to_iterator, cast_dataset_to_loader
from unot.utils.helpers import nest_dict


class GaussianDataset(IterableDataset):
    def __init__(self, mean=0, std=0.5):
        super(GaussianDataset).__init__()

        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std)

    def __iter__(self):
        return self.sample_generator()

    def __len__(self):
        # for completeness
        return 100000000

    def sample_generator(self):
        while True:
            sample = torch.normal(mean=self.mean, std=self.std).unsqueeze(dim=0)
            yield sample.squeeze(0)


def load_gaussian_data(
    config, perform_split=True, return_as="loader", include_model_kwargs=False
):

    if isinstance(return_as, str):
        return_as = [return_as]

    model_kwargs = {"input_dim": 2}

    if perform_split:
        dataset = {
            "train.source": GaussianDataset(config.data.x_mean, config.data.x_std),
            "train.target": GaussianDataset(config.data.y_mean, config.data.y_std),
            "test.source": GaussianDataset(config.data.x_mean, config.data.x_std),
            "test.target": GaussianDataset(config.data.y_mean, config.data.y_std),
        }

    else:
        dataset = {
            "source": GaussianDataset(config.data.x_mean, config.data.x_std),
            "target": GaussianDataset(config.data.y_mean, config.data.y_std),
        }

    dataset = nest_dict(dataset, as_dot_dict=True)

    kwargs = dict(config.data.loader)
    loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def get_iterable_gaussian_dataset(mean=0, std=0.1, batch_size=30):
    ds = GaussianDataset(mean, std)
    lo = DataLoader(ds, batch_size=batch_size)
    it = cast_loader_to_iterator(lo)
    return it


def load_g_data(config):
    x_mean = config["data"]["x_mean"]
    y_mean = config["data"]["y_mean"]
    x_std = config["data"]["x_std"]
    y_std = config["data"]["y_std"]
    batch_size = config["training"]["batch_size"]
    batch_size_test = config["training"]["batch_size_test"]

    x_iter = get_iterable_gaussian_dataset(x_mean, x_std, batch_size)
    y_iter = get_iterable_gaussian_dataset(y_mean, y_std, batch_size)
    x_test_iter = get_iterable_gaussian_dataset(x_mean, x_std, batch_size_test)
    y_test_iter = get_iterable_gaussian_dataset(y_mean, y_std, batch_size_test)

    # data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    return x_iter, y_iter, x_test_iter, y_test_iter


def compute_true_w2(config):
    w = 0.5 * np.sum(
        (np.array(config["data"]["x_mean"]) - np.array(config["data"]["y_mean"])) ** 2
    )
    return w
