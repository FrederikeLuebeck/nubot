import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from unot.data.utils import cast_loader_to_iterator
from unot.utils.helpers import nest_dict
from unot.data.utils import cast_dataset_to_loader


class ToyDataset(IterableDataset):
    def __init__(self, name, scale=5.0, sigma=0.71):
        self.scale = scale
        self.sigma = sigma
        self.name = name
        self.p = None

        if self.name == "simple":
            centers = np.array([0, 0])
            self.sigma = 1

        elif self.name == "circle":
            centers = np.array(
                [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                    (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                    (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                    (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                ]
            )

        elif self.name == "square_five":
            centers = np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]) + 1.1

        elif self.name == "square_four":

            centers = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]) + 1.1

        elif self.name == "gaussian_mixture_lower":
            centers = np.array([[1, 1], [3, 1]])
            self.p = np.array([0.6, 0.4])

        elif self.name == "gaussian_mixture_left_balanced":
            centers = np.array([[1, 1], [1, 3]])
            self.p = np.array([0.5, 0.5])

        elif self.name == "gaussian_mixture_right":
            centers = np.array([[2, 1], [2, 3]])
            self.p = np.array([0.8, 0.2])

        elif self.name == "three":
            centers = np.array([[0, 0], [1, 1], [0, 2]]) + 2
            self.p = np.array([0.45, 0.45, 0.1])

        elif self.name == "two":
            shift = 0.3
            centers = np.array([[0, 0], [1, 1]]) + shift + 2
            self.p = np.array([0.6, 0.4])

        elif self.name == "inner_triangle_balanced":
            centers = np.array([[0, -1], [1, 1], [-1, 1]])

        elif self.name == "outer_triangle_balanced":
            centers = np.array([[0, -1.5], [1.5, 1.5], [-1.5, 1.5]])

        elif self.name == "outer_triangle_unbalanced_1":
            centers = np.array([[0, -1.5], [1.5, 1.5], [-1.5, 1.5]])
            self.p = np.array([0.45, 0.45, 0.1])

        elif self.name == "outer_triangle_unbalanced_2":
            centers = np.array([[0, -1.5], [1.5, 1.5], [-1.5, 1.5]])
            self.p = np.array([0.5, 0.5, 0])

        elif self.name == "outer_triangle_unbalanced_3":
            centers = np.array([[0, -1.5], [1.5, 1.5], [-1.5, 1.5]])
            self.p = np.array([0.7, 0.2, 0.1])

        elif self.name == "inner_triangle_unbalanced_1_flipped":
            centers = np.array([[0, -1], [1, 1], [-1, 1]])
            self.p = np.array([0.1, 0.45, 0.45])

        elif self.name == "gaussian_upper":
            centers = np.array([[1.0, 2.0]])

        elif self.name == "gaussian_lower":
            centers = np.array([[1.0, 1.0]])

        else:
            raise NotImplementedError()

        self.centers = centers

    def __iter__(self):
        return self.create_sample_generators()

    def __len__(self):
        # for completeness
        return 10000000

    def create_sample_generators(self):
        # given name of dataset, select centers
        # create generator which randomly picks center and adds noise
        centers = self.scale * self.centers
        while True:
            center = centers[np.random.choice(len(centers), p=self.p)]
            point = center + self.sigma**2 * np.random.randn(2)

            yield np.float32(point)

    # def generate_finite_sample(self):
    #     # create "exact" sample (exact proportions of clusters)
    #     centers = self.scale * self.centers
    #     n = 400
    #     sample = []
    #     for i in len(centers):
    #         ncl = np.floor(self.p[i] * n)


def load_toy_data(
    config, perform_split=True, return_as="loader", include_model_kwargs=False
):

    if isinstance(return_as, str):
        return_as = [return_as]

    model_kwargs = {"input_dim": 2}

    if perform_split:
        dataset = {
            "train.source": ToyDataset(config.data.source),
            "train.target": ToyDataset(config.data.target),
            "test.source": ToyDataset(config.data.source),
            "test.target": ToyDataset(config.data.target),
        }

    else:
        dataset = {
            "source": ToyDataset(config.data.source),
            "target": ToyDataset(config.data.source),
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


def get_iterable_toy_dataset(name, batch_size=30):
    ds = ToyDataset(name=name)
    lo = DataLoader(ds, batch_size=batch_size)
    it = cast_loader_to_iterator(lo)
    return it
