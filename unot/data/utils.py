from torch.utils.data import DataLoader


# def cast_loader_to_iterator(loader, cycle_all=True):
#     def cycle(iterable):
#         while True:
#             for x in iterable:
#                 yield x.squeeze(1)

#     if isinstance(loader, DataLoader):
#         return cycle(loader)

from unot.utils.helpers import nest_dict, flat_dict
from torch.utils.data import DataLoader, Dataset
from itertools import groupby
from absl import logging


def cast_dataset_to_loader(dataset, **kwargs):
    # check if dataset is torch.utils.data.Dataset
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, **kwargs)

    batch_size = kwargs.pop("batch_size", 1)
    # batch_sizes = {"source": batch_size, "target": int(batch_size * kwargs.pop("batch_size_target_factor", 1))}

    flat_dataset = flat_dict(dataset)

    # minimum_batch_size = {
    #     key: min(len(flat_dataset[key]), batch_sizes[key.split(".")[1]])
    #     for key in flat_dataset.keys()
    # }

    minimum_batch_size = {
        group: min(*map(lambda x: len(flat_dataset[x]), keys), batch_size)
        for group, keys in groupby(flat_dataset.keys(), key=lambda x: x.split(".")[0])
    }

    scaling_factor = kwargs.pop("batch_size_target_factor", 1.0)
    if scaling_factor <= 1.0:
        factor = {"source": 1, "target": scaling_factor}
    else:
        factor = {"source": 1 / scaling_factor, "target": 1}

    final_bs = {
        key: int(minimum_batch_size[key.split(".")[0]] * factor[key.split(".")[1]])
        for key in flat_dataset.keys()
    }
    # TODO: adjust for varying min_bs for source & target (cell data)

    # min_bs = min(minimum_batch_size.values())
    # if batch_size != final_bs["train.source"]:
    #     logging.warn(f"Batch size adapted due to dataset size.")
    print("Batch_sizes: ", final_bs)

    loader = nest_dict(
        {
            key: DataLoader(val, batch_size=final_bs[key], **kwargs)
            for key, val in flat_dataset.items()
        },
        as_dot_dict=True,
    )

    return loader


def cast_loader_to_iterator(loader, cycle_all=True):
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    if isinstance(loader, DataLoader):
        return cycle(loader)

    iterator = nest_dict(
        {key: cycle(item) for key, item in flat_dict(loader).items()}, as_dot_dict=True
    )

    for value in flat_dict(loader).values():
        assert len(value) > 0

    return iterator
