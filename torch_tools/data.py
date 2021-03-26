import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np


class Base(Dataset):
    def __init__(self):
        self.name = None
        self.dim = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


class DatasetWrapper(Dataset):
    def __init__(self, X, Y=None):
        self.data = X
        self.name = X.__class__.__name__

        if Y is not None:
            self.labels = Y

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)


class TrainValLoader:
    def __init__(self, dataset, batch_size, fraction_val=0.2, seed=0, num_workers=0):
        if fraction_val > 1.0 or fraction_val < 0.0:
            raise ValueError("fraction_val must be a fraction between 0 and 1")

        self.name = dataset.name
        self.batch_size = batch_size
        self.fraction_val = fraction_val
        self.seed = seed
        self.num_workers = num_workers
        self.dataset = type(dataset)

        if fraction_val > 0.0:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(fraction_val * len(dataset)))

            np.random.seed(seed)
            np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            self.val = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=num_workers,
            )

        else:

            self.val = None
            train_sampler = None

        self.train = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )
