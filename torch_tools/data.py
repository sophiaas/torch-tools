import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import copy


class Base(Dataset):
    '''Not in use: August 3, 2021'''
    def __init__(self):
        self.name = None
        self.dim = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


class DatasetWrapper(Dataset):
    '''Not in use: August 3, 2021'''
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
    def __init__(self, batch_size, fraction_val=0.2, num_workers=0, seed=0):
        assert (
            fraction_val <= 1.0 and fraction_val >= 0.0
        ), "fraction_val must be a fraction between 0 and 1"

        np.random.seed(seed)

        self.batch_size = batch_size
        self.fraction_val = fraction_val
        self.seed = seed
        self.num_workers = num_workers

    def load(self, dataset):

        if self.fraction_val > 0.0:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.fraction_val * len(dataset)))

            np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            self.val = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=valid_sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )

        else:

            self.val = None
            train_sampler = None

        self.train = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
#     def to(self, device):
#         for k in dir(self.train.dataset):
#             v = getattr(self.train.dataset, k)
#             if type(v) == torch.Tensor:
#                 setattr(self.train.dataset, k, v.to(device))
#         if self.val is not None:
#             for k in dir(self.val.dataset):
#                 v = getattr(self.val.dataset, k)
#                 if type(v) == torch.Tensor:
#                     setattr(self.val.dataset, k, v.to(device))
                    
                    
    
# class TrainValLoader(torch.nn.Module):
#     def __init__(self, batch_size, fraction_val=0.2, num_workers=0, seed=0):
#         assert (
#             fraction_val <= 1.0 and fraction_val >= 0.0
#         ), "fraction_val must be a fraction between 0 and 1"

#         np.random.seed(seed)

#         self.batch_size = batch_size
#         self.fraction_val = fraction_val
#         self.seed = seed
#         self.num_workers = num_workers

#     def load(self, dataset):

#         if self.fraction_val > 0.0:
#             val_dataset = copy.deepcopy(dataset)
#             dataset_size = len(dataset)
#             indices = list(range(dataset_size))
#             split = int(np.floor(self.fraction_val * len(dataset)))

#             np.random.shuffle(indices)

#             train_indices, val_indices = indices[split:], indices[:split]
#             val_dataset.data = dataset.data[val_indices]
#             val_dataset.labels = dataset.data[val_indices]
            
#             self.val = torch.utils.data.DataLoader(
#                 dataset[val_indices],
#                 batch_size=self.batch_size,
#                 num_workers=self.num_workers,
#             )

#         else:

#             self.val = None
#             train_indices = np.arange(len(dataset))
            
#         import pdb; pdb.set_trace()

#         self.train = torch.utils.data.DataLoader(
#             dataset[train_indices],
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#         )
        
#     def to(self, device):
        
#         self.train.dataset.data = self.train.dataset.data.to(device)
#         self.train.dataset.labels = self.train.dataset.labels.to(device)
        
#         if self.val is not None:
#             self.val.dataset.data = self.val.dataset.data.to(device)
#             self.val.dataset.labels = self.val.dataset.labels.to(device)
