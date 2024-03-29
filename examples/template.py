import numpy as np
from sklearn.datasets import make_classification
import torch
from torch.optim import Adam

from torch_tools.data import TrainValLoader, DatasetWrapper
from torch_tools.experiment import Experiment
from torch_tools.generic import ParameterDict
from torch_tools.functional import l1_norm

# from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer, MultiRegularizer
from transform_datasets.torch.vector import Translation

from local_experiments import TestExperiment
from local_models import SimpleModel


in_dim = 20
out_dim = 3


# Data
# data_params = ParameterDict(
#     {
#         "n_classes": out_dim,
#         "transformation": "translation",
#         "max_transformation_steps": 10,
#         "dim": in_dim,
#         "noise": 0.2,
#         "seed": 0,
#     }
# )
# dataset = Translation(**data_params)

data_params = ParameterDict(
    {
        "n_samples": 1000,
        "n_features": in_dim,
        "n_classes": out_dim,
        "n_clusters_per_class": 1,
    }
)
X, y = make_classification(**data_params)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
dataset = DatasetWrapper(X, y)


# Data Loader
batch_size = 64
data_loader = TrainValLoader(dataset, batch_size)

# Model
model_params = ParameterDict({"in_dim": in_dim, "out_dim": out_dim})
model = SimpleModel(**model_params)

# Optimizer
# Issue here: have to initialize Adam with model params / values to be optimized
# Do we want to do this outside or inside Expt?
optimizer_params = ParameterDict({"lr": 1e-3})
optimizer = Adam(model.parameters(), **optimizer_params)

# Regularizer
reg1 = Regularizer(l1_norm, ["yh"], 15.0)
regularizer = MultiRegularizer([reg1])


# Loss
loss_function = torch.nn.CrossEntropyLoss()

# Experiment
experiment_name = "test"
device = "cpu"

experiment = TestExperiment(
    experiment_name, model, optimizer, regularizer, loss_function, device
)

# TRAIN
epochs = 50
experiment.train(data_loader, epochs)
