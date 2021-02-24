import numpy as np
import torch
from torch.optim import Adam

from torch_tools.data import TrainValLoader
from torch_tools.experiment import Experiment
from torch_tools.generic import ParameterDict
from torch_tools.functional import l1_norm, matrix_2_norm

# from torch_tools.model import Model
# from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer, MultiRegularizer
from transform_datasets.torch.vector import Translation

from local_experiments import TestExperiment


in_dim = 64
out_dim = 10


# Data
data_params = ParameterDict(
    {
        "n_classes": out_dim,
        "transformation": "translation",
        "max_transformation_steps": 10,
        "dim": in_dim,
        "noise": 0.2,
        "seed": 0,
    }
)
dataset = Translation(**data_params)

# Data Loader
batch_size = 32
data_loader = TrainValLoader(dataset, batch_size)

# Model
model_params = ParameterDict({"in_dim": in_dim, "out_dim": out_dim})
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=model_params.in_dim, out_features=model_params.out_dim)
)

# Optimizer
# Issue here: have to initialize Adam with model params / values to be optimized
# Do we want to do this outside or inside Expt?
optimizer_params = ParameterDict({"lr": 1e-3})
optimizer = Adam(model.parameters(), **optimizer_params)

# Regularizer
reg1 = Regularizer(l1_norm, "x", 0.7)
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
epochs = 20
experiment.train(data_loader, epochs)
