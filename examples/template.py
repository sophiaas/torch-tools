import numpy as np
import torch
from torch.optim import Adam

from torch_tools.experiment import Experiment
from torch_tools.functional import l1_norm, matrix_2_norm
from torch_tools.generic import ParameterDict
from torch_tools.model import Model
from torch_tools.optimizer import Optimizer
from torch_tools.regularizer import Regularizer
from transform_dataset.torch.vector import Translation


# Data
data_params = ParameterDict(
    {
        "c_classes": 10,
        "digits": np.arange(10),
        "max_transformation_steps": 10,
        "dim": 32,
        "noise": 0.2,
        "seed": 0,
    }
)
#
# dataset = Translation(n_classes=args.n_classes,
#                       transformation=args.dataset,
#                       max_transformation_steps=args.max_transformation_steps,
#                       dim=args.dim,
#                       seed=args.seed,
#                       noise=args.noise)
# dataset = Translation
#
# # Optimizer
# optimizer_params = ParameterDict({"lr": 1e-3})
# optimizer = Adam(**optimizer_params)
#
# # Regularizer
# # reg1 = Regularizer(l1_norm, "s", 0.7)
# # reg2 = Regularizer(matrix_2_norm, "A", 0.1)
# # regularizer = MultiRegularizer([reg1, reg2])
# regularizer = None
#
# # Model
# model_params = ParameterDict({"n_neurons": 64})
# model = torch.nn.Sequential([torch.nn.Linear(model_params.n_neurons)])
#
# # Loss
# loss_function = torch.nn.CrossEntropyLoss()
#
# # Experiment
# experiment_name = "test"
# device = "cpu"
#
# experiment = TestExperiment(
#     experiment_name, model, optimizer, regularizer, loss_function, device
# )
#
# # TRAIN
# experiment.train()
